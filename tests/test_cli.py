from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from multishell.__main__ import build_parser, main


def test_auto_login_requires_agent_or_all() -> None:
    parser = build_parser()
    args = parser.parse_args(["auto-login"])
    with pytest.raises(SystemExit, match="provide an agent name or pass --all"):
        args.func(args)


def test_login_opens_account_editor(monkeypatch) -> None:
    parser = build_parser()
    called: dict[str, bool] = {"writer": False, "editor": False}

    monkeypatch.setattr("multishell.__main__._write_default_config", lambda path, force: called.__setitem__("writer", True))
    monkeypatch.setattr("multishell.__main__.dotenv_path", lambda: Path("/tmp/multishell.env"))
    monkeypatch.setattr("multishell.account_login.run_login_editor", lambda: called.__setitem__("editor", True) or 0)

    args = parser.parse_args(["login"])

    assert args.func(args) == 0
    assert called == {"writer": True, "editor": True}


def test_auth_login_for_codex_agent_suppresses_node_warnings(monkeypatch, tmp_path) -> None:
    parser = build_parser()
    captured: dict[str, object] = {}

    monkeypatch.setattr("multishell.__main__.all_agent_names", lambda: ["worker-1"])
    monkeypatch.setattr("multishell.__main__.WORKER_SPECS", (SimpleNamespace(name="worker-1"),))
    monkeypatch.setattr("multishell.__main__.CLAUDE_WORKER_SPECS", ())
    monkeypatch.setattr("multishell.__main__.credential_source_agent", lambda agent: agent)
    monkeypatch.setattr("multishell.__main__.ensure_account_home", lambda *args, **kwargs: None)
    monkeypatch.setattr("multishell.__main__.account_home", lambda _agent: tmp_path / "worker-1")
    monkeypatch.setenv("NODE_NO_WARNINGS", "0")

    def fake_run(cmd, check, env):
        captured["cmd"] = cmd
        captured["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("multishell.__main__.subprocess.run", fake_run)

    args = parser.parse_args(["auth-login", "worker-1"])

    assert args.func(args) == 0
    assert captured["cmd"] == ["codex", "login", "--device-auth"]
    assert captured["env"]["HOME"] == str(tmp_path / "worker-1")
    assert captured["env"]["NODE_NO_WARNINGS"] == "1"


def test_auth_login_for_claude_agent_suppresses_node_warnings(monkeypatch, tmp_path) -> None:
    parser = build_parser()
    captured: dict[str, object] = {}

    monkeypatch.setattr("multishell.__main__.all_agent_names", lambda: ["claude-worker-1"])
    monkeypatch.setattr("multishell.__main__.WORKER_SPECS", ())
    monkeypatch.setattr(
        "multishell.__main__.CLAUDE_WORKER_SPECS",
        (SimpleNamespace(name="claude-worker-1", account_email="worker@example.com"),),
    )
    monkeypatch.setattr("multishell.__main__.credential_source_agent", lambda agent: agent)
    monkeypatch.setattr("multishell.__main__.ensure_claude_account_home", lambda *args, **kwargs: None)
    monkeypatch.setattr("multishell.__main__.claude_account_home", lambda _agent: tmp_path / "claude-worker-1")
    monkeypatch.setenv("NODE_NO_WARNINGS", "0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")

    def fake_run(cmd, check, env):
        captured["cmd"] = cmd
        captured["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("multishell.__main__.subprocess.run", fake_run)

    args = parser.parse_args(["auth-login", "claude-worker-1"])

    assert args.func(args) == 0
    assert captured["cmd"] == ["claude", "auth", "login", "--email", "worker@example.com"]
    assert captured["env"]["HOME"] == str(tmp_path / "claude-worker-1")
    assert captured["env"]["NODE_NO_WARNINGS"] == "1"
    assert "ANTHROPIC_API_KEY" not in captured["env"]


def test_logout_requires_agent_or_all() -> None:
    parser = build_parser()
    args = parser.parse_args(["logout"])
    with pytest.raises(SystemExit, match="provide an agent name or pass --all"):
        args.func(args)


def test_logout_clears_single_agent_state(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    auth_file = tmp_path / "worker-1-auth.json"
    browser_dir = tmp_path / "browser-profiles" / "worker-1"
    auth_file.write_text("{}", encoding="utf-8")
    browser_dir.mkdir(parents=True)

    monkeypatch.setattr("multishell.__main__.all_agent_names", lambda: ["worker-1"])
    monkeypatch.setattr("multishell.__main__.logout_paths", lambda agent: [auth_file, browser_dir] if agent == "worker-1" else [])

    args = parser.parse_args(["logout", "worker-1"])

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "cleared auth state for worker-1" in out
    assert not auth_file.exists()
    assert not browser_dir.exists()


def test_logout_all_clears_browser_root_and_deduplicates_paths(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    shared_auth = tmp_path / "shared-auth.json"
    worker_browser = tmp_path / "browser-profiles" / "worker-1"
    claude_browser = tmp_path / "browser-profiles" / "claude-worker-1"
    shared_auth.write_text("{}", encoding="utf-8")
    worker_browser.mkdir(parents=True)
    claude_browser.mkdir(parents=True)

    monkeypatch.setattr("multishell.__main__.all_agent_names", lambda: ["worker-1", "claude-worker-1"])
    monkeypatch.setattr(
        "multishell.__main__.logout_paths",
        lambda agent: [shared_auth, worker_browser] if agent == "worker-1" else [shared_auth, claude_browser],
    )
    monkeypatch.setattr("multishell.__main__.browser_profile_path", lambda _agent: tmp_path / "browser-profiles" / _agent)

    args = parser.parse_args(["logout", "--all"])

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "cleared auth state for all configured agents (2)" in out
    assert not shared_auth.exists()
    assert not (tmp_path / "browser-profiles").exists()


def test_run_fails_when_any_agent_login_is_missing(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    monkeypatch.setattr("multishell.__main__._maybe_reexec_into_venv", lambda _module: False)
    monkeypatch.setattr("multishell.__main__.check_startup_update", lambda: SimpleNamespace(message=None, restart_python=None))
    monkeypatch.setattr("multishell.__main__.state_root", lambda: tmp_path / ".multishell")
    monkeypatch.setattr("multishell.__main__.ensure_runtime_environment", lambda **_kwargs: "instance-1")
    monkeypatch.setattr("multishell.__main__.cleanup_stale_runtime", lambda: [])
    monkeypatch.setattr("multishell.__main__.ensure_agent_home", lambda *args, **kwargs: None)
    monkeypatch.setattr("multishell.__main__.ensure_claude_home", lambda *args, **kwargs: None)
    monkeypatch.setattr("multishell.__main__.missing_codex_logins", lambda: [])
    monkeypatch.setattr("multishell.__main__.missing_claude_logins", lambda: ["claude-worker-2"])

    args = parser.parse_args(["run"])

    assert args.func(args) == 1
    out = capsys.readouterr().out
    assert "missing Claude login for: claude-worker-2" in out
    assert "multishell refuses to start until every configured agent is logged in" in out


def test_run_prompts_for_startup_update_and_installs_on_yes(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    installed: dict[str, object] = {}

    monkeypatch.setattr("multishell.__main__._maybe_reexec_into_venv", lambda _module: False)
    monkeypatch.setattr("multishell.__main__.state_root", lambda: tmp_path / ".multishell")
    monkeypatch.setattr("multishell.__main__.ensure_runtime_environment", lambda **_kwargs: "instance-1")
    monkeypatch.setattr(
        "multishell.__main__.check_startup_update",
        lambda: SimpleNamespace(
            message="update available: 0.1.2 -> 0.1.3 (v0.1.3). Run `multishell update`.",
            restart_python=None,
            available_release=SimpleNamespace(tag_name="v0.1.3", version="0.1.3"),
        ),
    )
    monkeypatch.setattr("multishell.__main__._can_prompt_for_update", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: installed.__setitem__("prompt", prompt) or "y")
    monkeypatch.setattr("multishell.__main__._install_root", lambda: tmp_path / "install")
    monkeypatch.setattr("multishell.__main__._bin_dir", lambda: tmp_path / "bin")
    monkeypatch.setattr(
        "multishell.__main__.install_release",
        lambda release, **kwargs: installed.update({"release": release, "kwargs": kwargs}) or (tmp_path / "install" / "releases" / "0.1.3"),
    )

    args = parser.parse_args(["run"])

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "Install this update now and exit?" in installed["prompt"]
    assert installed["release"].tag_name == "v0.1.3"
    assert installed["kwargs"]["target_install_root"] == tmp_path / "install"
    assert installed["kwargs"]["target_bin_dir"] == tmp_path / "bin"
    assert "installed v0.1.3" in out
    assert "restart it to use the new version" in out


def test_run_continues_when_startup_update_is_declined(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    install_called = {"value": False}

    monkeypatch.setattr("multishell.__main__._maybe_reexec_into_venv", lambda _module: False)
    monkeypatch.setattr("multishell.__main__.state_root", lambda: tmp_path / ".multishell")
    monkeypatch.setattr("multishell.__main__.ensure_runtime_environment", lambda **_kwargs: "instance-1")
    monkeypatch.setattr("multishell.__main__.cleanup_stale_runtime", lambda: [])
    monkeypatch.setattr("multishell.__main__.ensure_agent_home", lambda *args, **kwargs: None)
    monkeypatch.setattr("multishell.__main__.ensure_claude_home", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "multishell.__main__.check_startup_update",
        lambda: SimpleNamespace(
            message="update available: 0.1.2 -> 0.1.3 (v0.1.3). Run `multishell update`.",
            restart_python=None,
            available_release=SimpleNamespace(tag_name="v0.1.3", version="0.1.3"),
        ),
    )
    monkeypatch.setattr("multishell.__main__._can_prompt_for_update", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    monkeypatch.setattr("multishell.__main__.install_release", lambda *args, **kwargs: install_called.__setitem__("value", True))
    monkeypatch.setattr("multishell.__main__.missing_codex_logins", lambda: [])
    monkeypatch.setattr("multishell.__main__.missing_claude_logins", lambda: ["claude-worker-2"])

    args = parser.parse_args(["run"])

    assert args.func(args) == 1
    assert install_called["value"] is False
    out = capsys.readouterr().out
    assert "missing Claude login for: claude-worker-2" in out


def test_check_update_prints_available_release(monkeypatch, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    monkeypatch.setattr("multishell.__main__.check_for_update", lambda repo=None: SimpleNamespace(version="0.2.0", tag_name="v0.2.0", html_url="https://example.invalid/release"))
    monkeypatch.setattr("multishell.__main__.current_version", lambda: "0.1.0")

    args = parser.parse_args(["check-update"])

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "update available: 0.1.0 -> 0.2.0 (v0.2.0)" in out
    assert "https://example.invalid/release" in out


def test_update_installs_latest_release(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    monkeypatch.setattr("multishell.__main__._install_root", lambda: tmp_path / "install")
    monkeypatch.setattr("multishell.__main__._bin_dir", lambda: tmp_path / "bin")
    monkeypatch.setattr(
        "multishell.__main__.install_latest_release",
        lambda **kwargs: SimpleNamespace(tag_name="v0.2.0"),
    )
    monkeypatch.setattr("multishell.__main__.current_release_version", lambda _root: "0.2.0")

    args = parser.parse_args(["update"])

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "installed v0.2.0" in out
    assert "wrapper now points to release 0.2.0" in out


def test_rollback_reports_selected_version(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    monkeypatch.setattr("multishell.__main__._install_root", lambda: tmp_path / "install")
    monkeypatch.setattr("multishell.__main__._bin_dir", lambda: tmp_path / "bin")
    monkeypatch.setattr("multishell.__main__.rollback_to_version", lambda **kwargs: "0.1.0")

    args = parser.parse_args(["rollback"])

    assert args.func(args) == 0
    assert "rolled back multishell to 0.1.0" in capsys.readouterr().out


def test_init_config_writes_template(monkeypatch, tmp_path: Path) -> None:
    parser = build_parser()
    target = tmp_path / ".multishell" / ".env"

    monkeypatch.setattr("multishell.__main__.dotenv_path", lambda: target)
    monkeypatch.setattr("multishell.__main__.dotenv_template_text", lambda: "MULTISHELL_MANAGER_EMAIL=manager@example.com\n")

    args = parser.parse_args(["init-config"])

    assert args.func(args) == 0
    assert target.read_text(encoding="utf-8") == "MULTISHELL_MANAGER_EMAIL=manager@example.com\n"


def test_install_browser_runs_playwright_install(monkeypatch) -> None:
    parser = build_parser()
    captured: dict[str, object] = {}

    monkeypatch.setattr("multishell.__main__._maybe_reexec_into_venv", lambda _module: False)
    monkeypatch.setenv("NODE_NO_WARNINGS", "0")
    monkeypatch.setenv("MULTISHELL_INSTALL_ROOT", "/tmp/multishell-install")

    def fake_run(cmd, check, env):
        captured["cmd"] = cmd
        captured["check"] = check
        captured["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("multishell.__main__.subprocess.run", fake_run)

    args = parser.parse_args(["install-browser"])

    assert args.func(args) == 0
    assert captured["cmd"][1:] == ["-m", "playwright", "install", "chromium"]
    assert captured["check"] is True
    assert captured["env"]["NODE_NO_WARNINGS"] == "1"
    assert captured["env"]["PLAYWRIGHT_BROWSERS_PATH"] == "/tmp/multishell-install/playwright-browsers"


def test_install_browser_can_request_system_deps(monkeypatch) -> None:
    parser = build_parser()
    captured: dict[str, object] = {}

    monkeypatch.setattr("multishell.__main__._maybe_reexec_into_venv", lambda _module: False)

    def fake_run(cmd, check, env):
        captured["cmd"] = cmd
        captured["check"] = check
        captured["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("multishell.__main__.subprocess.run", fake_run)

    args = parser.parse_args(["install-browser", "--with-deps"])

    assert args.func(args) == 0
    assert captured["cmd"][1:] == ["-m", "playwright", "install", "--with-deps", "chromium"]


def test_install_llama_cpp_runs_installer(monkeypatch) -> None:
    parser = build_parser()
    called: dict[str, object] = {"force": None}

    monkeypatch.setattr(
        "multishell.llama_cpp.install_llama_cpp",
        lambda *, force=False: called.__setitem__("force", force) or {},
    )

    args = parser.parse_args(["install-llama-cpp"])

    assert args.func(args) == 0
    assert called["force"] is False


def test_install_auth_model_runs_installer(monkeypatch) -> None:
    parser = build_parser()
    called: dict[str, object] = {"force": None}

    monkeypatch.setattr(
        "multishell.llama_cpp.install_auth_model",
        lambda *, force=False: called.__setitem__("force", force) or {},
    )

    args = parser.parse_args(["install-auth-model"])

    assert args.func(args) == 0
    assert called["force"] is False


def test_install_auth_model_passes_force(monkeypatch) -> None:
    parser = build_parser()
    called: dict[str, object] = {"force": None}

    monkeypatch.setattr(
        "multishell.llama_cpp.install_auth_model",
        lambda *, force=False: called.__setitem__("force", force) or {},
    )

    args = parser.parse_args(["install-auth-model", "--force"])

    assert args.func(args) == 0
    assert called["force"] is True


def test_auth_model_smoke_test_runs_requested_mode(monkeypatch) -> None:
    parser = build_parser()
    captured: dict[str, object] = {}
    entered: dict[str, bool] = {"value": False}

    class _ManagedServer:
        def __enter__(self):
            entered["value"] = True
            return True

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("multishell.__main__._maybe_reexec_into_venv", lambda _module: False)
    monkeypatch.setattr("multishell.llama_cpp.managed_auth_model_server", lambda **_kwargs: _ManagedServer())
    monkeypatch.setattr(
        "multishell.auth_flow_model.run_auth_model_smoke_test",
        lambda *, headed: captured.__setitem__("headed", headed),
    )

    args = parser.parse_args(["auth-model-smoke-test", "--headed"])

    assert args.func(args) == 0
    assert entered["value"] is True
    assert captured["headed"] is True


def test_auth_model_bench_prints_summary(monkeypatch, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    monkeypatch.setattr(
        "multishell.llama_cpp.bench_auth_model",
        lambda **_kwargs: SimpleNamespace(
            model_path=Path("/tmp/model.gguf"),
            prompt_tokens_per_second=150.5,
            decode_tokens_per_second=11.2,
            prompt_tokens=128,
            decode_tokens=64,
            runs=3,
        ),
    )

    args = parser.parse_args(["auth-model-bench"])

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "prompt_tps=150.50" in out
    assert "decode_tps=11.20" in out


def test_auto_login_manages_local_auth_model_server(monkeypatch) -> None:
    parser = build_parser()
    captured: dict[str, object] = {"entered": False, "credentials": None, "logger": None}

    class _ManagedServer:
        def __enter__(self):
            captured["entered"] = True
            return True

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("multishell.__main__._maybe_reexec_into_venv", lambda _module: False)
    monkeypatch.setattr("multishell.__main__.apply_node_warning_suppression", lambda: None)
    monkeypatch.setattr("multishell.__main__.all_agent_names", lambda: ["worker-1"])
    monkeypatch.setattr("multishell.__main__.missing_email_env_vars", lambda _agents: [])
    monkeypatch.setattr("multishell.autologin.resolve_credentials", lambda _agents: ["cred"])
    monkeypatch.setattr(
        "multishell.llama_cpp.managed_auth_model_server",
        lambda **kwargs: captured.__setitem__("logger", kwargs["log"]) or _ManagedServer(),
    )
    monkeypatch.setattr(
        "multishell.autologin.run_auto_login",
        lambda credentials, headed, timeout_seconds, max_parallel, verbose=False: captured.__setitem__(
            "credentials",
            (credentials, headed, timeout_seconds, max_parallel, verbose),
        ),
    )

    args = parser.parse_args(["auto-login", "--all"])

    assert args.func(args) == 0
    assert captured["entered"] is True
    assert captured["credentials"] == (["cred"], False, 180, 1, False)


def test_auto_login_passes_verbose_flag(monkeypatch) -> None:
    parser = build_parser()
    captured: dict[str, object] = {"verbose": None}

    class _ManagedServer:
        def __enter__(self):
            return True

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("multishell.__main__._maybe_reexec_into_venv", lambda _module: False)
    monkeypatch.setattr("multishell.__main__.apply_node_warning_suppression", lambda: None)
    monkeypatch.setattr("multishell.__main__.all_agent_names", lambda: ["worker-1"])
    monkeypatch.setattr("multishell.__main__.missing_email_env_vars", lambda _agents: [])
    monkeypatch.setattr("multishell.autologin.resolve_credentials", lambda _agents: ["cred"])
    monkeypatch.setattr("multishell.llama_cpp.managed_auth_model_server", lambda **_kwargs: _ManagedServer())
    monkeypatch.setattr(
        "multishell.autologin.run_auto_login",
        lambda credentials, headed, timeout_seconds, max_parallel, verbose=False: captured.__setitem__("verbose", verbose),
    )

    args = parser.parse_args(["auto-login", "--all", "--verbose"])

    assert args.func(args) == 0
    assert captured["verbose"] is True


def test_uninstall_removes_wrapper_and_install_root(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    install_root = tmp_path / "install"
    bin_dir = tmp_path / "bin"
    wrapper = bin_dir / "multishell"
    state_dir = tmp_path / "state"
    install_root.mkdir()
    bin_dir.mkdir()
    state_dir.mkdir()
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr("multishell.__main__.state_root", lambda: state_dir)

    args = parser.parse_args(
        [
            "uninstall",
            "--yes",
            "--install-root",
            str(install_root),
            "--bin-dir",
            str(bin_dir),
        ]
    )

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert f"removed {wrapper}" in out
    assert f"removed {install_root}" in out
    assert f"removed {state_dir}" in out
    assert not wrapper.exists()
    assert not install_root.exists()
    assert not state_dir.exists()


def test_uninstall_can_keep_state(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    monkeypatch.setattr("multishell.__main__.state_root", lambda: state_dir)

    args = parser.parse_args(["uninstall", "--yes", "--keep-state"])

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert f"removed {state_dir}" not in out
    assert state_dir.exists()


def test_uninstall_aborts_without_confirmation(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    install_root = tmp_path / "install"
    install_root.mkdir()
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    args = parser.parse_args(["uninstall", "--install-root", str(install_root)])

    assert args.func(args) == 1
    out = capsys.readouterr().out
    assert "aborted" in out
    assert install_root.exists()


def test_main_defaults_to_run_when_no_subcommand(monkeypatch) -> None:
    monkeypatch.setattr("multishell.__main__.cmd_run", lambda _args: 7)

    assert main([]) == 7


def test_main_returns_130_on_keyboard_interrupt(monkeypatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("multishell.__main__.cmd_run", lambda _args: (_ for _ in ()).throw(KeyboardInterrupt()))

    assert main([]) == 130
    assert "interrupted" in capsys.readouterr().out
