from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import (
    CLAUDE_WORKER_SPECS,
    MANAGER_SPEC,
    WORKER_SPECS,
    dotenv_path,
    dotenv_template_text,
    missing_email_env_vars,
    state_root,
)
from .homes import (
    agent_home,
    all_agent_names,
    claude_home,
    claude_logged_in,
    codex_logged_in,
    ensure_agent_home,
    ensure_claude_home,
    missing_claude_logins,
    missing_codex_logins,
)
from .runtime import (
    apply_node_warning_suppression,
    apply_playwright_browser_path,
    cleanup_stale_runtime,
    ensure_runtime_environment,
    install_root as runtime_install_root,
    playwright_browsers_path,
    suppress_node_warnings,
)
from .updater import (
    UpdaterError,
    check_for_update,
    check_startup_update,
    current_release_version,
    current_version,
    install_latest_release,
    rollback_to_version,
)


DEFAULT_BIN_DIR = Path.home() / ".local" / "bin"


def _bridge_command(role: str, agent: str) -> list[str]:
    from .config import app_root, socket_path

    launcher = (
        "import sys; "
        f"sys.path.insert(0, {str(app_root())!r}); "
        "from multishell.mcp_bridge import main; "
        "raise SystemExit(main())"
    )
    return [sys.executable, "-c", launcher, "--socket", str(socket_path()), "--role", role, "--agent", agent]


def _venv_python() -> Path:
    return Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"


def _maybe_reexec_into_venv(module_name: str) -> bool:
    venv_python = _venv_python()
    if not importlib.util.find_spec(module_name) and venv_python.exists() and Path(sys.executable) != venv_python:
        try:
            result = subprocess.run([str(venv_python), "-m", "multishell", *sys.argv[1:]], check=False)
        except KeyboardInterrupt:
            raise SystemExit(130)
        raise SystemExit(result.returncode)
    return False


def _write_default_config(target: Path, *, force: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        return
    target.write_text(dotenv_template_text().rstrip("\n") + "\n", encoding="utf-8")


def _install_root() -> Path:
    return runtime_install_root()


def _bin_dir() -> Path:
    override = os.environ.get("MULTISHELL_BIN_DIR", "").strip()
    return Path(override).expanduser() if override else DEFAULT_BIN_DIR


def cmd_run(_: argparse.Namespace) -> int:
    if _maybe_reexec_into_venv("websockets"):
        return 0
    from .orchestrator import MultiShellController
    from .tui import run_tui

    state_root().mkdir(parents=True, exist_ok=True)
    ensure_runtime_environment(role="controller", agent=MANAGER_SPEC.name)
    startup_update = check_startup_update()
    if startup_update.message:
        print(startup_update.message, flush=True)
    if startup_update.restart_python is not None:
        os.execv(str(startup_update.restart_python), [str(startup_update.restart_python), "-m", "multishell", *sys.argv[1:]])
    cleanup_messages = cleanup_stale_runtime()
    ensure_agent_home(MANAGER_SPEC.name, mcp_bridge_command=_bridge_command("manager", MANAGER_SPEC.name))
    for spec in WORKER_SPECS:
        ensure_agent_home(spec.name, mcp_bridge_command=_bridge_command("worker", spec.name))
    for spec in CLAUDE_WORKER_SPECS:
        ensure_claude_home(spec.name)

    missing_codex = missing_codex_logins()
    missing_claude = missing_claude_logins()
    if missing_codex or missing_claude:
        if missing_codex:
            print("missing Codex login for:", ", ".join(missing_codex))
        if missing_claude:
            print("missing Claude login for:", ", ".join(missing_claude))
        print("multishell refuses to start until every configured agent is logged in")
        print("run `multishell auto-login --all` first")
        print("add `--headed` only if you explicitly want visible browser windows for debugging")
        print("for a single lane, use `multishell auth-login <agent>`")
        return 1

    controller = MultiShellController()
    try:
        controller.start()
    except RuntimeError as exc:
        print(str(exc))
        return 1
    for message in cleanup_messages:
        controller.add_notice(message, level="warn")
    try:
        run_tui(controller)
    except KeyboardInterrupt:
        return 0
    finally:
        controller.stop()
    return 0


def cmd_check_update(args: argparse.Namespace) -> int:
    try:
        update = check_for_update(repo=args.repo)
    except UpdaterError as exc:
        print(str(exc))
        return 1
    current = current_version()
    if update is None:
        print(f"multishell is up to date ({current})")
        return 0
    print(f"update available: {current} -> {update.version} ({update.tag_name})")
    print(f"source: {update.html_url}")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    try:
        release = install_latest_release(repo=args.repo, python_bin=sys.executable, target_install_root=_install_root(), target_bin_dir=_bin_dir())
    except UpdaterError as exc:
        print(str(exc))
        return 1
    print(f"installed {release.tag_name} into {_install_root()}")
    print(f"wrapper now points to release {current_release_version(_install_root())}")
    print("restart multishell to use the new version if it is currently running")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    try:
        version = rollback_to_version(
            version=args.version,
            target_install_root=_install_root(),
            target_bin_dir=_bin_dir(),
        )
    except UpdaterError as exc:
        print(str(exc))
        return 1
    print(f"rolled back multishell to {version}")
    return 0


def cmd_init_config(args: argparse.Namespace) -> int:
    target = Path(args.path).expanduser() if args.path else dotenv_path()
    if target.exists() and not args.force:
        print(f"config already exists: {target}")
        return 0
    _write_default_config(target, force=True)
    print(f"wrote config template: {target}")
    return 0


def cmd_install_browser(args: argparse.Namespace) -> int:
    if _maybe_reexec_into_venv("playwright"):
        return 0
    print(
        f"installing Playwright Chromium browser into {playwright_browsers_path()}; this can take several minutes on first run",
        flush=True,
    )
    apply_node_warning_suppression()
    env = suppress_node_warnings(os.environ.copy())
    apply_playwright_browser_path(env)
    command = [sys.executable, "-m", "playwright", "install"]
    if args.with_deps:
        command.append("--with-deps")
    command.append("chromium")
    subprocess.run(command, check=True, env=env)
    print("browser install complete", flush=True)
    return 0


def cmd_install_llama_cpp(args: argparse.Namespace) -> int:
    from .llama_cpp import install_llama_cpp

    try:
        install_llama_cpp(force=args.force)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    return 0


def cmd_install_auth_model(args: argparse.Namespace) -> int:
    from .llama_cpp import install_auth_model

    try:
        install_auth_model(force=args.force)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    install_root = Path(args.install_root).expanduser() if args.install_root else _install_root()
    bin_dir = Path(args.bin_dir).expanduser() if args.bin_dir else _bin_dir()
    wrapper = bin_dir / "multishell"
    runtime_state = state_root()

    targets = [str(wrapper), str(install_root)]
    if not args.keep_state:
        targets.append(str(runtime_state))

    if not args.yes:
        print("This will remove:")
        for target in targets:
            print(f"  {target}")
        confirm = input("Continue? [y/N] ").strip().lower()
        if confirm not in {"y", "yes"}:
            print("aborted")
            return 1

    removed: list[str] = []
    missing: list[str] = []

    if wrapper.exists() or wrapper.is_symlink():
        wrapper.unlink()
        removed.append(str(wrapper))
    else:
        missing.append(str(wrapper))

    if install_root.exists():
        shutil.rmtree(install_root)
        removed.append(str(install_root))
    else:
        missing.append(str(install_root))

    if not args.keep_state:
        if runtime_state.exists():
            shutil.rmtree(runtime_state)
            removed.append(str(runtime_state))
        else:
            missing.append(str(runtime_state))

    for target in removed:
        print(f"removed {target}")
    for target in missing:
        print(f"not found {target}")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    ensure_agent_home(MANAGER_SPEC.name, mcp_bridge_command=_bridge_command("manager", MANAGER_SPEC.name))
    print(f"{MANAGER_SPEC.name}: provider=codex email={MANAGER_SPEC.account_email} home={agent_home(MANAGER_SPEC.name)} auth={'yes' if codex_logged_in(MANAGER_SPEC.name) else 'no'}")
    for spec in WORKER_SPECS:
        ensure_agent_home(spec.name, mcp_bridge_command=_bridge_command("worker", spec.name))
        print(f"{spec.name}: provider=codex email={spec.account_email} home={agent_home(spec.name)} auth={'yes' if codex_logged_in(spec.name) else 'no'}")
    for spec in CLAUDE_WORKER_SPECS:
        ensure_claude_home(spec.name)
        print(f"{spec.name}: provider=claude email={spec.account_email} home={claude_home(spec.name)} auth={'yes' if claude_logged_in(spec.name) else 'no'}")
    missing = missing_email_env_vars()
    if missing:
        print(f"warning: missing email env vars in .env: {', '.join(missing)}")
    return 0


def cmd_login(_: argparse.Namespace) -> int:
    from .account_login import run_login_editor

    _write_default_config(dotenv_path(), force=False)
    return run_login_editor()


def cmd_auth_login(args: argparse.Namespace) -> int:
    if args.agent not in all_agent_names():
        raise SystemExit(f"unknown agent: {args.agent}")

    if args.agent == MANAGER_SPEC.name:
        ensure_agent_home(args.agent, mcp_bridge_command=_bridge_command("manager", args.agent))
    elif args.agent in {spec.name for spec in WORKER_SPECS}:
        ensure_agent_home(args.agent, mcp_bridge_command=_bridge_command("worker", args.agent))
    else:
        ensure_claude_home(args.agent)

    env = suppress_node_warnings(os.environ.copy())
    if args.agent in {spec.name for spec in CLAUDE_WORKER_SPECS}:
        email_by_name = {spec.name: spec.account_email for spec in CLAUDE_WORKER_SPECS}
        env.pop("ANTHROPIC_API_KEY", None)
        env["HOME"] = str(claude_home(args.agent))
        cmd = ["claude", "auth", "login", "--email", email_by_name[args.agent]]
    else:
        env["HOME"] = str(agent_home(args.agent))
        cmd = ["codex", "login", "--device-auth"]
    subprocess.run(cmd, check=True, env=env)
    return 0


def _auto_login_server_logger(*, verbose: bool):
    waiting_logged = False

    def log(message: str) -> None:
        nonlocal waiting_logged
        if verbose:
            print(message, flush=True)
            return
        if message.startswith("auth model server log:"):
            return
        if message.startswith("waiting for local auth model server to finish loading"):
            if waiting_logged:
                return
            waiting_logged = True
        print(message, flush=True)

    return log


def cmd_auto_login(args: argparse.Namespace) -> int:
    if _maybe_reexec_into_venv("playwright"):
        return 0

    apply_playwright_browser_path()
    from .autologin import resolve_credentials, run_auto_login
    from .llama_cpp import managed_auth_model_server

    apply_node_warning_suppression()

    if args.all:
        target_agents = all_agent_names()
    else:
        if not args.agent:
            raise SystemExit("provide an agent name or pass --all")
        target_agents = [args.agent]
        if args.agent not in all_agent_names():
            raise SystemExit(f"unknown agent: {args.agent}")

    missing_emails = missing_email_env_vars(target_agents)
    if missing_emails:
        raise SystemExit(f"missing emails in .env: {', '.join(missing_emails)}")

    credentials = resolve_credentials(target_agents)
    with managed_auth_model_server(required=False, log=_auto_login_server_logger(verbose=args.verbose)):
        run_auto_login(
            credentials,
            headed=args.headed,
            timeout_seconds=args.timeout,
            max_parallel=args.parallel,
            verbose=args.verbose,
        )
    return 0


def cmd_auth_model_smoke_test(args: argparse.Namespace) -> int:
    if _maybe_reexec_into_venv("playwright"):
        return 0

    apply_playwright_browser_path()
    from .auth_flow_model import run_auth_model_smoke_test
    from .llama_cpp import managed_auth_model_server

    try:
        with managed_auth_model_server(required=True, log=lambda message: print(message, flush=True)):
            run_auth_model_smoke_test(headed=args.headed)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    return 0


def cmd_auth_model_bench(args: argparse.Namespace) -> int:
    from .llama_cpp import bench_auth_model

    try:
        result = bench_auth_model(
            prompt_tokens=args.prompt_tokens,
            decode_tokens=args.decode_tokens,
            runs=args.runs,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1
    print(
        "auth model bench: "
        f"model={result.model_path} "
        f"prompt_tps={result.prompt_tokens_per_second:.2f} "
        f"decode_tps={result.decode_tokens_per_second:.2f} "
        f"prompt_tokens={result.prompt_tokens} "
        f"decode_tokens={result.decode_tokens} "
        f"runs={result.runs}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multishell")
    subparsers = parser.add_subparsers(dest="command")
    parser.set_defaults(func=cmd_run)

    init_parser = subparsers.add_parser("init-config")
    init_parser.add_argument("--path")
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(func=cmd_init_config)

    browser_parser = subparsers.add_parser("install-browser")
    browser_parser.add_argument("--with-deps", action="store_true")
    browser_parser.set_defaults(func=cmd_install_browser)

    llama_cpp_parser = subparsers.add_parser("install-llama-cpp")
    llama_cpp_parser.add_argument("--force", action="store_true")
    llama_cpp_parser.set_defaults(func=cmd_install_llama_cpp)

    install_auth_model_parser = subparsers.add_parser("install-auth-model")
    install_auth_model_parser.add_argument("--force", action="store_true")
    install_auth_model_parser.set_defaults(func=cmd_install_auth_model)

    uninstall_parser = subparsers.add_parser("uninstall")
    uninstall_parser.add_argument("--yes", action="store_true")
    uninstall_parser.add_argument("--keep-state", action="store_true")
    uninstall_parser.add_argument("--install-root")
    uninstall_parser.add_argument("--bin-dir")
    uninstall_parser.set_defaults(func=cmd_uninstall)

    run_parser = subparsers.add_parser("run")
    run_parser.set_defaults(func=cmd_run)

    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(func=cmd_status)

    check_update_parser = subparsers.add_parser("check-update")
    check_update_parser.add_argument("--repo")
    check_update_parser.set_defaults(func=cmd_check_update)

    update_parser = subparsers.add_parser("update")
    update_parser.add_argument("--repo")
    update_parser.set_defaults(func=cmd_update)

    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("version", nargs="?")
    rollback_parser.set_defaults(func=cmd_rollback)

    login_parser = subparsers.add_parser("login")
    login_parser.set_defaults(func=cmd_login)

    auth_login_parser = subparsers.add_parser("auth-login")
    auth_login_parser.add_argument("agent")
    auth_login_parser.set_defaults(func=cmd_auth_login)

    auto_login_parser = subparsers.add_parser("auto-login")
    auto_login_parser.add_argument("agent", nargs="?")
    auto_login_parser.add_argument("--all", action="store_true")
    auto_login_parser.add_argument("--headed", action="store_true")
    auto_login_parser.add_argument("--timeout", type=int, default=180)
    auto_login_parser.add_argument("--parallel", type=int, default=1)
    auto_login_parser.add_argument("--verbose", action="store_true")
    auto_login_parser.set_defaults(func=cmd_auto_login)

    auth_model_smoke_parser = subparsers.add_parser("auth-model-smoke-test")
    auth_model_smoke_parser.add_argument("--headed", action="store_true")
    auth_model_smoke_parser.set_defaults(func=cmd_auth_model_smoke_test)

    auth_model_bench_parser = subparsers.add_parser("auth-model-bench")
    auth_model_bench_parser.add_argument("--prompt-tokens", type=int, default=128)
    auth_model_bench_parser.add_argument("--decode-tokens", type=int, default=64)
    auth_model_bench_parser.add_argument("--runs", type=int, default=3)
    auth_model_bench_parser.set_defaults(func=cmd_auth_model_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
