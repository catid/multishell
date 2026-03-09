from __future__ import annotations

from pathlib import Path

from multishell.updater import (
    AUTO_UPDATE_MODE_APPLY,
    ReleaseInfo,
    StartupUpdateResult,
    _default_rollback_target,
    _write_wrapper,
    check_for_update,
    check_startup_update,
    current_release_version,
    install_release,
    latest_release,
    rollback_to_version,
)


def test_latest_release_prefers_highest_semver(monkeypatch) -> None:
    monkeypatch.setattr(
        "multishell.updater._fetch_json",
        lambda _url: [
            {"name": "v0.1.9"},
            {"name": "v0.2.1"},
            {"name": "v0.2.0"},
        ],
    )

    release = latest_release(repo="catid/multishell")

    assert release is not None
    assert release.tag_name == "v0.2.1"
    assert release.version == "0.2.1"


def test_check_for_update_returns_none_when_current_is_latest(monkeypatch) -> None:
    monkeypatch.setattr(
        "multishell.updater.latest_release",
        lambda repo=None: ReleaseInfo(
            tag_name="v0.1.0",
            version="0.1.0",
            tarball_url="https://example.invalid/v0.1.0.tar.gz",
            html_url="https://example.invalid/v0.1.0",
        ),
    )

    assert check_for_update(current="0.1.0") is None


def test_install_release_reuses_existing_release_and_updates_current_link(tmp_path: Path) -> None:
    install_root = tmp_path / "install"
    bin_dir = tmp_path / "bin"
    release_dir = install_root / "releases" / "0.2.0"
    (release_dir / "app").mkdir(parents=True)
    (release_dir / "app" / "pyproject.toml").write_text("[project]\nname='multishell'\n", encoding="utf-8")
    (release_dir / "venv" / "bin").mkdir(parents=True)
    (release_dir / "venv" / "bin" / "python").write_text("", encoding="utf-8")

    release = ReleaseInfo(
        tag_name="v0.2.0",
        version="0.2.0",
        tarball_url="https://example.invalid/v0.2.0.tar.gz",
        html_url="https://example.invalid/v0.2.0",
    )

    installed = install_release(release, target_install_root=install_root, target_bin_dir=bin_dir)

    assert installed == release_dir
    assert current_release_version(install_root) == "0.2.0"
    wrapper = bin_dir / "multishell"
    assert wrapper.exists()
    assert "current/venv/bin/python" in wrapper.read_text(encoding="utf-8")


def test_rollback_to_previous_version_switches_current_release(tmp_path: Path) -> None:
    install_root = tmp_path / "install"
    bin_dir = tmp_path / "bin"
    for version in ("0.1.0", "0.2.0"):
        release_dir = install_root / "releases" / version
        (release_dir / "app").mkdir(parents=True)
        (release_dir / "app" / "pyproject.toml").write_text("[project]\nname='multishell'\n", encoding="utf-8")
        (release_dir / "venv" / "bin").mkdir(parents=True)
        (release_dir / "venv" / "bin" / "python").write_text("", encoding="utf-8")
    (install_root / "current").symlink_to(Path("releases") / "0.2.0", target_is_directory=True)

    version = rollback_to_version(target_install_root=install_root, target_bin_dir=bin_dir)

    assert version == "0.1.0"
    assert current_release_version(install_root) == "0.1.0"


def test_default_rollback_target_returns_previous_version() -> None:
    assert _default_rollback_target(["0.1.0", "0.2.0", "0.3.0"], "0.3.0") == "0.2.0"
    assert _default_rollback_target(["0.1.0"], "0.1.0") is None


def test_check_startup_update_returns_notice_when_update_is_available(monkeypatch) -> None:
    monkeypatch.setattr("multishell.updater._running_from_managed_install", lambda: True)
    monkeypatch.setattr("multishell.updater._should_check_for_update", lambda: True)
    monkeypatch.setattr(
        "multishell.updater.check_for_update",
        lambda repo=None: ReleaseInfo(
            tag_name="v0.2.0",
            version="0.2.0",
            tarball_url="https://example.invalid/v0.2.0.tar.gz",
            html_url="https://example.invalid/v0.2.0",
        ),
    )
    monkeypatch.setattr("multishell.updater.current_version", lambda: "0.1.0")
    monkeypatch.setattr("multishell.updater._record_update_check", lambda now=None: None)

    result = check_startup_update()

    assert isinstance(result, StartupUpdateResult)
    assert result.message == "update available: 0.1.0 -> 0.2.0 (v0.2.0). Run `multishell update`."
    assert result.restart_python is None


def test_check_startup_update_can_apply_and_request_restart(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MULTISHELL_AUTO_UPDATE", AUTO_UPDATE_MODE_APPLY)
    monkeypatch.setattr("multishell.updater._running_from_managed_install", lambda: True)
    monkeypatch.setattr("multishell.updater._should_check_for_update", lambda: True)
    monkeypatch.setattr(
        "multishell.updater.check_for_update",
        lambda repo=None: ReleaseInfo(
            tag_name="v0.2.0",
            version="0.2.0",
            tarball_url="https://example.invalid/v0.2.0.tar.gz",
            html_url="https://example.invalid/v0.2.0",
        ),
    )
    monkeypatch.setattr("multishell.updater.install_release", lambda release: tmp_path / "install" / "releases" / "0.2.0")
    monkeypatch.setattr("multishell.updater.current_release_python", lambda: tmp_path / "install" / "current" / "venv" / "bin" / "python")
    monkeypatch.setattr("multishell.updater._record_update_check", lambda now=None: None)

    result = check_startup_update()

    assert result.message == "updated multishell to v0.2.0; restarting"
    assert result.restart_python == tmp_path / "install" / "current" / "venv" / "bin" / "python"


def test_write_wrapper_uses_install_root_override(tmp_path: Path) -> None:
    wrapper = tmp_path / "bin" / "multishell"
    _write_wrapper(wrapper, target_install_root=tmp_path / "install")

    text = wrapper.read_text(encoding="utf-8")
    assert 'INSTALL_ROOT="${MULTISHELL_INSTALL_ROOT:-' in text
    assert 'exec "$INSTALL_ROOT/current/venv/bin/python" -m multishell "$@"' in text
