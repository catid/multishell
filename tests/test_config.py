from __future__ import annotations

from pathlib import Path

from multishell import config


def test_state_root_uses_env_override(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "state-root"
    monkeypatch.setenv(config.STATE_ROOT_ENV_VAR, str(target))

    assert config.state_root() == target


def test_workspace_root_defaults_to_current_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(config.WORKSPACE_ROOT_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    assert config.workspace_root() == tmp_path.resolve()


def test_workspace_root_uses_env_override(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "override"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv(config.WORKSPACE_ROOT_ENV_VAR, str(override))
    monkeypatch.chdir(elsewhere)

    assert config.workspace_root() == override.resolve()


def test_dotenv_path_prefers_state_root(monkeypatch, tmp_path: Path) -> None:
    state = tmp_path / "state-root"
    config_file = state / ".env"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("MULTISHELL_MANAGER_EMAIL=manager@example.com\n", encoding="utf-8")
    monkeypatch.setenv(config.STATE_ROOT_ENV_VAR, str(state))
    monkeypatch.delenv(config.ENV_FILE_ENV_VAR, raising=False)

    assert config.dotenv_path() == config_file
