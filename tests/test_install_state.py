from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from multishell.install_state import has_saved_account_credentials, missing_auth_agents, needs_account_login


def test_has_saved_account_credentials_requires_email_password_and_provider(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        'MULTISHELL_ACCOUNTS="[{\\"id\\":\\"account-1\\",\\"email\\":\\"\\",\\"password\\":\\"secret\\",\\"providers\\":[\\"openai\\"]}]"\n',
        encoding="utf-8",
    )

    assert has_saved_account_credentials(env_path) is False
    assert needs_account_login(env_path) is True

    env_path.write_text(
        'MULTISHELL_ACCOUNTS="[{\\"id\\":\\"account-1\\",\\"email\\":\\"user@example.com\\",\\"password\\":\\"secret\\",\\"providers\\":[\\"openai\\"]}]"\n',
        encoding="utf-8",
    )

    assert has_saved_account_credentials(env_path) is True
    assert needs_account_login(env_path) is False


def test_missing_auth_agents_reports_codex_and_claude(monkeypatch, tmp_path: Path) -> None:
    existing_auth = tmp_path / "existing-auth.json"
    existing_auth.write_text("{}", encoding="utf-8")
    missing_auth = tmp_path / "missing-auth.json"

    monkeypatch.setattr(
        "multishell.install_state.all_codex_specs",
        lambda: [SimpleNamespace(name="manager"), SimpleNamespace(name="worker-1")],
    )
    monkeypatch.setattr(
        "multishell.install_state.CLAUDE_WORKER_SPECS",
        [SimpleNamespace(name="claude-worker-1"), SimpleNamespace(name="claude-worker-2")],
    )
    monkeypatch.setattr(
        "multishell.install_state.auth_path",
        lambda agent: existing_auth if agent == "manager" else missing_auth,
    )
    monkeypatch.setattr(
        "multishell.install_state.claude_logged_in",
        lambda agent: agent == "claude-worker-2",
    )

    assert missing_auth_agents() == ["worker-1", "claude-worker-1"]
