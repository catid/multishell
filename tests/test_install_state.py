from __future__ import annotations

from pathlib import Path

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
    monkeypatch.setattr("multishell.install_state.missing_codex_logins", lambda: ["worker-1"])
    monkeypatch.setattr("multishell.install_state.missing_claude_logins", lambda: ["claude-worker-1"])

    assert missing_auth_agents() == ["worker-1", "claude-worker-1"]
