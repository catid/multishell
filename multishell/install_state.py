from __future__ import annotations

from pathlib import Path

from .accounts import load_accounts_from_dotenv
from .config import dotenv_path, dotenv_template_text
from .envfile import DotenvFile
from .homes import missing_claude_logins, missing_codex_logins


def has_saved_account_credentials(path: Path | None = None) -> bool:
    env_path = Path(path or dotenv_path()).expanduser()
    env_file = DotenvFile.load(env_path, dotenv_template_text())
    accounts = load_accounts_from_dotenv(env_file)
    return any(account.email.strip() and account.password and account.providers for account in accounts)


def needs_account_login(path: Path | None = None) -> bool:
    return not has_saved_account_credentials(path)


def missing_auth_agents() -> list[str]:
    missing = missing_codex_logins()
    missing.extend(missing_claude_logins())
    return missing
