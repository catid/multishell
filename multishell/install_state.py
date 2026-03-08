from __future__ import annotations

from pathlib import Path

from .accounts import load_accounts_from_dotenv
from .config import CLAUDE_WORKER_SPECS, all_codex_specs, dotenv_path, dotenv_template_text
from .envfile import DotenvFile
from .homes import auth_path, claude_logged_in


def has_saved_account_credentials(path: Path | None = None) -> bool:
    env_path = Path(path or dotenv_path()).expanduser()
    env_file = DotenvFile.load(env_path, dotenv_template_text())
    accounts = load_accounts_from_dotenv(env_file)
    return any(account.email.strip() and account.password and account.providers for account in accounts)


def needs_account_login(path: Path | None = None) -> bool:
    return not has_saved_account_credentials(path)


def missing_auth_agents() -> list[str]:
    missing = [spec.name for spec in all_codex_specs() if not auth_path(spec.name).exists()]
    missing.extend(spec.name for spec in CLAUDE_WORKER_SPECS if not claude_logged_in(spec.name))
    return missing
