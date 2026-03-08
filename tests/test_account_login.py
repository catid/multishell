from __future__ import annotations

from pathlib import Path

from multishell.account_login import mask_secret, run_login_editor
from multishell.accounts import (
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    AccountRecord,
    load_accounts_from_dotenv,
    save_accounts_to_dotenv,
)
from multishell.envfile import DotenvFile


def test_account_inventory_round_trips_through_dotenv(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_file = DotenvFile.load(env_path, 'MULTISHELL_ACCOUNTS="[]"\n')
    accounts = [
        AccountRecord(
            key="account-1",
            email="person@example.com",
            password="secret",
            providers=(PROVIDER_OPENAI, PROVIDER_ANTHROPIC),
        )
    ]

    save_accounts_to_dotenv(env_file, accounts)
    env_file.save()

    reloaded = DotenvFile.load(env_path, 'MULTISHELL_ACCOUNTS="[]"\n')
    assert load_accounts_from_dotenv(reloaded) == accounts


def test_mask_secret_hides_non_empty_values() -> None:
    assert mask_secret("") == "(missing)"
    assert mask_secret("hunter2").startswith("*")
    assert "hunter2" not in mask_secret("hunter2")


def test_run_login_editor_creates_config_before_curses(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    wrapper_args: dict[str, object] = {}

    def fake_wrapper(fn):
        wrapper_args["fn"] = fn
        return None

    monkeypatch.setattr("multishell.account_login.curses.wrapper", fake_wrapper)

    assert run_login_editor(env_path) == 0
    assert env_path.exists()
    assert callable(wrapper_args["fn"])
