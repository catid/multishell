from __future__ import annotations

from pathlib import Path

import multishell.account_login as account_login
from multishell.account_login import AccountState, _add_account, mask_secret, run_login_editor
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


def test_add_account_prompts_for_email_and_saves(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_file = DotenvFile.load(env_path, 'MULTISHELL_ACCOUNTS="[]"\n')
    state = AccountState(env_file=env_file, accounts=[])

    monkeypatch.setattr("multishell.account_login._edit_value", lambda *_args, **_kwargs: "person@example.com")

    _add_account(None, state)

    assert len(state.accounts) == 1
    assert state.accounts[0].email == "person@example.com"
    assert state.selected_col == 1
    reloaded = DotenvFile.load(env_path, 'MULTISHELL_ACCOUNTS="[]"\n')
    assert load_accounts_from_dotenv(reloaded)[0].email == "person@example.com"


def test_add_account_aborts_on_blank_or_escape(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_file = DotenvFile.load(env_path, 'MULTISHELL_ACCOUNTS="[]"\n')
    state = AccountState(env_file=env_file, accounts=[])

    monkeypatch.setattr("multishell.account_login._edit_value", lambda *_args, **_kwargs: None)
    _add_account(None, state)
    assert state.accounts == []

    monkeypatch.setattr("multishell.account_login._edit_value", lambda *_args, **_kwargs: "   ")
    _add_account(None, state)
    assert state.accounts == []
    reloaded = DotenvFile.load(env_path, 'MULTISHELL_ACCOUNTS="[]"\n')
    assert load_accounts_from_dotenv(reloaded) == []


class _TerminalWindow:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def keypad(self, enabled: bool) -> None:
        self.calls.append(("keypad", enabled))

    def timeout(self, value: int) -> None:
        self.calls.append(("timeout", value))


def test_restore_terminal_resets_modes_without_endwin(monkeypatch) -> None:
    window = _TerminalWindow()
    calls: list[str] = []

    monkeypatch.setattr(account_login.curses, "echo", lambda: calls.append("echo"))
    monkeypatch.setattr(account_login.curses, "nocbreak", lambda: calls.append("nocbreak"))
    monkeypatch.setattr(account_login.curses, "nl", lambda: calls.append("nl"))
    monkeypatch.setattr(account_login.curses, "qiflush", lambda: calls.append("qiflush"))
    monkeypatch.setattr(account_login.curses, "endwin", lambda: calls.append("endwin"))

    account_login._restore_terminal(window)

    assert window.calls == [("keypad", False), ("timeout", -1)]
    assert calls == ["echo", "nocbreak", "nl", "qiflush"]
