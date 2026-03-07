from __future__ import annotations

from pathlib import Path

from multishell.account_login import account_slots, mask_secret, run_login_editor


def test_account_slots_cover_manager_workers_and_gemini() -> None:
    slots = account_slots()

    assert [slot.key for slot in slots] == ["manager", "worker-1", "worker-2", "worker-3", "worker-4", "gemini"]


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
