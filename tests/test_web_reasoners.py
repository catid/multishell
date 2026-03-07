from __future__ import annotations

import time
from threading import Event

import pytest

from multishell.web_reasoners import (
    WebReasonerEvent,
    WebReasonerManager,
    _ensure_chatgpt_pro_workspace,
    _ensure_google_logged_in,
    _web_profile_name,
)


class _SuccessManager(WebReasonerManager):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.started = Event()
        self.release = Event()

    def _execute_job(self, job) -> str:
        self.started.set()
        assert self.release.wait(timeout=2)
        return f"answer for {job.provider}"


class _CancelManager(WebReasonerManager):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.started = Event()

    def _execute_job(self, job) -> str:
        self.started.set()
        cancel_flag = self._cancel_flags[job.id]
        deadline = time.time() + 2
        while time.time() < deadline:
            if cancel_flag.wait(timeout=0.01):
                raise RuntimeError("canceled")
        return "unexpected"


def _wait_for_status(manager: WebReasonerManager, job_id: str, expected: str, timeout: float = 3.0) -> dict[str, object]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = manager.snapshot(job_id)
        if snapshot["status"] == expected:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {expected!r}; last snapshot={manager.snapshot(job_id)!r}")


def test_start_job_supports_agent_alias_and_completes() -> None:
    events: list[WebReasonerEvent] = []
    manager = _SuccessManager(callback=events.append)

    job = manager.start_job("chatgpt_pro", agent="manager", prompt="Plan it", label="planner", timeout_seconds=45)
    assert job["agent"] == "manager"
    assert job["label"] == "planner"
    assert job["timeout_seconds"] == 45

    assert manager.started.wait(timeout=1)
    manager.release.set()
    snapshot = _wait_for_status(manager, str(job["job_id"]), "completed")

    assert snapshot["result_text"] == "answer for chatgpt_pro"
    assert [event.kind for event in events] == ["job_started", "job_completed"]


def test_cancel_job_finishes_as_canceled_without_completed_event() -> None:
    events: list[WebReasonerEvent] = []
    manager = _CancelManager(callback=events.append)

    job = manager.start_job("gemini_deepthink", "manager", "Think slowly")
    assert manager.started.wait(timeout=1)

    cancel_snapshot = manager.cancel_job(str(job["job_id"]))
    assert cancel_snapshot is not None
    assert cancel_snapshot["status"] in {"canceling", "canceled"}

    snapshot = _wait_for_status(manager, str(job["job_id"]), "canceled")
    assert snapshot["error"] == "canceled"
    assert [event.kind for event in events] == ["job_started", "job_canceled"]


def test_start_job_validates_required_fields() -> None:
    manager = WebReasonerManager()

    with pytest.raises(Exception, match="unsupported web reasoner provider"):
        manager.start_job("unknown", "manager", "hello")

    with pytest.raises(Exception, match="account_agent is required"):
        manager.start_job("chatgpt_pro", "", "hello")

    with pytest.raises(Exception, match="prompt is required"):
        manager.start_job("chatgpt_pro", "manager", "   ")


def test_web_profile_name_is_stable_per_provider_and_account() -> None:
    assert _web_profile_name("chatgpt_pro", "manager") == "web-chatgpt_pro-manager"
    assert _web_profile_name("chatgpt_pro", "manager") == _web_profile_name("chatgpt_pro", "manager")


class _FakeLocator:
    def __init__(self, selector: str) -> None:
        self.selector = selector
        self.first = self

    def click(self, timeout: int | None = None) -> None:
        if self.selector.startswith("text="):
            raise AssertionError("password screen should not click the account picker row")


class _FakePage:
    def __init__(self, url: str) -> None:
        self.url = url

    def locator(self, selector: str) -> _FakeLocator:
        return _FakeLocator(selector)


def test_google_password_screen_prefers_password_input_over_account_picker(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[tuple[str, tuple[str, ...]]] = []
    page = _FakePage("https://accounts.google.com/v3/signin/challenge/pwd")

    monkeypatch.setattr(
        "multishell.web_reasoners._body_text",
        lambda _: (
            "Sign in with Google\n"
            "Hi bot\n"
            "bot@kuang2.ai\n"
            "Enter your password\n"
            "Next"
        ),
    )
    monkeypatch.setattr("multishell.web_reasoners._page_title", lambda _: "Hi bot")
    monkeypatch.setattr(
        "multishell.web_reasoners._has_visible",
        lambda _page, selectors, timeout_ms: any("password" in selector or "Passwd" in selector for selector in selectors),
    )

    def fake_fill_first(_page: object, selectors: list[str], value: str) -> None:
        recorded.append((value, tuple(selectors)))
        raise RuntimeError("stop after password fill")

    monkeypatch.setattr("multishell.web_reasoners._fill_first", fake_fill_first)
    monkeypatch.setattr("multishell.web_reasoners._click_first", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("multishell.web_reasoners._click_optional", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="stop after password fill"):
        _ensure_google_logged_in(page, "bot@kuang2.ai", "secret", Event())

    assert recorded == [
        (
            "secret",
            (
                "input[type='password']:visible",
                "input[name='Passwd']:visible",
                "input[autocomplete='current-password']:visible",
            ),
        )
    ]


def test_chatgpt_workspace_reuses_openai_consent_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage("https://auth.openai.com/authorize")
    consent_calls: list[str] = []
    body_state = {"value": "Sign in to ChatGPT with ChatGPT"}

    monkeypatch.setattr("multishell.web_reasoners._body_text", lambda _page: body_state["value"])
    monkeypatch.setattr("multishell.web_reasoners._has_visible", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "multishell.web_reasoners._complete_workspace_consent",
        lambda _page: consent_calls.append("consent") or body_state.__setitem__("value", "ChatGPT 5.4 Pro"),
    )
    monkeypatch.setattr("multishell.web_reasoners._click_optional", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("multishell.web_reasoners._check_cancel", lambda _flag: None)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)
    page.wait_for_timeout = lambda *_args, **_kwargs: None

    _ensure_chatgpt_pro_workspace(page, Event())

    assert consent_calls == ["consent"]
