from __future__ import annotations

import time
from threading import Event

import pytest

from multishell.web_reasoners import WebReasonerEvent, WebReasonerManager


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
