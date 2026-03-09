from __future__ import annotations

import io
import queue
import threading
from pathlib import Path

from multishell.codex_session import CodexSession, SessionEvent, TranscriptEntry, TurnRequest
from multishell.config import AgentSpec


def _spec(name: str = "worker-1") -> AgentSpec:
    return AgentSpec(
        name=name,
        account_email="worker@example.com",
        role="worker",
        personality="Test worker.",
        accent_color=2,
    )


def test_start_session_enqueues_startup_prompt(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir(parents=True)
    workspace.mkdir(parents=True)

    launches: list[str] = []

    def fake_launch(self: CodexSession) -> None:
        launches.append(str(self.working_dir))
        with self._state_lock:
            self.state.thread_id = "thread-1"
            self.state.session_active = True
            self.state.process_alive = True
            self.state.status = "idle"
            self.state.current_cwd = str(self.working_dir)

    monkeypatch.setattr("multishell.codex_session.ensure_agent_home", lambda *args, **kwargs: home)
    monkeypatch.setattr("multishell.codex_session.workspace_root", lambda: workspace)
    monkeypatch.setattr(CodexSession, "_launch_session", fake_launch)

    session = CodexSession(_spec(), "initial", startup_prompt="Reply once that you are ready.")
    session.start_session()

    assert launches == [str(workspace)]
    overview = session.overview()
    assert overview["status"] == "idle"
    assert overview["pending_tasks"] == 1
    transcript = session.recent_transcript(2)
    assert transcript[-1].source == "system"
    assert transcript[-1].text == "Reply once that you are ready."
    assert overview["persona_label"] == "worker-1"


def test_start_session_can_override_prompt_and_persona(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir(parents=True)
    workspace.mkdir(parents=True)

    monkeypatch.setattr("multishell.codex_session.ensure_agent_home", lambda *args, **kwargs: home)
    monkeypatch.setattr("multishell.codex_session.workspace_root", lambda: workspace)
    monkeypatch.setattr(CodexSession, "_launch_session", lambda self: None)

    session = CodexSession(_spec(), "initial", persona_label="Grace Hopper")
    session.start_session(system_prompt="custom prompt", persona_label="Ada Lovelace")

    assert session.initial_prompt == "custom prompt"
    assert session.overview()["persona_label"] == "Ada Lovelace"


def test_run_turn_omits_null_cwd_and_updates_metrics(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir(parents=True)
    workspace.mkdir(parents=True)

    requests: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr("multishell.codex_session.ensure_agent_home", lambda *args, **kwargs: home)
    monkeypatch.setattr("multishell.codex_session.workspace_root", lambda: workspace)

    session = CodexSession(_spec(), "initial")
    session.state.thread_id = "thread-123"
    session.state.session_active = True
    session.state.process_alive = True
    session.state.status = "idle"

    def fake_request(method: str, params: dict[str, object], timeout: float) -> dict[str, object]:
        requests.append((method, params))
        if method == "turn/start":
            threading.Timer(
                0.01,
                lambda: session._handle_notification(
                    {
                        "method": "turn/completed",
                        "params": {"turn": {"id": "turn-1", "status": "completed", "error": None}},
                    }
                ),
            ).start()
            return {"turn": {"id": "turn-1"}}
        raise AssertionError(f"unexpected request: {method}")

    monkeypatch.setattr(session, "_request", fake_request)

    session._run_turn(TurnRequest(prompt="ship it", source="user"))

    assert requests == [
        (
            "turn/start",
            {
                "threadId": "thread-123",
                "input": [{"type": "text", "text": "ship it", "text_elements": []}],
                "model": "gpt-5.4",
                "effort": "high",
                "personality": "pragmatic",
            },
        )
    ]
    overview = session.overview()
    assert overview["status"] == "idle"
    assert overview["completed_turns"] == 1
    assert overview["failed_turns"] == 0
    assert overview["last_turn_duration"] is not None
    assert overview["current_task_source"] == ""


def test_worker_notifications_emit_message_and_event_callbacks(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir(parents=True)
    workspace.mkdir(parents=True)

    messages: list[TranscriptEntry] = []
    events: list[SessionEvent] = []

    monkeypatch.setattr("multishell.codex_session.ensure_agent_home", lambda *args, **kwargs: home)
    monkeypatch.setattr("multishell.codex_session.workspace_root", lambda: workspace)

    session = CodexSession(
        _spec(),
        "initial",
        message_callback=messages.append,
        event_callback=events.append,
    )

    session._handle_notification(
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "agentMessage",
                    "text": "Ready for assignments.",
                    "phase": "final_answer",
                }
            },
        }
    )

    assert messages == [TranscriptEntry(ts=messages[0].ts, source="assistant", text="Ready for assignments.")]
    assert events[-1].kind == "assistant_message"
    assert events[-1].message == "Ready for assignments."
    transcript = session.recent_transcript(1)
    assert transcript[-1].source == "assistant"


def test_mcp_failure_and_stop_session_emit_events(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir(parents=True)
    workspace.mkdir(parents=True)

    events: list[SessionEvent] = []
    calls: list[str] = []

    monkeypatch.setattr("multishell.codex_session.ensure_agent_home", lambda *args, **kwargs: home)
    monkeypatch.setattr("multishell.codex_session.workspace_root", lambda: workspace)

    session = CodexSession(_spec(), "initial", event_callback=events.append)
    session.state.thread_id = "thread-1"
    session.state.session_active = True
    session.state.process_alive = True
    session.state.status = "idle"

    monkeypatch.setattr(session, "_drain_pending_queue", lambda: calls.append("drain"))
    monkeypatch.setattr(session, "interrupt", lambda *args, **kwargs: calls.append("interrupt"))
    monkeypatch.setattr(session, "_close_transport", lambda: calls.append("close"))

    session._handle_notification(
        {
            "method": "codex/event/mcp_startup_update",
            "params": {
                "msg": {
                    "server": "multishell",
                    "status": {"state": "failed", "error": "timed out"},
                }
            },
        }
    )

    session.stop_session(clear_pending=True)

    assert calls == ["drain", "interrupt", "close"]
    assert session.overview()["status"] == "stopped"
    assert [event.kind for event in events] == ["mcp_failed", "session_stopped"]
    assert "timed out" in events[0].message


def test_unexpected_transport_close_marks_running_turn_failed_and_unblocks_waiters(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir(parents=True)
    workspace.mkdir(parents=True)

    events: list[SessionEvent] = []

    monkeypatch.setattr("multishell.codex_session.ensure_agent_home", lambda *args, **kwargs: home)
    monkeypatch.setattr("multishell.codex_session.workspace_root", lambda: workspace)

    session = CodexSession(_spec(), "initial", event_callback=events.append)
    session.state.thread_id = "thread-1"
    session.state.turn_id = "turn-1"
    session.state.session_active = True
    session.state.process_alive = True
    session.state.status = "running"

    waiter: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)
    session._response_waiters["req-1"] = waiter

    session._handle_transport_closed()

    overview = session.overview()
    assert overview["status"] == "error"
    assert overview["failed_turns"] == 1
    assert overview["consecutive_failures"] == 1
    assert overview["turn_id"] is None
    assert overview["session_active"] is False
    assert overview["process_alive"] is False
    assert overview["last_error"] == "app-server disconnected unexpectedly; during active turn"
    assert overview["last_disconnect_kind"] == "abrupt_disconnect"
    assert overview["last_disconnect_retryable"] is True
    assert session._current_turn_error == "app-server disconnected unexpectedly; during active turn"
    assert session._turn_done.is_set() is True
    assert waiter.get_nowait() == {
        "error": {
            "message": "app-server disconnected unexpectedly; during active turn",
            "retryable": True,
            "disconnect_kind": "abrupt_disconnect",
            "recommended_action": "restart_session",
            "during_turn": True,
        }
    }
    assert [event.kind for event in events] == ["transport_closed"]


def test_interrupted_turn_records_interrupt_reason(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir(parents=True)
    workspace.mkdir(parents=True)

    events: list[SessionEvent] = []
    requests: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr("multishell.codex_session.ensure_agent_home", lambda *args, **kwargs: home)
    monkeypatch.setattr("multishell.codex_session.workspace_root", lambda: workspace)

    session = CodexSession(_spec(), "initial", event_callback=events.append)
    session.state.thread_id = "thread-1"
    session.state.turn_id = "turn-1"
    session.state.session_active = True
    session.state.process_alive = True
    session.state.status = "running"
    session.state.last_turn_started_at = 10.0

    monkeypatch.setattr(
        session,
        "_request",
        lambda method, params, timeout: requests.append((method, params)) or {"ok": True},
    )

    session.interrupt(reason="session restart was requested", requested_by="controller")
    session._handle_notification(
        {
            "method": "turn/completed",
            "params": {"turn": {"id": "turn-1", "status": "interrupted", "error": None}},
        }
    )

    assert requests == [("turn/interrupt", {"threadId": "thread-1", "turnId": "turn-1"})]
    overview = session.overview()
    assert overview["status"] == "error"
    assert overview["last_error"] == "turn interrupted because session restart was requested"
    assert overview["failed_turns"] == 1
    assert events[-1].kind == "turn_failed"
    assert events[-1].message == "turn interrupted because session restart was requested"
    assert events[-1].data["interrupted"] is True
    assert events[-1].data["interrupt_reason"] == "session restart was requested"
    assert events[-1].data["turn_status"] == "interrupted"


def test_thread_closed_notification_stops_session_without_transport_error(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir(parents=True)
    workspace.mkdir(parents=True)

    events: list[SessionEvent] = []

    monkeypatch.setattr("multishell.codex_session.ensure_agent_home", lambda *args, **kwargs: home)
    monkeypatch.setattr("multishell.codex_session.workspace_root", lambda: workspace)

    session = CodexSession(_spec(), "initial", event_callback=events.append)
    session.state.thread_id = "thread-1"
    session.state.turn_id = "turn-1"
    session.state.session_active = True
    session.state.process_alive = True
    session.state.status = "running"

    session._handle_notification({"method": "thread/closed", "params": {}})

    overview = session.overview()
    assert overview["status"] == "stopped"
    assert overview["thread_id"] is None
    assert overview["turn_id"] is None
    assert overview["session_active"] is False
    assert overview["process_alive"] is False
    assert overview["failed_turns"] == 0
    assert [event.kind for event in events] == ["thread_closed"]


def test_stderr_reader_ignores_node_deprecation_noise(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir(parents=True)
    workspace.mkdir(parents=True)

    monkeypatch.setattr("multishell.codex_session.ensure_agent_home", lambda *args, **kwargs: home)
    monkeypatch.setattr("multishell.codex_session.workspace_root", lambda: workspace)

    session = CodexSession(_spec(), "initial")
    session._stderr_reader(
        io.StringIO(
            "(node:825330) [DEP0169] DeprecationWarning: `url.parse()` behavior is not standardized\n"
            "Use `node --trace-deprecation ...` to show where the warning was created)\n"
            "actual error line\n"
        )
    )

    texts = [entry.text for entry in session.recent_transcript(8)]
    assert "actual error line" in texts
    assert not any("DeprecationWarning" in text for text in texts)


def test_transport_close_captures_abrupt_disconnect_diagnostics(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir(parents=True)
    workspace.mkdir(parents=True)

    events: list[SessionEvent] = []

    monkeypatch.setattr("multishell.codex_session.ensure_agent_home", lambda *args, **kwargs: home)
    monkeypatch.setattr("multishell.codex_session.workspace_root", lambda: workspace)

    session = CodexSession(_spec(), "initial", event_callback=events.append)
    session.state.thread_id = "thread-1"
    session.state.turn_id = "turn-1"
    session.state.session_active = True
    session.state.process_alive = True
    session.state.status = "running"
    session.state.current_task_source = "user"
    session.state.last_turn_started_at = 1.0
    session._stderr_tail.extend(["fatal: worker crashed", "trace: socket reset"])

    class FakeProcess:
        def poll(self) -> int:
            return 137

    class FakeClose:
        code = 1006
        reason = "abnormal closure"

    session._process = FakeProcess()  # type: ignore[assignment]
    detail = session._handle_transport_closed(connection_closed=FakeClose())  # type: ignore[arg-type]

    assert "disconnected unexpectedly" in detail
    overview = session.overview()
    assert overview["status"] == "error"
    assert overview["failed_turns"] == 1
    assert overview["last_disconnect_kind"] == "abrupt_disconnect"
    assert overview["last_disconnect_retryable"] is True
    assert overview["last_disconnect_code"] == 1006
    assert overview["last_process_exit_code"] == 137
    assert "fatal: worker crashed" in str(overview["last_transport_diagnostics"])
    assert events[-1].kind == "transport_closed"
    assert events[-1].data["retryable"] is True
    assert events[-1].data["recommended_action"] == "restart_session"


def test_transport_close_wakes_waiters_with_detailed_retryable_error(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir(parents=True)
    workspace.mkdir(parents=True)

    monkeypatch.setattr("multishell.codex_session.ensure_agent_home", lambda *args, **kwargs: home)
    monkeypatch.setattr("multishell.codex_session.workspace_root", lambda: workspace)

    session = CodexSession(_spec(), "initial")
    waiter: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)
    session._response_waiters["req-1"] = waiter
    session._stderr_tail.append("panic: app-server died")

    detail = session._handle_transport_closed(reader_error=OSError("broken pipe"))
    payload = waiter.get_nowait()

    assert "reading events" in detail
    assert payload["error"]["message"] == detail
    assert payload["error"]["retryable"] is True
    assert payload["error"]["recommended_action"] == "restart_session"


def test_clean_transport_close_is_classified_distinctly(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir(parents=True)
    workspace.mkdir(parents=True)

    events: list[SessionEvent] = []

    monkeypatch.setattr("multishell.codex_session.ensure_agent_home", lambda *args, **kwargs: home)
    monkeypatch.setattr("multishell.codex_session.workspace_root", lambda: workspace)

    session = CodexSession(_spec(), "initial", event_callback=events.append)
    session.state.thread_id = "thread-1"
    session.state.session_active = True
    session.state.process_alive = True
    session.state.status = "idle"

    class FakeProcess:
        def poll(self) -> int:
            return 0

    class FakeClose:
        code = 1000
        reason = "ok"

    session._process = FakeProcess()  # type: ignore[assignment]
    detail = session._handle_transport_closed(connection_closed=FakeClose())  # type: ignore[arg-type]

    assert "closed the websocket cleanly" in detail
    assert session.overview()["last_disconnect_kind"] == "clean_disconnect"
    assert session.overview()["last_disconnect_retryable"] is False
    assert events[-1].data["retryable"] is False


def test_intentional_transport_close_does_not_mark_turn_failed(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir(parents=True)
    workspace.mkdir(parents=True)

    events: list[SessionEvent] = []

    monkeypatch.setattr("multishell.codex_session.ensure_agent_home", lambda *args, **kwargs: home)
    monkeypatch.setattr("multishell.codex_session.workspace_root", lambda: workspace)

    session = CodexSession(_spec(), "initial", event_callback=events.append)
    session.state.thread_id = "thread-1"
    session.state.turn_id = "turn-1"
    session.state.session_active = True
    session.state.process_alive = True
    session.state.status = "running"
    session._intentional_transport_close.set()

    detail = session._handle_transport_closed()

    assert "during shutdown" in detail
    assert session._current_turn_error is None
    assert session._turn_done.is_set() is True
    assert session.overview()["last_disconnect_kind"] == "intentional_shutdown"
    assert events == []
