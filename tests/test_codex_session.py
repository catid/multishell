from __future__ import annotations

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
                "effort": "medium",
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
    monkeypatch.setattr(session, "interrupt", lambda: calls.append("interrupt"))
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
