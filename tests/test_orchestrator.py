from __future__ import annotations

from dataclasses import dataclass

import pytest

import multishell.orchestrator as orch
from multishell.codex_session import SessionEvent


@dataclass
class _FakeEntry:
    ts: float
    source: str
    text: str


class FakeSession:
    def __init__(
        self,
        spec,
        initial_prompt,
        mcp_bridge_command=None,
        working_dir=None,
        startup_prompt=None,
        persona_label=None,
        message_callback=None,
        event_callback=None,
        turn_timeout_seconds=None,
        model="gpt-5.4",
        reasoning_effort="high",
        auth_source_agent=None,
        **kwargs,
    ) -> None:
        self.spec = spec
        self.initial_prompt = initial_prompt
        self.startup_prompt = startup_prompt
        self.persona_label = persona_label or spec.name
        self.message_callback = message_callback
        self.event_callback = event_callback
        self.turn_timeout_seconds = turn_timeout_seconds
        self.model = model
        self._overview = {
            "name": spec.name,
            "account_key": getattr(spec, "account_key", spec.name),
            "account_email": spec.account_email,
            "accent_color": spec.accent_color,
            "status": "idle",
            "thread_id": None,
            "turn_id": None,
            "pending_tasks": 0,
            "updated_at": 0.0,
            "last_error": None,
            "last_message": "",
            "started_turns": 0,
            "completed_turns": 0,
            "failed_turns": 0,
            "consecutive_failures": 0,
            "last_turn_started_at": None,
            "last_turn_finished_at": None,
            "last_turn_duration": None,
            "running_for_seconds": None,
            "current_task_source": "",
            "cwd": str(working_dir),
            "session_active": False,
            "process_alive": False,
            "auth_mode": None,
            "plan_type": None,
            "persona_label": self.persona_label,
            "engine": getattr(spec, "engine", "codex"),
            "model": self.model,
        }
        self.enqueued: list[tuple[str, str, str | None]] = []
        self.started_calls: list[tuple[str | None, str | None, str | None]] = []
        self.restarted_calls: list[tuple[str | None, str | None, str | None]] = []
        self.stopped = 0
        self.start_count = 0
        self.interrupt_count = 0

    def start(self, *, start_session: bool = True) -> None:
        self.start_count += 1
        self._overview["status"] = "idle" if start_session else "stopped"

    def stop(self) -> None:
        self._overview["status"] = "stopped"

    def start_session(
        self,
        cwd: str | None = None,
        system_prompt: str | None = None,
        persona_label: str | None = None,
    ) -> None:
        self.started_calls.append((cwd, system_prompt, persona_label))
        if cwd:
            self._overview["cwd"] = cwd
        if system_prompt is not None:
            self.initial_prompt = system_prompt
        if persona_label is not None:
            self._overview["persona_label"] = persona_label
        self._overview["status"] = "idle"

    def restart_session(
        self,
        cwd: str | None = None,
        system_prompt: str | None = None,
        persona_label: str | None = None,
    ) -> None:
        self.restarted_calls.append((cwd, system_prompt, persona_label))
        if cwd:
            self._overview["cwd"] = cwd
        if system_prompt is not None:
            self.initial_prompt = system_prompt
        if persona_label is not None:
            self._overview["persona_label"] = persona_label
        self._overview["status"] = "idle"

    def stop_session(self, clear_pending: bool = False, reason: str | None = None) -> None:
        self.stopped += 1
        self._overview["status"] = "stopped"

    def interrupt(self, reason: str | None = None, *, requested_by: str | None = None) -> None:
        self.interrupt_count += 1
        self._overview["status"] = "idle"

    def enqueue(self, prompt: str, source: str = "system", cwd: str | None = None) -> None:
        self.enqueued.append((prompt, source, cwd))
        self._overview["pending_tasks"] += 1

    def queue_priority_prompt(self, prompt: str, source: str = "system", cwd: str | None = None) -> None:
        self.enqueued.insert(0, (prompt, source, cwd))
        self._overview["pending_tasks"] += 1

    def bind_account(self, account_key: str, account_email: str | None = None) -> None:
        self._overview["account_key"] = account_key
        if account_email is not None:
            self._overview["account_email"] = account_email

    def overview(self) -> dict[str, object]:
        return dict(self._overview)

    def recent_transcript(self, lines: int = 12) -> list[_FakeEntry]:
        return []


class FakeControlServer:
    def __init__(self, socket_path, controller) -> None:
        self.socket_path = socket_path
        self.controller = controller

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


class FakeSparkCoordinator:
    def __init__(self, sessions, callback=None) -> None:
        self.sessions = sessions
        self.callback = callback
        self.jobs: dict[str, dict[str, object]] = {}

    def start_job(self, worker, prompt, *, cwd=None, label=None, timeout_seconds=900):
        job = {
            "job_id": "spark-1",
            "worker": worker,
            "status": "running",
            "cwd": cwd,
            "label": label or "spark",
            "prompt_preview": prompt,
            "timeout_seconds": timeout_seconds,
            "result_text": "",
            "error": None,
            "out_of_tokens": False,
        }
        self.jobs[job["job_id"]] = job
        return dict(job)

    def snapshot(self, job_id):
        return dict(self.jobs[job_id])

    def list_jobs(self, worker=None):
        jobs = list(self.jobs.values())
        if worker is not None:
            jobs = [job for job in jobs if job["worker"] == worker]
        return [dict(job) for job in jobs]

    def cancel_job(self, job_id):
        job = self.jobs[job_id]
        job["status"] = "cancelled"
        return dict(job)

    def cancel_active_for_worker(self, worker):
        return None


class FakeWebReasonerManager:
    def __init__(self, callback=None, event_callback=None, **kwargs) -> None:
        self.callback = callback or event_callback
        self.jobs = {}

    def start_job(self, provider, prompt, *, agent, label=None, timeout_seconds=1200):
        job = {
            "job_id": f"{provider}-1",
            "provider": provider,
            "agent": agent,
            "label": label or provider,
            "status": "running",
            "timeout_seconds": timeout_seconds,
            "prompt_preview": prompt,
            "result_text": "",
            "result_capture_source": "",
            "result_quality": "",
            "result_validation_note": "",
            "error": None,
        }
        self.jobs[job["job_id"]] = job
        return dict(job)

    def snapshot(self, job_id):
        return dict(self.jobs[job_id])

    def list_jobs(self, provider=None):
        jobs = list(self.jobs.values())
        if provider is not None:
            jobs = [job for job in jobs if job["provider"] == provider]
        return [dict(job) for job in jobs]

    def cancel_job(self, job_id):
        job = self.jobs[job_id]
        job["status"] = "cancelled"
        return dict(job)


def test_build_worker_event_prompt_ignores_ready_messages(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    controller._user_message_count = 1
    controller._last_user_message = "check status"

    ready_event = SessionEvent(ts=0.0, agent="worker-1", kind="assistant_message", message="Ready for assignments.")
    done_event = SessionEvent(ts=0.0, agent="worker-1", kind="assistant_message", message="Checked the logs.")

    assert controller._build_worker_event_prompt(ready_event) is None
    prompt = controller._build_worker_event_prompt(done_event)
    assert prompt is not None
    assert "worker-1 assistant_message: Checked the logs." in prompt


def test_ready_messages_are_condensed_in_main_chat(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    controller._handle_worker_event(
        SessionEvent(
            ts=0.0,
            agent="claude-worker-3",
            kind="assistant_message",
            message="**Don Norman here — ready for assignments.** I will write a very long introduction.",
        )
    )

    messages = controller.recent_messages(1)
    assert messages[-1].source == "claude-worker-3"
    assert messages[-1].text == "Ready for assignments."


def test_fanout_guidance_for_top_k_requests(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    guidance = controller._fanout_guidance("Find the best approach and compare alternatives for optimization.")

    assert "Use multiple Codex and Claude workers in parallel" in guidance


def test_fanout_guidance_for_non_trivial_requests_prefers_small_swarm(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    guidance = controller._fanout_guidance("Implement a small Rust command that prints the first ten thousand primes and verify it locally.")

    assert "do not over-fan out by default" in guidance
    assert "one implementer plus one verifier or reviewer" in guidance


def test_manager_has_no_hard_turn_timeout(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()

    assert controller.manager.turn_timeout_seconds is None


def test_send_user_message_includes_dependency_and_completion_gates(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    controller.send_user_message("Build a file, then verify it.")

    prompt, source, _cwd = controller.manager.enqueued[-1]
    assert source == "user"
    assert "do not ask workers to poll in loops" in prompt
    assert "Do not send a final success update" in prompt


def test_handle_control_request_passes_cwd_and_persona_to_worker(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    worker = controller.workers["worker-1"]

    response = controller.handle_control_request(
        {"tool": "delegate_to_worker", "arguments": {"worker": "worker-1", "task": "inspect logs", "cwd": "/tmp/repo"}}
    )
    assert response["ok"] is True
    assert worker.enqueued[-1] == ("inspect logs", "manager", "/tmp/repo")

    worker._overview["status"] = "stopped"
    response = controller.handle_control_request(
        {
            "tool": "start_worker_session",
            "arguments": {
                "worker": "worker-1",
                "cwd": "/tmp/repo-a",
                "persona_name": "Ada Lovelace",
                "task_context": "top-k candidate focused on elegant algorithm structure",
                "extra_instructions": "Bias toward a mathematically clean formulation.",
            },
        }
    )
    assert response["ok"] is True
    start_cwd, start_prompt, start_persona = worker.started_calls[-1]
    assert start_cwd == "/tmp/repo-a"
    assert start_persona == "Ada Lovelace"
    assert start_prompt is not None
    assert "Session persona: Ada Lovelace." in start_prompt
    assert "top-k candidate focused on elegant algorithm structure" in start_prompt

    response = controller.handle_control_request(
        {
            "tool": "restart_worker_session",
            "arguments": {
                "worker": "worker-1",
                "cwd": "/tmp/repo-b",
                "persona_name": "John von Neumann",
                "task_context": "top-k candidate focused on raw implementation speed",
                "extra_instructions": "Attack the problem with an aggressive performance mindset.",
            },
        }
    )
    assert response["ok"] is True
    restart_cwd, restart_prompt, restart_persona = worker.restarted_calls[-1]
    assert restart_cwd == "/tmp/repo-b"
    assert restart_persona == "John von Neumann"
    assert restart_prompt is not None
    assert "Session persona: John von Neumann." in restart_prompt


def test_notify_user_rejects_premature_final_completion(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    controller.workers["worker-1"]._overview["status"] = "running"

    response = controller.handle_control_request(
        {"tool": "notify_user", "arguments": {"message": "Done. Everything succeeded."}}
    )

    assert response["ok"] is False
    assert "cannot send a final completion update" in str(response["error"])


def test_notify_user_allows_final_completion_once_work_is_idle(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()

    response = controller.handle_control_request(
        {"tool": "notify_user", "arguments": {"message": "Done. Everything succeeded."}}
    )

    assert response["ok"] is True
    assert controller.recent_messages(1)[-1].text == "Done. Everything succeeded."


def test_worker_spark_tool_uses_paired_worker_name(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    response = controller.handle_control_request(
        {
            "tool": "gpt_5_3_spark",
            "role": "worker",
            "agent": "worker-1",
            "arguments": {"action": "start", "prompt": "draft a first pass", "cwd": "/tmp/repo"},
        }
    )

    assert response["ok"] is True
    assert response["job"]["worker"] == "worker-1"
    assert response["job"]["status"] == "running"


def test_controller_start_starts_all_workers_when_logins_are_present(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)
    monkeypatch.setattr(orch, "missing_codex_logins", lambda: [])
    monkeypatch.setattr(orch, "missing_claude_logins", lambda: [])

    controller = orch.MultiShellController()
    controller.start()

    assert controller.manager.start_count == 1
    assert all(worker.start_count == 1 for worker in controller.codex_workers.values())
    assert all(worker.start_count == 1 for worker in controller.claude_workers.values())


def test_controller_start_fails_when_any_agent_login_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)
    monkeypatch.setattr(orch, "missing_codex_logins", lambda: ["worker-2"])
    monkeypatch.setattr(orch, "missing_claude_logins", lambda: ["claude-worker-3"])

    controller = orch.MultiShellController()

    with pytest.raises(RuntimeError, match="missing Codex login for: worker-2; missing Claude login for: claude-worker-3"):
        controller.start()

    assert controller.manager.start_count == 0
    assert all(worker.start_count == 0 for worker in controller.codex_workers.values())
    assert all(worker.start_count == 0 for worker in controller.claude_workers.values())


def test_monitor_items_use_compact_swarm_labels(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    labels = [item["label"] for item in controller.monitor_items()]

    assert labels[0] == "manager"
    assert "codex-1" in labels
    assert "claude-1" in labels
    assert "spark-1" in labels
    assert "gptpro-1" in labels
    assert "gptpro-2" in labels
    assert "deepthink" in labels
    assert "codex-2" not in labels
    assert "claude-2" not in labels
    assert "spark-2" not in labels


def test_worker_slots_materialize_only_when_used(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()

    assert "worker-1" in controller.workers
    assert "claude-worker-1" in controller.workers
    assert "worker-2" not in controller.workers
    assert "claude-worker-2" not in controller.workers
    assert "worker-2" not in controller.spark_workers

    response = controller.handle_control_request(
        {
            "tool": "restart_worker_session",
            "arguments": {
                "worker": "worker-2",
                "cwd": "/tmp/repo-b",
                "persona_name": "John von Neumann",
                "task_context": "top-k candidate focused on raw implementation speed",
                "extra_instructions": "Attack the problem with an aggressive performance mindset.",
            },
        }
    )

    assert response["ok"] is True
    assert "worker-2" in controller.workers
    assert "worker-2" in controller.codex_workers
    assert "worker-2" in controller.spark_workers
    assert controller.codex_workers["worker-2"].restarted_calls[-1][0] == "/tmp/repo-b"


def test_session_rows_hide_last_error_for_intentional_stop(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    worker = controller.workers["worker-1"]
    worker._overview["status"] = "stopped"
    worker._overview["last_error"] = "stale transport failure"
    controller._mark_session_stopped("worker-1")

    row = next(item for item in controller.session_rows() if item["name"] == "worker-1")

    assert row["status"] == "stopped"
    assert row["last_error"] is None
    assert "stopped intentionally" in str(row["failure_context"])


def test_handle_transport_closed_worker_event_surfaces_error_and_supervision_prompt(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    controller._user_message_count = 1
    controller._last_user_message = "keep the swarm healthy"

    controller._handle_worker_event(
        SessionEvent(
            ts=0.0,
            agent="worker-1",
            kind="transport_closed",
            message="app-server connection closed",
        )
    )

    messages = controller.recent_messages(1)
    assert messages[-1].source == "system"
    assert messages[-1].level == "warn"
    assert messages[-1].text == "worker-1: app-server disconnected; automatically restarted idle session"
    assert controller.manager.enqueued == []


def test_build_worker_event_prompt_ignores_non_terminal_progress_messages(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    controller._user_message_count = 1
    controller._last_user_message = "build and verify"

    prompt = controller._build_worker_event_prompt(
        SessionEvent(ts=0.0, agent="claude-worker-2", kind="assistant_message", message="File doesn't exist yet. Let me poll briefly and retry.")
    )

    assert prompt is None


def test_build_worker_event_prompt_keeps_terminal_worker_messages(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    controller._user_message_count = 1
    controller._last_user_message = "build and verify"

    prompt = controller._build_worker_event_prompt(
        SessionEvent(ts=0.0, agent="worker-2", kind="assistant_message", message="Verification complete. Final line: 104729.")
    )

    assert prompt is not None
    assert "worker-2 assistant_message: Verification complete. Final line: 104729." in prompt


def test_build_worker_event_prompt_keeps_structured_implementer_verification_reports(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    controller._user_message_count = 1
    controller._last_user_message = "build and verify"

    prompt = controller._build_worker_event_prompt(
        SessionEvent(
            ts=0.0,
            agent="worker-3",
            kind="assistant_message",
            message=(
                "Reused the existing file; no rewrite needed.\n\n"
                "- Compile command: `rustc /tmp/prime.rs -O -o /tmp/prime`\n"
                "- Run command: `/tmp/prime > /tmp/prime.out`\n"
                "- Line count result: `10000`\n"
                "- Last line: `104729`"
            ),
        )
    )

    assert prompt is not None
    assert "worker-3 assistant_message: Reused the existing file; no rewrite needed." in prompt


def test_build_worker_event_prompt_keeps_structured_verifier_success_reports(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    controller._user_message_count = 1
    controller._last_user_message = "build and verify"

    prompt = controller._build_worker_event_prompt(
        SessionEvent(
            ts=0.0,
            agent="claude-worker-4",
            kind="assistant_message",
            message=(
                "Verification PASSED - all checks green.\n\n"
                "Compile command: `rustc prime.rs`\n"
                "Line count result: `10000`\n"
                "Last line: `104729`"
            ),
        )
    )

    assert prompt is not None
    assert "claude-worker-4 assistant_message: Verification PASSED - all checks green." in prompt


def test_handle_worker_event_interrupts_stale_manager_turn_for_significant_completion(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    controller._user_message_count = 1
    controller._last_user_message = "build and verify"
    controller.manager._overview["status"] = "running"
    controller.manager._overview["running_for_seconds"] = 12.0
    controller.manager._overview["turn_id"] = "turn-1"

    controller._handle_worker_event(
        SessionEvent(ts=0.0, agent="worker-2", kind="assistant_message", message="Verification complete. Final line: 104729.")
    )

    assert controller.manager.interrupt_count == 1
    prompt, source, _cwd = controller.manager.enqueued[-1]
    assert source == "system"
    assert "worker-2 assistant_message: Verification complete. Final line: 104729." in prompt
    notice = controller.recent_messages(1)[-1]
    assert notice.source == "system"
    assert notice.level == "warn"
    assert notice.text == (
        "interrupted manager turn turn-1 after 12.0s because "
        "worker-2 reported assistant_message: Verification complete. Final line: 104729."
    )


def test_handle_web_reasoner_event_marks_weak_capture_as_warning_and_passes_context(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    controller._user_message_count = 1
    controller._last_user_message = "run a reasoner smoke test"

    event = orch.WebReasonerEvent(
        ts=0.0,
        kind="job_completed",
        provider="gemini_deepthink",
        job_id="gemini-1",
        account_agent="gemini-account-1",
        message="gemini_deepthink job completed",
        data={
            "job_id": "gemini-1",
            "provider": "gemini_deepthink",
            "label": "smoke",
            "result": "Gemini UI chrome",
            "result_quality": "weak",
            "result_capture_source": "main_fallback",
            "result_validation_note": "result was captured via main_fallback; page-level text may include UI chrome",
        },
    )

    controller._handle_web_reasoner_event(event)

    notice = controller.recent_messages(1)[-1]
    assert notice.source == "system"
    assert notice.level == "warn"
    assert notice.text == (
        "gemini_deepthink: completed smoke "
        "(result was captured via main_fallback; page-level text may include UI chrome)"
    )

    prompt, source, _cwd = controller.manager.enqueued[-1]
    assert source == "system"
    assert "Result quality: weak" in prompt
    assert "Capture source: main_fallback" in prompt
    assert "describe it as an extraction/capture issue" in prompt


def test_handle_worker_event_does_not_interrupt_fresh_manager_turn(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    controller._user_message_count = 1
    controller._last_user_message = "build and verify"
    controller.manager._overview["status"] = "running"
    controller.manager._overview["running_for_seconds"] = 3.0
    controller.manager._overview["turn_id"] = "turn-1"

    controller._handle_worker_event(
        SessionEvent(ts=0.0, agent="worker-2", kind="assistant_message", message="Verification complete. Final line: 104729.")
    )

    assert controller.manager.interrupt_count == 0
    prompt, source, _cwd = controller.manager.enqueued[-1]
    assert source == "system"
    assert "worker-2 assistant_message: Verification complete. Final line: 104729." in prompt


def test_monitoring_does_not_interrupt_long_running_manager_turn(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    controller.manager._overview["status"] = "running"
    controller.manager._overview["running_for_seconds"] = 80.0
    controller.manager._overview["turn_id"] = "turn-1"

    rows = controller.session_rows()
    controller._emit_health_alerts(rows)
    controller._check_for_stalls(rows)

    assert controller.manager.interrupt_count == 0


def test_handle_worker_event_logs_interrupted_turn_reason(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()

    controller._handle_worker_event(
        SessionEvent(
            ts=0.0,
            agent="worker-1",
            kind="turn_failed",
            message="turn interrupted because session restart was requested",
            data={
                "interrupted": True,
                "interrupt_reason": "session restart was requested",
                "duration_seconds": 8.25,
                "turn_status": "interrupted",
            },
        )
    )

    notice = controller.recent_messages(1)[-1]
    assert notice.source == "system"
    assert notice.level == "warn"
    assert notice.text == "worker-1: turn interrupted after 8.2s because session restart was requested"


def test_transport_closed_idle_codex_worker_auto_restarts_and_updates_failure_context(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    worker = controller.codex_workers["worker-1"]
    worker._overview["status"] = "idle"
    worker._overview["cwd"] = "/tmp/repo"
    worker._overview["persona_label"] = "Grace Hopper"

    controller._handle_worker_event(
        SessionEvent(
            ts=10.0,
            agent="worker-1",
            kind="transport_closed",
            message="app-server connection closed",
        )
    )

    assert worker.restarted_calls[-1] == ("/tmp/repo", worker.initial_prompt, "Grace Hopper")

    message = controller.recent_messages(1)[-1]
    assert message.source == "system"
    assert message.level == "warn"
    assert message.text == "worker-1: app-server disconnected; automatically restarted idle session"

    row = next(row for row in controller.session_rows() if row["name"] == "worker-1")
    assert row["auto_restarts"] == 1
    assert row["disconnect_streak"] == 1
    assert row["last_failure_kind"] == "transport_closed"
    assert row["last_recovery_action"] == "auto_restart"
    assert row["failure_context"] == "app-server connection closed; auto-restarted once"
    assert row["last_error"] == "app-server connection closed; auto-restarted once"


def test_repeated_transport_closed_suppresses_auto_restart_and_flags_manual_recovery(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)

    controller = orch.MultiShellController()
    worker = controller.codex_workers["worker-1"]
    worker._overview["status"] = "idle"
    worker._overview["cwd"] = "/tmp/repo"
    worker._overview["persona_label"] = "Grace Hopper"

    first = SessionEvent(ts=10.0, agent="worker-1", kind="transport_closed", message="app-server connection closed")
    second = SessionEvent(ts=11.0, agent="worker-1", kind="transport_closed", message="app-server connection closed")

    controller._handle_worker_event(first)
    controller._handle_worker_event(second)

    assert len(worker.restarted_calls) == 1

    message = controller.recent_messages(1)[-1]
    assert message.source == "system"
    assert message.level == "error"
    assert message.text == (
        "worker-1: app-server connection closed "
        "(manual restart recommended after repeated or in-flight disconnect)"
    )

    row = next(row for row in controller.session_rows() if row["name"] == "worker-1")
    assert row["auto_restarts"] == 1
    assert row["disconnect_streak"] == 2
    assert row["last_failure_kind"] == "transport_closed"
    assert row["last_recovery_action"] == "auto_restart"
    assert row["failure_context"] == (
        "app-server disconnected 2 times; auto-restart suppressed; manual restart recommended"
    )
    assert row["last_error"] == row["failure_context"]
