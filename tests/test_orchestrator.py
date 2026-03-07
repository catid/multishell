from __future__ import annotations

from dataclasses import dataclass

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
        reasoning_effort="medium",
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

    def start(self) -> None:
        self.start_count += 1
        self._overview["status"] = "idle"

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

    def stop_session(self, clear_pending: bool = False) -> None:
        self.stopped += 1
        self._overview["status"] = "stopped"

    def interrupt(self) -> None:
        self._overview["status"] = "idle"

    def enqueue(self, prompt: str, source: str = "system", cwd: str | None = None) -> None:
        self.enqueued.append((prompt, source, cwd))
        self._overview["pending_tasks"] += 1

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


def test_controller_start_skips_claude_workers_without_login(monkeypatch) -> None:
    monkeypatch.setattr(orch, "CodexSession", FakeSession)
    monkeypatch.setattr(orch, "ClaudeSession", FakeSession)
    monkeypatch.setattr(orch, "SparkCoordinator", FakeSparkCoordinator)
    monkeypatch.setattr(orch, "WebReasonerManager", FakeWebReasonerManager)
    monkeypatch.setattr(orch, "ControlServer", FakeControlServer)
    monkeypatch.setattr(orch, "claude_logged_in", lambda name: False)

    controller = orch.MultiShellController()
    controller.start()

    assert controller.manager.start_count == 1
    assert all(worker.start_count == 1 for worker in controller.codex_workers.values())
    assert all(worker.start_count == 0 for worker in controller.claude_workers.values())
