from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

from .claude_session import ClaudeSession
from .codex_session import CodexSession, SessionEvent, TranscriptEntry
from .config import (
    CLAUDE_WORKER_SPECS,
    MANAGER_SPEC,
    SPARK_MODEL,
    SPARK_REASONING_EFFORT,
    WORKER_SPECS,
    AgentSpec,
    app_root,
    manager_workspace_root,
    socket_path,
    spark_agent_name,
    workspace_root,
)
from .control import ControlServer
from .homes import claude_logged_in
from .spark_pool import SparkCoordinator, SparkError
from .web_reasoners import WebReasonerError, WebReasonerEvent, WebReasonerManager


ManagedSession = CodexSession | ClaudeSession


def _bridge_command(role: str, agent: str) -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import sys; "
            f"sys.path.insert(0, {str(app_root())!r}); "
            "from multishell.mcp_bridge import main; "
            "raise SystemExit(main())"
        ),
        "--socket",
        str(socket_path()),
        "--role",
        role,
        "--agent",
        agent,
    ]


@dataclass
class UiMessage:
    ts: float
    source: str
    text: str
    level: str = "info"


@dataclass(frozen=True)
class WorkerArchetype:
    persona_name: str
    domain: str
    approach: str
    coaching: str


WORKER_ARCHETYPES = {
    "worker-1": WorkerArchetype(
        persona_name="Grace Hopper",
        domain="practical programming languages and decisive implementation",
        approach="ship a correct first cut quickly, then tighten it without drifting into ceremony",
        coaching="Prefer concrete code and direct progress over long speculation. Keep the implementation moving.",
    ),
    "worker-2": WorkerArchetype(
        persona_name="Edsger Dijkstra",
        domain="systems reasoning, invariants, and operational correctness",
        approach="treat state, failure cases, and interface contracts as first-class design material",
        coaching="Surface hidden assumptions early. Protect correctness before convenience.",
    ),
    "worker-3": WorkerArchetype(
        persona_name="Douglas Engelbart",
        domain="interactive computing, tools, and human-centered workflows",
        approach="optimize for usability, clarity, and operator leverage in the terminal",
        coaching="Make the workflow easier to understand and drive. Bias toward lucid interaction design.",
    ),
    "worker-4": WorkerArchetype(
        persona_name="Katherine Johnson",
        domain="precise verification, cross-checking, and integration confidence",
        approach="stress the joins between pieces and validate the final answer against edge cases",
        coaching="Act as a sharp closer. Verify, compare, and call out subtle mistakes quickly.",
    ),
    "claude-worker-5": WorkerArchetype(
        persona_name="Richard Feynman",
        domain="cross-disciplinary reasoning and sharp explanation",
        approach="generate alternate framings, critique assumptions, and look for elegant idea pivots",
        coaching="Lean into diverse reasoning paths and code review. Be explicit where confidence is lower.",
    ),
    "claude-worker-1": WorkerArchetype(
        persona_name="Leonardo da Vinci",
        domain="creative invention and divergent solution generation",
        approach="explore different implementation shapes before converging on the most interesting candidates",
        coaching="Generate diverse options instead of one average answer. Value novelty, then practicality.",
    ),
    "claude-worker-2": WorkerArchetype(
        persona_name="Donald Knuth",
        domain="program analysis, critique, and algorithmic reflection",
        approach="review for subtle reasoning errors, unclear invariants, and more interesting algorithmic choices",
        coaching="Use the different model family to pressure-test code and arguments rather than mirror Codex.",
    ),
    "claude-worker-3": WorkerArchetype(
        persona_name="Don Norman",
        domain="interaction design and human factors",
        approach="push on usability, operator empathy, and alternative UX directions that improve leverage",
        coaching="Offer surprising but coherent UX alternatives and review terminal flows critically.",
    ),
    "claude-worker-4": WorkerArchetype(
        persona_name="Barbara Liskov",
        domain="interfaces, abstraction, and rigorous review",
        approach="review joins between components and propose alternative abstractions when the current design is brittle",
        coaching="Be a creative reviewer and closer. Find mismatches and suggest cleaner directions.",
    ),
}


def manager_prompt() -> str:
    codex_lines = [f"- {spec.name}: {spec.personality}" for spec in WORKER_SPECS]
    claude_lines = [f"- {spec.name}: {spec.personality}" for spec in CLAUDE_WORKER_SPECS]
    return f"""You are the multishell manager.

Operate only through delegation, session management, long-running reasoning tools, and user communication.

Rules:
- Your own working directory is isolated on purpose. Do not inspect or edit project files yourself.
- Use `delegate_to_worker` to assign concrete tasks to named workers.
- Use `start_worker_session`, `stop_worker_session`, and `restart_worker_session` to manage worker memory and working directories.
- Use `get_workers_overview` and `get_worker_transcript` to monitor progress and compare candidates.
- Use `notify_user` for all user-facing messages. Do not assume plain assistant text reaches the user.
- Use `gpt_5_4_pro` and `gemini_deepthink` for slow planning, research, world knowledge, or math-heavy reasoning in parallel with worker execution.
- Start slow web reasoners early on hard tasks, continue delegating while they run, then incorporate useful revisions after they complete.
- Restart workers for unrelated tasks so stale memory does not leak across problems.
- When starting or restarting a worker, craft a fresh session persona using `persona_name`, `task_context`, and `extra_instructions`.
- For non-trivial tasks, use more than one worker. For top-k or uncertain work, fan out aggressively across Codex and Claude workers.
- Codex workers are generally more reliable for execution. Claude workers are more creative and often useful for alternative ideas and code review.
- Use cross-review patterns for diversity: Claude drafts with Codex review, Codex drafts with Claude review, or parallel candidates from both families.
- Claude is especially useful for code review, idea expansion, alternative framings, and different-model perspective. Treat it as creative but less reliable.
- Use Gemini Deep Think when you want slower but broader world knowledge or math-heavy parallel reasoning. It only runs on the manager/bot account.
- Choose worker working directories intentionally. You can pass `cwd` when starting, restarting, or delegating.
- Keep messages concise and operational.

Codex workers:
{chr(10).join(codex_lines)}

Claude workers:
{chr(10).join(claude_lines)}
"""


def _worker_prompt(
    worker_name: str,
    personality: str,
    *,
    engine: str,
    persona_name: str | None = None,
    task_context: str | None = None,
    extra_instructions: str | None = None,
) -> str:
    archetype = WORKER_ARCHETYPES.get(
        worker_name,
        WorkerArchetype(
            persona_name=worker_name,
            domain="software engineering",
            approach="solve the assigned task directly",
            coaching="Work concretely and note blockers quickly.",
        ),
    )
    resolved_persona = persona_name or archetype.persona_name
    resolved_task_context = task_context or "Handle the assigned work using this session's specialty and angle."
    resolved_extra = extra_instructions or archetype.coaching
    base = [
        f"You are {worker_name}.",
        "",
        f"Session persona: {resolved_persona}.",
        f"Adopt the strongest traits of {resolved_persona}, historically brilliant in {archetype.domain}.",
        f"Base specialty: {personality}",
        f"Primary working angle: {archetype.approach}",
        f"Session objective: {resolved_task_context}",
        f"Additional instructions: {resolved_extra}",
        "",
    ]
    if engine == "claude":
        base.extend(
            [
                "You are one persistent Claude worker session in a coordinated swarm.",
                "Claude is more creative than Codex but less reliable, so lean into alternative ideas, review, and fresh perspectives.",
                "Good uses: creative exploration, generating candidate implementations, code review for Codex output, and broader idea search.",
                "When asked for top-k or comparison work, generate materially different options instead of converging too early.",
                "Be explicit about uncertainty and things that need Codex or the manager to verify.",
                "If you have no task yet, reply once that you are ready for assignments.",
            ]
        )
    else:
        base.extend(
            [
                "You are one persistent Codex worker session in a coordinated swarm. Work directly in the assigned repository when given a task.",
                "Before drafting substantial code or design work, use `gpt_5_3_spark` to get a first draft, alternative angle, or quick review on the same account.",
                "Use `gpt_5_3_spark` with `action=start`, keep inspecting the repo while it runs, then use `action=status` or `action=list` to collect the result.",
                "If spark reports OUT_OF_TOKENS, says it is out of tokens, or the tool fails, continue the work yourself without blocking.",
                "Do not trust spark to modify existing files. Treat spark output as draft or review material only. Review any proposed edits before writing them.",
                "Spark may create new scratch files only when explicitly asked; it should not be trusted to edit existing files directly.",
                "Be concrete, state what you changed, and note blockers quickly.",
                "If you are part of a top-k exploration, lean into your assigned angle instead of averaging toward the other workers.",
                "If you have no task yet, reply once that you are ready for assignments.",
            ]
        )
    return "\n".join(base)


def _spark_prompt(worker_name: str, worker_personality: str, *, persona_name: str) -> str:
    return f"""You are {spark_agent_name(worker_name)}, the paired draft delegate for {worker_name}.

Session persona: {persona_name}.
You are a persistent Codex session using gpt-5.3-spark at xhigh reasoning.
Primary job: produce first drafts, alternative approaches, review notes, candidate code, and scratch artifacts for {worker_name}.
Base specialty of the paired owner worker: {worker_personality}
You are not the final authority. Your output will be reviewed by {worker_name} before it reaches disk.

Rules:
- Do not edit existing files.
- If asked to write something, prefer inline draft content or new scratch files only.
- If the task exceeds your token or context budget, reply starting with `OUT_OF_TOKENS:` and stop.
- Keep answers direct, practical, and draft-oriented.
"""


class MultiShellController:
    def __init__(self) -> None:
        self.manager = CodexSession(
            MANAGER_SPEC,
            manager_prompt(),
            mcp_bridge_command=_bridge_command("manager", MANAGER_SPEC.name),
            working_dir=manager_workspace_root(),
            persona_label="Multishell Manager",
            message_callback=self._handle_manager_message,
            turn_timeout_seconds=90,
        )
        self.workers: dict[str, ManagedSession] = {}
        self.codex_workers: dict[str, CodexSession] = {}
        self.claude_workers: dict[str, ClaudeSession] = {}
        self.spark_workers: dict[str, CodexSession] = {}
        self._codex_worker_names = {spec.name for spec in WORKER_SPECS}

        for spec in WORKER_SPECS:
            session = CodexSession(
                spec,
                _worker_prompt(spec.name, spec.personality, engine="codex"),
                mcp_bridge_command=_bridge_command("worker", spec.name),
                working_dir=workspace_root(),
                startup_prompt="Reply once that you are ready for assignments.",
                persona_label=self._default_persona_label(spec.name),
                event_callback=self._handle_worker_event,
                turn_timeout_seconds=900,
            )
            self.codex_workers[spec.name] = session
            self.workers[spec.name] = session

        for spec in CLAUDE_WORKER_SPECS:
            session = ClaudeSession(
                spec,
                _worker_prompt(spec.name, spec.personality, engine="claude"),
                working_dir=workspace_root(),
                startup_prompt="Reply once that you are ready for assignments.",
                persona_label=self._default_persona_label(spec.name),
                event_callback=self._handle_worker_event,
                turn_timeout_seconds=900,
            )
            self.claude_workers[spec.name] = session
            self.workers[spec.name] = session

        for spec in WORKER_SPECS:
            spark_spec = AgentSpec(
                name=spark_agent_name(spec.name),
                account_email=spec.account_email,
                role="spark",
                personality=f"Draft delegate paired with {spec.name}",
                accent_color=spec.accent_color,
            )
            self.spark_workers[spec.name] = CodexSession(
                spark_spec,
                _spark_prompt(spec.name, spec.personality, persona_name=f"{self._default_persona_label(spec.name)} Draft Partner"),
                working_dir=workspace_root(),
                persona_label=f"{self._default_persona_label(spec.name)} Spark",
                event_callback=self._handle_spark_session_event,
                turn_timeout_seconds=900,
                model=SPARK_MODEL,
                reasoning_effort=SPARK_REASONING_EFFORT,
                auth_source_agent=spec.name,
            )

        self.spark_pool = SparkCoordinator(self.spark_workers, callback=self._handle_spark_update)
        self.web_reasoners = WebReasonerManager(callback=self._handle_web_reasoner_event)
        self.messages: list[UiMessage] = []
        self._lock = threading.Lock()
        self._control_server = ControlServer(socket_path(), self)
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="multishell-monitor", daemon=True)
        self._stop = threading.Event()
        self._last_session_alerts: dict[str, tuple[object, ...]] = {}
        self._last_stall_alerts: dict[str, int] = {}
        self._last_manager_event_signatures: dict[tuple[str, str], float] = {}
        self._user_message_count = 0
        self._last_user_message = ""
        self._last_manager_interrupt_turn: str | None = None
        self._shutting_down = False

    def start(self) -> None:
        self._shutting_down = False
        self._control_server.start()
        self.manager.start()
        for worker in self.codex_workers.values():
            worker.start()
        for name, worker in self.claude_workers.items():
            if claude_logged_in(name):
                worker.start()
            else:
                self._push_message("system", f"{name}: Claude login missing; worker left unavailable", level="warn")
        self._push_message("system", "multishell started", level="info")
        self._monitor_thread.start()

    def stop(self) -> None:
        self._shutting_down = True
        self._stop.set()
        self._control_server.stop()
        self.web_reasoners.stop()
        self.spark_pool.stop()
        for worker in self.workers.values():
            worker.stop()
        for spark in self.spark_workers.values():
            spark.stop()
        self.manager.stop()
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2)

    def handle_control_request(self, request: dict[str, object]) -> dict[str, object]:
        tool = request.get("tool")
        bridge_role = str(request.get("bridge_role") or request.get("role") or "manager")
        bridge_agent = str(request.get("bridge_agent") or request.get("agent") or "")
        arguments = request.get("arguments", {})
        if not isinstance(arguments, dict):
            return {"ok": False, "error": "arguments must be an object"}

        try:
            if tool == "gpt_5_3_spark":
                if bridge_role != "worker" or not bridge_agent:
                    return {"ok": False, "error": "spark jobs can only be started from a worker bridge"}
                return self._handle_spark_request(bridge_agent, arguments)
            return self._handle_manager_request(tool, arguments)
        except (SparkError, WebReasonerError, KeyError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    def _handle_manager_request(self, tool: object, arguments: dict[str, object]) -> dict[str, object]:
        if tool == "notify_user":
            message = str(arguments.get("message", "")).strip()
            level = str(arguments.get("level", "info"))
            self._push_message("manager", message, level=level)
            return {"ok": True, "message": "user notified"}

        if tool == "delegate_to_worker":
            worker_name = str(arguments.get("worker", "")).strip()
            task = str(arguments.get("task", "")).strip()
            cwd = str(arguments.get("cwd", "")).strip() or None
            worker = self.workers.get(worker_name)
            if worker is None:
                return {"ok": False, "error": f"unknown worker: {worker_name}"}
            if worker.overview()["status"] == "stopped":
                return {"ok": False, "error": f"{worker_name} session is stopped; start or restart it first"}
            worker.enqueue(task, source="manager", cwd=cwd)
            target = cwd or worker.overview()["cwd"]
            self._push_message("manager", f"delegated to {worker_name} ({target}): {task}")
            return {"ok": True, "message": f"queued for {worker_name}"}

        if tool == "start_worker_session":
            worker_name = str(arguments.get("worker", "")).strip()
            cwd = str(arguments.get("cwd", "")).strip() or None
            worker = self.workers.get(worker_name)
            if worker is None:
                return {"ok": False, "error": f"unknown worker: {worker_name}"}
            if worker.overview()["status"] != "stopped":
                return {"ok": False, "error": f"{worker_name} session is already running; use restart_worker_session"}
            system_prompt, persona_label = self._resolve_worker_session_prompt(worker_name, arguments)
            worker.start_session(cwd, system_prompt=system_prompt, persona_label=persona_label)
            self._sync_paired_spark_session(
                worker_name,
                cwd=cwd,
                persona_label=persona_label,
                task_context=str(arguments.get("task_context", "")).strip() or None,
                action="start",
            )
            self._push_message("manager", f"started {worker_name} session in {worker.overview()['cwd']} as {worker.overview()['persona_label']}")
            return {"ok": True, "message": f"started {worker_name}"}

        if tool == "stop_worker_session":
            worker_name = str(arguments.get("worker", "")).strip()
            worker = self.workers.get(worker_name)
            if worker is None:
                return {"ok": False, "error": f"unknown worker: {worker_name}"}
            worker.stop_session(clear_pending=True)
            self._sync_paired_spark_session(worker_name, action="stop")
            self._push_message("manager", f"stopped {worker_name} session")
            return {"ok": True, "message": f"stopped {worker_name}"}

        if tool == "restart_worker_session":
            worker_name = str(arguments.get("worker", "")).strip()
            cwd = str(arguments.get("cwd", "")).strip() or None
            worker = self.workers.get(worker_name)
            if worker is None:
                return {"ok": False, "error": f"unknown worker: {worker_name}"}
            system_prompt, persona_label = self._resolve_worker_session_prompt(worker_name, arguments)
            worker.restart_session(cwd, system_prompt=system_prompt, persona_label=persona_label)
            self._sync_paired_spark_session(
                worker_name,
                cwd=cwd,
                persona_label=persona_label,
                task_context=str(arguments.get("task_context", "")).strip() or None,
                action="restart",
            )
            self._push_message("manager", f"restarted {worker_name} session in {worker.overview()['cwd']} as {worker.overview()['persona_label']}")
            return {"ok": True, "message": f"restarted {worker_name}"}

        if tool == "get_workers_overview":
            return {"ok": True, "workers": [worker.overview() for worker in self.workers.values()]}

        if tool == "get_worker_transcript":
            worker_name = str(arguments.get("worker", "")).strip()
            try:
                line_count = int(arguments.get("lines", 12))
            except (TypeError, ValueError):
                return {"ok": False, "error": "lines must be an integer"}
            line_count = max(1, min(40, line_count))
            worker = self.workers.get(worker_name)
            if worker is None:
                return {"ok": False, "error": f"unknown worker: {worker_name}"}
            entries = [{"ts": entry.ts, "source": entry.source, "text": entry.text} for entry in worker.recent_transcript(line_count)]
            return {"ok": True, "worker": worker_name, "entries": entries}

        if tool == "gpt_5_4_pro":
            return self._handle_web_reasoner_request("chatgpt_pro", arguments)

        if tool == "gemini_deepthink":
            return self._handle_web_reasoner_request("gemini_deepthink", arguments)

        return {"ok": False, "error": f"unknown tool: {tool}"}

    def _handle_spark_request(self, owner_worker: str, arguments: dict[str, object]) -> dict[str, object]:
        if owner_worker not in self.codex_workers:
            return {"ok": False, "error": f"{owner_worker} does not have a paired spark delegate"}
        action = str(arguments.get("action", "")).strip().lower()
        if action == "start":
            prompt = str(arguments.get("prompt", "")).strip()
            label = str(arguments.get("label", "")).strip() or None
            cwd = str(arguments.get("cwd", "")).strip() or None
            if not prompt:
                return {"ok": False, "error": "prompt is required for action=start"}
            job = self.spark_pool.start_job(owner_worker, prompt, cwd=cwd, label=label)
            return {"ok": True, "job": job}
        if action == "status":
            job_id = str(arguments.get("job_id", "")).strip()
            if not job_id:
                return {"ok": False, "error": "job_id is required for action=status"}
            job = self.spark_pool.job_snapshot(job_id)
            if job is None or str(job["worker"]) != owner_worker:
                return {"ok": False, "error": f"unknown spark job: {job_id}"}
            return {"ok": True, "job": job}
        if action == "list":
            return {"ok": True, "jobs": self.spark_pool.list_jobs(owner_worker)}
        if action == "cancel":
            job_id = str(arguments.get("job_id", "")).strip()
            if not job_id:
                return {"ok": False, "error": "job_id is required for action=cancel"}
            job = self.spark_pool.job_snapshot(job_id)
            if job is None or str(job["worker"]) != owner_worker:
                return {"ok": False, "error": f"unknown spark job: {job_id}"}
            return {"ok": True, "job": self.spark_pool.cancel_job(job_id)}
        return {"ok": False, "error": "action must be one of: start, status, list, cancel"}

    def _handle_web_reasoner_request(self, provider: str, arguments: dict[str, object]) -> dict[str, object]:
        action = str(arguments.get("action", "")).strip().lower()
        if action == "start":
            prompt = str(arguments.get("prompt", "")).strip()
            label = str(arguments.get("label", "")).strip() or None
            agent = str(arguments.get("agent", "")).strip() or MANAGER_SPEC.name
            try:
                timeout_seconds = int(arguments.get("timeout_seconds", 900))
            except (TypeError, ValueError):
                return {"ok": False, "error": "timeout_seconds must be an integer"}
            if not prompt:
                return {"ok": False, "error": "prompt is required for action=start"}
            if provider == "gemini_deepthink":
                agent = MANAGER_SPEC.name
            job = self.web_reasoners.start_job(provider, agent, prompt, label=label, timeout_seconds=timeout_seconds)
            self._push_message("manager", f"started {provider} job {job['id'][:8]} on {agent}: {job['label']}")
            return {"ok": True, "job": job}
        if action == "status":
            job_id = str(arguments.get("job_id", "")).strip()
            if not job_id:
                return {"ok": False, "error": "job_id is required for action=status"}
            job = self.web_reasoners.job_snapshot(job_id)
            if job is None:
                return {"ok": False, "error": f"unknown job: {job_id}"}
            return {"ok": True, "job": job}
        if action == "list":
            return {"ok": True, "jobs": self.web_reasoners.list_jobs(provider=provider)}
        if action == "cancel":
            job_id = str(arguments.get("job_id", "")).strip()
            if not job_id:
                return {"ok": False, "error": "job_id is required for action=cancel"}
            job = self.web_reasoners.cancel_job(job_id)
            if job is None:
                return {"ok": False, "error": f"unknown job: {job_id}"}
            return {"ok": True, "job": job}
        return {"ok": False, "error": "action must be one of: start, status, list, cancel"}

    def send_user_message(self, text: str) -> None:
        manager_overview = self.manager.overview()
        running_for = manager_overview.get("running_for_seconds")
        if manager_overview["status"] == "running" and isinstance(running_for, float) and running_for >= 15:
            self.manager.interrupt()
            self._push_message("system", "interrupted a stale manager turn to handle new user input", level="warn")
        fanout_guidance = self._fanout_guidance(text)
        prompt = (
            "User message:\n"
            f"{text}\n\n"
            "Current worker overview:\n"
            f"{self._overview_text()}\n\n"
            "Respond using tools only. If the task is unrelated to ongoing worker memory, restart or stop the relevant worker "
            "sessions first. Use Codex workers for reliable implementation and Claude workers for diversity, review, and cross-model "
            "perspective. Start slow web reasoners early when long-horizon planning, outside knowledge, or heavy reasoning may help later. "
            "For substantial work, split it into distinct parallel assignments. Choose worker working directories intentionally and notify the user."
            f"{fanout_guidance}"
        )
        self._user_message_count += 1
        self._last_user_message = text
        self._push_message("user", text, level="info")
        self.manager.enqueue(prompt, source="user")

    def session_rows(self) -> list[dict[str, object]]:
        rows = [self.manager.overview()]
        rows.extend(worker.overview() for worker in self.codex_workers.values())
        rows.extend(worker.overview() for worker in self.claude_workers.values())
        return rows

    def recent_messages(self, limit: int = 80) -> list[UiMessage]:
        with self._lock:
            return self.messages[-limit:]

    def add_notice(self, text: str, level: str = "info") -> None:
        self._push_message("system", text, level=level)

    def active_reasoner_counts(self) -> dict[str, int]:
        jobs = self.web_reasoners.list_jobs()
        running = sum(1 for job in jobs if job["status"] == "running")
        queued = sum(1 for job in jobs if job["status"] == "queued")
        return {"running": running, "queued": queued}

    def transcripts_for_debug(self) -> dict[str, list[TranscriptEntry]]:
        data = {self.manager.spec.name: self.manager.recent_transcript(18)}
        for name, worker in self.workers.items():
            data[name] = worker.recent_transcript(18)
        return data

    def _monitor_loop(self) -> None:
        while not self._stop.wait(5):
            rows = self.session_rows()
            self._enforce_manager_timeout(rows[0])
            self._emit_health_alerts(rows)
            self._check_for_stalls(rows)

    def _overview_text(self) -> str:
        lines = []
        for worker in self.workers.values():
            overview = worker.overview()
            timing = ""
            if overview["status"] == "running" and overview["running_for_seconds"] is not None:
                timing = f" running_for={overview['running_for_seconds']:.1f}s"
            elif overview["last_turn_duration"] is not None:
                timing = f" last_turn={overview['last_turn_duration']:.1f}s"
            lines.append(
                f"- {overview['name']}: engine={overview.get('engine', 'codex')} model={overview.get('model', '-')!r} "
                f"persona={overview['persona_label']!r} status={overview['status']} cwd={overview['cwd']!r} "
                f"pending={overview['pending_tasks']} completed={overview['completed_turns']} failed={overview['failed_turns']}{timing} "
                f"last_message={overview['last_message']!r}"
            )
        return "\n".join(lines)

    def _push_message(self, source: str, text: str, level: str = "info") -> None:
        with self._lock:
            self.messages.append(UiMessage(ts=time.time(), source=source, text=text, level=level))
            self.messages = self.messages[-260:]

    def _emit_health_alerts(self, rows: list[dict[str, object]]) -> None:
        for row in rows:
            signature = (row["status"], row["failed_turns"], row["last_error"], row.get("cwd"))
            previous = self._last_session_alerts.get(str(row["name"]))
            if signature == previous:
                continue
            self._last_session_alerts[str(row["name"])] = signature
            if row["status"] == "error":
                detail = str(row["last_error"] or "unknown error")
                self._push_message("system", f"{row['name']} entered error state: {detail}", level="error")
            elif previous and previous[0] == "error":
                self._push_message("system", f"{row['name']} recovered to {row['status']}", level="info")

    def _enforce_manager_timeout(self, manager_overview: dict[str, object]) -> None:
        running_for = manager_overview.get("running_for_seconds")
        turn_id = str(manager_overview.get("turn_id") or "")
        if manager_overview["status"] != "running" or not isinstance(running_for, float) or running_for < 90:
            return
        if turn_id and turn_id == self._last_manager_interrupt_turn:
            return
        self._last_manager_interrupt_turn = turn_id or None
        self.manager.interrupt()
        self._push_message("system", "interrupted a manager turn that exceeded 90 seconds", level="warn")

    def _handle_manager_message(self, entry: TranscriptEntry) -> None:
        self._push_message("manager", entry.text, level="info")

    def _default_persona_label(self, worker_name: str) -> str:
        archetype = WORKER_ARCHETYPES.get(worker_name)
        return archetype.persona_name if archetype is not None else worker_name

    def _fanout_guidance(self, text: str) -> str:
        normalized = text.lower()
        top_k_keywords = (
            "top-k",
            "top@k",
            "best",
            "better",
            "improve",
            "optimization",
            "optimize",
            "alternative",
            "alternatives",
            "compare",
            "comparison",
            "explore",
            "brainstorm",
            "review",
            "diverse",
        )
        if any(keyword in normalized for keyword in top_k_keywords):
            return (
                "\n\nThis request matches top-k exploration. Use multiple Codex and Claude workers in parallel with fresh personas and "
                "different attack angles. Start one or both slow web reasoners early if long-horizon planning or outside knowledge may revise the path later."
            )
        if len(text.split()) >= 18:
            return "\n\nThis is not a trivial request. Prefer parallel delegation across several Codex and Claude workers rather than serial execution."
        return ""

    def _resolve_worker_session_prompt(self, worker_name: str, arguments: dict[str, object]) -> tuple[str, str]:
        worker = self.workers[worker_name]
        persona_name = str(arguments.get("persona_name", "")).strip() or self._default_persona_label(worker_name)
        task_context = str(arguments.get("task_context", "")).strip() or None
        extra_instructions = str(arguments.get("extra_instructions", "")).strip() or None
        system_prompt = str(arguments.get("system_prompt", "")).strip() or None
        if system_prompt:
            return system_prompt, persona_name
        engine = "claude" if worker_name.startswith("claude-") else "codex"
        return (
            _worker_prompt(
                worker_name,
                worker.spec.personality,
                engine=engine,
                persona_name=persona_name,
                task_context=task_context,
                extra_instructions=extra_instructions,
            ),
            persona_name,
        )

    def _sync_paired_spark_session(
        self,
        worker_name: str,
        *,
        cwd: str | None = None,
        persona_label: str | None = None,
        task_context: str | None = None,
        action: str,
    ) -> None:
        if worker_name not in self.spark_workers:
            return
        self.spark_pool.cancel_active_for_worker(worker_name)
        spark = self.spark_workers[worker_name]
        spark_prompt = _spark_prompt(
            worker_name,
            self.codex_workers[worker_name].spec.personality,
            persona_name=f"{persona_label or self._default_persona_label(worker_name)} Draft Partner",
        )
        if task_context:
            spark_prompt = spark_prompt + f"\nCurrent owner task context: {task_context}\n"
        if action == "start":
            if spark.overview()["status"] == "stopped":
                spark.start_session(cwd, system_prompt=spark_prompt, persona_label=f"{persona_label or self._default_persona_label(worker_name)} Spark")
        elif action == "restart":
            spark.restart_session(cwd, system_prompt=spark_prompt, persona_label=f"{persona_label or self._default_persona_label(worker_name)} Spark")
        elif action == "stop":
            spark.stop_session(clear_pending=True)

    def _handle_worker_event(self, event: SessionEvent) -> None:
        if event.kind == "assistant_message":
            self._push_message(event.agent, self._display_worker_message(event), level="info")
        elif event.kind in {"turn_failed", "transport_closed", "auth_error", "mcp_failed", "error"}:
            self._push_message("system", f"{event.agent}: {event.message}", level="error")
        elif event.kind in {"session_started", "session_stopped"}:
            self._push_message("system", f"{event.agent}: {event.message}", level="info")

        if self._shutting_down:
            return
        prompt = self._build_worker_event_prompt(event)
        if prompt:
            self.manager.enqueue(prompt, source="system")

    def _handle_spark_session_event(self, event: SessionEvent) -> None:
        # Spark jobs are polled via SparkCoordinator; keep the event hook in place for future deeper integration.
        return None

    def _handle_spark_update(self, kind: str, snapshot: dict[str, object]) -> None:
        worker = str(snapshot["worker"])
        job_id = str(snapshot["job_id"])
        if kind == "failed":
            self._push_message("system", f"{worker} spark job {job_id} failed: {snapshot.get('error')}", level="warn")
        elif kind == "completed" and bool(snapshot.get("out_of_tokens")):
            self._push_message("system", f"{worker} spark job {job_id} reported token exhaustion", level="warn")

    def _handle_web_reasoner_event(self, event: WebReasonerEvent) -> None:
        snapshot = event.data
        provider = event.provider
        label = str(snapshot.get("label") or provider)
        if event.kind == "job_started":
            self._push_message("system", f"{provider}: started {label}", level="info")
            return
        if event.kind == "job_completed":
            self._push_message("system", f"{provider}: completed {label}", level="info")
            result = str(snapshot.get("result") or "")
            if self._user_message_count > 0 and not self._shutting_down:
                self.manager.enqueue(
                    (
                        "Parallel web reasoning job completed.\n"
                        f"Latest user request: {self._last_user_message}\n"
                        f"Provider: {provider}\n"
                        f"Label: {label}\n"
                        f"Result:\n{result}\n\n"
                        "Decide whether to revise delegation, restart workers, or send a better user-facing update. Use tools only.\n\n"
                        f"{self._overview_text()}"
                    ),
                    source="system",
                )
            return
        self._push_message("system", f"{provider}: {event.message}", level="warn" if event.kind == "job_canceled" else "error")
        if self._user_message_count > 0 and event.kind == "job_failed" and not self._shutting_down:
            self.manager.enqueue(
                (
                    "Parallel web reasoning job failed.\n"
                    f"Latest user request: {self._last_user_message}\n"
                    f"Provider: {provider}\n"
                    f"Error: {event.message}\n\n"
                    "React only if this changes delegation or user-visible status. Use tools only.\n\n"
                    f"{self._overview_text()}"
                ),
                source="system",
            )

    def _build_worker_event_prompt(self, event: SessionEvent) -> str | None:
        if self._shutting_down or self._user_message_count == 0 or event.agent == self.manager.spec.name:
            return None
        if event.kind not in {"assistant_message", "turn_failed", "transport_closed", "auth_error", "mcp_failed"}:
            return None
        message = event.message.strip()
        if event.kind == "assistant_message" and not message:
            return None
        if self._is_ready_message(message):
            return None
        signature = (event.agent, f"{event.kind}:{message[:160]}")
        now = time.time()
        last_sent = self._last_manager_event_signatures.get(signature)
        if last_sent is not None and now - last_sent < 3:
            return None
        self._last_manager_event_signatures[signature] = now
        preview = message if len(message) <= 400 else f"{message[:397]}..."
        return (
            "Worker supervision event.\n"
            f"Latest user request: {self._last_user_message}\n"
            f"Event: {event.agent} {event.kind}: {preview}\n"
            "React only if this changes delegation, worker lifetime, worker cwd, or user-visible status. Use tools only.\n\n"
            f"{self._overview_text()}"
        )

    def _check_for_stalls(self, rows: list[dict[str, object]]) -> None:
        if self._shutting_down or self._user_message_count == 0:
            return
        for row in rows[1:]:
            name = str(row["name"])
            if row["status"] != "running":
                self._last_stall_alerts.pop(name, None)
                continue
            running_for = row.get("running_for_seconds")
            if not isinstance(running_for, float) or running_for < 120:
                continue
            bucket = int(running_for // 120)
            if self._last_stall_alerts.get(name) == bucket:
                continue
            self._last_stall_alerts[name] = bucket
            self.manager.enqueue(
                (
                    "Worker appears stalled.\n"
                    f"Latest user request: {self._last_user_message}\n"
                    f"Worker: {name}\n"
                    f"Running for: {running_for:.1f}s\n\n"
                    f"{self._overview_text()}\n\n"
                    "Decide whether to wait, restart, stop, change cwd, or send a sharper follow-up via tools."
                ),
                source="system",
            )

    def _display_worker_message(self, event: SessionEvent) -> str:
        message = event.message.strip()
        if self._is_ready_message(message):
            return "Ready for assignments."
        return message

    def _is_ready_message(self, message: str) -> bool:
        normalized = " ".join(message.strip().lower().split())
        return "ready for assignments" in normalized
