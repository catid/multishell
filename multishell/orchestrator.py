from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field

from .claude_session import ClaudeSession
from .codex_session import CodexSession, SessionEvent, TranscriptEntry
from .config import (
    GEMINI_ACCOUNT_SPECS,
    MANAGER_SPEC,
    SPARK_MODEL,
    SPARK_REASONING_EFFORT,
    AgentSpec,
    ProviderAccountSpec,
    app_root,
    claude_account_specs,
    codex_account_specs,
    manager_workspace_root,
    runtime_claude_worker_specs,
    runtime_codex_worker_specs,
    socket_path,
    spark_agent_name,
    workspace_root,
)
from .control import ControlServer
from .homes import missing_claude_logins, missing_codex_logins
from .spark_pool import SparkCoordinator, SparkError
from .web_reasoners import WebReasonerError, WebReasonerEvent, WebReasonerManager


ManagedSession = CodexSession | ClaudeSession

_CODEX_ARCHETYPE_ROTATION = ("worker-1", "worker-2", "worker-3", "worker-4")
_CLAUDE_ARCHETYPE_ROTATION = (
    "claude-worker-1",
    "claude-worker-2",
    "claude-worker-3",
    "claude-worker-4",
    "claude-worker-5",
)


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


@dataclass
class SessionLifecycleMetadata:
    last_failure_at: float | None = None
    last_failure_detail: str | None = None
    last_failure_kind: str | None = None
    disconnect_streak: int = 0
    auto_restarts: int = 0
    auto_restart_suppressed: bool = False
    last_recovery_at: float | None = None
    last_recovery_action: str | None = None

    def record_failure(self, *, kind: str, detail: str, now: float) -> None:
        self.last_failure_at = now
        self.last_failure_kind = kind
        self.last_failure_detail = detail.strip() or kind
        if kind == "transport_closed":
            self.disconnect_streak += 1

    def note_recovery(self, *, action: str, now: float) -> None:
        self.last_recovery_at = now
        self.last_recovery_action = action
        if action in {"manual_restart", "session_started"}:
            self.disconnect_streak = 0
            self.auto_restart_suppressed = False

    def note_stop(self, *, now: float) -> None:
        self.last_recovery_at = now
        self.last_recovery_action = "stopped"
        self.disconnect_streak = 0
        self.auto_restart_suppressed = False

    def failure_context(self) -> str | None:
        parts: list[str] = []
        if self.last_failure_kind == "transport_closed":
            if self.disconnect_streak > 1:
                parts.append(f"app-server disconnected {self.disconnect_streak} times")
            elif self.last_failure_detail:
                parts.append(self.last_failure_detail)
            if self.auto_restart_suppressed:
                parts.append("auto-restart suppressed")
                parts.append("manual restart recommended")
            elif self.last_recovery_action == "auto_restart":
                parts.append("auto-restarted once")
        elif self.last_failure_detail:
            parts.append(self.last_failure_detail)
        if self.last_failure_kind != "transport_closed" and self.auto_restart_suppressed:
            parts.append("auto-restart suppressed; manual restart recommended")
        elif self.last_recovery_action == "manual_restart":
            parts.append("manually restarted")
        elif self.last_recovery_action == "account_failover":
            parts.append("continued on another account")
        elif self.last_recovery_action == "stopped":
            parts.append("stopped intentionally")
        return "; ".join(parts) if parts else None


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


@dataclass
class ProviderAccountState:
    spec: ProviderAccountSpec
    leased_sessions: set[str] = field(default_factory=set)
    last_exhausted_at: float | None = None
    exhausted_until: float | None = None

    def lease_count(self) -> int:
        return len(self.leased_sessions)


class ProviderAccountPool:
    def __init__(self, accounts: list[ProviderAccountSpec], *, cooldown_seconds: float) -> None:
        self._accounts = list(accounts)
        self._states = {spec.account_key: ProviderAccountState(spec=spec) for spec in accounts}
        self._bindings: dict[str, str] = {}
        self._cooldown_seconds = cooldown_seconds
        self._lock = threading.RLock()

    def assigned_spec(self, session_name: str) -> ProviderAccountSpec | None:
        with self._lock:
            account_key = self._bindings.get(session_name)
            state = self._states.get(account_key or "")
            return state.spec if state is not None else None

    def release(self, session_name: str) -> None:
        with self._lock:
            account_key = self._bindings.pop(session_name, None)
            if account_key and account_key in self._states:
                self._states[account_key].leased_sessions.discard(session_name)

    def mark_exhausted(self, session_name: str, *, now: float | None = None) -> ProviderAccountSpec | None:
        with self._lock:
            account_key = self._bindings.get(session_name)
            if not account_key:
                return None
            state = self._states.get(account_key)
            if state is None:
                return None
            timestamp = time.time() if now is None else now
            state.last_exhausted_at = timestamp
            state.exhausted_until = timestamp + self._cooldown_seconds
            return state.spec

    def bind(
        self,
        session_name: str,
        *,
        preferred_key: str | None = None,
        avoid_keys: set[str] | None = None,
        reserve_keys: set[str] | None = None,
        allow_exhausted_fallback: bool = True,
    ) -> ProviderAccountSpec | None:
        with self._lock:
            if not self._accounts:
                return None
            avoid = set(avoid_keys or ())
            reserve = set(reserve_keys or ())
            if preferred_key and preferred_key not in avoid:
                preferred = self._states.get(preferred_key)
                if preferred is not None:
                    self._apply_binding(session_name, preferred.spec.account_key)
                    return preferred.spec

            chosen = self._choose_candidate(avoid=avoid, reserve=reserve, allow_exhausted_fallback=allow_exhausted_fallback)
            if chosen is None:
                return None
            self._apply_binding(session_name, chosen.account_key)
            return chosen

    def _apply_binding(self, session_name: str, account_key: str) -> None:
        previous = self._bindings.get(session_name)
        if previous == account_key:
            self._states[account_key].leased_sessions.add(session_name)
            return
        if previous and previous in self._states:
            self._states[previous].leased_sessions.discard(session_name)
        self._bindings[session_name] = account_key
        self._states[account_key].leased_sessions.add(session_name)

    def _choose_candidate(
        self,
        *,
        avoid: set[str],
        reserve: set[str],
        allow_exhausted_fallback: bool,
    ) -> ProviderAccountSpec | None:
        now = time.time()
        viable = [state for key, state in self._states.items() if key not in avoid]
        if not viable:
            return None

        def sort_key(state: ProviderAccountState) -> tuple[float, int, int, str]:
            exhausted = bool(state.exhausted_until and state.exhausted_until > now)
            exhausted_rank = 1 if exhausted else 0
            reserve_rank = 1 if state.spec.account_key in reserve else 0
            exhausted_until = state.exhausted_until or 0.0
            return (exhausted_rank, reserve_rank, state.lease_count(), exhausted_until, state.spec.account_key)

        preferred = [state for state in viable if not (state.exhausted_until and state.exhausted_until > now)]
        if preferred:
            return min(preferred, key=sort_key).spec
        if not allow_exhausted_fallback:
            return None
        return min(viable, key=sort_key).spec


def _archetype_for_worker(worker_name: str) -> WorkerArchetype:
    archetype = WORKER_ARCHETYPES.get(worker_name)
    if archetype is not None:
        return archetype
    if worker_name.startswith("claude-worker-"):
        try:
            index = max(1, int(worker_name.split("-")[-1]))
        except ValueError:
            index = 1
        return WORKER_ARCHETYPES[_CLAUDE_ARCHETYPE_ROTATION[(index - 1) % len(_CLAUDE_ARCHETYPE_ROTATION)]]
    if worker_name.startswith("worker-"):
        try:
            index = max(1, int(worker_name.split("-")[-1]))
        except ValueError:
            index = 1
        return WORKER_ARCHETYPES[_CODEX_ARCHETYPE_ROTATION[(index - 1) % len(_CODEX_ARCHETYPE_ROTATION)]]
    return WorkerArchetype(
        persona_name=worker_name,
        domain="software engineering",
        approach="solve the assigned task directly",
        coaching="Work concretely and note blockers quickly.",
    )


def manager_prompt(
    codex_specs: list[AgentSpec],
    claude_specs: list[AgentSpec],
    *,
    max_codex_workers: int,
    max_claude_workers: int,
) -> str:
    codex_lines = [f"- {spec.name}: {spec.personality}" for spec in codex_specs]
    claude_lines = [f"- {spec.name}: {spec.personality}" for spec in claude_specs]
    expansion_lines: list[str] = []
    if max_codex_workers > len(codex_specs):
        expansion_lines.append(
            f"- Additional Codex lanes are available on demand as worker-2 through worker-{max_codex_workers}. "
            "Name a higher-numbered worker only when you actually need more parallelism; the controller will materialize it on first use."
        )
    if max_claude_workers > len(claude_specs):
        expansion_lines.append(
            f"- Additional Claude lanes are available on demand as claude-worker-2 through claude-worker-{max_claude_workers}. "
            "Create them only when a different-model perspective is actually needed."
        )
    expansion_block = "\n".join(expansion_lines)
    return f"""You are the multishell manager.

Operate only through delegation, session management, long-running reasoning tools, and user communication.

Rules:
- Your own working directory is isolated on purpose. Do not inspect or edit project files yourself.
- Use `delegate_to_worker` to assign concrete tasks to named workers.
- Use `start_worker_session`, `stop_worker_session`, and `restart_worker_session` to manage worker memory and working directories.
- Use `get_workers_overview` and `get_worker_transcript` to monitor progress and compare candidates.
- Worker sessions may begin stopped. Start the specific lanes you need instead of assuming the full swarm is already live.
- Do not poll worker transcripts in tight loops. After delegating, prefer to return and wait for significant worker events unless there is a concrete blocker that requires an immediate check.
- Use `notify_user` for all user-facing messages. Do not assume plain assistant text reaches the user.
- Use `gpt_5_4_pro` and `gemini_deepthink` for slow planning, research, world knowledge, or math-heavy reasoning in parallel with worker execution.
- Start slow web reasoners early on hard tasks, continue delegating while they run, then incorporate useful revisions after they complete.
- Restart workers for unrelated tasks so stale memory does not leak across problems.
- When starting or restarting a worker, craft a fresh session persona using `persona_name`, `task_context`, and `extra_instructions`.
- The controller may rebind a worker or the manager to a different account after token, context, or rate-limit exhaustion. When that happens, resume from the supplied handoff context instead of restarting blindly.
- Worker personalities are intentionally narrow. Do not assume one strong worker will also cover security, performance, UX, operability, and integration concerns automatically.
- Assign explicit complementary roles when the task warrants it. Useful splits include implementer, correctness reviewer, security reviewer, performance reviewer, UX/operator reviewer, integration closer, and creative alternative generator.
- Default to a small swarm. For straightforward single-file, single-bug, or otherwise bounded tasks, start with one implementer plus one verifier or reviewer. Add more workers only when the task spans multiple files, has meaningful uncertainty, or clearly benefits from diversity.
- Gate dependent work. If verification, review, or integration depends on a file, binary, report, or other artifact that does not exist yet, wait for the prerequisite to be produced before delegating that dependent step. Do not ask workers to poll in loops unless there is no better option.
- For non-trivial tasks, use more than one worker. For top-k or uncertain work, fan out aggressively across Codex and Claude workers.
- For top-k work, maximize diversity of attack angle, not just worker count. Give each parallel worker a materially different persona, task framing, or review role.
- For top-k work, keep the diversity emphasis explicit: run diverse candidates in parallel, align them on the exact target and success criteria, validate the winner quickly, and stop once further coordination is lower-value than delivery.
- Codex workers are generally more reliable for execution. Claude workers are more creative and often useful for alternative ideas and code review.
- Use cross-review patterns for diversity: Claude drafts with Codex review, Codex drafts with Claude review, or parallel candidates from both families.
- Claude is especially useful for code review, idea expansion, alternative framings, and different-model perspective. Treat it as creative but less reliable.
- Use Gemini Deep Think when you want slower but broader world knowledge or math-heavy parallel reasoning. It runs on the configured Gemini AI Ultra account pool.
- Choose worker working directories intentionally. You can pass `cwd` when starting, restarting, or delegating.
- Do not send a final success update until every required worker and reasoner for the task is terminal and the required verification has actually completed.
- Keep messages concise and operational.
- The controller only materializes a small starter set of workers at startup. Add higher-numbered workers only when the task actually needs more lanes.

Codex workers:
{chr(10).join(codex_lines)}

Claude workers:
{chr(10).join(claude_lines)}

Additional worker capacity:
{expansion_block or "- No extra worker capacity is currently configured."}
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
    archetype = _archetype_for_worker(worker_name)
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
        "Stay within this session's angle instead of pretending to cover every concern at once.",
        "If the task clearly needs complementary review from another angle such as security, performance, UX, or integration, say so explicitly.",
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
                "Do not silently claim coverage outside your assigned angle. Ask for complementary peer review when other concerns matter.",
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
                "Do not silently claim coverage outside your assigned angle. Ask for complementary peer review when security, performance, UX, or integration risk matters.",
                "If you have no task yet, reply once that you are ready for assignments.",
            ]
        )
    return "\n".join(base)


def _spark_prompt(worker_name: str, worker_personality: str, *, persona_name: str) -> str:
    return f"""You are {spark_agent_name(worker_name)}, the paired draft delegate for {worker_name}.

Session persona: {persona_name}.
You are a persistent Codex session using {SPARK_MODEL} at xhigh reasoning.
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
    _AUTO_RECOVERY_COOLDOWN_SECONDS = 120.0
    _AUTO_RECOVERY_MAX_DISCONNECTS = 1
    _MANAGER_EVENT_INTERRUPT_AFTER_SECONDS = 5.0
    _ACCOUNT_EXHAUSTION_COOLDOWN_SECONDS = 900.0
    _MAX_HANDOFF_ENTRIES = 18
    _MAX_HANDOFF_CHARS = 6000

    def __init__(self) -> None:
        self._all_codex_worker_specs = runtime_codex_worker_specs()
        self._all_claude_worker_specs = runtime_claude_worker_specs()
        self._codex_account_specs = codex_account_specs()
        self._claude_account_specs = claude_account_specs()
        self._codex_accounts = ProviderAccountPool(
            self._codex_account_specs,
            cooldown_seconds=self._ACCOUNT_EXHAUSTION_COOLDOWN_SECONDS,
        )
        self._claude_accounts = ProviderAccountPool(
            self._claude_account_specs,
            cooldown_seconds=self._ACCOUNT_EXHAUSTION_COOLDOWN_SECONDS,
        )
        manager_account = self._codex_accounts.bind(
            MANAGER_SPEC.name,
            preferred_key=self._codex_account_specs[0].account_key if self._codex_account_specs else MANAGER_SPEC.account_key,
        )
        self.manager = CodexSession(
            MANAGER_SPEC,
            manager_prompt(
                self._all_codex_worker_specs[:1],
                self._all_claude_worker_specs[:1],
                max_codex_workers=len(self._all_codex_worker_specs),
                max_claude_workers=len(self._all_claude_worker_specs),
            ),
            mcp_bridge_command=_bridge_command("manager", MANAGER_SPEC.name),
            working_dir=manager_workspace_root(),
            persona_label="Multishell Manager",
            message_callback=self._handle_manager_message,
            event_callback=self._handle_manager_event,
            turn_timeout_seconds=None,
            auth_source_agent=manager_account.account_key if manager_account is not None else MANAGER_SPEC.account_key,
            account_email=manager_account.account_email if manager_account is not None else MANAGER_SPEC.account_email,
        )
        self.workers: dict[str, ManagedSession] = {}
        self.codex_workers: dict[str, CodexSession] = {}
        self.claude_workers: dict[str, ClaudeSession] = {}
        self.spark_workers: dict[str, CodexSession] = {}
        self._codex_worker_names = {spec.name for spec in self._all_codex_worker_specs}
        self._claude_worker_names = {spec.name for spec in self._all_claude_worker_specs}
        self._started_controller = False
        self._lifecycle: dict[str, SessionLifecycleMetadata] = {
            self.manager.spec.name: SessionLifecycleMetadata()
        }

        if self._all_codex_worker_specs:
            self._materialize_worker(self._all_codex_worker_specs[0].name)
        if self._all_claude_worker_specs:
            self._materialize_worker(self._all_claude_worker_specs[0].name)

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

        for spark in self.spark_workers.values():
            self._lifecycle.setdefault(spark.spec.name, SessionLifecycleMetadata())

    def _codex_worker_spec(self, worker_name: str) -> AgentSpec | None:
        for spec in self._all_codex_worker_specs:
            if spec.name == worker_name:
                return spec
        return None

    def _claude_worker_spec(self, worker_name: str) -> AgentSpec | None:
        for spec in self._all_claude_worker_specs:
            if spec.name == worker_name:
                return spec
        return None

    def _materialized_codex_specs(self) -> list[AgentSpec]:
        return [spec for spec in self._all_codex_worker_specs if spec.name in self.codex_workers]

    def _materialized_claude_specs(self) -> list[AgentSpec]:
        return [spec for spec in self._all_claude_worker_specs if spec.name in self.claude_workers]

    def _materialize_worker(self, worker_name: str) -> ManagedSession | None:
        existing = self.workers.get(worker_name)
        if existing is not None:
            return existing

        codex_spec = self._codex_worker_spec(worker_name)
        if codex_spec is not None:
            initial_codex_account = self._codex_account_specs[0] if self._codex_account_specs else None
            session = CodexSession(
                codex_spec,
                _worker_prompt(codex_spec.name, codex_spec.personality, engine="codex"),
                mcp_bridge_command=_bridge_command("worker", codex_spec.name),
                working_dir=workspace_root(),
                startup_prompt="Reply once that you are ready for assignments.",
                persona_label=self._default_persona_label(codex_spec.name),
                event_callback=self._handle_worker_event,
                turn_timeout_seconds=900,
                auth_source_agent=initial_codex_account.account_key if initial_codex_account is not None else codex_spec.name,
                account_email=codex_spec.account_email,
            )
            self.codex_workers[codex_spec.name] = session
            self.workers[codex_spec.name] = session

            spark_spec = AgentSpec(
                name=spark_agent_name(codex_spec.name),
                account_email=codex_spec.account_email,
                role="spark",
                personality=f"Draft delegate paired with {codex_spec.name}",
                accent_color=codex_spec.accent_color,
                account_key=codex_spec.account_key,
            )
            self.spark_workers[codex_spec.name] = CodexSession(
                spark_spec,
                _spark_prompt(
                    codex_spec.name,
                    codex_spec.personality,
                    persona_name=f"{self._default_persona_label(codex_spec.name)} Draft Partner",
                ),
                working_dir=workspace_root(),
                persona_label=f"{self._default_persona_label(codex_spec.name)} Spark",
                event_callback=self._handle_spark_session_event,
                turn_timeout_seconds=900,
                model=SPARK_MODEL,
                reasoning_effort=SPARK_REASONING_EFFORT,
                auth_source_agent=initial_codex_account.account_key if initial_codex_account is not None else codex_spec.name,
                account_email=codex_spec.account_email,
            )
            self._lifecycle[codex_spec.name] = SessionLifecycleMetadata()
            self._lifecycle[spark_spec.name] = SessionLifecycleMetadata()
            if self._started_controller and not self._shutting_down:
                session.start(start_session=False)
                self.spark_workers[codex_spec.name].start(start_session=False)
            return session

        claude_spec = self._claude_worker_spec(worker_name)
        if claude_spec is None:
            return None
        initial_claude_account = self._claude_account_specs[0] if self._claude_account_specs else None
        session = ClaudeSession(
            claude_spec,
            _worker_prompt(claude_spec.name, claude_spec.personality, engine="claude"),
            working_dir=workspace_root(),
            startup_prompt="Reply once that you are ready for assignments.",
            persona_label=self._default_persona_label(claude_spec.name),
            event_callback=self._handle_worker_event,
            turn_timeout_seconds=900,
            auth_source_agent=initial_claude_account.account_key if initial_claude_account is not None else claude_spec.name,
            account_email=claude_spec.account_email,
        )
        self.claude_workers[claude_spec.name] = session
        self.workers[claude_spec.name] = session
        self._lifecycle[claude_spec.name] = SessionLifecycleMetadata()
        if self._started_controller and not self._shutting_down:
            session.start(start_session=False)
        return session

    def start(self) -> None:
        missing_codex = missing_codex_logins()
        missing_claude = missing_claude_logins()
        if missing_codex or missing_claude:
            problems: list[str] = []
            if missing_codex:
                problems.append(f"missing Codex login for: {', '.join(missing_codex)}")
            if missing_claude:
                problems.append(f"missing Claude login for: {', '.join(missing_claude)}")
            raise RuntimeError("; ".join(problems))
        self._shutting_down = False
        self._started_controller = True
        self._control_server.start()
        self.manager.start()
        for worker in self.codex_workers.values():
            worker.start(start_session=False)
        for worker in self.claude_workers.values():
            worker.start(start_session=False)
        for spark in self.spark_workers.values():
            spark.start(start_session=False)
        self._push_message("system", "multishell started", level="info")
        self._monitor_thread.start()

    def stop(self) -> None:
        self._shutting_down = True
        self._started_controller = False
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

    def _lookup_worker(self, worker_name: str) -> ManagedSession | None:
        worker = self.workers.get(worker_name)
        if worker is not None:
            return worker
        return self._materialize_worker(worker_name)

    def _handle_manager_request(self, tool: object, arguments: dict[str, object]) -> dict[str, object]:
        if tool == "notify_user":
            message = str(arguments.get("message", "")).strip()
            level = str(arguments.get("level", "info"))
            if self._looks_like_final_user_update(message):
                active_workers, active_reasoners = self._active_incomplete_work()
                if active_workers or active_reasoners:
                    waiting_on = ", ".join([*active_workers, *active_reasoners])
                    return {
                        "ok": False,
                        "error": f"cannot send a final completion update while work is still active: {waiting_on}",
                    }
            self._push_message("manager", message, level=level)
            return {"ok": True, "message": "user notified"}

        if tool == "delegate_to_worker":
            worker_name = str(arguments.get("worker", "")).strip()
            task = str(arguments.get("task", "")).strip()
            cwd = str(arguments.get("cwd", "")).strip() or None
            worker = self._lookup_worker(worker_name)
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
            worker = self._lookup_worker(worker_name)
            if worker is None:
                return {"ok": False, "error": f"unknown worker: {worker_name}"}
            if worker.overview()["status"] != "stopped":
                return {"ok": False, "error": f"{worker_name} session is already running; use restart_worker_session"}
            account = self._ensure_worker_binding(worker_name, prefer_current=False)
            system_prompt, persona_label = self._resolve_worker_session_prompt(worker_name, arguments)
            worker.start_session(cwd, system_prompt=system_prompt, persona_label=persona_label)
            self._mark_manual_recovery(worker_name, action="session_started")
            self._sync_paired_spark_session(
                worker_name,
                cwd=cwd,
                persona_label=persona_label,
                task_context=str(arguments.get("task_context", "")).strip() or None,
                action="start",
            )
            self._push_message(
                "manager",
                f"started {worker_name} session in {worker.overview()['cwd']} as {worker.overview()['persona_label']} "
                f"on {account.account_email}",
            )
            return {"ok": True, "message": f"started {worker_name}"}

        if tool == "stop_worker_session":
            worker_name = str(arguments.get("worker", "")).strip()
            worker = self._lookup_worker(worker_name)
            if worker is None:
                return {"ok": False, "error": f"unknown worker: {worker_name}"}
            worker.stop_session(clear_pending=True)
            self._mark_session_stopped(worker_name)
            self._sync_paired_spark_session(worker_name, action="stop")
            self._release_worker_binding(worker_name)
            self._push_message("manager", f"stopped {worker_name} session")
            return {"ok": True, "message": f"stopped {worker_name}"}

        if tool == "restart_worker_session":
            worker_name = str(arguments.get("worker", "")).strip()
            cwd = str(arguments.get("cwd", "")).strip() or None
            worker = self._lookup_worker(worker_name)
            if worker is None:
                return {"ok": False, "error": f"unknown worker: {worker_name}"}
            account = self._ensure_worker_binding(worker_name, prefer_current=True)
            system_prompt, persona_label = self._resolve_worker_session_prompt(worker_name, arguments)
            worker.restart_session(cwd, system_prompt=system_prompt, persona_label=persona_label)
            self._mark_manual_recovery(worker_name, action="manual_restart")
            self._sync_paired_spark_session(
                worker_name,
                cwd=cwd,
                persona_label=persona_label,
                task_context=str(arguments.get("task_context", "")).strip() or None,
                action="restart",
            )
            self._push_message(
                "manager",
                f"restarted {worker_name} session in {worker.overview()['cwd']} as {worker.overview()['persona_label']} "
                f"on {account.account_email}",
            )
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
            worker = self._lookup_worker(worker_name)
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
                agent = self._select_gemini_account()
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

    def _ensure_worker_binding(self, worker_name: str, *, prefer_current: bool) -> ProviderAccountSpec:
        pool = self._account_pool_for_session(worker_name)
        current = pool.assigned_spec(worker_name)
        if current is not None and prefer_current:
            self._bind_session_to_account(worker_name, current)
            return current
        reserve: set[str] = set()
        manager_account = self._codex_accounts.assigned_spec(self.manager.spec.name)
        if worker_name != self.manager.spec.name and manager_account is not None:
            reserve.add(manager_account.account_key)
        preferred_key = current.account_key if current is not None and prefer_current else None
        chosen = pool.bind(worker_name, preferred_key=preferred_key, reserve_keys=reserve)
        if chosen is None:
            raise ValueError(f"no accounts available for {worker_name}")
        self._bind_session_to_account(worker_name, chosen)
        return chosen

    def _release_worker_binding(self, worker_name: str) -> None:
        self._account_pool_for_session(worker_name).release(worker_name)

    def _account_pool_for_session(self, session_name: str) -> ProviderAccountPool:
        if session_name == self.manager.spec.name or session_name in self.codex_workers:
            return self._codex_accounts
        if session_name in self.claude_workers:
            return self._claude_accounts
        raise KeyError(session_name)

    def _bind_session_to_account(self, session_name: str, account: ProviderAccountSpec) -> None:
        if session_name == self.manager.spec.name:
            self.manager.bind_account(account.account_key, account.account_email)
            return
        worker = self.workers[session_name]
        worker.bind_account(account.account_key, account.account_email)
        if session_name in self.spark_workers:
            self.spark_workers[session_name].bind_account(account.account_key, account.account_email)

    def _active_incomplete_work(self) -> tuple[list[str], list[str]]:
        active_workers: list[str] = []
        for name, worker in self.workers.items():
            overview = worker.overview()
            if overview.get("status") == "running" or int(overview.get("pending_tasks", 0)) > 0:
                active_workers.append(name)

        active_reasoners: list[str] = []
        for job in self.web_reasoners.list_jobs():
            if str(job.get("status") or "") not in {"queued", "running", "canceling"}:
                continue
            provider = str(job.get("provider") or "reasoner")
            label = str(job.get("label") or job.get("id") or provider)
            active_reasoners.append(f"{provider}:{label}")
        return active_workers, active_reasoners

    def _looks_like_final_user_update(self, message: str) -> bool:
        normalized = message.strip().lower()
        if not normalized:
            return False
        if normalized.startswith(("done", "completed", "brief summary", "final summary", "task complete")):
            return True
        return "verification summary" in normalized or "outcome: succeeded" in normalized

    def send_user_message(self, text: str) -> None:
        manager_overview = self.manager.overview()
        running_for = manager_overview.get("running_for_seconds")
        if manager_overview["status"] == "running" and isinstance(running_for, float) and running_for >= 15:
            self._interrupt_manager_turn(manager_overview, reason="a new user message arrived")
        fanout_guidance = self._fanout_guidance(text)
        prompt = (
            "User message:\n"
            f"{text}\n\n"
            "Current worker overview:\n"
            f"{self._overview_text()}\n\n"
            "Respond using tools only. If the task is unrelated to ongoing worker memory, restart or stop the relevant worker "
            "sessions first. Use Codex workers for reliable implementation and Claude workers for diversity, review, and cross-model "
            "perspective. Agents stay narrow to their prompted roles, so explicitly assign complementary jobs instead of assuming one worker covers "
            "everything. Use role splits like implementer, reviewer, performance checker, security skeptic, UX/operator critic, and integration closer "
            "when appropriate. Default to one implementer plus one verifier or reviewer unless the task clearly benefits from more lanes. "
            "If later stages depend on a file, binary, or report that does not exist yet, wait for that prerequisite before delegating the dependent work; do not ask workers to poll in loops unless there is no better option. "
            "Do not sit in a long transcript-polling loop after delegation; return control and wait for significant worker events unless there is a concrete blocker. "
            "Start slow web reasoners early when long-horizon planning, outside knowledge, or heavy reasoning may help later. "
            "Do not send a final success update until required workers and reasoners are terminal and the required verification has actually completed. "
            "Choose worker working directories intentionally and notify the user."
            f"{fanout_guidance}"
        )
        self._user_message_count += 1
        self._last_user_message = text
        self._push_message("user", text, level="info")
        self.manager.enqueue(prompt, source="user")

    def session_rows(self) -> list[dict[str, object]]:
        rows = [self._decorate_overview(self.manager.spec.name, self.manager.overview())]
        rows.extend(
            self._decorate_overview(spec.name, self.codex_workers[spec.name].overview())
            for spec in self._materialized_codex_specs()
        )
        rows.extend(
            self._decorate_overview(spec.name, self.claude_workers[spec.name].overview())
            for spec in self._materialized_claude_specs()
        )
        return rows

    def monitor_items(self) -> list[dict[str, object]]:
        items = [self._monitor_from_session("manager", self.manager.overview())]

        for spec in self._materialized_codex_specs():
            worker_index = spec.name.split("-")[-1]
            items.append(self._monitor_from_session(f"codex-{worker_index}", self.codex_workers[spec.name].overview()))

        for spec in self._materialized_claude_specs():
            worker_index = spec.name.split("-")[-1]
            items.append(self._monitor_from_session(f"claude-{worker_index}", self.claude_workers[spec.name].overview()))

        for spec in self._materialized_codex_specs():
            worker_index = spec.name.split("-")[-1]
            items.append(self._monitor_from_session(f"spark-{worker_index}", self.spark_workers[spec.name].overview()))

        gptpro_jobs = self.web_reasoners.list_jobs(provider="chatgpt_pro")
        items.append(self._monitor_from_reasoner("gptpro-1", gptpro_jobs[0] if len(gptpro_jobs) >= 1 else None, accent_color=1))
        items.append(self._monitor_from_reasoner("gptpro-2", gptpro_jobs[1] if len(gptpro_jobs) >= 2 else None, accent_color=1))

        deepthink_jobs = self.web_reasoners.list_jobs(provider="gemini_deepthink")
        items.append(self._monitor_from_reasoner("deepthink", deepthink_jobs[0] if deepthink_jobs else None, accent_color=3))
        return items

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
            self._emit_health_alerts(rows)
            self._check_for_stalls(rows)

    def _overview_text(self) -> str:
        lines = []
        for worker in self.workers.values():
            overview = self._decorate_overview(worker.spec.name, worker.overview())
            timing = ""
            if overview["status"] == "running" and overview["running_for_seconds"] is not None:
                timing = f" running_for={overview['running_for_seconds']:.1f}s"
            elif overview["last_turn_duration"] is not None:
                timing = f" last_turn={overview['last_turn_duration']:.1f}s"
            failure_context = str(overview.get("failure_context") or "").strip()
            failure_suffix = f" failure_context={failure_context!r}" if failure_context else ""
            lines.append(
                f"- {overview['name']}: engine={overview.get('engine', 'codex')} model={overview.get('model', '-')!r} "
                f"account={overview.get('account_email', '-')!r} persona={overview['persona_label']!r} "
                f"status={overview['status']} cwd={overview['cwd']!r} "
                f"pending={overview['pending_tasks']} completed={overview['completed_turns']} failed={overview['failed_turns']}{timing} "
                f"last_message={overview['last_message']!r}{failure_suffix}"
            )
        return "\n".join(lines)

    def _decorate_overview(self, session_name: str, overview: dict[str, object]) -> dict[str, object]:
        meta = self._lifecycle.get(session_name)
        row = dict(overview)
        failure_context = meta.failure_context() if meta is not None else None
        row["failure_context"] = failure_context
        row["disconnect_streak"] = meta.disconnect_streak if meta is not None else 0
        row["auto_restarts"] = meta.auto_restarts if meta is not None else 0
        row["last_failure_kind"] = meta.last_failure_kind if meta is not None else None
        row["last_failure_at"] = meta.last_failure_at if meta is not None else None
        row["last_recovery_action"] = meta.last_recovery_action if meta is not None else None
        row["last_recovery_at"] = meta.last_recovery_at if meta is not None else None
        if row.get("status") == "stopped":
            row["last_error"] = None
        elif failure_context and not row.get("last_error"):
            row["last_error"] = failure_context
        return row

    def _push_message(self, source: str, text: str, level: str = "info") -> None:
        with self._lock:
            self.messages.append(UiMessage(ts=time.time(), source=source, text=text, level=level))
            self.messages = self.messages[-260:]

    def _mark_manual_recovery(self, session_name: str, *, action: str) -> None:
        meta = self._lifecycle.get(session_name)
        if meta is not None:
            meta.note_recovery(action=action, now=time.time())

    def _mark_session_stopped(self, session_name: str) -> None:
        meta = self._lifecycle.get(session_name)
        if meta is not None:
            meta.note_stop(now=time.time())

    def _record_session_failure(self, event: SessionEvent) -> None:
        meta = self._lifecycle.get(event.agent)
        if meta is None:
            return
        meta.record_failure(kind=event.kind, detail=event.message, now=event.ts or time.time())

    def _maybe_auto_recover_transport(self, event: SessionEvent) -> bool:
        worker = self.codex_workers.get(event.agent)
        if worker is None or self._shutting_down:
            return False
        meta = self._lifecycle.get(event.agent)
        if meta is None:
            return False
        overview = worker.overview()
        if overview.get("status") == "stopped":
            return False
        if overview.get("turn_id") or overview.get("status") == "running" or int(overview.get("pending_tasks", 0)) > 0:
            meta.auto_restart_suppressed = True
            return False
        now = time.time()
        if meta.disconnect_streak > self._AUTO_RECOVERY_MAX_DISCONNECTS:
            meta.auto_restart_suppressed = True
            return False
        if meta.last_recovery_action == "auto_restart" and meta.last_recovery_at is not None:
            if now - meta.last_recovery_at < self._AUTO_RECOVERY_COOLDOWN_SECONDS:
                meta.auto_restart_suppressed = True
                return False
        cwd = str(overview.get("cwd") or worker.working_dir)
        persona_label = str(overview.get("persona_label") or worker.persona_label)
        system_prompt = getattr(worker, "initial_prompt", None)
        worker.restart_session(cwd=cwd, system_prompt=system_prompt, persona_label=persona_label)
        meta.auto_restarts += 1
        meta.note_recovery(action="auto_restart", now=now)
        return True

    def _emit_health_alerts(self, rows: list[dict[str, object]]) -> None:
        for row in rows:
            signature = (row["status"], row["failed_turns"], row["last_error"], row.get("cwd"), row.get("failure_context"))
            previous = self._last_session_alerts.get(str(row["name"]))
            if signature == previous:
                continue
            self._last_session_alerts[str(row["name"])] = signature
            if row["status"] == "error":
                if self._maybe_recover_limit_error_row(row):
                    continue
                detail = str(row["last_error"] or "unknown error")
                self._push_message("system", f"{row['name']} entered error state: {detail}", level="error")
            elif previous and previous[0] == "error":
                self._push_message("system", f"{row['name']} recovered to {row['status']}", level="info")

    def _maybe_recover_limit_error_row(self, row: dict[str, object]) -> bool:
        session_name = str(row.get("name") or "").strip()
        last_error = str(row.get("last_error") or "").strip()
        if not session_name or not last_error:
            return False
        if self._message_limit_kind(last_error, assistant_message=False) is None:
            return False
        try:
            session = self._session_for_name(session_name)
        except KeyError:
            return False
        current = session.overview()
        if str(current.get("status") or "") != "error":
            return True
        if str(current.get("last_error") or "").strip() != last_error:
            return False
        return self._maybe_failover_account(
            SessionEvent(
                ts=time.time(),
                agent=session_name,
                kind="error",
                message=last_error,
                data={"recovery_source": "health_monitor"},
            )
        )

    def _maybe_interrupt_manager_for_worker_event(self, event: SessionEvent) -> bool:
        manager_overview = self.manager.overview()
        running_for = manager_overview.get("running_for_seconds")
        turn_id = str(manager_overview.get("turn_id") or "")
        if (
            manager_overview.get("status") != "running"
            or not isinstance(running_for, float)
            or running_for < self._MANAGER_EVENT_INTERRUPT_AFTER_SECONDS
        ):
            return False
        if turn_id and turn_id == self._last_manager_interrupt_turn:
            return False
        reason = f"{event.agent} reported {event.kind}: {self._single_line_snippet(event.message, limit=120)}"
        self._interrupt_manager_turn(manager_overview, reason=reason)
        return True

    def _handle_manager_message(self, entry: TranscriptEntry) -> None:
        self._push_message("manager", entry.text, level="info")

    def _default_persona_label(self, worker_name: str) -> str:
        return _archetype_for_worker(worker_name).persona_name

    def _monitor_from_session(self, label: str, overview: dict[str, object]) -> dict[str, object]:
        return {
            "label": label,
            "status": str(overview.get("status") or "stopped"),
            "accent_color": int(overview.get("accent_color", 1)),
            "pending_tasks": int(overview.get("pending_tasks", 0)),
            "completed_turns": int(overview.get("completed_turns", 0)),
            "failed_turns": int(overview.get("failed_turns", 0)),
        }

    def _monitor_from_reasoner(self, label: str, job: dict[str, object] | None, *, accent_color: int) -> dict[str, object]:
        status = "idle"
        pending = 0
        completed = 0
        failed = 0
        if job is not None:
            status = str(job.get("status") or "idle")
            pending = 1 if status in {"queued", "running", "canceling"} else 0
            completed = 1 if status == "completed" else 0
            failed = 1 if status == "failed" else 0
        return {
            "label": label,
            "status": status,
            "accent_color": accent_color,
            "pending_tasks": pending,
            "completed_turns": completed,
            "failed_turns": failed,
        }

    def _select_gemini_account(self) -> str:
        if not GEMINI_ACCOUNT_SPECS:
            raise WebReasonerError("no Gemini AI Ultra accounts configured; use `multishell login` first")
        active_counts = {spec.name: 0 for spec in GEMINI_ACCOUNT_SPECS}
        for job in self.web_reasoners.list_jobs(provider="gemini_deepthink"):
            if str(job.get("status") or "") not in {"queued", "running", "canceling"}:
                continue
            name = str(job.get("account_agent") or "")
            if name in active_counts:
                active_counts[name] += 1
        selected = min(GEMINI_ACCOUNT_SPECS, key=lambda spec: (active_counts[spec.name], spec.name))
        return selected.name

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
                "materially different attack angles. Do not assign near-duplicate roles. Split across complementary jobs such as implementation, "
                "correctness review, performance review, security scrutiny, UX/operator critique, integration validation, and creative alternatives "
                "as appropriate. Start one or both slow web reasoners early if long-horizon planning or outside knowledge may revise the path later."
            )
        if len(text.split()) >= 12:
            return (
                "\n\nThis is not a trivial request, but do not over-fan out by default. Start with one implementer plus one verifier or reviewer, "
                "then add more workers only if the task spans multiple files, has genuine uncertainty, or clearly benefits from an extra review angle."
            )
        return ""

    def _assistant_message_needs_supervision(self, message: str) -> bool:
        normalized = message.strip().lower()
        if not normalized or self._is_ready_message(message):
            return False
        if normalized.startswith(
            (
                "let me ",
                "i'll ",
                "i will ",
                "checking ",
                "good, ",
                "file doesn't exist yet",
                "file appeared",
                "compiles cleanly",
                "now let me ",
            )
        ):
            return False
        if normalized.startswith(
            (
                "done",
                "status:",
                "verification complete",
                "blocked",
                "need",
                "cannot",
                "can't",
                "failed",
                "error",
                "checked ",
                "inspected ",
                "reviewed ",
                "created ",
                "wrote ",
            )
        ):
            return True
        return any(
            marker in normalized
            for marker in (
                "verification passed",
                "verification complete",
                "all checks green",
                "compile failed",
                "run failed",
                "line-count check",
                "line count result",
                "final-value check",
                "last line:",
                "compile command:",
                "run command:",
                "reused the existing file",
                "created:",
                "created file:",
                "blocked",
                "cannot ",
                "can't ",
                " failed",
                " error",
            )
        )

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
        worker_overview = self.codex_workers[worker_name].overview()
        account_key = str(worker_overview.get("account_key") or "").strip()
        account_email = str(worker_overview.get("account_email") or "").strip() or None
        if account_key:
            spark.bind_account(account_key, account_email)
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

    def _handle_manager_event(self, event: SessionEvent) -> None:
        if event.kind in {"turn_failed", "transport_closed", "auth_error", "mcp_failed", "error"}:
            self._record_session_failure(event)
        if self._maybe_failover_account(event):
            return
        if event.kind in {"turn_failed", "transport_closed", "auth_error", "mcp_failed", "error"}:
            self._push_message("system", f"manager: {event.message}", level="warn" if event.kind == "turn_failed" else "error")

    def _maybe_failover_account(self, event: SessionEvent) -> bool:
        limit_kind = self._event_limit_kind(event)
        if limit_kind is None:
            return False
        try:
            pool = self._account_pool_for_session(event.agent)
        except KeyError:
            return False
        current = pool.assigned_spec(event.agent)
        if current is None:
            return False
        pool.mark_exhausted(event.agent, now=event.ts)
        reserve: set[str] = set()
        manager_account = self._codex_accounts.assigned_spec(self.manager.spec.name)
        if event.agent != self.manager.spec.name and manager_account is not None:
            reserve.add(manager_account.account_key)
        replacement = pool.bind(
            event.agent,
            avoid_keys={current.account_key},
            reserve_keys=reserve,
            allow_exhausted_fallback=True,
        )
        if replacement is None or replacement.account_key == current.account_key:
            self._push_message(
                "system",
                f"{event.agent}: {limit_kind.replace('_', ' ')} on {current.account_email}; no alternate account available",
                level="error",
            )
            return False
        session = self._session_for_name(event.agent)
        overview = session.overview()
        handoff = self._build_account_failover_prompt(
            event=event,
            session=session,
            previous_account=current,
            next_account=replacement,
            limit_kind=limit_kind,
        )
        self._bind_session_to_account(event.agent, replacement)
        cwd = str(overview.get("cwd") or workspace_root())
        persona_label = str(overview.get("persona_label") or event.agent)
        system_prompt = getattr(session, "initial_prompt", None)
        if isinstance(session, CodexSession):
            session.stop_session(clear_pending=False, reason="account failover was requested")
        else:
            session.stop_session(clear_pending=False)
        original_startup_prompt = getattr(session, "startup_prompt", None)
        if original_startup_prompt is not None:
            session.startup_prompt = None
        try:
            session.start_session(cwd=cwd, system_prompt=system_prompt, persona_label=persona_label)
        finally:
            if original_startup_prompt is not None:
                session.startup_prompt = original_startup_prompt
        session.queue_priority_prompt(handoff, source="system", cwd=cwd)
        if event.agent in self.codex_workers:
            self._sync_paired_spark_session(
                event.agent,
                cwd=cwd,
                persona_label=persona_label,
                action="restart",
            )
        meta = self._lifecycle.get(event.agent)
        if meta is not None:
            meta.note_recovery(action="account_failover", now=time.time())
        self._push_message(
            "system",
            f"{event.agent}: switched from {current.account_email} to {replacement.account_email} after {limit_kind.replace('_', ' ')}",
            level="warn",
        )
        return True

    def _event_limit_kind(self, event: SessionEvent) -> str | None:
        message = event.message.strip()
        if message:
            assistant_message = event.kind == "assistant_message"
            matched = self._message_limit_kind(message, assistant_message=assistant_message)
            if matched is not None:
                return matched
        if event.kind == "status_changed" and str(event.data.get("status") or "") == "error":
            current_error = self._current_session_error_message(event.agent)
            if current_error:
                return self._message_limit_kind(current_error, assistant_message=False)
        return None

    def _message_limit_kind(self, message: str, *, assistant_message: bool) -> str | None:
        lowered = " ".join(message.lower().split())
        if assistant_message and not (
            lowered.startswith("out_of_tokens:")
            or lowered.startswith("out of tokens")
            or lowered.startswith("token limit")
            or lowered.startswith("rate limit")
            or lowered.startswith("rate limited")
            or lowered.startswith("context limit")
        ):
            return None
        if any(marker in lowered for marker in ("maximum context length", "context window", "context limit", "prompt is too long")):
            return "context_limit"
        if any(marker in lowered for marker in ("too many requests", "rate limit", "rate limited", " 429 ", "status 429")):
            return "rate_limit"
        if any(marker in lowered for marker in ("out_of_tokens", "out of tokens", "token limit", "usage limit", "token budget")):
            return "token_limit"
        return None

    def _current_session_error_message(self, session_name: str) -> str:
        try:
            session = self._session_for_name(session_name)
        except KeyError:
            return ""
        return str(session.overview().get("last_error") or "").strip()

    def _build_account_failover_prompt(
        self,
        *,
        event: SessionEvent,
        session: CodexSession | ClaudeSession,
        previous_account: ProviderAccountSpec,
        next_account: ProviderAccountSpec,
        limit_kind: str,
    ) -> str:
        overview = session.overview()
        transcript_lines: list[str] = []
        total_chars = 0
        for entry in reversed(session.recent_transcript(self._MAX_HANDOFF_ENTRIES)):
            clean = " ".join(entry.text.split())
            if not clean:
                continue
            snippet = clean if len(clean) <= 360 else f"{clean[:357]}..."
            rendered = f"- [{entry.source}] {snippet}"
            total_chars += len(rendered)
            if total_chars > self._MAX_HANDOFF_CHARS:
                break
            transcript_lines.append(rendered)
        transcript_lines.reverse()
        transcript_block = "\n".join(transcript_lines) if transcript_lines else "- <no recent transcript captured>"
        return (
            "Account failover handoff.\n"
            f"Previous account: {previous_account.account_email}\n"
            f"New account: {next_account.account_email}\n"
            f"Reason: {limit_kind.replace('_', ' ')} while handling this session.\n"
            f"Latest failure: {event.message}\n"
            f"Current cwd: {overview.get('cwd')}\n"
            f"Persona: {overview.get('persona_label')}\n"
            "Resume the same task without restarting from scratch. Preserve completed work, re-check only what is necessary, "
            "and continue from the latest useful point.\n\n"
            "Recent session transcript:\n"
            f"{transcript_block}"
        )

    def _session_for_name(self, session_name: str) -> CodexSession | ClaudeSession:
        if session_name == self.manager.spec.name:
            return self.manager
        return self.workers[session_name]

    def _handle_worker_event(self, event: SessionEvent) -> None:
        if event.kind in {"turn_failed", "transport_closed", "auth_error", "mcp_failed", "error"}:
            self._record_session_failure(event)

        if self._maybe_failover_account(event):
            return

        auto_recovered_transport = False
        if event.kind == "assistant_message":
            self._push_message(event.agent, self._display_worker_message(event), level="info")
        elif event.kind in {"turn_failed", "transport_closed", "auth_error", "mcp_failed", "error"}:
            if event.kind == "transport_closed" and self._maybe_auto_recover_transport(event):
                auto_recovered_transport = True
                self._push_message(
                    "system",
                    f"{event.agent}: app-server disconnected; automatically restarted idle session",
                    level="warn",
                )
            else:
                level = "error"
                detail = event.message
                if event.kind == "turn_failed" and bool(event.data.get("interrupted")):
                    level = "warn"
                    detail = "turn interrupted"
                    duration = event.data.get("duration_seconds")
                    if isinstance(duration, (int, float)):
                        detail = f"{detail} after {float(duration):.1f}s"
                    interrupt_reason = str(event.data.get("interrupt_reason") or "").strip()
                    if interrupt_reason:
                        detail = f"{detail} because {interrupt_reason}"
                    else:
                        turn_status = str(event.data.get("turn_status") or "").strip()
                        if turn_status:
                            detail = f"{detail} (session reported status={turn_status})"
                meta = self._lifecycle.get(event.agent)
                if event.kind == "transport_closed" and meta is not None and meta.auto_restart_suppressed:
                    detail = f"{detail} (manual restart recommended after repeated or in-flight disconnect)"
                self._push_message("system", f"{event.agent}: {detail}", level=level)
        elif event.kind in {"session_started", "session_stopped"}:
            if event.kind == "session_started":
                self._mark_manual_recovery(event.agent, action="session_started")
            else:
                self._mark_session_stopped(event.agent)
            self._push_message("system", f"{event.agent}: {event.message}", level="info")

        if self._shutting_down:
            return
        if auto_recovered_transport:
            return
        prompt = self._build_worker_event_prompt(event)
        if prompt:
            self._maybe_interrupt_manager_for_worker_event(event)
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
            result_quality = str(snapshot.get("result_quality") or "ok")
            capture_source = str(snapshot.get("result_capture_source") or "unknown")
            validation_note = str(snapshot.get("result_validation_note") or "")
            if result_quality == "weak":
                detail = validation_note or f"low-confidence capture via {capture_source}"
                self._push_message("system", f"{provider}: completed {label} ({detail})", level="warn")
            else:
                self._push_message("system", f"{provider}: completed {label}", level="info")
            result = str(snapshot.get("result") or "")
            if self._user_message_count > 0 and not self._shutting_down:
                self.manager.enqueue(
                    (
                        "Parallel web reasoning job completed.\n"
                        f"Latest user request: {self._last_user_message}\n"
                        f"Provider: {provider}\n"
                        f"Label: {label}\n"
                        f"Result quality: {result_quality}\n"
                        f"Capture source: {capture_source}\n"
                        f"Validation note: {validation_note or '<none>'}\n"
                        f"Result:\n{result}\n\n"
                        "Decide whether to revise delegation, restart workers, or send a better user-facing update. "
                        "If result quality is weak or the capture source is page-level, describe it as an extraction/capture issue rather than claiming the provider is broken. "
                        "Use tools only.\n\n"
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
                    "React only if this changes delegation or user-visible status. "
                    "If the error points to response extraction or page-level UI capture, describe it as a capture failure rather than as a provider malfunction. "
                    "Use tools only.\n\n"
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
        if event.kind == "assistant_message" and not self._assistant_message_needs_supervision(message):
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

    def _single_line_snippet(self, text: str, *, limit: int = 96) -> str:
        clean = " ".join(text.strip().split())
        if len(clean) <= limit:
            return clean
        return f"{clean[: max(0, limit - 3)]}..."

    def _interrupt_manager_turn(self, manager_overview: dict[str, object], *, reason: str) -> None:
        running_for = manager_overview.get("running_for_seconds")
        turn_id = str(manager_overview.get("turn_id") or "")
        self._last_manager_interrupt_turn = turn_id or None
        self.manager.interrupt(reason=reason, requested_by="controller")
        duration_suffix = f" after {running_for:.1f}s" if isinstance(running_for, float) else ""
        turn_suffix = f" {turn_id}" if turn_id else ""
        self._push_message("system", f"interrupted manager turn{turn_suffix}{duration_suffix} because {reason}", level="warn")

    def _is_ready_message(self, message: str) -> bool:
        normalized = " ".join(message.strip().lower().split())
        return "ready for assignments" in normalized
