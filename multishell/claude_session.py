from __future__ import annotations

from collections.abc import Callable
import json
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from .codex_session import SessionEvent, TranscriptEntry
from .codex_session import _ignore_process_noise
from .config import CLAUDE_MODEL, CLAUDE_REASONING_EFFORT, AgentSpec, workspace_root
from .homes import ensure_claude_home
from .runtime import child_env


def _now() -> float:
    return time.time()


@dataclass(frozen=True)
class TurnRequest:
    prompt: str
    source: str
    cwd: str | None = None
    enqueued_at: float = field(default_factory=_now)


@dataclass
class SessionState:
    spec: AgentSpec
    home: Path
    thread_id: str | None = None
    turn_id: str | None = None
    status: str = "stopped"
    last_error: str | None = None
    last_message: str = ""
    updated_at: float = field(default_factory=_now)
    transcript: list[TranscriptEntry] = field(default_factory=list)
    pending_tasks: int = 0
    started_turns: int = 0
    completed_turns: int = 0
    failed_turns: int = 0
    consecutive_failures: int = 0
    last_turn_started_at: float | None = None
    last_turn_finished_at: float | None = None
    last_turn_duration: float | None = None
    current_task_source: str = ""
    current_cwd: str = ""
    session_active: bool = False
    process_alive: bool = False
    auth_mode: str | None = None
    plan_type: str | None = None
    persona_label: str = ""

    def push(self, source: str, text: str) -> TranscriptEntry | None:
        clean = text.strip()
        if not clean:
            return None
        entry = TranscriptEntry(ts=_now(), source=source, text=clean)
        self.transcript.append(entry)
        self.transcript = self.transcript[-240:]
        self.last_message = clean
        self.updated_at = _now()
        return entry


@dataclass
class _TurnOutcome:
    session_id: str | None = None
    result_text: str | None = None
    error_text: str | None = None
    assistant_message_count: int = 0
    saw_result: bool = False
    result_is_error: bool = False
    last_diagnostic: str | None = None


class ClaudeSession:
    def __init__(
        self,
        spec: AgentSpec,
        initial_prompt: str,
        working_dir: Path | None = None,
        startup_prompt: str | None = None,
        persona_label: str | None = None,
        message_callback: Callable[[TranscriptEntry], None] | None = None,
        event_callback: Callable[[SessionEvent], None] | None = None,
        turn_timeout_seconds: float | None = None,
        model: str = CLAUDE_MODEL,
        reasoning_effort: str = CLAUDE_REASONING_EFFORT,
        auth_source_agent: str | None = None,
        permission_mode: str = "bypassPermissions",
    ) -> None:
        self.spec = spec
        self.initial_prompt = initial_prompt
        self.persona_label = persona_label or spec.name
        self.startup_prompt = startup_prompt
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.permission_mode = permission_mode
        self.home = ensure_claude_home(spec.name, auth_source_agent=auth_source_agent)
        self.working_dir = Path(working_dir or workspace_root())
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.state = SessionState(spec=spec, home=self.home, current_cwd=str(self.working_dir), persona_label=self.persona_label)

        self._state_lock = threading.RLock()
        self._queue: queue.Queue[TurnRequest | None] = queue.Queue()
        self._turn_thread = threading.Thread(target=self._worker_loop, name=f"multishell-{spec.name}", daemon=True)
        self._stop = threading.Event()
        self._started = False
        self._message_callback = message_callback
        self._event_callback = event_callback
        self.turn_timeout_seconds = turn_timeout_seconds

        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._interrupt_requested = threading.Event()
        self._session_generation = 0

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._turn_thread.start()
        self.start_session()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        self.stop_session(clear_pending=True)
        self._queue.put(None)
        self._turn_thread.join(timeout=5)

    def start_session(
        self,
        cwd: str | None = None,
        system_prompt: str | None = None,
        persona_label: str | None = None,
    ) -> None:
        self.interrupt()
        target_cwd = Path(cwd) if cwd else self.working_dir
        target_cwd.mkdir(parents=True, exist_ok=True)
        with self._state_lock:
            self._session_generation += 1
            self.working_dir = target_cwd
            if system_prompt is not None:
                self.initial_prompt = system_prompt
            if persona_label is not None:
                self.persona_label = persona_label
                self.state.persona_label = persona_label
            self.state.thread_id = None
            self.state.turn_id = None
            self.state.status = "idle"
            self.state.current_task_source = ""
            self.state.current_cwd = str(self.working_dir)
            self.state.session_active = True
            self.state.process_alive = False
            self.state.last_error = None
            self.state.updated_at = _now()
            current_cwd = self.state.current_cwd
            resolved_persona = self.state.persona_label
        self._push_line("system", f"session started in {current_cwd} as {resolved_persona}")
        self._emit_event(
            "session_started",
            f"session started in {current_cwd} as {resolved_persona}",
            cwd=current_cwd,
            persona_label=resolved_persona,
        )
        if self.startup_prompt:
            self.enqueue(self.startup_prompt, source="system")

    def restart_session(
        self,
        cwd: str | None = None,
        system_prompt: str | None = None,
        persona_label: str | None = None,
    ) -> None:
        self.stop_session(clear_pending=True)
        self.start_session(cwd=cwd, system_prompt=system_prompt, persona_label=persona_label)

    def stop_session(self, clear_pending: bool = False) -> None:
        if clear_pending:
            self._drain_pending_queue()
        self._terminate_process(wait=True)
        with self._state_lock:
            self._session_generation += 1
            self.state.thread_id = None
            self.state.turn_id = None
            self.state.status = "stopped"
            self.state.last_error = None
            self.state.current_task_source = ""
            self.state.session_active = False
            self.state.process_alive = False
            self.state.updated_at = _now()
            current_cwd = self.state.current_cwd
        self._emit_event("session_stopped", "session stopped", cwd=current_cwd)

    def interrupt(self) -> None:
        self._interrupt_requested.set()
        with self._process_lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            process.terminate()

    def enqueue(self, prompt: str, source: str = "system", cwd: str | None = None) -> None:
        item = TurnRequest(prompt=prompt, source=source, cwd=cwd)
        with self._state_lock:
            self.state.pending_tasks += 1
            self.state.push(source, prompt)
        self._queue.put(item)

    def overview(self) -> dict[str, object]:
        with self._state_lock:
            running_for = None
            if self.state.status == "running" and self.state.last_turn_started_at is not None:
                running_for = max(0.0, _now() - self.state.last_turn_started_at)
            return {
                "name": self.spec.name,
                "account_email": self.spec.account_email,
                "accent_color": self.spec.accent_color,
                "status": self.state.status,
                "thread_id": self.state.thread_id,
                "turn_id": self.state.turn_id,
                "pending_tasks": self.state.pending_tasks,
                "updated_at": self.state.updated_at,
                "last_error": self.state.last_error,
                "last_message": self.state.last_message,
                "started_turns": self.state.started_turns,
                "completed_turns": self.state.completed_turns,
                "failed_turns": self.state.failed_turns,
                "consecutive_failures": self.state.consecutive_failures,
                "last_turn_started_at": self.state.last_turn_started_at,
                "last_turn_finished_at": self.state.last_turn_finished_at,
                "last_turn_duration": self.state.last_turn_duration,
                "running_for_seconds": running_for,
                "current_task_source": self.state.current_task_source,
                "cwd": self.state.current_cwd,
                "session_active": self.state.session_active,
                "process_alive": self.state.process_alive,
                "auth_mode": self.state.auth_mode,
                "plan_type": self.state.plan_type,
                "persona_label": self.state.persona_label,
                "engine": self.spec.engine,
                "model": self.model,
            }

    def recent_transcript(self, lines: int = 12) -> list[TranscriptEntry]:
        with self._state_lock:
            return list(self.state.transcript[-lines:])

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                self._queue.task_done()
                continue
            try:
                self._ensure_session_ready()
                self._run_turn(item)
            except Exception as exc:
                self._mark_internal_failure(str(exc))
            finally:
                with self._state_lock:
                    self.state.pending_tasks = max(0, self.state.pending_tasks - 1)
                self._queue.task_done()

    def _ensure_session_ready(self) -> None:
        with self._state_lock:
            session_active = self.state.session_active
        if not session_active:
            self.start_session()

    def _run_turn(self, turn: TurnRequest) -> None:
        started_at = _now()
        with self._state_lock:
            generation = self._session_generation
            resume_session_id = self.state.thread_id
            self.state.status = "running"
            self.state.last_error = None
            self.state.updated_at = started_at
            self.state.started_turns += 1
            self.state.last_turn_started_at = started_at
            self.state.current_task_source = turn.source
            if turn.cwd:
                self.working_dir = Path(turn.cwd)
                self.working_dir.mkdir(parents=True, exist_ok=True)
                self.state.current_cwd = str(self.working_dir)
            current_cwd = self.state.current_cwd

        self._interrupt_requested.clear()
        self._emit_event("turn_started", "turn started", cwd=current_cwd)
        process = self._spawn_turn_process(turn.prompt, resume_session_id)
        with self._process_lock:
            self._process = process
        with self._state_lock:
            self.state.turn_id = str(process.pid)
            self.state.process_alive = True

        outcome = _TurnOutcome()
        reader = threading.Thread(
            target=self._stream_output,
            args=(process.stdout, outcome, generation),
            name=f"multishell-{self.spec.name}-stream",
            daemon=True,
        )
        reader.start()

        exit_code: int | None = None
        timed_out = False
        try:
            if self.turn_timeout_seconds is None:
                exit_code = process.wait()
            else:
                exit_code = process.wait(timeout=self.turn_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            self.interrupt()
            exit_code = self._wait_after_termination(process)
        finally:
            reader.join(timeout=10)
            with self._process_lock:
                if self._process is process:
                    self._process = None
            with self._state_lock:
                if self.state.turn_id == str(process.pid):
                    self.state.turn_id = None
                    self.state.process_alive = False

        if not self._generation_matches(generation):
            return

        if timed_out:
            self._record_turn_failure(
                f"turn timed out after {self.turn_timeout_seconds}s",
                outcome=outcome,
                started_at=started_at,
                generation=generation,
            )
            return

        interrupted = self._interrupt_requested.is_set()
        failure = self._build_failure_text(outcome, exit_code, interrupted=interrupted)
        if failure is not None:
            self._record_turn_failure(failure, outcome=outcome, started_at=started_at, generation=generation)
            return

        if outcome.assistant_message_count == 0 and outcome.result_text:
            assistant_text = outcome.result_text.strip()
            if assistant_text:
                self._push_line("assistant", assistant_text)
                self._emit_event("assistant_message", assistant_text)

        self._record_turn_success(outcome=outcome, started_at=started_at, generation=generation)

    def _spawn_turn_process(self, prompt: str, resume_session_id: str | None) -> subprocess.Popen[str]:
        env = child_env(os.environ.copy(), role="claude-session", agent=self.spec.name)
        env.pop("ANTHROPIC_API_KEY", None)
        env["HOME"] = str(self.home)
        command = [
            "claude",
            "-p",
            "--verbose",
            "--output-format",
            "stream-json",
            "--permission-mode",
            self.permission_mode,
            "--model",
            self.model,
            "--effort",
            self.reasoning_effort,
            "--system-prompt",
            self.initial_prompt,
        ]
        if resume_session_id:
            command.extend(["--resume", resume_session_id])
        command.append(prompt)
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=str(self.working_dir),
            env=env,
            start_new_session=True,
        )

    def _stream_output(self, stream: IO[str] | None, outcome: _TurnOutcome, generation: int) -> None:
        if stream is None:
            return
        try:
            for raw_line in stream:
                self._handle_stream_line(raw_line, outcome, generation)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _handle_stream_line(self, raw_line: str, outcome: _TurnOutcome, generation: int) -> None:
        line = raw_line.strip()
        if not line:
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if _ignore_process_noise(line):
                return
            outcome.last_diagnostic = line
            if self._generation_matches(generation):
                source = "error" if "error" in line.lower() else "event"
                self._push_line(source, line)
            return

        if not isinstance(payload, dict):
            outcome.last_diagnostic = line
            if self._generation_matches(generation):
                self._push_line("event", line)
            return

        session_id = payload.get("session_id")
        if isinstance(session_id, str):
            outcome.session_id = session_id
            if self._generation_matches(generation):
                with self._state_lock:
                    self.state.thread_id = session_id
                    self.state.session_active = True
                    self.state.updated_at = _now()

        payload_type = str(payload.get("type") or "")
        if payload_type == "system":
            self._handle_system_payload(payload, outcome, generation)
            return
        if payload_type == "assistant":
            self._handle_assistant_payload(payload, outcome, generation)
            return
        if payload_type == "result":
            self._handle_result_payload(payload, outcome, generation)
            return

        diagnostic = json.dumps(payload, ensure_ascii=True)
        outcome.last_diagnostic = diagnostic
        if self._generation_matches(generation):
            self._push_line("event", diagnostic)

    def _handle_system_payload(self, payload: dict[str, object], outcome: _TurnOutcome, generation: int) -> None:
        subtype = str(payload.get("subtype") or "")
        diagnostic_parts: list[str] = []
        if subtype:
            diagnostic_parts.append(subtype)
        model = payload.get("model")
        if isinstance(model, str) and model:
            diagnostic_parts.append(f"model={model}")
        permission_mode = payload.get("permissionMode")
        if isinstance(permission_mode, str) and permission_mode:
            diagnostic_parts.append(f"permission={permission_mode}")
        cwd = payload.get("cwd")
        if isinstance(cwd, str) and cwd and self._generation_matches(generation):
            with self._state_lock:
                self.state.current_cwd = cwd
                self.state.updated_at = _now()
        api_key_source = payload.get("apiKeySource")
        if isinstance(api_key_source, str) and self._generation_matches(generation):
            with self._state_lock:
                self.state.auth_mode = api_key_source
                self.state.updated_at = _now()
        if diagnostic_parts:
            outcome.last_diagnostic = " ".join(diagnostic_parts)
            if self._generation_matches(generation):
                self._push_line("meta", " ".join(diagnostic_parts))

    def _handle_assistant_payload(self, payload: dict[str, object], outcome: _TurnOutcome, generation: int) -> None:
        message = payload.get("message")
        if not isinstance(message, dict):
            return
        assistant_text = self._extract_assistant_text(message).strip()
        if not assistant_text:
            return
        outcome.assistant_message_count += 1
        outcome.last_diagnostic = assistant_text
        if self._generation_matches(generation):
            self._push_line("assistant", assistant_text)
            self._emit_event("assistant_message", assistant_text)

    def _handle_result_payload(self, payload: dict[str, object], outcome: _TurnOutcome, generation: int) -> None:
        outcome.saw_result = True
        outcome.result_is_error = bool(payload.get("is_error"))
        result = payload.get("result")
        if result is not None:
            outcome.result_text = str(result)
            if outcome.result_text.strip():
                outcome.last_diagnostic = outcome.result_text.strip()
        if outcome.result_is_error:
            outcome.error_text = (outcome.result_text or "Claude session failed").strip()
        elif self._generation_matches(generation):
            usage = payload.get("usage")
            if isinstance(usage, dict):
                output_tokens = usage.get("output_tokens")
                if output_tokens is not None:
                    self._push_line("meta", f"turn completed; output_tokens={output_tokens}")

    def _extract_assistant_text(self, message: dict[str, object]) -> str:
        content = message.get("content")
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
        return "\n\n".join(parts).strip()

    def _build_failure_text(self, outcome: _TurnOutcome, exit_code: int | None, *, interrupted: bool) -> str | None:
        if outcome.error_text:
            return outcome.error_text
        if interrupted:
            return "turn interrupted"
        if exit_code not in (None, 0):
            if outcome.last_diagnostic:
                return outcome.last_diagnostic
            return f"claude exited with code {exit_code}"
        return None

    def _record_turn_success(self, *, outcome: _TurnOutcome, started_at: float, generation: int) -> None:
        if not self._generation_matches(generation):
            return
        finished_at = _now()
        with self._state_lock:
            self.state.updated_at = finished_at
            self.state.last_turn_finished_at = finished_at
            self.state.last_turn_duration = finished_at - started_at
            self.state.current_task_source = ""
            self.state.status = "idle"
            self.state.completed_turns += 1
            self.state.consecutive_failures = 0
            self.state.last_error = None
            self.state.process_alive = False
            if outcome.session_id:
                self.state.thread_id = outcome.session_id
            duration = self.state.last_turn_duration
        self._emit_event("turn_completed", "turn completed", duration_seconds=duration)

    def _record_turn_failure(
        self,
        error_text: str,
        *,
        outcome: _TurnOutcome,
        started_at: float,
        generation: int,
    ) -> None:
        if not self._generation_matches(generation):
            return
        finished_at = _now()
        detail = error_text.strip() or "Claude session failed"
        with self._state_lock:
            self.state.updated_at = finished_at
            self.state.last_turn_finished_at = finished_at
            self.state.last_turn_duration = finished_at - started_at
            self.state.current_task_source = ""
            self.state.status = "error"
            self.state.failed_turns += 1
            self.state.consecutive_failures += 1
            self.state.last_error = detail
            self.state.process_alive = False
            if outcome.session_id:
                self.state.thread_id = outcome.session_id
            self.state.push("error", detail)
            duration = self.state.last_turn_duration
        kind = "auth_error" if self._looks_like_auth_error(detail) else "turn_failed"
        self._emit_event(kind, detail, duration_seconds=duration)

    def _mark_internal_failure(self, detail: str) -> None:
        clean = detail.strip() or "Claude session failed"
        finished_at = _now()
        with self._state_lock:
            self.state.status = "error"
            self.state.last_error = clean
            self.state.failed_turns += 1
            self.state.consecutive_failures += 1
            self.state.current_task_source = ""
            self.state.process_alive = False
            self.state.turn_id = None
            self.state.updated_at = finished_at
            self.state.last_turn_finished_at = finished_at
            if self.state.last_turn_started_at is not None:
                self.state.last_turn_duration = finished_at - self.state.last_turn_started_at
        self._push_line("error", clean)
        kind = "auth_error" if self._looks_like_auth_error(clean) else "turn_failed"
        self._emit_event(kind, clean)

    def _generation_matches(self, generation: int) -> bool:
        with self._state_lock:
            return generation == self._session_generation

    def _wait_after_termination(self, process: subprocess.Popen[str]) -> int:
        try:
            return process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()
            return process.wait(timeout=10)

    def _looks_like_auth_error(self, text: str) -> bool:
        lowered = text.lower()
        auth_markers = (
            "auth",
            "login",
            "logged in",
            "not logged",
            "unauthorized",
            "invalid api key",
            "api key",
        )
        return any(marker in lowered for marker in auth_markers)

    def _terminate_process(self, *, wait: bool) -> None:
        self._interrupt_requested.set()
        with self._process_lock:
            process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                process.terminate()
        if not wait:
            return
        try:
            self._wait_after_termination(process)
        except Exception:
            return
        with self._process_lock:
            if self._process is process:
                self._process = None

    def _drain_pending_queue(self) -> None:
        removed = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                removed += 1
            self._queue.task_done()
        if removed:
            with self._state_lock:
                self.state.pending_tasks = max(0, self.state.pending_tasks - removed)

    def _push_line(self, source: str, text: str) -> None:
        with self._state_lock:
            entry = self.state.push(source, text)
        if entry is not None and self._message_callback is not None and source == "assistant":
            self._message_callback(entry)

    def _emit_event(self, kind: str, message: str, **data: object) -> None:
        if self._event_callback is None:
            return
        event = SessionEvent(ts=_now(), agent=self.spec.name, kind=kind, message=message, data=dict(data))
        self._event_callback(event)
