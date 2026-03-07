from __future__ import annotations

from collections.abc import Callable
import json
import os
import queue
import signal
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect

from .config import MODEL, MODEL_REASONING_EFFORT, AgentSpec, workspace_root
from .homes import ensure_agent_home
from .runtime import child_env


OPT_OUT_NOTIFICATION_METHODS = [
    "codex/event/agent_message_content_delta",
    "codex/event/agent_message_delta",
    "codex/event/agent_reasoning_delta",
    "codex/event/reasoning_content_delta",
    "codex/event/reasoning_raw_content_delta",
    "codex/event/exec_command_output_delta",
    "codex/event/item_started",
    "codex/event/item_completed",
    "item/agentMessage/delta",
    "item/commandExecution/outputDelta",
    "item/fileChange/outputDelta",
    "item/plan/delta",
    "item/reasoning/summaryTextDelta",
    "item/reasoning/textDelta",
]


def _now() -> float:
    return time.time()


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class TranscriptEntry:
    ts: float
    source: str
    text: str


@dataclass(frozen=True)
class SessionEvent:
    ts: float
    agent: str
    kind: str
    message: str
    data: dict[str, object] = field(default_factory=dict)


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


class CodexSession:
    def __init__(
        self,
        spec: AgentSpec,
        initial_prompt: str,
        mcp_bridge_command: list[str] | None = None,
        working_dir: Path | None = None,
        startup_prompt: str | None = None,
        persona_label: str | None = None,
        message_callback: Callable[[TranscriptEntry], None] | None = None,
        event_callback: Callable[[SessionEvent], None] | None = None,
        turn_timeout_seconds: float | None = None,
        model: str = MODEL,
        reasoning_effort: str = MODEL_REASONING_EFFORT,
        auth_source_agent: str | None = None,
    ) -> None:
        self.spec = spec
        self._default_initial_prompt = initial_prompt
        self.initial_prompt = initial_prompt
        self.persona_label = persona_label or spec.name
        self.startup_prompt = startup_prompt
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.home = ensure_agent_home(
            spec.name,
            mcp_bridge_command=mcp_bridge_command,
            model=model,
            reasoning_effort=reasoning_effort,
            auth_source_agent=auth_source_agent,
        )
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
        self._ws_lock = threading.Lock()
        self._ws: ClientConnection | None = None
        self._request_lock = threading.Lock()
        self._response_waiters: dict[str, queue.Queue[dict[str, object]]] = {}
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._turn_done = threading.Event()
        self._current_turn_error: str | None = None
        self._intentional_transport_close = threading.Event()

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
        target_cwd = Path(cwd) if cwd else self.working_dir
        self.working_dir = target_cwd
        self.working_dir.mkdir(parents=True, exist_ok=True)
        if system_prompt is not None:
            self.initial_prompt = system_prompt
        if persona_label is not None:
            self.persona_label = persona_label
            with self._state_lock:
                self.state.persona_label = self.persona_label
        self._launch_session()
        if self.startup_prompt:
            self.enqueue(self.startup_prompt, source="system")

    def restart_session(
        self,
        cwd: str | None = None,
        system_prompt: str | None = None,
        persona_label: str | None = None,
    ) -> None:
        self.stop_session(clear_pending=True)
        self.start_session(cwd, system_prompt=system_prompt, persona_label=persona_label)

    def stop_session(self, clear_pending: bool = False) -> None:
        if clear_pending:
            self._drain_pending_queue()
        self.interrupt()
        self._close_transport()
        with self._state_lock:
            self.state.process_alive = False
            self.state.session_active = False
            self.state.turn_id = None
            self.state.thread_id = None
            self.state.status = "stopped" if not self._stop.is_set() else "stopped"
            self.state.current_task_source = ""
            self.state.updated_at = _now()
            current_cwd = self.state.current_cwd
        self._emit_event("session_stopped", "session stopped", cwd=current_cwd)

    def interrupt(self) -> None:
        thread_id = None
        turn_id = None
        with self._state_lock:
            thread_id = self.state.thread_id
            turn_id = self.state.turn_id
        if thread_id and turn_id:
            try:
                self._request("turn/interrupt", {"threadId": thread_id, "turnId": turn_id}, timeout=10)
                return
            except Exception:
                pass
        self._terminate_process()

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
            except Exception as exc:  # pragma: no cover - defensive
                with self._state_lock:
                    self.state.status = "error"
                    self.state.last_error = str(exc)
                    self.state.failed_turns += 1
                    self.state.consecutive_failures += 1
                    self.state.push("error", str(exc))
                    self.state.updated_at = _now()
                self._emit_event("turn_failed", str(exc))
            finally:
                with self._state_lock:
                    self.state.pending_tasks = max(0, self.state.pending_tasks - 1)
                self._queue.task_done()

    def _ensure_session_ready(self) -> None:
        with self._state_lock:
            active = self.state.session_active and self.state.process_alive and self.state.thread_id is not None
        if active:
            return
        self._launch_session()

    def _launch_session(self) -> None:
        self._close_transport()
        port = _reserve_port()
        env = child_env(os.environ.copy(), role="codex-session", agent=self.spec.name)
        env["HOME"] = str(self.home)
        process = subprocess.Popen(
            ["codex", "app-server", "--listen", f"ws://127.0.0.1:{port}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=str(self.working_dir),
            env=env,
            start_new_session=True,
        )
        ws = self._connect_websocket(port)

        with self._process_lock:
            self._process = process
        with self._ws_lock:
            self._ws = ws

        self._turn_done.clear()
        self._current_turn_error = None
        self._reader_thread = threading.Thread(target=self._socket_reader, name=f"multishell-{self.spec.name}-ws", daemon=True)
        self._reader_thread.start()
        if process.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._stderr_reader,
                args=(process.stderr,),
                name=f"multishell-{self.spec.name}-stderr",
                daemon=True,
            )
            self._stderr_thread.start()

        self._request(
            "initialize",
            {
                "clientInfo": {"name": "multishell", "title": "Multishell", "version": "0.2.0"},
                "capabilities": {"experimentalApi": True, "optOutNotificationMethods": OPT_OUT_NOTIFICATION_METHODS},
            },
            timeout=20,
        )
        self._notify({"method": "initialized"})
        result = self._request(
            "thread/start",
            {
                "model": self.model,
                "cwd": str(self.working_dir),
                "approvalPolicy": "never",
                "sandbox": "danger-full-access",
                "baseInstructions": self.initial_prompt,
                "personality": "pragmatic",
            },
            timeout=30,
        )

        thread = result.get("thread", {})
        thread_id = str(thread.get("id", ""))
        with self._state_lock:
            self.state.thread_id = thread_id or None
            self.state.turn_id = None
            self.state.process_alive = True
            self.state.session_active = True
            self.state.status = "idle"
            self.state.current_cwd = str(thread.get("cwd") or self.working_dir)
            self.state.updated_at = _now()
            self.state.last_error = None
            current_cwd = self.state.current_cwd
            persona_label = self.state.persona_label
        self._push_line("system", f"session started in {current_cwd} as {persona_label}")
        self._emit_event("session_started", f"session started in {current_cwd} as {persona_label}", cwd=current_cwd, persona_label=persona_label)

    def _connect_websocket(self, port: int) -> ClientConnection:
        last_error: Exception | None = None
        url = f"ws://127.0.0.1:{port}"
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                return connect(url, open_timeout=5, close_timeout=1)
            except Exception as exc:  # pragma: no cover - environment dependent
                last_error = exc
                time.sleep(0.25)
        raise RuntimeError(f"failed to connect to app-server websocket at {url}: {last_error}")

    def _request(self, method: str, params: dict[str, object], timeout: float) -> dict[str, object]:
        request_id = str(uuid.uuid4())
        waiter: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)
        with self._request_lock:
            self._response_waiters[request_id] = waiter
            self._send({"id": request_id, "method": method, "params": params})
        try:
            response = waiter.get(timeout=timeout)
        except queue.Empty as exc:
            with self._request_lock:
                self._response_waiters.pop(request_id, None)
            raise RuntimeError(f"timed out waiting for response to {method}") from exc
        if "error" in response:
            raise RuntimeError(f"{method} failed: {json.dumps(response['error'], ensure_ascii=True)}")
        result = response.get("result", {})
        return result if isinstance(result, dict) else {"value": result}

    def _notify(self, payload: dict[str, object]) -> None:
        self._send(payload)

    def _send(self, payload: dict[str, object]) -> None:
        message = json.dumps(payload, ensure_ascii=True)
        with self._ws_lock:
            if self._ws is None:
                raise RuntimeError("app-server websocket is not connected")
            self._ws.send(message)

    def _run_turn(self, turn: TurnRequest) -> None:
        started_at = _now()
        with self._state_lock:
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
            thread_id = self.state.thread_id
        if thread_id is None:
            raise RuntimeError("cannot start turn without a loaded thread")

        self._turn_done.clear()
        self._current_turn_error = None
        result = self._request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": turn.prompt, "text_elements": []}],
                "model": self.model,
                "effort": self.reasoning_effort,
                "personality": "pragmatic",
                **({"cwd": turn.cwd} if turn.cwd else {}),
            },
            timeout=30,
        )
        turn_info = result.get("turn", {})
        with self._state_lock:
            self.state.turn_id = str(turn_info.get("id") or self.state.turn_id)

        wait_timeout = self.turn_timeout_seconds
        completed = self._turn_done.wait(timeout=wait_timeout)
        if not completed:
            self.interrupt()
            raise RuntimeError(f"turn timed out after {wait_timeout}s")
        if self._current_turn_error:
            raise RuntimeError(self._current_turn_error)

    def _socket_reader(self) -> None:
        try:
            while not self._stop.is_set():
                with self._ws_lock:
                    ws = self._ws
                if ws is None:
                    return
                raw = ws.recv()
                if raw is None:
                    break
                message = json.loads(raw)
                if "id" in message and ("result" in message or "error" in message):
                    request_id = str(message.get("id"))
                    with self._request_lock:
                        waiter = self._response_waiters.pop(request_id, None)
                    if waiter is not None:
                        waiter.put(message)
                    continue
                if "method" in message:
                    self._handle_notification(message)
        except ConnectionClosed:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            self._push_line("error", f"app-server reader failed: {exc}")
        finally:
            self._handle_transport_closed()

    def _stderr_reader(self, stderr: subprocess.PIPE[str]) -> None:
        try:
            for line in stderr:
                clean = line.strip()
                if not clean:
                    continue
                source = "event"
                if "error" in clean.lower():
                    source = "error"
                self._push_line(source, clean)
        except Exception:  # pragma: no cover - defensive
            return

    def _handle_transport_closed(self) -> None:
        intentional = self._intentional_transport_close.is_set()
        with self._state_lock:
            was_running = self.state.turn_id is not None
            self.state.process_alive = False
            self.state.session_active = False
            self.state.turn_id = None
            if not intentional and not self._stop.is_set() and self.state.status != "stopped":
                self.state.status = "error"
                self.state.last_error = self.state.last_error or "app-server connection closed"
                if was_running:
                    self.state.failed_turns += 1
                    self.state.consecutive_failures += 1
            self.state.updated_at = _now()
            detail = self.state.last_error or "app-server connection closed"
        self._current_turn_error = self._current_turn_error or "app-server connection closed"
        self._turn_done.set()
        if not intentional:
            self._emit_event("transport_closed", detail)
        with self._request_lock:
            waiters = list(self._response_waiters.values())
            self._response_waiters.clear()
        for waiter in waiters:
            waiter.put({"error": {"message": "app-server connection closed"}})

    def _handle_notification(self, message: dict[str, object]) -> None:
        method = str(message.get("method", ""))
        params = message.get("params", {})
        if not isinstance(params, dict):
            return

        if method == "thread/started":
            thread = params.get("thread", {})
            if isinstance(thread, dict):
                with self._state_lock:
                    self.state.thread_id = str(thread.get("id") or self.state.thread_id)
                    self.state.current_cwd = str(thread.get("cwd") or self.state.current_cwd)
                    self.state.session_active = True
                    self.state.process_alive = True
            return

        if method == "thread/status/changed":
            status = params.get("status", {})
            status_type = status.get("type") if isinstance(status, dict) else None
            previous_status = None
            with self._state_lock:
                previous_status = self.state.status
                if status_type == "active":
                    self.state.status = "running"
                elif status_type == "idle" and self.state.turn_id is None:
                    self.state.status = "idle"
                elif status_type == "systemError":
                    self.state.status = "error"
                current_status = self.state.status
            if status_type and current_status != previous_status:
                self._emit_event("status_changed", f"status changed to {current_status}", status=current_status)
            return

        if method == "turn/started":
            turn = params.get("turn", {})
            if isinstance(turn, dict):
                with self._state_lock:
                    self.state.turn_id = str(turn.get("id") or self.state.turn_id)
                    self.state.status = "running"
                    turn_id = self.state.turn_id
                    current_cwd = self.state.current_cwd
                self._emit_event("turn_started", "turn started", turn_id=turn_id, cwd=current_cwd)
            return

        if method == "turn/completed":
            turn = params.get("turn", {})
            turn_error = None
            turn_status = None
            if isinstance(turn, dict):
                turn_status = turn.get("status")
                if turn.get("error") is not None:
                    turn_error = json.dumps(turn.get("error"), ensure_ascii=True)
            finished_at = _now()
            with self._state_lock:
                self.state.updated_at = finished_at
                self.state.last_turn_finished_at = finished_at
                if self.state.last_turn_started_at is not None:
                    self.state.last_turn_duration = finished_at - self.state.last_turn_started_at
                self.state.turn_id = None
                self.state.current_task_source = ""
                if turn_error or turn_status != "completed":
                    self.state.status = "error"
                    self.state.failed_turns += 1
                    self.state.consecutive_failures += 1
                    self.state.last_error = turn_error or f"turn completed with status={turn_status}"
                    self.state.push("error", self.state.last_error)
                    self._current_turn_error = self.state.last_error
                else:
                    self.state.status = "idle"
                    self.state.completed_turns += 1
                    self.state.consecutive_failures = 0
                    self._current_turn_error = None
                duration = self.state.last_turn_duration
                failure_text = self.state.last_error
            self._turn_done.set()
            if turn_error or turn_status != "completed":
                self._emit_event(
                    "turn_failed",
                    failure_text or "turn failed",
                    duration_seconds=duration,
                )
            else:
                self._emit_event(
                    "turn_completed",
                    "turn completed",
                    duration_seconds=duration,
                )
            return

        if method == "thread/closed":
            with self._state_lock:
                self.state.thread_id = None
                self.state.turn_id = None
                self.state.session_active = False
                self.state.process_alive = False
                self.state.status = "stopped"
                self.state.updated_at = _now()
            self._turn_done.set()
            self._emit_event("thread_closed", "thread closed")
            return

        if method == "item/completed":
            item = params.get("item", {})
            if not isinstance(item, dict):
                return
            item_type = item.get("type")
            if item_type == "agentMessage":
                text = str(item.get("text", ""))
                self._push_line("assistant", text)
                if text.strip():
                    self._emit_event("assistant_message", text, phase=item.get("phase"))
            elif item_type == "reasoning":
                pass
            else:
                self._push_line("event", json.dumps(item, ensure_ascii=True))
            return

        if method == "account/updated":
            with self._state_lock:
                self.state.auth_mode = params.get("authMode") if params.get("authMode") is not None else self.state.auth_mode
                self.state.plan_type = params.get("planType") if params.get("planType") is not None else self.state.plan_type
                auth_mode = self.state.auth_mode
                plan_type = self.state.plan_type
            self._emit_event("account_updated", "account updated", auth_mode=auth_mode, plan_type=plan_type)
            return

        if method == "account/login/completed":
            if not params.get("success", False):
                detail = str(params.get("error") or "account login failed")
                with self._state_lock:
                    self.state.status = "error"
                    self.state.last_error = detail
                self._push_line("error", detail)
                self._emit_event("auth_error", detail)
            return

        if method == "error":
            detail = json.dumps(params, ensure_ascii=True)
            with self._state_lock:
                self.state.status = "error"
                self.state.last_error = detail
            self._push_line("error", detail)
            self._emit_event("error", detail)
            return

        if method == "codex/event/mcp_startup_update":
            msg = params.get("msg", {})
            if isinstance(msg, dict):
                server = str(msg.get("server") or "")
                status = msg.get("status", {})
                state = status.get("state") if isinstance(status, dict) else None
                error = status.get("error") if isinstance(status, dict) else None
                if state == "failed":
                    detail = str(error or f"{server} startup failed")
                    self._push_line("error", f"mcp {server} failed: {detail}")
                    self._emit_event("mcp_failed", f"mcp {server} failed: {detail}", server=server)
            return

        if method == "codex/event/mcp_startup_complete":
            msg = params.get("msg", {})
            if isinstance(msg, dict):
                ready = ", ".join(msg.get("ready", [])) or "none"
                failed_entries = msg.get("failed", [])
                failed = ", ".join(
                    entry.get("server", "") if isinstance(entry, dict) else str(entry)
                    for entry in failed_entries
                ) or "none"
                self._push_line("mcp", f"mcp ready={ready} failed={failed}")
                if failed_entries:
                    self._emit_event(
                        "mcp_failed",
                        f"mcp ready={ready} failed={failed}",
                        ready=msg.get("ready", []),
                        failed=failed_entries,
                    )
                else:
                    self._emit_event("mcp_ready", f"mcp ready={ready}", ready=msg.get("ready", []))
            return

        if method == "codex/event/task_complete":
            msg = params.get("msg", {})
            if isinstance(msg, dict):
                last = str(msg.get("last_agent_message") or "").strip()
                if last:
                    self._push_line("meta", f"task complete; last_agent_message={last}")
            return

        if method == "codex/event/task_started":
            self._push_line("meta", "task started")
            return

    def _push_line(self, source: str, text: str) -> None:
        entry = None
        with self._state_lock:
            entry = self.state.push(source, text)
        if entry is not None and self._message_callback is not None and source == "assistant":
            self._message_callback(entry)

    def _emit_event(self, kind: str, message: str, **data: object) -> None:
        if self._event_callback is None:
            return
        event = SessionEvent(ts=_now(), agent=self.spec.name, kind=kind, message=message, data=dict(data))
        self._event_callback(event)

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

    def _terminate_process(self) -> None:
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

    def _close_transport(self) -> None:
        self._intentional_transport_close.set()
        with self._ws_lock:
            ws = self._ws
            self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        self._terminate_process()
        with self._process_lock:
            process = self._process
            self._process = None
        if process is not None:
            try:
                process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except Exception:
                    pass
        reader_thread = self._reader_thread
        self._reader_thread = None
        if reader_thread is not None and reader_thread.is_alive():
            reader_thread.join(timeout=1)
        stderr_thread = self._stderr_thread
        self._stderr_thread = None
        if stderr_thread is not None and stderr_thread.is_alive():
            stderr_thread.join(timeout=1)
        self._intentional_transport_close.clear()
