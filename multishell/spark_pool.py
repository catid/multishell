from __future__ import annotations

from collections.abc import Callable
import threading
import time
import uuid
from dataclasses import dataclass, field

from .codex_session import CodexSession


class SparkError(RuntimeError):
    pass


@dataclass
class SparkJob:
    job_id: str
    worker: str
    prompt: str
    cwd: str | None
    label: str
    timeout_seconds: int
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result_text: str = ""
    error: str | None = None
    out_of_tokens: bool = False
    cancel_requested: bool = False


class SparkCoordinator:
    def __init__(
        self,
        sessions: dict[str, CodexSession],
        callback: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        self._sessions = sessions
        self._callback = callback
        self._lock = threading.Lock()
        self._jobs: dict[str, SparkJob] = {}
        self._active_by_worker: dict[str, str] = {}

    def start_job(
        self,
        worker: str,
        prompt: str,
        *,
        cwd: str | None = None,
        label: str | None = None,
        timeout_seconds: int = 900,
    ) -> dict[str, object]:
        if worker not in self._sessions:
            raise SparkError(f"unknown spark worker: {worker}")
        if not prompt.strip():
            raise SparkError("prompt is required")
        with self._lock:
            active_job_id = self._active_by_worker.get(worker)
            if active_job_id is not None:
                active_job = self._jobs[active_job_id]
                if active_job.status in {"queued", "running"}:
                    raise SparkError(f"{worker} already has a spark job in progress")
            job = SparkJob(
                job_id=str(uuid.uuid4()),
                worker=worker,
                prompt=prompt.strip(),
                cwd=cwd,
                label=(label or "spark").strip() or "spark",
                timeout_seconds=max(30, min(3600, int(timeout_seconds))),
            )
            self._jobs[job.job_id] = job
            self._active_by_worker[worker] = job.job_id
        session = self._sessions[worker]
        if session.overview()["status"] == "stopped":
            session.start()
            if cwd:
                session.restart_session(cwd=cwd)
        threading.Thread(target=self._run_job, args=(job.job_id,), name=f"spark-{worker}", daemon=True).start()
        self._emit("started", job)
        return self.snapshot(job.job_id)

    def start(self) -> None:
        return None

    def stop(self) -> None:
        for worker in list(self._sessions):
            self.cancel_active_for_worker(worker)

    def snapshot(self, job_id: str) -> dict[str, object]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise SparkError(f"unknown job: {job_id}")
            return _snapshot(job)

    def job_snapshot(self, job_id: str) -> dict[str, object]:
        return self.snapshot(job_id)

    def job_snapshot(self, job_id: str) -> dict[str, object] | None:
        try:
            return self.snapshot(job_id)
        except SparkError:
            return None

    def list_jobs(self, worker: str | None = None) -> list[dict[str, object]]:
        with self._lock:
            jobs = list(self._jobs.values())
        if worker:
            jobs = [job for job in jobs if job.worker == worker]
        jobs.sort(key=lambda item: item.created_at, reverse=True)
        return [_snapshot(job) for job in jobs]

    def cancel_job(self, job_id: str) -> dict[str, object]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise SparkError(f"unknown job: {job_id}")
            if job.status in {"completed", "failed", "cancelled"}:
                return _snapshot(job)
            job.cancel_requested = True
        self._sessions[job.worker].interrupt()
        return self.snapshot(job_id)

    def cancel_active_for_worker(self, worker: str) -> None:
        with self._lock:
            job_id = self._active_by_worker.get(worker)
        if job_id is not None:
            self.cancel_job(job_id)

    def stop(self) -> None:
        for worker in list(self._sessions):
            self.cancel_active_for_worker(worker)

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            session = self._sessions[job.worker]
            baseline = session.overview()
            baseline_completed = int(baseline["completed_turns"])
            baseline_failed = int(baseline["failed_turns"])
            start_ts = time.time()
            job.status = "running"
            job.started_at = start_ts
        session.enqueue(job.prompt, source="spark", cwd=job.cwd)
        deadline = start_ts + job.timeout_seconds
        while time.time() < deadline:
            with self._lock:
                job = self._jobs[job_id]
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.finished_at = time.time()
                    self._active_by_worker.pop(job.worker, None)
                    self._emit("cancelled", job)
                    return
            overview = session.overview()
            if int(overview["completed_turns"]) > baseline_completed:
                result_text = _latest_assistant_text(session, since=start_ts)
                with self._lock:
                    job = self._jobs[job_id]
                    job.status = "completed"
                    job.finished_at = time.time()
                    job.result_text = result_text
                    lowered = result_text.lower()
                    job.out_of_tokens = "out_of_tokens" in lowered or "out of tokens" in lowered
                    self._active_by_worker.pop(job.worker, None)
                self._emit("completed", self._jobs[job_id])
                return
            if int(overview["failed_turns"]) > baseline_failed:
                error = str(overview.get("last_error") or "spark turn failed")
                with self._lock:
                    job = self._jobs[job_id]
                    job.status = "failed"
                    job.error = error
                    job.finished_at = time.time()
                    self._active_by_worker.pop(job.worker, None)
                self._emit("failed", self._jobs[job_id])
                return
            time.sleep(1)
        session.interrupt()
        with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.error = f"timed out after {job.timeout_seconds}s"
            job.finished_at = time.time()
            self._active_by_worker.pop(job.worker, None)
        self._emit("failed", self._jobs[job_id])

    def _emit(self, kind: str, job: SparkJob) -> None:
        if self._callback is not None:
            self._callback(kind, _snapshot(job))


def _latest_assistant_text(session: CodexSession, *, since: float) -> str:
    entries = session.recent_transcript(80)
    for entry in reversed(entries):
        if entry.ts < since:
            break
        if entry.source == "assistant" and entry.text.strip():
            return entry.text.strip()
    for entry in reversed(entries):
        if entry.source == "assistant" and entry.text.strip():
            return entry.text.strip()
    return ""


def _snapshot(job: SparkJob) -> dict[str, object]:
    preview = job.prompt if len(job.prompt) <= 160 else f"{job.prompt[:157]}..."
    return {
        "job_id": job.job_id,
        "worker": job.worker,
        "label": job.label,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "timeout_seconds": job.timeout_seconds,
        "cwd": job.cwd,
        "prompt_preview": preview,
        "result_text": job.result_text,
        "error": job.error,
        "out_of_tokens": job.out_of_tokens,
    }
