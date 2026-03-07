from __future__ import annotations

import os
import signal
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import socket_path, state_root


ENV_STATE_ROOT = "MULTISHELL_STATE_ROOT"
ENV_INSTANCE_ID = "MULTISHELL_INSTANCE_ID"
ENV_ROLE = "MULTISHELL_ROLE"
ENV_AGENT = "MULTISHELL_AGENT"
ENV_NODE_NO_WARNINGS = "NODE_NO_WARNINGS"


@dataclass(frozen=True)
class ObservedProcess:
    pid: int
    cmdline: str
    cwd: str
    home: str
    instance_id: str
    runtime_state_root: str
    role: str
    agent: str


@dataclass(frozen=True)
class StaleProcess:
    pid: int
    cmdline: str
    reasons: tuple[str, ...]


def ensure_runtime_environment(*, role: str | None = None, agent: str | None = None) -> str:
    os.environ[ENV_STATE_ROOT] = str(state_root().resolve())
    instance_id = os.environ.get(ENV_INSTANCE_ID, "").strip()
    if not instance_id:
        instance_id = str(uuid.uuid4())
        os.environ[ENV_INSTANCE_ID] = instance_id
    if role is not None:
        os.environ[ENV_ROLE] = role
    if agent is not None:
        os.environ[ENV_AGENT] = agent
    return instance_id


def child_env(
    base_env: dict[str, str] | None = None,
    *,
    role: str | None = None,
    agent: str | None = None,
) -> dict[str, str]:
    ensure_runtime_environment()
    env = suppress_node_warnings(base_env)
    env[ENV_STATE_ROOT] = os.environ[ENV_STATE_ROOT]
    env[ENV_INSTANCE_ID] = os.environ[ENV_INSTANCE_ID]
    if role is not None:
        env[ENV_ROLE] = role
    if agent is not None:
        env[ENV_AGENT] = agent
    return env


def suppress_node_warnings(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env if base_env is not None else os.environ)
    env[ENV_NODE_NO_WARNINGS] = "1"
    return env


def cleanup_stale_runtime(grace_seconds: float = 3.0) -> list[str]:
    root = state_root().resolve()
    candidates = _discover_stale_processes(
        list(_iter_processes()),
        state_root_path=root,
        current_pid=os.getpid(),
        current_instance_id=os.environ.get(ENV_INSTANCE_ID, "").strip(),
        socket_owner_pids=_control_socket_owner_pids(socket_path()),
    )
    if not candidates:
        return []

    pending = {candidate.pid: candidate for candidate in candidates}
    _signal_processes(pending, signal.SIGTERM)
    pending = _wait_for_exit(pending, deadline=time.time() + max(0.2, grace_seconds))
    if pending:
        _signal_processes(pending, signal.SIGKILL)
        pending = _wait_for_exit(pending, deadline=time.time() + max(0.2, grace_seconds))

    cleaned: list[str] = []
    for candidate in candidates:
        if candidate.pid in pending:
            cleaned.append(f"stale pid={candidate.pid} survived cleanup: {', '.join(candidate.reasons)}")
        else:
            cleaned.append(
                f"cleaned stale pid={candidate.pid}: {', '.join(candidate.reasons)}"
                f" [{_preview(candidate.cmdline)}]"
            )
    return cleaned


def _discover_stale_processes(
    processes: list[ObservedProcess],
    *,
    state_root_path: Path,
    current_pid: int,
    current_instance_id: str,
    socket_owner_pids: set[int],
) -> list[StaleProcess]:
    root = str(state_root_path)
    homes_root = str((state_root_path / "homes").resolve())
    browser_profiles_root = str((state_root_path / "browser-profiles").resolve())
    control_socket = str(socket_path().resolve())

    stale: list[StaleProcess] = []
    seen: set[int] = set()
    for process in processes:
        if process.pid == current_pid or process.pid in seen:
            continue

        reasons: list[str] = []
        if process.pid in socket_owner_pids:
            reasons.append("owns control socket")
        if process.runtime_state_root == root and process.instance_id and process.instance_id != current_instance_id:
            reasons.append("previous multishell runtime")
        if process.home.startswith(homes_root):
            reasons.append("isolated agent HOME")
        if browser_profiles_root in process.cmdline or process.cwd.startswith(browser_profiles_root):
            reasons.append("isolated browser profile")
        if control_socket and control_socket in process.cmdline:
            reasons.append("multishell control subprocess")

        if reasons:
            stale.append(StaleProcess(pid=process.pid, cmdline=process.cmdline, reasons=tuple(dict.fromkeys(reasons))))
            seen.add(process.pid)
    return stale


def _iter_processes() -> list[ObservedProcess]:
    observed: list[ObservedProcess] = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return observed
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            cmdline = _read_cmdline(entry)
            env = _read_environ(entry)
            cwd = os.readlink(entry / "cwd")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        observed.append(
            ObservedProcess(
                pid=pid,
                cmdline=cmdline,
                cwd=cwd,
                home=env.get("HOME", ""),
                instance_id=env.get(ENV_INSTANCE_ID, ""),
                runtime_state_root=env.get(ENV_STATE_ROOT, ""),
                role=env.get(ENV_ROLE, ""),
                agent=env.get(ENV_AGENT, ""),
            )
        )
    return observed


def _read_cmdline(proc_entry: Path) -> str:
    raw = (proc_entry / "cmdline").read_bytes()
    if not raw:
        return ""
    parts = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
    return " ".join(parts)


def _read_environ(proc_entry: Path) -> dict[str, str]:
    raw = (proc_entry / "environ").read_bytes()
    env: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode("utf-8", errors="replace")] = value.decode("utf-8", errors="replace")
    return env


def _control_socket_owner_pids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    inode = _socket_inode(path)
    if not inode:
        return set()

    owners: set[int] = set()
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        fd_dir = entry / "fd"
        if not fd_dir.exists():
            continue
        try:
            for fd_entry in fd_dir.iterdir():
                try:
                    target = os.readlink(fd_entry)
                except (FileNotFoundError, PermissionError, OSError):
                    continue
                if target == f"socket:[{inode}]":
                    owners.add(int(entry.name))
                    break
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return owners


def _socket_inode(path: Path) -> str:
    needle = str(path.resolve())
    try:
        lines = Path("/proc/net/unix").read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        maybe_path = parts[-1] if parts[-1].startswith("/") else ""
        if maybe_path != needle:
            continue
        return parts[-2] if maybe_path else parts[-1]
    return ""


def _signal_processes(processes: dict[int, StaleProcess], sig: int) -> None:
    for pid in list(processes):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            processes.pop(pid, None)
        except PermissionError:
            continue


def _wait_for_exit(processes: dict[int, StaleProcess], *, deadline: float) -> dict[int, StaleProcess]:
    pending = dict(processes)
    while pending and time.time() < deadline:
        time.sleep(0.1)
        for pid in list(pending):
            if not _pid_exists(pid):
                pending.pop(pid, None)
    for pid in list(pending):
        if not _pid_exists(pid):
            pending.pop(pid, None)
    return pending


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _preview(text: str, limit: int = 120) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned or "<no cmdline>"
    return f"{cleaned[: limit - 3]}..."
