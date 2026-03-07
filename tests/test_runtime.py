from __future__ import annotations

from pathlib import Path

from multishell.config import state_root
from multishell.runtime import (
    ENV_AGENT,
    ENV_INSTANCE_ID,
    ENV_NODE_NO_WARNINGS,
    ENV_ROLE,
    ENV_STATE_ROOT,
    ObservedProcess,
    _discover_stale_processes,
    child_env,
    suppress_node_warnings,
)


def test_child_env_inherits_runtime_markers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_STATE_ROOT, str(tmp_path / ".multishell"))
    monkeypatch.setenv(ENV_INSTANCE_ID, "instance-1")

    env = child_env({"PATH": "/usr/bin"}, role="codex-session", agent="worker-1")

    assert env[ENV_STATE_ROOT] == str(state_root().resolve())
    assert env[ENV_INSTANCE_ID] == "instance-1"
    assert env[ENV_ROLE] == "codex-session"
    assert env[ENV_AGENT] == "worker-1"
    assert env[ENV_NODE_NO_WARNINGS] == "1"
    assert env["PATH"] == "/usr/bin"


def test_suppress_node_warnings_overrides_inherited_setting() -> None:
    env = suppress_node_warnings({"PATH": "/usr/bin", ENV_NODE_NO_WARNINGS: "0"})

    assert env[ENV_NODE_NO_WARNINGS] == "1"
    assert env["PATH"] == "/usr/bin"


def test_discover_stale_processes_finds_previous_runtime_homes_and_browser_profiles(tmp_path: Path) -> None:
    state = tmp_path / ".multishell"
    processes = [
        ObservedProcess(
            pid=101,
            cmdline="python3 -m multishell run",
            cwd="/home/catid/multishell",
            home="/home/catid",
            instance_id="old-instance",
            runtime_state_root=str(state),
            role="controller",
            agent="manager",
        ),
        ObservedProcess(
            pid=102,
            cmdline="codex app-server --listen ws://127.0.0.1:1",
            cwd="/home/catid/multishell",
            home=str(state / "homes" / "worker-1"),
            instance_id="",
            runtime_state_root="",
            role="",
            agent="",
        ),
        ObservedProcess(
            pid=103,
            cmdline=f"google-chrome --user-data-dir={state / 'browser-profiles' / 'manager'}",
            cwd="/tmp",
            home="/home/catid",
            instance_id="",
            runtime_state_root="",
            role="",
            agent="",
        ),
        ObservedProcess(
            pid=200,
            cmdline="python3 -m multishell run",
            cwd="/home/catid/multishell",
            home="/home/catid",
            instance_id="current-instance",
            runtime_state_root=str(state),
            role="controller",
            agent="manager",
        ),
    ]

    stale = _discover_stale_processes(
        processes,
        state_root_path=state,
        current_pid=200,
        current_instance_id="current-instance",
        socket_owner_pids={101},
    )

    assert [item.pid for item in stale] == [101, 102, 103]
    assert stale[0].reasons == ("owns control socket", "previous multishell runtime")
    assert stale[1].reasons == ("isolated agent HOME",)
    assert stale[2].reasons == ("isolated browser profile",)
