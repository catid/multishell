from __future__ import annotations

import json
import os
import subprocess
import shutil
from pathlib import Path

from .config import MODEL, MODEL_REASONING_EFFORT, all_agent_specs, credential_source_agent, state_root
from .runtime import suppress_node_warnings


def agent_home(agent_name: str) -> Path:
    return state_root() / "homes" / agent_name


def claude_home_owner(agent_name: str) -> str:
    return credential_source_agent(agent_name)


def claude_home(agent_name: str) -> Path:
    return state_root() / "homes" / claude_home_owner(agent_name)


def codex_dir(agent_name: str) -> Path:
    return agent_home(agent_name) / ".codex"


def auth_path(agent_name: str) -> Path:
    return codex_dir(agent_name) / "auth.json"


def config_path(agent_name: str) -> Path:
    return codex_dir(agent_name) / "config.toml"


def claude_dir(agent_name: str) -> Path:
    return claude_home(agent_name) / ".claude"


def claude_auth_path(agent_name: str) -> Path:
    return claude_dir(agent_name) / ".credentials.json"


def claude_root_auth_path(agent_name: str) -> Path:
    return claude_home(agent_name) / ".claude.json"


def has_claude_auth(agent_name: str) -> bool:
    return claude_auth_path(agent_name).exists() or claude_root_auth_path(agent_name).exists()


def claude_logged_in(agent_name: str) -> bool:
    if not has_claude_auth(agent_name):
        return False
    env = suppress_node_warnings(os.environ.copy())
    env.pop("ANTHROPIC_API_KEY", None)
    env["HOME"] = str(claude_home(agent_name))
    result = subprocess.run(
        ["claude", "auth", "status"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    if result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    return bool(payload.get("loggedIn"))


def ensure_agent_home(
    agent_name: str,
    mcp_bridge_command: list[str] | None = None,
    manager_bridge_command: list[str] | None = None,
    model: str = MODEL,
    reasoning_effort: str = MODEL_REASONING_EFFORT,
    auth_source_agent: str | None = None,
) -> Path:
    codex_home = codex_dir(agent_name)
    codex_home.mkdir(parents=True, exist_ok=True)

    base_lines = [
        f'model = "{model}"',
        f'model_reasoning_effort = "{reasoning_effort}"',
        'personality = "pragmatic"',
        "",
        "[features]",
        "multi_agent = true",
        "",
    ]

    bridge_command = mcp_bridge_command or manager_bridge_command
    if bridge_command:
        escaped_args = ", ".join(json.dumps(arg) for arg in bridge_command[1:])
        base_lines.extend(
            [
                "[mcp_servers.multishell]",
                f"command = {json.dumps(bridge_command[0])}",
                f"args = [{escaped_args}]",
                "",
            ]
        )

    config_path(agent_name).write_text("\n".join(base_lines), encoding="utf-8")
    resolved_auth_source = auth_source_agent or credential_source_agent(agent_name)
    _link_auth_state(agent_name, resolved_auth_source)
    return agent_home(agent_name)


def ensure_claude_home(agent_name: str, auth_source_agent: str | None = None) -> Path:
    _migrate_legacy_claude_state(agent_name)
    claude_dir(agent_name).mkdir(parents=True, exist_ok=True)
    resolved_auth_source = auth_source_agent or credential_source_agent(agent_name)
    _link_claude_auth_state(agent_name, resolved_auth_source)
    return claude_home(agent_name)


def _link_auth_state(agent_name: str, auth_source_agent: str) -> None:
    if auth_source_agent == agent_name:
        return

    source = auth_path(auth_source_agent)
    target = auth_path(agent_name)
    if not source.exists():
        return
    if target.exists() or target.is_symlink():
        try:
            if target.samefile(source):
                return
        except FileNotFoundError:
            pass
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _link_claude_auth_state(agent_name: str, auth_source_agent: str) -> None:
    if claude_home(agent_name) == claude_home(auth_source_agent):
        return
    for target, source in (
        (claude_auth_path(agent_name), claude_auth_path(auth_source_agent)),
        (claude_root_auth_path(agent_name), claude_root_auth_path(auth_source_agent)),
    ):
        if not source.exists():
            continue
        if target.exists() or target.is_symlink():
            try:
                if target.samefile(source):
                    continue
            except FileNotFoundError:
                pass
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)


def _migrate_legacy_claude_state(agent_name: str) -> None:
    legacy_home = state_root() / "homes" / agent_name
    target_home = claude_home(agent_name)
    if legacy_home == target_home or not legacy_home.exists():
        return

    migrations = (
        (legacy_home / ".claude.json", target_home / ".claude.json"),
        (legacy_home / ".claude" / ".credentials.json", target_home / ".claude" / ".credentials.json"),
    )
    for source, target in migrations:
        if not source.exists() or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)


def all_agent_names() -> list[str]:
    return [spec.name for spec in all_agent_specs()]
