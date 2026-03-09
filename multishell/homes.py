from __future__ import annotations

import json
import os
import subprocess
import shutil
from pathlib import Path

from .config import (
    CLAUDE_WORKER_SPECS,
    MANAGER_SPEC,
    MODEL,
    MODEL_REASONING_EFFORT,
    WORKER_SPECS,
    all_agent_specs,
    credential_source_agent,
    home_owner_name,
    state_root,
)
from .runtime import suppress_node_warnings


def agent_home(agent_name: str) -> Path:
    return state_root() / "homes" / home_owner_name(agent_name)


def account_home(account_name: str) -> Path:
    return state_root() / "accounts" / account_name


def claude_home_owner(agent_name: str) -> str:
    return home_owner_name(agent_name)


def claude_home(agent_name: str) -> Path:
    return state_root() / "homes" / claude_home_owner(agent_name)


def codex_dir(agent_name: str) -> Path:
    return agent_home(agent_name) / ".codex"


def account_codex_dir(account_name: str) -> Path:
    return account_home(account_name) / ".codex"


def auth_path(agent_name: str) -> Path:
    return codex_dir(agent_name) / "auth.json"


def account_auth_path(account_name: str) -> Path:
    return account_codex_dir(account_name) / "auth.json"


def config_path(agent_name: str) -> Path:
    return codex_dir(agent_name) / "config.toml"


def account_config_path(account_name: str) -> Path:
    return account_codex_dir(account_name) / "config.toml"


def claude_dir(agent_name: str) -> Path:
    return claude_home(agent_name) / ".claude"


def claude_account_home(account_name: str) -> Path:
    return account_home(account_name)


def claude_account_dir(account_name: str) -> Path:
    return claude_account_home(account_name) / ".claude"


def claude_auth_path(agent_name: str) -> Path:
    return claude_dir(agent_name) / ".credentials.json"


def claude_account_auth_path(account_name: str) -> Path:
    return claude_account_dir(account_name) / ".credentials.json"


def claude_root_auth_path(agent_name: str) -> Path:
    return claude_home(agent_name) / ".claude.json"


def claude_account_root_auth_path(account_name: str) -> Path:
    return claude_account_home(account_name) / ".claude.json"


def browser_profile_path(agent_name: str) -> Path:
    return state_root() / "browser-profiles" / agent_name


def has_claude_auth(agent_name: str) -> bool:
    account_name = credential_source_agent(agent_name)
    _migrate_legacy_claude_state(account_name)
    return claude_account_auth_path(account_name).exists() or claude_account_root_auth_path(account_name).exists()


def codex_logged_in(agent_name: str) -> bool:
    account_name = credential_source_agent(agent_name)
    ensure_account_home(account_name)
    if not account_auth_path(account_name).exists():
        return False
    env = suppress_node_warnings(os.environ.copy())
    env["HOME"] = str(account_home(account_name))
    result = subprocess.run(
        ["codex", "login", "status"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    if result.returncode != 0:
        return False
    combined = f"{result.stdout}\n{result.stderr}".strip().lower()
    return "logged in" in combined


def claude_logged_in(agent_name: str) -> bool:
    if not has_claude_auth(agent_name):
        return False
    account_name = credential_source_agent(agent_name)
    ensure_claude_account_home(account_name)
    env = suppress_node_warnings(os.environ.copy())
    env.pop("ANTHROPIC_API_KEY", None)
    env["HOME"] = str(claude_account_home(account_name))
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


def missing_codex_logins() -> list[str]:
    return [spec.name for spec in [MANAGER_SPEC, *WORKER_SPECS] if not codex_logged_in(spec.name)]


def missing_claude_logins() -> list[str]:
    return [spec.name for spec in CLAUDE_WORKER_SPECS if not claude_logged_in(spec.name)]


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
    ensure_account_home(resolved_auth_source, model=model, reasoning_effort=reasoning_effort)
    _link_auth_state(agent_name, resolved_auth_source)
    return agent_home(agent_name)


def ensure_claude_home(agent_name: str, auth_source_agent: str | None = None) -> Path:
    claude_dir(agent_name).mkdir(parents=True, exist_ok=True)
    resolved_auth_source = auth_source_agent or credential_source_agent(agent_name)
    ensure_claude_account_home(resolved_auth_source)
    _link_claude_auth_state(agent_name, resolved_auth_source)
    return claude_home(agent_name)


def ensure_account_home(
    account_name: str,
    *,
    model: str = MODEL,
    reasoning_effort: str = MODEL_REASONING_EFFORT,
) -> Path:
    _migrate_legacy_codex_account_state(account_name)
    codex_root = account_codex_dir(account_name)
    codex_root.mkdir(parents=True, exist_ok=True)
    account_config_path(account_name).write_text(
        "\n".join(
            [
                f'model = "{model}"',
                f'model_reasoning_effort = "{reasoning_effort}"',
                'personality = "pragmatic"',
                "",
                "[features]",
                "multi_agent = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return account_home(account_name)


def ensure_claude_account_home(account_name: str) -> Path:
    _migrate_legacy_claude_state(account_name)
    claude_account_dir(account_name).mkdir(parents=True, exist_ok=True)
    return claude_account_home(account_name)


def _link_auth_state(agent_name: str, auth_source_agent: str) -> None:
    source = account_auth_path(auth_source_agent)
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
    for target, source in (
        (claude_auth_path(agent_name), claude_account_auth_path(auth_source_agent)),
        (claude_root_auth_path(agent_name), claude_account_root_auth_path(auth_source_agent)),
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


def _migrate_legacy_codex_account_state(account_name: str) -> None:
    legacy_home = state_root() / "homes" / account_name
    target_home = account_home(account_name)
    if legacy_home == target_home or not legacy_home.exists():
        return

    migrations = (
        (legacy_home / ".codex" / "auth.json", account_auth_path(account_name)),
        (legacy_home / ".codex" / "config.toml", account_config_path(account_name)),
    )
    for source, target in migrations:
        if not source.exists() or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _migrate_legacy_claude_state(account_name: str) -> None:
    legacy_home = state_root() / "homes" / account_name
    target_home = claude_account_home(account_name)
    if legacy_home == target_home or not legacy_home.exists():
        return

    migrations = (
        (legacy_home / ".claude.json", claude_account_root_auth_path(account_name)),
        (legacy_home / ".claude" / ".credentials.json", claude_account_auth_path(account_name)),
    )
    for source, target in migrations:
        if not source.exists() or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def all_agent_names() -> list[str]:
    return [spec.name for spec in all_agent_specs()]


def logout_paths(agent_name: str) -> list[Path]:
    account_name = credential_source_agent(agent_name)
    legacy_home = state_root() / "homes" / account_name
    paths: set[Path] = {browser_profile_path(agent_name)}

    if agent_name == MANAGER_SPEC.name or agent_name in {spec.name for spec in WORKER_SPECS}:
        paths.update(
            {
                auth_path(agent_name),
                account_auth_path(account_name),
                legacy_home / ".codex" / "auth.json",
            }
        )
    elif agent_name in {spec.name for spec in CLAUDE_WORKER_SPECS}:
        paths.update(
            {
                claude_auth_path(agent_name),
                claude_root_auth_path(agent_name),
                claude_account_auth_path(account_name),
                claude_account_root_auth_path(account_name),
                legacy_home / ".claude" / ".credentials.json",
                legacy_home / ".claude.json",
            }
        )
    else:
        raise KeyError(agent_name)

    return sorted(paths)
