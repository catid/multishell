from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .control import send_control_request


LATEST_PROTOCOL_VERSION = "2025-06-18"


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, object]


def _async_reasoner_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["start", "status", "list", "cancel"]},
            "prompt": {"type": "string"},
            "job_id": {"type": "string"},
            "label": {"type": "string"},
            "agent": {"type": "string"},
            "cwd": {"type": "string"},
            "timeout_seconds": {"type": "integer", "minimum": 30, "maximum": 3600},
        },
        "required": ["action"],
        "additionalProperties": False,
    }


MANAGER_TOOLS = [
    ToolDef(
        name="notify_user",
        description="Send a concise manager-authored message into the main TUI chat transcript.",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "level": {"type": "string", "enum": ["info", "warn", "error"]},
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    ),
    ToolDef(
        name="delegate_to_worker",
        description="Queue a new task or nudge for a named worker session. Prefer task-specific complementary roles instead of near-duplicate assignments.",
        input_schema={
            "type": "object",
            "properties": {
                "worker": {"type": "string"},
                "task": {"type": "string"},
                "cwd": {"type": "string"},
            },
            "required": ["worker", "task"],
            "additionalProperties": False,
        },
    ),
    ToolDef(
        name="start_worker_session",
        description="Start a stopped worker session, optionally in a specific working directory and with a task-specific persona.",
        input_schema={
            "type": "object",
            "properties": {
                "worker": {"type": "string"},
                "cwd": {"type": "string"},
                "persona_name": {"type": "string"},
                "task_context": {"type": "string"},
                "extra_instructions": {"type": "string"},
                "system_prompt": {"type": "string"},
            },
            "required": ["worker"],
            "additionalProperties": False,
        },
    ),
    ToolDef(
        name="stop_worker_session",
        description="Stop a worker session and clear its in-memory state until restarted.",
        input_schema={
            "type": "object",
            "properties": {
                "worker": {"type": "string"},
            },
            "required": ["worker"],
            "additionalProperties": False,
        },
    ),
    ToolDef(
        name="restart_worker_session",
        description="Restart a worker session to clear memory, optionally changing its working directory and resetting it to a task-specific persona.",
        input_schema={
            "type": "object",
            "properties": {
                "worker": {"type": "string"},
                "cwd": {"type": "string"},
                "persona_name": {"type": "string"},
                "task_context": {"type": "string"},
                "extra_instructions": {"type": "string"},
                "system_prompt": {"type": "string"},
            },
            "required": ["worker"],
            "additionalProperties": False,
        },
    ),
    ToolDef(
        name="get_workers_overview",
        description="Return high-level status for all worker sessions.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    ToolDef(
        name="get_worker_transcript",
        description="Return the latest transcript lines for one worker.",
        input_schema={
            "type": "object",
            "properties": {
                "worker": {"type": "string"},
                "lines": {"type": "integer", "minimum": 1, "maximum": 40},
            },
            "required": ["worker"],
            "additionalProperties": False,
        },
    ),
    ToolDef(
        name="gpt_5_4_pro",
        description=(
            "Launch, inspect, or cancel a long-running ChatGPT 5.4 Pro web reasoning job. "
            "Use start to begin the job asynchronously, continue delegating, then check status later."
        ),
        input_schema=_async_reasoner_schema(),
    ),
    ToolDef(
        name="gemini_deepthink",
        description=(
            "Launch, inspect, or cancel a long-running Gemini Deep Think web reasoning job. "
            "Use it for slow parallel pathing, world knowledge, or math-heavy reasoning."
        ),
        input_schema=_async_reasoner_schema(),
    ),
]

WORKER_TOOLS: list[ToolDef] = []


def tools_for_role(role: str) -> list[ToolDef]:
    return MANAGER_TOOLS if role == "manager" else WORKER_TOOLS


def read_message() -> tuple[dict[str, object] | None, str | None]:
    while True:
        first_line = sys.stdin.buffer.readline()
        if not first_line:
            return None, None
        if first_line in (b"\r\n", b"\n"):
            continue
        break

    stripped = first_line.lstrip()
    if stripped.startswith((b"{", b"[")):
        return json.loads(first_line.decode("utf-8")), "line"

    headers: dict[str, str] = {}
    line = first_line
    while True:
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode("utf-8").split(":", 1)
        headers[key.strip().lower()] = value.strip()
        line = sys.stdin.buffer.readline()
        if not line:
            return None, None
    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        return None, None
    body = sys.stdin.buffer.read(content_length)
    return json.loads(body.decode("utf-8")), "header"


def write_message(payload: dict[str, object], mode: str) -> None:
    body = json.dumps(payload).encode("utf-8")
    if mode == "header":
        sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
        sys.stdout.buffer.write(body)
    else:
        sys.stdout.buffer.write(body + b"\n")
    sys.stdout.buffer.flush()


def success_result(text: str) -> dict[str, object]:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def error_result(text: str) -> dict[str, object]:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def handle_request(
    socket_path: Path,
    message: dict[str, object],
    *,
    role: str = "manager",
    agent_name: str = "manager",
) -> dict[str, object] | None:
    method = message.get("method")
    params = message.get("params", {})
    if method == "initialize":
        protocol_version = LATEST_PROTOCOL_VERSION
        if isinstance(params, dict):
            requested = params.get("protocolVersion")
            if isinstance(requested, str) and requested:
                protocol_version = requested
        return {
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "protocolVersion": protocol_version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "multishell-bridge", "version": "0.2.0"},
            },
        }
    if method in {"notifications/initialized", "initialized"}:
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": message["id"], "result": {}}
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.input_schema,
                    }
                    for tool in tools_for_role(role)
                ]
            },
        }
    if method == "tools/call":
        if not isinstance(params, dict):
            return {"jsonrpc": "2.0", "id": message["id"], "result": error_result("invalid params")}
        payload = {
            "tool": params.get("name"),
            "arguments": params.get("arguments", {}),
            "bridge_role": role,
            "bridge_agent": agent_name,
        }
        response = send_control_request(socket_path, payload)
        text = json.dumps(response, ensure_ascii=True, indent=2)
        result = success_result(text) if response.get("ok", False) else error_result(text)
        return {"jsonrpc": "2.0", "id": message["id"], "result": result}
    return {
        "jsonrpc": "2.0",
        "id": message.get("id"),
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--role", choices=["manager", "worker"], required=True)
    parser.add_argument("--agent", required=True)
    args = parser.parse_args(argv)
    socket_path = Path(args.socket)

    while True:
        message, mode = read_message()
        if message is None or mode is None:
            return 0
        response = handle_request(socket_path, message, role=args.role, agent_name=args.agent)
        if response is not None:
            write_message(response, mode)


if __name__ == "__main__":
    raise SystemExit(main())
