from __future__ import annotations

from pathlib import Path

from multishell.config import SPARK_MODEL
from multishell.mcp_bridge import handle_request


def test_initialize_echoes_requested_protocol_version() -> None:
    response = handle_request(
        Path("/tmp/control.sock"),
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
    )

    assert response is not None
    assert response["result"]["protocolVersion"] == "2025-06-18"


def test_ping_returns_empty_result() -> None:
    response = handle_request(Path("/tmp/control.sock"), {"jsonrpc": "2.0", "id": 2, "method": "ping"})

    assert response == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_worker_tools_list_reports_current_spark_model_name() -> None:
    response = handle_request(
        Path("/tmp/control.sock"),
        {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
        role="worker",
        agent_name="worker-1",
    )

    assert response is not None
    tools = response["result"]["tools"]
    spark_tool = next(tool for tool in tools if tool["name"] == "gpt_5_3_spark")
    assert SPARK_MODEL in spark_tool["description"]
