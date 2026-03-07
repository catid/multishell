from __future__ import annotations

from pathlib import Path

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
