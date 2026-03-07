import socket
from pathlib import Path

import pytest

from multishell.control import ControlServer, send_control_request


def test_send_control_request_empty_socket(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sock"
    response = send_control_request(missing, {"tool": "noop"})
    assert response["ok"] is False
    assert "control socket not found" in str(response["error"])


def test_control_server_refuses_live_socket(tmp_path: Path) -> None:
    socket_path = tmp_path / "control.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)
    try:
        server = ControlServer(socket_path, controller=object())
        with pytest.raises(RuntimeError, match="already in use"):
            server.start()
    finally:
        listener.close()
