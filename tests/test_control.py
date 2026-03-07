import json
import socket
import struct
import time
from pathlib import Path

import pytest

from multishell.control import ControlServer, send_control_request


class _Controller:
    def handle_control_request(self, request: dict[str, object]) -> dict[str, object]:
        return {"ok": True, "echo": request}


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


def test_control_server_ignores_broken_pipe_on_client_disconnect(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    socket_path = tmp_path / "control.sock"
    server = ControlServer(socket_path, controller=_Controller())
    server.start()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            linger = struct.pack("ii", 1, 0)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
            sock.connect(str(socket_path))
            sock.sendall((json.dumps({"tool": "ping"}) + "\n").encode("utf-8"))
            sock.shutdown(socket.SHUT_RDWR)
        time.sleep(0.1)
        captured = capsys.readouterr()
        assert "Broken pipe" not in captured.err
        assert "Traceback" not in captured.err
    finally:
        server.stop()
