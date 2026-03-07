from __future__ import annotations

import errno
import json
import os
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path


class ControlHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            try:
                request = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                response = {"ok": False, "error": f"invalid control payload: {exc}"}
            else:
                try:
                    response = self.server.controller.handle_control_request(request)  # type: ignore[attr-defined]
                except Exception as exc:  # pragma: no cover - defensive
                    response = {"ok": False, "error": f"control handler crashed: {exc}"}
            try:
                self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))
                self.wfile.flush()
            except OSError as exc:
                if _is_disconnect_error(exc):
                    return
                raise


class ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request: object, client_address: object) -> None:  # pragma: no cover - defensive
        _, exc, _ = sys.exc_info()
        if isinstance(exc, OSError) and _is_disconnect_error(exc):
            return
        super().handle_error(request, client_address)


class ControlServer:
    def __init__(self, socket_path: Path, controller: object) -> None:
        self.socket_path = socket_path
        self.controller = controller
        self._server: ThreadedUnixServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.socket_path.exists():
            if _socket_is_live(self.socket_path):
                raise RuntimeError(f"control socket already in use: {self.socket_path}")
            os.unlink(self.socket_path)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._server = ThreadedUnixServer(str(self.socket_path), ControlHandler)
        self._server.controller = self.controller  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self.socket_path.exists():
            os.unlink(self.socket_path)
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def send_control_request(socket_path: Path, payload: dict[str, object]) -> dict[str, object]:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5)
            sock.connect(str(socket_path))
            sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            received = b""
            while not received.endswith(b"\n"):
                chunk = sock.recv(65536)
                if not chunk:
                    break
                received += chunk
    except FileNotFoundError:
        return {"ok": False, "error": f"control socket not found: {socket_path}"}
    except OSError as exc:
        return {"ok": False, "error": f"control socket error: {exc}"}

    if not received:
        return {"ok": False, "error": "empty response"}
    try:
        return json.loads(received.decode("utf-8"))
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"invalid control response: {exc}"}


def format_timestamp(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _is_disconnect_error(exc: OSError) -> bool:
    return exc.errno in {
        errno.EPIPE,
        errno.ECONNRESET,
        errno.ENOTCONN,
        errno.EBADF,
    }


def _socket_is_live(socket_path: Path) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            sock.connect(str(socket_path))
        return True
    except OSError:
        return False
