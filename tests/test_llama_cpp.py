from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from multishell.llama_cpp import (
    LlamaBackend,
    auth_model_cache_root,
    auth_model_log_dir,
    bench_auth_model,
    download_auth_model,
    install_auth_model,
    install_llama_cpp,
    managed_auth_model_server,
    _managed_server_matches,
    _probe_auth_model_endpoint,
    _resolved_gpu_layers,
)


def test_install_auth_model_runs_build_and_download(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    called: dict[str, bool] = {"build": False, "download": False}

    monkeypatch.setattr(
        "multishell.llama_cpp.install_llama_cpp",
        lambda **_kwargs: called.__setitem__("build", True) or {"llama-server": tmp_path / "llama-server"},
    )
    monkeypatch.setattr(
        "multishell.llama_cpp.download_auth_model",
        lambda **_kwargs: called.__setitem__("download", True) or model_path,
    )

    installed = install_auth_model()

    assert called == {"build": True, "download": True}
    assert installed["model"] == model_path


def test_install_llama_cpp_reuses_existing_binaries(monkeypatch, tmp_path: Path) -> None:
    build_dir = tmp_path / "llama.cpp" / "build" / "bin"
    build_dir.mkdir(parents=True)
    for name in ("llama-server", "llama-cli", "llama-bench"):
        (build_dir / name).write_text("x", encoding="utf-8")
    (tmp_path / "llama.cpp" / "build" / "CMakeCache.txt").write_text("GGML_CUDA:BOOL=ON\n", encoding="utf-8")

    monkeypatch.setattr("multishell.llama_cpp.llama_cpp_root", lambda: tmp_path / "llama.cpp")
    monkeypatch.setattr(
        "multishell.llama_cpp._detect_best_backend",
        lambda logger=None: LlamaBackend(name="cuda", uses_gpu=True, cmake_args=("-DGGML_CUDA=ON",)),
    )

    seen: dict[str, bool] = {"cmake": False}

    def fake_run(*_args, **_kwargs):
        seen["cmake"] = True
        raise AssertionError("cmake should not run when binaries already exist")

    monkeypatch.setattr("multishell.llama_cpp.subprocess.run", fake_run)

    installed = install_llama_cpp()

    assert seen["cmake"] is False
    assert installed["llama-server"] == build_dir / "llama-server"


def test_download_auth_model_reuses_existing_target(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr("multishell.llama_cpp.auth_model_file", lambda: model_path)

    downloaded = download_auth_model()

    assert downloaded == model_path


def test_auth_model_paths_use_state_root_cache(monkeypatch, tmp_path: Path) -> None:
    state = tmp_path / ".multishell"
    monkeypatch.setenv("MULTISHELL_STATE_ROOT", str(state))

    assert auth_model_cache_root() == state / "cache" / "auth-model"
    assert auth_model_log_dir() == state / "cache" / "auth-model" / "logs"


def test_install_llama_cpp_migrates_legacy_cache(monkeypatch, tmp_path: Path) -> None:
    state = tmp_path / ".multishell"
    legacy_home = tmp_path / "legacy-home"
    legacy_build_dir = legacy_home / ".local" / "share" / "multishell" / "llama.cpp" / "build" / "bin"
    legacy_build_dir.mkdir(parents=True)
    for name in ("llama-server", "llama-cli", "llama-bench"):
        (legacy_build_dir / name).write_text("x", encoding="utf-8")
    (legacy_home / ".local" / "share" / "multishell" / "llama.cpp" / "build" / "CMakeCache.txt").write_text(
        "GGML_CUDA:BOOL=ON\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("MULTISHELL_STATE_ROOT", str(state))
    monkeypatch.setattr("pathlib.Path.home", lambda: legacy_home)
    monkeypatch.setattr(
        "multishell.llama_cpp._detect_best_backend",
        lambda logger=None: LlamaBackend(name="cuda", uses_gpu=True, cmake_args=("-DGGML_CUDA=ON",)),
    )

    installed = install_llama_cpp()

    assert installed["llama-server"] == state / "cache" / "auth-model" / "llama.cpp" / "build" / "bin" / "llama-server"
    assert not (legacy_home / ".local" / "share" / "multishell" / "llama.cpp").exists()


def test_download_auth_model_migrates_legacy_cache(monkeypatch, tmp_path: Path) -> None:
    state = tmp_path / ".multishell"
    legacy_home = tmp_path / "legacy-home"
    legacy_model = legacy_home / ".local" / "share" / "multishell" / "models" / "Qwen_Qwen3.5-9B-Q4_K_M.gguf"
    legacy_model.parent.mkdir(parents=True, exist_ok=True)
    legacy_model.write_text("cached-model", encoding="utf-8")

    monkeypatch.setenv("MULTISHELL_STATE_ROOT", str(state))
    monkeypatch.setattr("pathlib.Path.home", lambda: legacy_home)

    downloaded = download_auth_model()

    assert downloaded == state / "cache" / "auth-model" / "models" / "Qwen_Qwen3.5-9B-Q4_K_M.gguf"
    assert downloaded.read_text(encoding="utf-8") == "cached-model"
    assert not legacy_model.exists()


def test_bench_auth_model_parses_llama_bench_json(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_text("x", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr("multishell.llama_cpp.auth_model_file", lambda: model_path)
    monkeypatch.setattr("multishell.llama_cpp.llama_bench_binary", lambda: tmp_path / "llama-bench")
    monkeypatch.setattr("multishell.llama_cpp._installed_backend", lambda _build_dir: "cuda")
    monkeypatch.setattr(
        "multishell.llama_cpp.subprocess.check_output",
        lambda args, **_kwargs: captured.__setitem__("args", args) or """
[
  {"n_prompt": 128, "n_gen": 0, "avg_ts": 148.7},
  {"n_prompt": 0, "n_gen": 64, "avg_ts": 11.2}
]
""",
    )

    result = bench_auth_model()

    assert result.model_path == model_path
    assert result.prompt_tokens_per_second == 148.7
    assert result.decode_tokens_per_second == 11.2
    assert result.prompt_tokens == 128
    assert result.decode_tokens == 64
    assert "-ngl" in captured["args"]
    assert captured["args"][captured["args"].index("-ngl") + 1] == "999"


def test_resolved_gpu_layers_normalizes_all_to_numeric(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("multishell.llama_cpp.llama_cpp_root", lambda: tmp_path / "llama.cpp")
    monkeypatch.setattr("multishell.llama_cpp._installed_backend", lambda _build_dir: "cuda")
    monkeypatch.setenv("MULTISHELL_AUTH_MODEL_GPU_LAYERS", "all")

    assert _resolved_gpu_layers() == "999"


class _FakeDownloadResponse:
    def __init__(self, chunks: list[bytes], *, status: int, headers: dict[str, str]) -> None:
        self._chunks = list(chunks)
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, _size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class _FakeModelsResponse:
    def __init__(self, body: str, *, status: int = 200) -> None:
        self._body = body.encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, _size: int = -1) -> bytes:
        return self._body


def test_download_auth_model_resumes_partial_file(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    part_path = model_path.with_suffix(".gguf.part")
    part_path.write_bytes(b"hello")
    seen: dict[str, object] = {"range": None}

    def fake_urlopen(request):
        seen["range"] = request.headers.get("Range")
        return _FakeDownloadResponse(
            [b" ", b"world"],
            status=206,
            headers={"Content-Range": "bytes 5-10/11", "Content-Length": "6"},
        )

    monkeypatch.setattr("multishell.llama_cpp.auth_model_file", lambda: model_path)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    downloaded = download_auth_model()

    assert seen["range"] == "bytes=5-"
    assert downloaded == model_path
    assert model_path.read_bytes() == b"hello world"
    assert not part_path.exists()


def test_probe_auth_model_endpoint_rejects_non_json_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _FakeModelsResponse("<html>not the auth model</html>"),
    )

    assert _probe_auth_model_endpoint("http://127.0.0.1:8080/v1", timeout_seconds=2.0) is False


def test_managed_server_matches_requires_backend_and_model(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    monkeypatch.setattr(
        "multishell.llama_cpp._read_server_meta",
        lambda: {
            "pid": 1234,
            "api_base": "http://127.0.0.1:8080/v1",
            "threads": 32,
            "context": 16384,
            "backend": "cpu",
            "model_path": str(model_path),
        },
    )
    monkeypatch.setattr("multishell.llama_cpp._pid_alive", lambda pid: pid == 1234)

    assert _managed_server_matches(
        api_base="http://127.0.0.1:8080/v1",
        desired_threads=32,
        desired_context=16384,
        desired_backend="cuda",
        desired_model_path=model_path,
    ) is False


def test_managed_auth_model_server_uses_existing_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "multishell.auth_flow_model.load_auth_model_settings",
        lambda: SimpleNamespace(enabled=True, api_base="http://127.0.0.1:8080/v1"),
    )
    monkeypatch.setattr("multishell.llama_cpp._probe_auth_model_endpoint", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("multishell.llama_cpp._managed_server_matches", lambda **_kwargs: True)

    with managed_auth_model_server(required=True) as ready:
        assert ready is True


def test_managed_auth_model_server_can_fallback_when_runtime_missing(monkeypatch) -> None:
    logs: list[str] = []
    monkeypatch.setattr(
        "multishell.auth_flow_model.load_auth_model_settings",
        lambda: SimpleNamespace(enabled=True, api_base="http://127.0.0.1:8080/v1"),
    )
    monkeypatch.setattr("multishell.llama_cpp._probe_auth_model_endpoint", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("multishell.llama_cpp._is_loopback_api_base", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("multishell.llama_cpp.llama_server_binary", lambda **_kwargs: None)
    monkeypatch.setattr("multishell.llama_cpp.install_llama_cpp", lambda **_kwargs: {})
    monkeypatch.setattr("multishell.llama_cpp.download_auth_model", lambda **_kwargs: Path("/tmp/missing-model.gguf"))

    with managed_auth_model_server(required=False, log=logs.append) as ready:
        assert ready is False
    assert any("attempting to build llama.cpp" in line for line in logs)


def test_managed_auth_model_server_can_auto_repair_missing_runtime(monkeypatch, tmp_path: Path) -> None:
    logs: list[str] = []
    model_path = tmp_path / "model.gguf"
    model_path.write_text("x", encoding="utf-8")
    state = {"installed": False}
    captured: dict[str, object] = {}

    class _FakeProcess:
        pid = 12345

        def poll(self):
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: int | None = None) -> int:
            return 0

    monkeypatch.setattr(
        "multishell.auth_flow_model.load_auth_model_settings",
        lambda: SimpleNamespace(enabled=True, api_base="http://127.0.0.1:8080/v1"),
    )
    monkeypatch.setattr("multishell.llama_cpp.os.cpu_count", lambda: 96)
    monkeypatch.setattr(
        "multishell.llama_cpp._detect_best_backend",
        lambda logger=None: LlamaBackend(name="cuda", uses_gpu=True, cmake_args=("-DGGML_CUDA=ON",)),
    )
    monkeypatch.setattr("multishell.llama_cpp._probe_auth_model_endpoint", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("multishell.llama_cpp._is_loopback_api_base", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("multishell.llama_cpp._installed_backend", lambda _build_dir: "cuda")
    monkeypatch.setattr(
        "multishell.llama_cpp.llama_server_binary",
        lambda **_kwargs: (tmp_path / "llama-server") if state["installed"] else None,
    )
    monkeypatch.setattr("multishell.llama_cpp.auth_model_file", lambda: model_path)
    monkeypatch.setattr(
        "multishell.llama_cpp.install_llama_cpp",
        lambda **_kwargs: state.__setitem__("installed", True) or {"llama-server": tmp_path / "llama-server"},
    )
    monkeypatch.setattr(
        "multishell.llama_cpp.subprocess.Popen",
        lambda args, **_kwargs: captured.__setitem__("args", args) or _FakeProcess(),
    )
    monkeypatch.setattr("multishell.llama_cpp._wait_for_auth_model_server", lambda **_kwargs: True)

    with managed_auth_model_server(required=False, log=logs.append) as ready:
        assert ready is True
    assert any("attempting to build llama.cpp" in line for line in logs)
    assert "-c" in captured["args"]
    assert captured["args"][captured["args"].index("-c") + 1] == "16384"
    assert "-t" in captured["args"]
    assert captured["args"][captured["args"].index("-t") + 1] == "96"
    assert "-ngl" in captured["args"]
    assert captured["args"][captured["args"].index("-ngl") + 1] == "999"
    assert any("threads=96 context=16384 backend=cuda gpu_layers=999" in line for line in logs)


def test_managed_auth_model_server_uses_fallback_port_when_default_is_busy(monkeypatch, tmp_path: Path) -> None:
    logs: list[str] = []
    model_path = tmp_path / "model.gguf"
    model_path.write_text("x", encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeProcess:
        pid = 23456

        def poll(self):
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: int | None = None) -> int:
            return 0

    monkeypatch.delenv("MULTISHELL_AUTH_MODEL_API_BASE", raising=False)
    monkeypatch.setattr(
        "multishell.auth_flow_model.load_auth_model_settings",
        lambda: SimpleNamespace(enabled=True, api_base="http://127.0.0.1:8080/v1"),
    )
    monkeypatch.setattr("multishell.llama_cpp._managed_server_matches", lambda **_kwargs: False)
    monkeypatch.setattr("multishell.llama_cpp._probe_auth_model_endpoint", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("multishell.llama_cpp._host_port_available", lambda host, port: port != 8080)
    monkeypatch.setattr("multishell.llama_cpp._reserve_loopback_port", lambda _host: 18080)
    monkeypatch.setattr("multishell.llama_cpp.llama_server_binary", lambda **_kwargs: tmp_path / "llama-server")
    monkeypatch.setattr("multishell.llama_cpp.auth_model_file", lambda: model_path)
    monkeypatch.setattr("multishell.llama_cpp.subprocess.Popen", lambda *_args, **_kwargs: _FakeProcess())

    def fake_wait_for_auth_model_server(*, process, api_base, timeout_seconds, log):
        captured["api_base"] = api_base
        return True

    monkeypatch.setattr("multishell.llama_cpp._wait_for_auth_model_server", fake_wait_for_auth_model_server)

    with managed_auth_model_server(required=True, log=logs.append) as ready:
        assert ready is True
        assert os.environ["MULTISHELL_AUTH_MODEL_API_BASE"] == "http://127.0.0.1:18080/v1"

    assert captured["api_base"] == "http://127.0.0.1:18080/v1"
    assert "MULTISHELL_AUTH_MODEL_API_BASE" not in os.environ
    assert any("default auth model port was busy; using http://127.0.0.1:18080/v1" in line for line in logs)
