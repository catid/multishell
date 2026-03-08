from __future__ import annotations

import json
import os
import socket
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .config import state_root


LLAMA_CPP_SOURCE_URL_ENV_VAR = "MULTISHELL_LLAMA_CPP_SOURCE_URL"
LLAMA_CPP_ROOT_ENV_VAR = "MULTISHELL_LLAMA_CPP_ROOT"
AUTH_MODEL_CACHE_ROOT_ENV_VAR = "MULTISHELL_AUTH_MODEL_CACHE_ROOT"
AUTH_MODEL_FILE_ENV_VAR = "MULTISHELL_AUTH_MODEL_FILE"
AUTH_MODEL_DOWNLOAD_URL_ENV_VAR = "MULTISHELL_AUTH_MODEL_DOWNLOAD_URL"
AUTH_MODEL_THREADS_ENV_VAR = "MULTISHELL_AUTH_MODEL_THREADS"
AUTH_MODEL_CONTEXT_ENV_VAR = "MULTISHELL_AUTH_MODEL_CONTEXT"
AUTH_MODEL_STARTUP_TIMEOUT_ENV_VAR = "MULTISHELL_AUTH_MODEL_STARTUP_TIMEOUT_SECONDS"
AUTH_MODEL_GPU_LAYERS_ENV_VAR = "MULTISHELL_AUTH_MODEL_GPU_LAYERS"
DEFAULT_LLAMA_CPP_SOURCE_URL = "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/master"
DEFAULT_AUTH_MODEL_FILENAME = "Qwen_Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_AUTH_MODEL_DOWNLOAD_URL = (
    "https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF/resolve/main/Qwen_Qwen3.5-9B-Q4_K_M.gguf"
)


@dataclass(frozen=True)
class AuthModelBenchResult:
    prompt_tokens_per_second: float
    decode_tokens_per_second: float
    prompt_tokens: int
    decode_tokens: int
    runs: int
    model_path: Path


@dataclass(frozen=True)
class LlamaBackend:
    name: str
    uses_gpu: bool
    cmake_args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()


def auth_model_cache_root() -> Path:
    override = os.environ.get(AUTH_MODEL_CACHE_ROOT_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    return state_root() / "cache" / "auth-model"


def install_root() -> Path:
    return auth_model_cache_root()


def _legacy_install_root() -> Path:
    return Path.home() / ".local" / "share" / "multishell"


def llama_cpp_root() -> Path:
    override = os.environ.get(LLAMA_CPP_ROOT_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    return auth_model_cache_root() / "llama.cpp"


def models_root() -> Path:
    return auth_model_cache_root() / "models"


def auth_model_log_dir() -> Path:
    return auth_model_cache_root() / "logs"


def auth_model_server_meta_file() -> Path:
    return auth_model_log_dir() / "llama-server.json"


def auth_model_file() -> Path:
    override = os.environ.get(AUTH_MODEL_FILE_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    return models_root() / DEFAULT_AUTH_MODEL_FILENAME


def install_llama_cpp(*, force: bool = False, log: Callable[[str], None] | None = None) -> dict[str, Path]:
    logger = log or (lambda message: print(message, flush=True))
    source_url = os.environ.get(LLAMA_CPP_SOURCE_URL_ENV_VAR, DEFAULT_LLAMA_CPP_SOURCE_URL).strip() or DEFAULT_LLAMA_CPP_SOURCE_URL
    root = llama_cpp_root()
    src_dir = root / "src"
    build_dir = root / "build"
    desired_backend = _detect_best_backend(logger=logger)

    if not force:
        _migrate_legacy_llama_cpp_cache(logger)

    if not force:
        existing = _installed_binaries(build_dir)
        installed_backend = _installed_backend(build_dir)
        if existing is not None and installed_backend == desired_backend.name:
            logger("llama.cpp already installed")
            logger(f"backend: {desired_backend.name}")
            for name, path in existing.items():
                logger(f"{name}: {path}")
            return existing
        if existing is not None:
            logger(f"llama.cpp backend changed; rebuilding from {installed_backend} to {desired_backend.name}")

    _require_tool("cmake")
    _require_cxx()

    with tempfile.TemporaryDirectory(prefix="multishell-llama-cpp-") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        tarball = tmp_dir / "llama.cpp.tar.gz"
        extract_root = tmp_dir / "extract"

        logger(f"installing llama.cpp from {source_url}")
        extract_root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(source_url, tarball)
        with tarfile.open(tarball, mode="r:gz") as archive:
            archive.extractall(path=extract_root)

        entries = [path for path in extract_root.iterdir() if path.is_dir()]
        if not entries:
            raise RuntimeError("llama.cpp source archive did not contain an extracted directory")
        extracted = entries[0]

        root.mkdir(parents=True, exist_ok=True)
        if src_dir.exists():
            shutil.rmtree(src_dir)
        shutil.copytree(extracted, src_dir)

    logger(f"configuring llama.cpp in {build_dir}")
    configure_env = os.environ.copy()
    for key, value in desired_backend.env:
        configure_env[key] = value
    subprocess.run(
        [
            "cmake",
            "-S",
            str(src_dir),
            "-B",
            str(build_dir),
            "-DBUILD_SHARED_LIBS=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            *desired_backend.cmake_args,
        ],
        check=True,
        env=configure_env,
    )
    logger("building llama.cpp binaries: llama-server, llama-cli, llama-bench")
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "llama-server",
            "llama-cli",
            "llama-bench",
            "--parallel",
            str(_build_jobs()),
        ],
        check=True,
    )

    binaries = {
        "llama-server": _find_binary(build_dir, "llama-server"),
        "llama-cli": _find_binary(build_dir, "llama-cli"),
        "llama-bench": _find_binary(build_dir, "llama-bench"),
    }
    logger("llama.cpp install complete")
    logger(f"backend: {desired_backend.name}")
    for name, path in binaries.items():
        logger(f"{name}: {path}")
    return binaries


def download_auth_model(*, force: bool = False, log: Callable[[str], None] | None = None) -> Path:
    logger = log or (lambda message: print(message, flush=True))
    source_url = os.environ.get(AUTH_MODEL_DOWNLOAD_URL_ENV_VAR, DEFAULT_AUTH_MODEL_DOWNLOAD_URL).strip() or DEFAULT_AUTH_MODEL_DOWNLOAD_URL
    target = auth_model_file()
    if not force:
        _migrate_legacy_auth_model_cache(logger)
    if target.exists() and target.stat().st_size > 0 and not force:
        logger(f"auth model already present: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_suffix(target.suffix + ".part")
    if force:
        target.unlink(missing_ok=True)
        temp_target.unlink(missing_ok=True)

    resume_from = temp_target.stat().st_size if temp_target.exists() else 0
    if resume_from > 0:
        logger(
            f"resuming auth model download from {source_url} at "
            f"{resume_from / (1024**3):.2f} GiB"
        )
    else:
        logger(f"downloading auth model from {source_url}")
    try:
        request = urllib.request.Request(source_url)
        if resume_from > 0:
            request.add_header("Range", f"bytes={resume_from}-")
        with urllib.request.urlopen(request) as response:
            status = int(getattr(response, "status", 200) or 200)
            append_mode = resume_from > 0 and status == 206
            if append_mode:
                total_bytes = _total_bytes_from_headers(response.headers, fallback_completed=resume_from)
                downloaded = resume_from
                handle = temp_target.open("ab")
            else:
                total_bytes = _total_bytes_from_headers(response.headers)
                downloaded = 0
                handle = temp_target.open("wb")
            last_log_at = 0.0
            with handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_log_at >= 2.0:
                        logger(_download_progress_message(downloaded, total_bytes))
                        last_log_at = now
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"failed to download auth model from {source_url}: {reason}") from exc

    temp_target.replace(target)
    logger(f"auth model download complete: {target}")
    return target


def install_auth_model(*, force: bool = False, log: Callable[[str], None] | None = None) -> dict[str, Path]:
    logger = log or (lambda message: print(message, flush=True))
    installed = install_llama_cpp(force=force, log=logger)
    installed["model"] = download_auth_model(force=force, log=logger)
    return installed


def bench_auth_model(*, prompt_tokens: int = 128, decode_tokens: int = 64, runs: int = 3) -> AuthModelBenchResult:
    bench_binary = llama_bench_binary()
    model_path = _require_existing_path(
        auth_model_file(),
        instruction="Run `multishell install-auth-model` first.",
    )
    output = subprocess.check_output(
        [
            str(bench_binary),
            "-m",
            str(model_path),
            "-t",
            str(_server_threads()),
            "-ngl",
            _resolved_gpu_layers(),
            "-p",
            str(prompt_tokens),
            "-n",
            str(decode_tokens),
            "-r",
            str(max(1, runs)),
            "-o",
            "json",
        ],
        text=True,
    )
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"llama-bench did not return valid JSON: {output[:240]!r}") from exc
    if not isinstance(payload, list):
        raise RuntimeError(f"llama-bench returned unexpected output: {payload!r}")

    prompt_sample = next((item for item in payload if isinstance(item, dict) and int(item.get("n_prompt", 0) or 0) > 0), None)
    decode_sample = next((item for item in payload if isinstance(item, dict) and int(item.get("n_gen", 0) or 0) > 0), None)
    if not isinstance(prompt_sample, dict) or not isinstance(decode_sample, dict):
        raise RuntimeError(f"llama-bench output missing prompt/decode samples: {payload!r}")

    return AuthModelBenchResult(
        prompt_tokens_per_second=float(prompt_sample.get("avg_ts") or 0.0),
        decode_tokens_per_second=float(decode_sample.get("avg_ts") or 0.0),
        prompt_tokens=int(prompt_sample.get("n_prompt") or 0),
        decode_tokens=int(decode_sample.get("n_gen") or 0),
        runs=max(1, runs),
        model_path=model_path,
    )


@contextmanager
def managed_auth_model_server(
    *,
    required: bool,
    log: Callable[[str], None] | None = None,
) -> Iterator[bool]:
    logger = log or (lambda message: print(message, flush=True))

    from .auth_flow_model import AUTH_MODEL_API_BASE_ENV_VAR, load_auth_model_settings
    settings = load_auth_model_settings()
    if not settings.enabled:
        if required:
            raise RuntimeError("auth model is disabled; unset MULTISHELL_AUTH_MODEL_ENABLED=0 to use the local auth model")
        logger("auth model is disabled; continuing without local browser AI")
        yield False
        return

    configured_api_base = settings.api_base
    api_base = configured_api_base
    host, port = _api_host_port(api_base)
    had_api_base_override = bool(os.environ.get(AUTH_MODEL_API_BASE_ENV_VAR, "").strip())
    restore_api_base = os.environ.get(AUTH_MODEL_API_BASE_ENV_VAR)
    changed_api_base = False
    desired_threads = _server_threads()
    desired_context = _server_context()
    desired_backend = _detect_best_backend(logger=logger)
    desired_model_path = auth_model_file()
    if _is_loopback_api_base(api_base) and not had_api_base_override and not _probe_auth_model_endpoint(api_base, timeout_seconds=2.0):
        if not _host_port_available(host, port):
            port = _reserve_loopback_port(host)
            api_base = _replace_api_base_port(api_base, host=host, port=port)
            os.environ[AUTH_MODEL_API_BASE_ENV_VAR] = api_base
            changed_api_base = True
            logger(f"default auth model port was busy; using {api_base} for this run")

    if _probe_auth_model_endpoint(api_base, timeout_seconds=2.0):
        if had_api_base_override or not _is_loopback_api_base(api_base) or _managed_server_matches(
            api_base=api_base,
            desired_threads=desired_threads,
            desired_context=desired_context,
            desired_backend=desired_backend.name,
            desired_model_path=desired_model_path,
        ):
            logger(f"auth model server already running at {api_base}")
            try:
                yield True
            finally:
                if changed_api_base:
                    _restore_api_base_env(AUTH_MODEL_API_BASE_ENV_VAR, restore_api_base)
            return
        port = _reserve_loopback_port(host)
        api_base = _replace_api_base_port(configured_api_base, host=host, port=port)
        os.environ[AUTH_MODEL_API_BASE_ENV_VAR] = api_base
        changed_api_base = True
        logger(
            "existing auth model endpoint did not match desired config; "
            f"using {api_base} for this run"
        )

    if not _is_loopback_api_base(api_base):
        message = f"auth model endpoint is unreachable at {api_base}"
        if required:
            raise RuntimeError(message)
        logger(f"{message}; continuing without local browser AI")
        try:
            yield False
        finally:
            if changed_api_base:
                _restore_api_base_env(AUTH_MODEL_API_BASE_ENV_VAR, restore_api_base)
        return

    server_binary, model_path = _ensure_local_auth_model_runtime(logger=logger, required=required)
    if server_binary is None or not model_path.exists():
        message = (
            "local auth model runtime is not installed. "
            "Run `multishell install-auth-model` first."
        )
        if required:
            if changed_api_base:
                _restore_api_base_env(AUTH_MODEL_API_BASE_ENV_VAR, restore_api_base)
            raise RuntimeError(message)
        logger(f"{message} Continuing without local browser AI.")
        try:
            yield False
        finally:
            if changed_api_base:
                _restore_api_base_env(AUTH_MODEL_API_BASE_ENV_VAR, restore_api_base)
        return

    timeout_seconds = _safe_int(os.environ.get(AUTH_MODEL_STARTUP_TIMEOUT_ENV_VAR), default=180, minimum=10, maximum=600)
    log_path = auth_model_log_dir() / "llama-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    installed_backend = _installed_backend(llama_cpp_root() / "build")
    logger(
        "starting local auth model server at "
        f"{api_base} (threads={desired_threads} context={desired_context} backend={installed_backend} gpu_layers={_resolved_gpu_layers()})"
    )
    logger(f"auth model server log: {log_path}")
    command = [
        str(server_binary),
        "--host",
        host,
        "--port",
        str(port),
        "-m",
        str(model_path),
        "-t",
        str(desired_threads),
        "-c",
        str(desired_context),
        "-ngl",
        _resolved_gpu_layers(),
        "--reasoning-budget",
        "0",
        "--reasoning-format",
        "none",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
    ]

    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        _write_server_meta(
            {
                "pid": process.pid,
                "api_base": api_base,
                "threads": desired_threads,
                "context": desired_context,
                "backend": installed_backend,
                "model_path": str(model_path),
            }
        )
        started = False
        try:
            started = _wait_for_auth_model_server(
                process=process,
                api_base=api_base,
                timeout_seconds=timeout_seconds,
                log=logger,
            )
            if not started:
                tail = _tail_log(log_path)
                message = (
                    f"local auth model server did not become ready within {timeout_seconds} seconds. "
                    f"Recent log output:\n{tail}"
                )
                if required:
                    raise RuntimeError(message)
                logger(f"{message}\ncontinuing without local browser AI")
                yield False
                return
            yield True
        finally:
            if process.poll() is None:
                logger("stopping local auth model server")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            _clear_server_meta(process.pid)
            if changed_api_base:
                _restore_api_base_env(AUTH_MODEL_API_BASE_ENV_VAR, restore_api_base)


def _ensure_local_auth_model_runtime(*, logger: Callable[[str], None], required: bool) -> tuple[Path | None, Path]:
    server_binary = llama_server_binary(required=False)
    model_path = auth_model_file()
    desired_backend = _detect_best_backend(logger=logger)
    missing_server = server_binary is None
    missing_model = not model_path.exists()
    backend_mismatch = False
    if not missing_server:
        backend_mismatch = _installed_backend(llama_cpp_root() / "build") != desired_backend.name
    if not missing_server and not missing_model and not backend_mismatch:
        return server_binary, model_path

    try:
        if missing_server or backend_mismatch:
            if backend_mismatch:
                logger(
                    "local auth model runtime backend mismatch; attempting to rebuild "
                    f"for {desired_backend.name}"
                )
            else:
                logger("local auth model runtime missing; attempting to build llama.cpp")
            install_llama_cpp(force=False, log=logger)
        if missing_model:
            logger("local auth model file missing; attempting to download the GGUF")
            download_auth_model(force=False, log=logger)
    except Exception as exc:
        if required:
            raise
        logger(f"local auth model auto-repair failed: {exc}")

    return llama_server_binary(required=False), auth_model_file()


def llama_server_binary(*, required: bool = True) -> Path | None:
    return _binary_path("llama-server", required=required)


def llama_cli_binary(*, required: bool = True) -> Path | None:
    return _binary_path("llama-cli", required=required)


def llama_bench_binary(*, required: bool = True) -> Path | None:
    return _binary_path("llama-bench", required=required)


def _binary_path(name: str, *, required: bool) -> Path | None:
    build_dir = llama_cpp_root() / "build"
    try:
        return _find_binary(build_dir, name)
    except RuntimeError:
        if required:
            raise RuntimeError(f"{name} is not installed. Run `multishell install-auth-model` first.")
        return None


def _probe_auth_model_endpoint(api_base: str, *, timeout_seconds: float) -> bool:
    request = urllib.request.Request(f"{api_base.rstrip('/')}/models", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if not 200 <= getattr(response, "status", 200) < 300:
                return False
            payload = json.load(response)
            if isinstance(payload, dict):
                data = payload.get("data")
                return isinstance(data, list)
            return False
    except Exception:
        return False


def _wait_for_auth_model_server(
    *,
    process: subprocess.Popen[str],
    api_base: str,
    timeout_seconds: int,
    log: Callable[[str], None],
) -> bool:
    deadline = time.time() + max(1, timeout_seconds)
    last_log_at = 0.0
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        if _probe_auth_model_endpoint(api_base, timeout_seconds=2.0):
            log(f"local auth model server ready at {api_base}")
            return True
        now = time.time()
        if now - last_log_at >= 5.0:
            log(f"waiting for local auth model server to finish loading at {api_base}")
            last_log_at = now
        time.sleep(0.5)
    return False


def _is_loopback_api_base(api_base: str) -> bool:
    parsed = urlparse(api_base)
    host = (parsed.hostname or "").strip().lower()
    return parsed.scheme == "http" and host in {"127.0.0.1", "localhost", "::1"}


def _api_host_port(api_base: str) -> tuple[str, int]:
    parsed = urlparse(api_base)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8080
    return host, port


def _host_port_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _reserve_loopback_port(host: str) -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _replace_api_base_port(api_base: str, *, host: str, port: int) -> str:
    parsed = urlparse(api_base)
    netloc = f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"
    return parsed._replace(netloc=netloc).geturl()


def _restore_api_base_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _download_progress_message(downloaded: int, total_bytes: int | None) -> str:
    if total_bytes and total_bytes > 0:
        percent = min(100.0, downloaded * 100.0 / total_bytes)
        return (
            f"downloading auth model: {downloaded / (1024**3):.2f} GiB / "
            f"{total_bytes / (1024**3):.2f} GiB ({percent:.1f}%)"
        )
    return f"downloading auth model: {downloaded / (1024**3):.2f} GiB"


def _total_bytes_from_headers(headers: object, *, fallback_completed: int = 0) -> int | None:
    if headers is None:
        return None
    try:
        content_range = headers.get("Content-Range")
    except Exception:
        content_range = None
    if content_range and "/" in str(content_range):
        total_part = str(content_range).rsplit("/", 1)[-1].strip()
        total = _coerce_int(total_part)
        if total is not None and total > 0:
            return total
    try:
        content_length = headers.get("Content-Length")
    except Exception:
        content_length = None
    length = _coerce_int(content_length)
    if length is None:
        return None
    if fallback_completed > 0:
        return fallback_completed + length
    return length


def _tail_log(path: Path, *, max_lines: int = 20) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "<log unavailable>"
    tail = lines[-max_lines:]
    return "\n".join(tail) if tail else "<log empty>"


def _migrate_legacy_llama_cpp_cache(logger: Callable[[str], None]) -> None:
    if os.environ.get(LLAMA_CPP_ROOT_ENV_VAR, "").strip() or os.environ.get(AUTH_MODEL_CACHE_ROOT_ENV_VAR, "").strip():
        return
    target = llama_cpp_root()
    source = _legacy_install_root() / "llama.cpp"
    if target.exists() or not source.exists() or target == source:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    logger(f"migrated llama.cpp cache from {source} to {target}")


def _migrate_legacy_auth_model_cache(logger: Callable[[str], None]) -> None:
    if os.environ.get(AUTH_MODEL_FILE_ENV_VAR, "").strip() or os.environ.get(AUTH_MODEL_CACHE_ROOT_ENV_VAR, "").strip():
        return
    target = auth_model_file()
    source = _legacy_install_root() / "models" / target.name
    if target.exists() or not source.exists() or target == source:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    logger(f"migrated auth model cache from {source} to {target}")


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"required build tool is missing: {name}")


def _require_cxx() -> None:
    if shutil.which("c++") is None and shutil.which("g++") is None and shutil.which("clang++") is None:
        raise RuntimeError("required C++ compiler is missing: install g++ or clang++")


def _build_jobs() -> int:
    try:
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1
    return max(1, min(16, cpu_count))


def _server_threads() -> int:
    try:
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1
    return _safe_int(os.environ.get(AUTH_MODEL_THREADS_ENV_VAR), default=max(1, cpu_count), minimum=1, maximum=512)


def _server_context() -> int:
    return _safe_int(os.environ.get(AUTH_MODEL_CONTEXT_ENV_VAR), default=16384, minimum=512, maximum=32768)


def _resolved_gpu_layers() -> str:
    backend = _installed_backend(llama_cpp_root() / "build")
    default = "999" if backend in {"cuda", "hip"} else "0"
    raw = os.environ.get(AUTH_MODEL_GPU_LAYERS_ENV_VAR)
    text = str(raw).strip() if raw is not None else default
    if not text:
        return default
    lowered = text.lower()
    if lowered in {"all", "auto"}:
        return default
    return text


def _find_binary(build_dir: Path, name: str) -> Path:
    candidates = [
        build_dir / "bin" / name,
        build_dir / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(f"built llama.cpp binary not found: {name}")


def _installed_binaries(build_dir: Path) -> dict[str, Path] | None:
    binaries: dict[str, Path] = {}
    for name in ("llama-server", "llama-cli", "llama-bench"):
        try:
            binaries[name] = _find_binary(build_dir, name)
        except RuntimeError:
            return None
    return binaries


def _require_existing_path(path: Path, *, instruction: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"required file is missing: {path}. {instruction}")
    return path


def _coerce_int(raw: str | None) -> int | None:
    try:
        return int(str(raw).strip())
    except Exception:
        return None


def _safe_int(raw: str | None, *, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(raw).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _detect_best_backend(*, logger: Callable[[str], None] | None = None) -> LlamaBackend:
    log = logger or (lambda _message: None)
    if _cuda_available():
        log("detected NVIDIA GPU and CUDA toolkit; preferring CUDA backend for llama.cpp")
        return LlamaBackend(name="cuda", uses_gpu=True, cmake_args=("-DGGML_CUDA=ON",))
    hip_env = _hip_build_env()
    if hip_env is not None:
        log("detected AMD ROCm toolchain; preferring HIP backend for llama.cpp")
        return LlamaBackend(name="hip", uses_gpu=True, cmake_args=("-DGGML_HIP=ON",), env=tuple(sorted(hip_env.items())))
    log("no usable CUDA or ROCm toolchain detected; using CPU backend for llama.cpp")
    return LlamaBackend(name="cpu", uses_gpu=False)


def _cuda_available() -> bool:
    if shutil.which("nvidia-smi") is None or shutil.which("nvcc") is None:
        return False
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT, timeout=10)
    except Exception:
        return False
    return "GPU " in output


def _hip_build_env() -> dict[str, str] | None:
    if shutil.which("hipconfig") is None:
        return None
    try:
        hip_root = subprocess.check_output(["hipconfig", "-R"], text=True, stderr=subprocess.STDOUT, timeout=10).strip()
        compiler_root = subprocess.check_output(["hipconfig", "-l"], text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except Exception:
        return None
    clang_path = Path(compiler_root) / "clang"
    if not hip_root or not compiler_root or not clang_path.exists():
        return None
    return {
        "HIPCXX": str(clang_path),
        "HIP_PATH": hip_root,
    }


def _installed_backend(build_dir: Path) -> str:
    cache = build_dir / "CMakeCache.txt"
    try:
        text = cache.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "cpu"
    if "GGML_CUDA:BOOL=ON" in text:
        return "cuda"
    if "GGML_HIP:BOOL=ON" in text:
        return "hip"
    return "cpu"


def _managed_server_matches(
    *,
    api_base: str,
    desired_threads: int,
    desired_context: int,
    desired_backend: str,
    desired_model_path: Path,
) -> bool:
    payload = _read_server_meta()
    if not payload:
        return False
    try:
        pid = int(payload.get("pid") or 0)
    except Exception:
        return False
    return (
        pid > 0
        and _pid_alive(pid)
        and str(payload.get("api_base") or "") == api_base
        and int(payload.get("threads") or 0) == desired_threads
        and int(payload.get("context") or 0) == desired_context
        and str(payload.get("backend") or "") == desired_backend
        and str(payload.get("model_path") or "") == str(desired_model_path)
    )


def _read_server_meta() -> dict[str, object] | None:
    path = auth_model_server_meta_file()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_server_meta(payload: dict[str, object]) -> None:
    path = auth_model_server_meta_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _clear_server_meta(pid: int) -> None:
    path = auth_model_server_meta_file()
    payload = _read_server_meta()
    if not payload:
        return
    try:
        recorded_pid = int(payload.get("pid") or 0)
    except Exception:
        recorded_pid = 0
    if recorded_pid == pid:
        path.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
