from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pexpect

from .auth_flow_model import (
    AuthModelFailure,
    AuthModelProtocolError,
    AuthModelUnavailable,
    capture_auth_snapshot,
    drive_google_auth_with_model,
    snapshot_requires_model,
)
from .config import AgentSpec, account_for_agent, all_agent_specs, credential_source_agent, state_root
from .homes import (
    agent_home,
    auth_path,
    claude_auth_path,
    claude_home,
    claude_root_auth_path,
    ensure_agent_home,
    ensure_claude_home,
    has_claude_auth,
)
from .runtime import apply_node_warning_suppression, apply_playwright_browser_path, child_env, suppress_node_warnings


DEVICE_URL = "https://auth.openai.com/codex/device"
GOOGLE_EMAIL_SELECTORS = [
    "input[type='email']:visible",
    "input[name='identifier']:visible",
    "input[autocomplete='username']:visible",
]
GOOGLE_PASSWORD_SELECTORS = [
    "input[type='password']:visible",
    "input[name='Passwd']:visible",
    "input[autocomplete='current-password']:visible",
]
GOOGLE_EMAIL_NEXT_SELECTORS = ["#identifierNext", "button:has-text('Next')"]
GOOGLE_PASSWORD_NEXT_SELECTORS = ["#passwordNext", "button:has-text('Next')"]
GOOGLE_CODE_SELECTORS = ["input[name*=code]", "input[autocomplete='one-time-code']", "input[inputmode='numeric']"]
GOOGLE_ERROR_SNIPPETS = (
    "Couldn’t find your Google Account",
    "Couldn't find your Google Account",
    "Enter a valid email or phone number",
    "Enter an email or phone number",
    "Wrong password",
    "This browser or app may not be secure",
    "Couldn’t sign you in",
    "Couldn't sign you in",
    "2-Step Verification",
    "Verify it’s you",
    "Verify it's you",
)
GOOGLE_MANUAL_CHALLENGE_SNIPPETS = {
    "2-Step Verification",
    "Verify it’s you",
    "Verify it's you",
}
_LOG_LOCK = threading.RLock()
LOGIN_ERROR_RETRY_ATTEMPTS = 10
LOGIN_ERROR_RETRY_DELAY_SECONDS = 15
_AUTO_LOGIN_VERBOSE = False


class CodexDeviceAuthRateLimit(RuntimeError):
    pass


class RetryableLoginError(RuntimeError):
    pass


class NoAuthLoginError(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentCredentials:
    spec: AgentSpec
    password: str


def _log_progress(agent_name: str, message: str) -> None:
    with _LOG_LOCK:
        print(f"[{agent_name}] {message}", flush=True)


def _auto_login_verbose() -> bool:
    return _AUTO_LOGIN_VERBOSE


def _verbose_progress_label(agent_name: str) -> str | None:
    return agent_name if _auto_login_verbose() else None


def _log_verbose(agent_name: str, message: str) -> None:
    if _auto_login_verbose():
        _log_progress(agent_name, message)


def _log_coarse_progress(agent_name: str, message: str) -> None:
    if not _auto_login_verbose():
        _log_progress(agent_name, message)


def _log_stage_once(agent_name: str | None, seen: set[str], stage: str, message: str) -> None:
    if agent_name is None or stage in seen:
        return
    seen.add(stage)
    _log_progress(agent_name, message)


def _periodic_progress(agent_name: str, last_logged_at: float, message: str, *, interval_seconds: float = 10.0) -> float:
    now = time.time()
    if now - last_logged_at >= interval_seconds:
        _log_progress(agent_name, message)
        return now
    return last_logged_at


def _auth_model_trace_logger(agent_name: str, flow_name: str, progress_label: str | None) -> tuple[Callable[[str], None], Path]:
    debug_dir = state_root() / "debug" / agent_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    slug = re.sub(r"[^a-z0-9]+", "-", flow_name.lower()).strip("-") or "auth-model"
    trace_path = debug_dir / f"{timestamp}-{slug}.auth-model.log"
    with _LOG_LOCK:
        trace_path.write_text(f"agent={agent_name}\nflow={flow_name}\n", encoding="utf-8")

    if progress_label is not None:
        _log_progress(progress_label, f"auth model trace: {trace_path}")

    def logger(message: str) -> None:
        with _LOG_LOCK:
            if progress_label is not None:
                _log_progress(progress_label, message)
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")

    return logger, trace_path


def specs_by_name() -> dict[str, AgentSpec]:
    return {spec.name: spec for spec in all_agent_specs()}


def resolve_credentials(agent_names: list[str]) -> list[AgentCredentials]:
    by_name = specs_by_name()
    credentials: list[AgentCredentials] = []
    missing: list[str] = []

    for agent_name in agent_names:
        spec = by_name[agent_name]
        try:
            password = account_for_agent(agent_name).password
        except KeyError:
            password = ""
        if not password:
            missing.append(agent_name)
            continue
        credentials.append(AgentCredentials(spec=spec, password=password))

    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"missing passwords for: {missing_text}")
    return credentials


def run_auto_login(
    credentials: list[AgentCredentials],
    headed: bool = False,
    timeout_seconds: int = 180,
    *,
    max_parallel: int = 1,
    verbose: bool = False,
) -> None:
    global _AUTO_LOGIN_VERBOSE
    apply_node_warning_suppression()
    apply_playwright_browser_path()
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run `python3 -m pip install playwright pexpect` "
            "and `python3 -m playwright install chromium` first."
        ) from exc

    if max_parallel < 1:
        raise RuntimeError("max_parallel must be at least 1")

    total = len(credentials)
    failures: list[tuple[int, str, str, str]] = []
    max_workers = min(max_parallel, total) if total else 0
    engine_limits = {engine: _engine_parallel_limit(engine, max_parallel) for engine in {credential.spec.engine for credential in credentials}}

    previous_verbose = _AUTO_LOGIN_VERBOSE
    _AUTO_LOGIN_VERBOSE = bool(verbose)
    try:
        def run_one(credential: AgentCredentials) -> None:
            for attempt in range(1, LOGIN_ERROR_RETRY_ATTEMPTS + 1):
                _log_progress(
                    credential.spec.name,
                    f"starting login round {attempt}/{LOGIN_ERROR_RETRY_ATTEMPTS}",
                )
                try:
                    with sync_playwright() as playwright:
                        if credential.spec.role == "claude-worker":
                            _login_claude_one(playwright, credential, timeout_seconds, headed=headed)
                        else:
                            _login_codex_one(playwright, credential, timeout_seconds, headed=headed)
                    return
                except NoAuthLoginError:
                    raise
                except RetryableLoginError as exc:
                    if attempt >= LOGIN_ERROR_RETRY_ATTEMPTS:
                        raise RetryableLoginError(str(exc)) from exc
                    _log_progress(
                        credential.spec.name,
                        f"retryable login error; retrying in {LOGIN_ERROR_RETRY_DELAY_SECONDS}s "
                        f"(attempt {attempt + 1}/{LOGIN_ERROR_RETRY_ATTEMPTS}): {exc}",
                    )
                    _reset_login_attempt_state(credential.spec.name)
                    time.sleep(LOGIN_ERROR_RETRY_DELAY_SECONDS)
                except Exception as exc:
                    if attempt >= LOGIN_ERROR_RETRY_ATTEMPTS:
                        raise RetryableLoginError(str(exc).strip() or exc.__class__.__name__) from exc
                    _log_progress(
                        credential.spec.name,
                        f"unexpected login error; retrying in {LOGIN_ERROR_RETRY_DELAY_SECONDS}s "
                        f"(attempt {attempt + 1}/{LOGIN_ERROR_RETRY_ATTEMPTS}): {exc}",
                    )
                    _reset_login_attempt_state(credential.spec.name)
                    time.sleep(LOGIN_ERROR_RETRY_DELAY_SECONDS)

        def record_failure(index: int, credential: AgentCredentials, exc: BaseException) -> None:
            message = str(exc).strip() or exc.__class__.__name__
            kind = _login_failure_kind(exc)
            failures.append((index, credential.spec.name, kind, message))
            _log_progress(credential.spec.name, f"login failed [{kind}]: {message}")

        if max_workers <= 1:
            for index, credential in enumerate(credentials, start=1):
                print(
                    f"auto-login {index}/{total}: {credential.spec.name} "
                    f"({credential.spec.engine}, {credential.spec.account_email})"
                )
                try:
                    run_one(credential)
                except Exception as exc:
                    record_failure(index, credential, exc)
        else:
            remaining: list[tuple[int, AgentCredentials]] = list(enumerate(credentials, start=1))
            pending: dict[Future[None], tuple[int, AgentCredentials]] = {}
            active_by_engine: dict[str, int] = {}
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="multishell-auth") as executor:
                while remaining or pending:
                    while remaining and len(pending) < max_workers:
                        deferred: list[tuple[int, AgentCredentials]] = []
                        submitted = False
                        for index, credential in remaining:
                            engine = credential.spec.engine
                            if active_by_engine.get(engine, 0) >= engine_limits.get(engine, max_parallel):
                                deferred.append((index, credential))
                                continue
                            print(
                                f"auto-login {index}/{total}: {credential.spec.name} "
                                f"({credential.spec.engine}, {credential.spec.account_email})"
                            )
                            pending[executor.submit(run_one, credential)] = (index, credential)
                            active_by_engine[engine] = active_by_engine.get(engine, 0) + 1
                            submitted = True
                            deferred.extend(remaining[remaining.index((index, credential)) + 1 :])
                            remaining = deferred
                            break
                        if not submitted:
                            break
                    done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        index, credential = pending.pop(future)
                        engine = credential.spec.engine
                        active_by_engine[engine] = max(0, active_by_engine.get(engine, 1) - 1)
                        try:
                            future.result()
                        except Exception as exc:
                            record_failure(index, credential, exc)

        if failures:
            lines = ["auto-login completed with failures:"]
            for _index, agent_name, kind, message in sorted(failures):
                lines.append(f"- {agent_name} [{kind}]: {message}")
            raise RuntimeError("\n".join(lines))
    finally:
        _AUTO_LOGIN_VERBOSE = previous_verbose


def _engine_parallel_limit(engine: str, max_parallel: int) -> int:
    if engine == "codex":
        return 1
    return max_parallel


def _reset_login_attempt_state(agent_name: str) -> None:
    for path in (
        auth_path(agent_name),
        claude_auth_path(agent_name),
        claude_root_auth_path(agent_name),
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except Exception:
            continue

    profile_dir = state_root() / "browser-profiles" / agent_name
    if profile_dir.exists():
        _cleanup_stale_chrome_profile(profile_dir)
        try:
            shutil.rmtree(profile_dir)
        except Exception:
            pass


def _login_failure_kind(exc: BaseException) -> str:
    if isinstance(exc, NoAuthLoginError):
        return "no_auth"
    return "error"


def _raise_auth_model_exception(flow_label: str, exc: Exception, logger: Callable[[str], None] | None = None) -> None:
    if isinstance(exc, AuthModelFailure):
        if logger is not None:
            logger(f"auth model failed the {flow_label} flow ({exc}) failure_kind={exc.failure_kind}")
        if exc.failure_kind == "no_auth":
            raise NoAuthLoginError(str(exc)) from exc
        raise RetryableLoginError(str(exc)) from exc
    if isinstance(exc, AuthModelUnavailable):
        raise RetryableLoginError(f"auth model unavailable for {flow_label} flow: {exc}") from exc
    if isinstance(exc, AuthModelProtocolError):
        raise RetryableLoginError(f"auth model returned unusable output for {flow_label} flow: {exc}") from exc
    raise RuntimeError(str(exc).strip() or exc.__class__.__name__) from exc


def _login_codex_one(playwright: object, credential: AgentCredentials, timeout_seconds: int, headed: bool) -> None:
    agent_name = credential.spec.name
    progress_label = _verbose_progress_label(agent_name)
    page = None
    child = None
    try:
        _log_verbose(agent_name, "preparing Codex login state")
        _ensure_home(credential.spec.name)
        if auth_path(credential.spec.name).exists():
            _log_progress(agent_name, "already logged in; skipping")
            return
        _log_coarse_progress(agent_name, "running headless browser auth")

        env = suppress_node_warnings(os.environ.copy())
        env["HOME"] = str(agent_home(credential.spec.name))
        _log_verbose(agent_name, "starting `codex login --device-auth`")
        child, url, device_code = _start_codex_device_auth(agent_name, env)
        _log_verbose(agent_name, f"received device code {device_code}; launching browser")
        with _isolated_chrome(playwright, credential.spec.name, headed=headed, progress_label=progress_label) as (browser, context, page):
            try:
                _log_verbose(agent_name, "browser ready; completing Google/OpenAI sign-in")
                _complete_openai_google_sign_in(page, url, device_code, credential, progress_label=progress_label)
                _log_verbose(agent_name, "waiting for Codex auth.json to be created")
                _wait_for_codex_auth(child, credential.spec.name, timeout_seconds, progress_label=progress_label)
            except Exception as exc:
                raise _enrich_login_error(exc, page, credential.spec.name) from exc
        _log_progress(agent_name, "Codex login complete")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(str(exc).strip() or exc.__class__.__name__) from exc
    finally:
        if child is not None and child.isalive():
            child.terminate(force=True)


def _start_codex_device_auth(agent_name: str, env: dict[str, str], *, max_attempts: int = 3) -> tuple[pexpect.spawn, str, str]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        child = pexpect.spawn(
            "codex",
            ["login", "--device-auth"],
            env=env,
            encoding="utf-8",
            timeout=30,
        )
        output = _collect_child_output(child, seconds=5)
        try:
            url, device_code = _extract_device_flow(output)
            return child, url, device_code
        except CodexDeviceAuthRateLimit as exc:
            last_error = exc
            if child.isalive():
                child.terminate(force=True)
            if attempt >= max_attempts:
                break
            delay_seconds = min(30, attempt * 10)
            _log_progress(
                agent_name,
                f"codex device-auth hit a rate limit; retrying in {delay_seconds}s (attempt {attempt + 1}/{max_attempts})",
            )
            time.sleep(delay_seconds)
        except Exception as exc:
            if child.isalive():
                child.terminate(force=True)
            raise RetryableLoginError(str(exc).strip() or exc.__class__.__name__) from exc
    detail = str(last_error).strip() if last_error is not None else "rate limited"
    raise RetryableLoginError(f"codex device-auth hit OpenAI rate limits after {max_attempts} attempts: {detail}")


def _ensure_home(agent_name: str) -> None:
    from .__main__ import _bridge_command

    if agent_name == "manager":
        ensure_agent_home(agent_name, mcp_bridge_command=_bridge_command("manager", agent_name))
        return
    if agent_name.startswith("worker-"):
        ensure_agent_home(agent_name, mcp_bridge_command=_bridge_command("worker", agent_name))
        return
    ensure_agent_home(agent_name)


def _login_claude_one(playwright: object, credential: AgentCredentials, timeout_seconds: int, headed: bool) -> None:
    agent_name = credential.spec.name
    progress_label = _verbose_progress_label(agent_name)
    page = None
    child = None
    try:
        _log_verbose(agent_name, "preparing Claude login state")
        ensure_claude_home(credential.spec.name)
        if _claude_logged_in(credential.spec.name):
            _log_progress(agent_name, "already logged in; skipping")
            return
        _log_coarse_progress(agent_name, "running headless browser auth")

        env = suppress_node_warnings(os.environ.copy())
        env.pop("ANTHROPIC_API_KEY", None)
        env["HOME"] = str(claude_home(credential.spec.name))
        _log_verbose(agent_name, f"starting `claude auth login --email {credential.spec.account_email}`")
        child = pexpect.spawn(
            "claude",
            ["auth", "login", "--email", credential.spec.account_email],
            env=env,
            encoding="utf-8",
            timeout=30,
        )
        output = _collect_child_output(child, seconds=5)
        manual_url = _extract_claude_login_url(output)
        _log_verbose(agent_name, "waiting for Claude local callback URL")
        auth_url = _wait_for_claude_local_callback_url(child.pid, manual_url, timeout_seconds=10, progress_label=progress_label) or manual_url
        _log_verbose(agent_name, "launching browser for Claude/Google sign-in")
        with _isolated_chrome(playwright, credential.spec.name, headed=headed, progress_label=progress_label) as (browser, context, page):
            try:
                _log_verbose(agent_name, "browser ready; completing Claude/Google sign-in")
                callback_code = _complete_claude_google_sign_in(page, auth_url, credential, progress_label=progress_label)
                if callback_code:
                    _log_verbose(agent_name, "received Claude callback code; sending it back to the CLI")
                    child.sendline(callback_code)
                _log_verbose(agent_name, "waiting for Claude auth status to become logged in")
                _wait_for_claude_auth(child, credential.spec.name, timeout_seconds, progress_label=progress_label)
            except Exception as exc:
                raise _enrich_login_error(exc, page, credential.spec.name) from exc
        _log_progress(agent_name, "Claude login complete")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(str(exc).strip() or exc.__class__.__name__) from exc
    finally:
        if child is not None and child.isalive():
            child.terminate(force=True)


def _extract_device_flow(output: str) -> tuple[str, str]:
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    clean = ansi_re.sub("", output)
    if "429 Too Many Requests" in clean or "rate limit" in clean.lower():
        raise CodexDeviceAuthRateLimit(_truncate_output(clean))
    url_match = re.search(r"https://auth\.openai\.com/codex/device", clean)
    code_match = re.search(r"\b([A-Z0-9]{4}-[A-Z0-9]{5})\b", clean)
    if not url_match or not code_match:
        raise RuntimeError(f"unable to parse device login flow from codex output: {clean!r}")
    return DEVICE_URL, code_match.group(1)


def _extract_claude_login_url(output: str) -> str:
    match = re.search(r"https://claude\.ai/oauth/authorize\S+", output)
    if not match:
        raise RuntimeError(f"unable to parse Claude auth login URL from output: {output!r}")
    return match.group(0)


def _collect_child_output(child: pexpect.spawn, seconds: int) -> str:
    deadline = time.time() + max(1, seconds)
    chunks: list[str] = []
    while time.time() < deadline:
        try:
            chunk = child.read_nonblocking(size=4096, timeout=1)
        except Exception:
            continue
        if chunk:
            chunks.append(chunk)
    return "".join(chunks)


def _truncate_output(text: str, *, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _wait_for_claude_local_callback_url(
    child_pid: int,
    manual_url: str,
    timeout_seconds: int,
    *,
    progress_label: str | None = None,
) -> str | None:
    deadline = time.time() + max(1, timeout_seconds)
    last_progress = 0.0
    while time.time() < deadline:
        port = _claude_listen_port(child_pid)
        if port is not None:
            return _build_claude_local_callback_url(manual_url, port)
        if progress_label is not None:
            last_progress = _periodic_progress(progress_label, last_progress, "still waiting for Claude local callback server")
        time.sleep(0.25)
    return None


def _claude_listen_port(child_pid: int) -> int | None:
    try:
        output = subprocess.check_output(["ss", "-ltnp"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return _extract_claude_listen_port(output, child_pid)


def _extract_claude_listen_port(ss_output: str, child_pid: int) -> int | None:
    for line in ss_output.splitlines():
        if f"pid={child_pid}," not in line or "127.0.0.1:" not in line:
            continue
        match = re.search(r"127\.0\.0\.1:(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def _build_claude_local_callback_url(manual_url: str, port: int) -> str:
    parsed = urlparse(manual_url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    replaced = False
    updated: list[tuple[str, str]] = []
    for key, value in query:
        if key == "redirect_uri":
            updated.append((key, f"http://localhost:{port}/callback"))
            replaced = True
        else:
            updated.append((key, value))
    if not replaced:
        updated.append(("redirect_uri", f"http://localhost:{port}/callback"))
    return urlunparse(parsed._replace(query=urlencode(updated)))


def _complete_openai_google_sign_in(
    page: object,
    url: str,
    device_code: str,
    credential: AgentCredentials,
    *,
    progress_label: str | None = None,
) -> None:
    page.goto(url, wait_until="domcontentloaded")
    _normalize_openai_login_entry(page)
    flow_page = page
    if _has_visible(page, ["button:has-text('Continue with Google')", "text=Continue with Google"], timeout_ms=5000):
        flow_page = _click_google_and_capture_page(page)

    _advance_auth_flow(flow_page, credential, device_code, progress_label=progress_label)


def _complete_claude_google_sign_in(
    page: object,
    url: str,
    credential: AgentCredentials,
    *,
    progress_label: str | None = None,
) -> str | None:
    page.goto(url, wait_until="domcontentloaded")
    flow_page = _preferred_claude_auth_page(page)
    if _has_visible(flow_page, ["button:has-text('Continue with Google')", "text=Continue with Google"], timeout_ms=5000):
        flow_page = _click_google_and_capture_page(flow_page)
        try:
            flow_page.wait_for_load_state("domcontentloaded", timeout=15000)
        except Exception:
            pass

    auth_model_logger, _trace_path = _auth_model_trace_logger(credential.spec.name, "claude-google", progress_label)
    try:
        drive_google_auth_with_model(
            flow_page,
            flow_label="claude/google",
            account_email=credential.spec.account_email,
            password=credential.password,
            logger=auth_model_logger,
        )
    except AuthModelFailure as exc:
        _raise_auth_model_exception("Claude/Google", exc, auth_model_logger)
    except (AuthModelUnavailable, AuthModelProtocolError) as exc:
        _raise_auth_model_exception("Claude/Google", exc, auth_model_logger)

    deadline = time.time() + 20
    last_progress = 0.0
    while time.time() < deadline:
        flow_page = _preferred_claude_auth_page(flow_page)
        try:
            snapshot = capture_auth_snapshot(flow_page)
        except Exception:
            snapshot = None
        if snapshot is not None and snapshot_requires_model(snapshot):
            auth_model_logger(
                "Claude auth handoff still has interactive auth controls; re-entering auth model drive "
                f"({snapshot.get('url') or ''})"
            )
            try:
                drive_google_auth_with_model(
                    flow_page,
                    flow_label="claude/google",
                    account_email=credential.spec.account_email,
                    password=credential.password,
                    logger=auth_model_logger,
                )
            except AuthModelFailure as exc:
                _raise_auth_model_exception("Claude/Google", exc, auth_model_logger)
            except (AuthModelUnavailable, AuthModelProtocolError) as exc:
                _raise_auth_model_exception("Claude/Google", exc, auth_model_logger)
            continue
        body = _body_text(flow_page)
        if progress_label is not None:
            last_progress = _periodic_progress(
                progress_label,
                last_progress,
                f"waiting for Claude auth handoff: {_describe_auth_surface(flow_page, body=body, title=_page_title(flow_page))}",
                interval_seconds=5.0,
            )
        if _claude_logged_in(credential.spec.name):
            return None
        callback_code = _extract_claude_callback_code(body)
        if callback_code:
            return callback_code
        if "platform.claude.com/oauth/code/success" in getattr(flow_page, "url", "") or "You’re all set up for Claude Code" in body or "You're all set up for Claude Code" in body:
            return None
        time.sleep(0.5)

    raise RuntimeError(f"timed out waiting for Claude auth handoff at {flow_page.url!r} with title={_page_title(flow_page)!r}")


def _preferred_claude_auth_page(page: object) -> object:
    context = getattr(page, "context", None)
    if context is None:
        return page

    fallback = page
    for candidate in context.pages:
        try:
            if candidate.is_closed():
                continue
            url = candidate.url
        except Exception:
            continue
        if "accounts.google.com" in url:
            return candidate
        if any(host in url for host in ("claude.ai/oauth", "platform.claude.com/oauth", "claude.ai/login")):
            fallback = candidate
    return fallback


def _extract_claude_callback_code(body: str) -> str | None:
    match = re.search(r"Paste this into Claude Code:\s*([^\s]+)", body)
    if not match:
        return None
    return match.group(1).strip()


def _click_claude_organization_option(page: object) -> bool:
    for selector in ("text=Kuang2", "button:has-text('Kuang2')", "[role='button']:has-text('Kuang2')"):
        locator = page.locator(selector).first
        try:
            if locator.is_visible():
                locator.click(timeout=3000)
                return True
        except Exception:
            continue

    candidates = [page.locator("[role='button']:visible"), page.locator("button:visible")]
    skip_labels = {"Switch account"}
    for locator in candidates:
        count = locator.count()
        for index in range(count):
            element = locator.nth(index)
            try:
                label = element.inner_text().strip()
            except Exception:
                continue
            if not label or label in skip_labels or "Logged in as" in label:
                continue
            try:
                if element.is_enabled():
                    element.click(timeout=3000)
                    return True
            except Exception:
                continue
    return False


def _click_google_and_capture_page(page: object) -> object:
    from playwright.sync_api import TimeoutError

    original_url = page.url
    try:
        with page.expect_popup(timeout=5000) as popup_info:
            _click_first(page, ["button:has-text('Continue with Google')", "text=Continue with Google"])
        return popup_info.value
    except TimeoutError:
        deadline = time.time() + 15
        while time.time() < deadline:
            current_url = page.url
            if "accounts.google.com" in current_url or current_url != original_url:
                break
            if _has_visible(
                page,
                [
                    "input[type='email']:visible",
                    "input[name='identifier']:visible",
                    "input[autocomplete='username']:visible",
                    "input[type='password']:visible",
                    "input[name='Passwd']:visible",
                    "text=Choose an account",
                ],
                timeout_ms=500,
            ):
                break
            try:
                page.wait_for_timeout(250)
            except Exception:
                break
        try:
            page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            pass
        return page


def _normalize_openai_login_entry(page: object) -> None:
    try:
        _wait_for_live_login_surface(page, timeout_seconds=30)
    except RuntimeError:
        body = _body_text(page)
        title = _page_title(page)
        url = getattr(page, "url", "")
        if "auth.openai.com" in url and (
            title == "Just a moment..."
            or "Performing security verification" in body
            or "This website uses a security service to protect against malicious bots" in body
        ):
            return
        raise
    body = _body_text(page)
    if "Your session has ended" in body and _has_visible(page, ["text=Log in"], timeout_ms=5000):
        page.locator("text=Log in").first.click()
        _wait_for_live_login_surface(page, timeout_seconds=45)
    if "Oops, an error occurred!" in body and _has_visible(page, ["button:has-text('Try again')"], timeout_ms=5000):
        page.locator("button:has-text('Try again')").first.click()
        _wait_for_live_login_surface(page, timeout_seconds=30)


def _advance_auth_flow(
    page: object,
    credential: AgentCredentials,
    device_code: str,
    *,
    progress_label: str | None = None,
) -> None:
    auth_model_logger, _trace_path = _auth_model_trace_logger(credential.spec.name, "codex-google", progress_label)
    try:
        drive_google_auth_with_model(
            page,
            flow_label="codex/google",
            account_email=credential.spec.account_email,
            password=credential.password,
            device_code=device_code,
            logger=auth_model_logger,
        )
    except AuthModelFailure as exc:
        _raise_auth_model_exception("Codex/Google", exc, auth_model_logger)
    except (AuthModelUnavailable, AuthModelProtocolError) as exc:
        _raise_auth_model_exception("Codex/Google", exc, auth_model_logger)


def _complete_workspace_consent(page: object) -> None:
    continue_button = page.locator("button:has-text('Continue')").first
    try:
        continue_button.wait_for(state="visible", timeout=10000)
    except Exception as exc:
        raise RuntimeError(f"workspace consent page missing Continue button at {page.url!r}") from exc

    for _ in range(8):
        if continue_button.is_enabled():
            continue_button.click()
            return
        time.sleep(0.5)

    if not _click_first_workspace_option(page):
        raise RuntimeError(f"unable to choose a workspace on consent page at {page.url!r}")

    for _ in range(20):
        if continue_button.is_enabled():
            continue_button.click()
            return
        time.sleep(0.5)

    raise RuntimeError(f"workspace Continue button stayed disabled at {page.url!r}")


def _advance_google_form_step(
    page: object,
    *,
    step_name: str,
    field_selectors: list[str],
    value: str,
    next_selectors: list[str],
    progress_label: str | None = None,
    submitted_value: str | None = None,
    hide_value: bool = False,
) -> None:
    _fill_first(page, field_selectors, value)
    if _submit_google_form_step(
        page,
        step_name=step_name,
        next_selectors=next_selectors,
        progress_label=progress_label,
        submitted_value=submitted_value,
        hide_value=hide_value,
    ):
        return

    if progress_label is not None:
        _log_progress(progress_label, f"{step_name} step did not advance after a normal submit; retrying with typed input")
    _type_first(page, field_selectors, value)
    if _submit_google_form_step(
        page,
        step_name=step_name,
        next_selectors=next_selectors,
        progress_label=progress_label,
        submitted_value=submitted_value,
        hide_value=hide_value,
    ):
        return

    error_text = _google_inline_error(page)
    raise RuntimeError(_format_google_step_error(step_name, submitted_value, error_text, hide_value=hide_value))


def _submit_google_form_step(
    page: object,
    *,
    step_name: str,
    next_selectors: list[str],
    progress_label: str | None = None,
    submitted_value: str | None = None,
    hide_value: bool = False,
) -> bool:
    actions = [
        ("clicking the Google Next button", lambda: _click_first(page, next_selectors)),
        ("pressing Enter", lambda: page.keyboard.press("Enter")),
    ]
    for index, (description, action) in enumerate(actions, start=1):
        try:
            action()
        except Exception:
            pass
        if _wait_for_google_step_transition(page, step_name):
            return True
        error_text = _google_inline_error(page)
        if error_text:
            raise RuntimeError(_format_google_step_error(step_name, submitted_value, error_text, hide_value=hide_value))
        if progress_label is not None and index < len(actions):
            _log_progress(progress_label, f"{step_name} step did not advance after {description}; retrying")
    return False


def _wait_for_google_step_transition(page: object, step_name: str, timeout_seconds: int = 6) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        body = _body_text(page)
        if step_name == "email":
            if _visible_now(page, GOOGLE_PASSWORD_SELECTORS):
                return True
            if _visible_now(page, GOOGLE_CODE_SELECTORS):
                return True
            if "Choose an account" in body or "Use another account" in body:
                return True
            if "Select a workspace" in body or "Sign in to Codex with ChatGPT" in body:
                return True
            if not _visible_now(page, GOOGLE_EMAIL_SELECTORS):
                return True
        elif step_name == "password":
            if _visible_now(page, GOOGLE_CODE_SELECTORS):
                return True
            if "Select a workspace" in body or "Sign in to Codex with ChatGPT" in body:
                return True
            if "/deviceauth/callback" in getattr(page, "url", ""):
                return True
            if not _visible_now(page, GOOGLE_PASSWORD_SELECTORS):
                return True
        else:
            return True
        time.sleep(0.25)
    return False


def _google_inline_error(page: object) -> str | None:
    body = " ".join(_body_text(page).split())
    for snippet in GOOGLE_ERROR_SNIPPETS:
        if snippet in body:
            return snippet
    return None


def _raise_if_google_error(page: object, account_email: str) -> None:
    error_text = _google_inline_error(page)
    if not error_text:
        return
    if error_text in GOOGLE_MANUAL_CHALLENGE_SNIPPETS:
        raise RuntimeError(f"Google requires manual verification and cannot be automated here: {error_text}")
    if error_text == "This browser or app may not be secure":
        raise RuntimeError("Google rejected the browser session as not secure")
    if _visible_now(page, GOOGLE_PASSWORD_SELECTORS):
        raise RuntimeError(_format_google_step_error("password", None, error_text, hide_value=True))
    if _visible_now(page, GOOGLE_EMAIL_SELECTORS):
        raise RuntimeError(_format_google_step_error("email", account_email, error_text))
    if _visible_now(page, GOOGLE_CODE_SELECTORS):
        raise RuntimeError(_format_google_step_error("device-code", None, error_text))
    raise RuntimeError(f"Google rejected the sign-in flow: {error_text}")


def _format_google_step_error(
    step_name: str,
    submitted_value: str | None,
    error_text: str | None,
    *,
    hide_value: bool = False,
) -> str:
    detail = error_text or "Google kept the step open without returning a clearer error"
    if step_name == "email" and submitted_value and not hide_value:
        return f"Google rejected the configured email {submitted_value!r}: {detail}"
    if step_name == "password":
        return f"Google rejected the configured password or requested extra verification: {detail}"
    return f"Google rejected the {step_name} step: {detail}"


def _click_first_workspace_option(page: object) -> bool:
    candidates = []

    role_buttons = page.locator("[role='button']:visible")
    candidates.append(role_buttons)

    button_locator = page.locator("button:visible")
    candidates.append(button_locator)

    label_locator = page.locator("label:visible")
    candidates.append(label_locator)

    for locator in candidates:
        count = locator.count()
        for index in range(count):
            element = locator.nth(index)
            try:
                label = element.inner_text().strip()
            except Exception:
                continue
            if not label or label in {"Continue", "Cancel"}:
                continue
            if not element.is_enabled():
                continue
            element.click()
            return True

    text_candidates = ["Personal account", "Kuang2"]
    for text in text_candidates:
        element = page.locator(f"text={text}").first
        try:
            element.wait_for(state="visible", timeout=1000)
            element.click()
            return True
        except Exception:
            continue
    return False


def _enter_device_code(page: object, device_code: str) -> None:
    visible_inputs = page.locator("input:visible")
    input_count = visible_inputs.count()
    if input_count == 0:
        raise RuntimeError(f"device code page has no visible inputs at {page.url!r}")

    normalized = device_code.replace("-", "")

    if input_count == 1:
        input_field = visible_inputs.first
        input_field.fill(device_code)
        if _continue_enabled(page):
            return
        input_field.fill(normalized)
        return

    max_lengths = []
    for index in range(input_count):
        max_length = visible_inputs.nth(index).get_attribute("maxlength")
        max_lengths.append(max_length)

    if all(length in {None, "1", "2"} for length in max_lengths):
        try:
            visible_inputs.first.click()
            page.keyboard.type(normalized, delay=60)
            if _continue_enabled(page):
                return
        except Exception:
            pass
        for index, char in enumerate(normalized[:input_count]):
            visible_inputs.nth(index).fill(char)
        if _continue_enabled(page):
            return
        for index, char in enumerate(normalized[:input_count]):
            visible_inputs.nth(index).click()
            page.keyboard.press("Control+A")
            page.keyboard.type(char, delay=20)
        return

    visible_inputs.first.fill(device_code)
    if _continue_enabled(page):
        return
    visible_inputs.first.fill(normalized)


def _click_enabled_continue(page: object) -> None:
    continue_button = page.locator("button:has-text('Continue'), button:has-text('Submit'), button:has-text('Authorize')").first
    for _ in range(40):
        if continue_button.is_visible() and continue_button.is_enabled():
            continue_button.click()
            return
        time.sleep(0.25)
    raise RuntimeError(
        f"device code Continue button never enabled; title={page.title()!r}; url={page.url!r}; body={_body_text(page)[:240]!r}"
    )


def _continue_enabled(page: object) -> bool:
    button = page.locator("button:has-text('Continue'), button:has-text('Submit'), button:has-text('Authorize')").first
    try:
        return button.is_visible() and button.is_enabled()
    except Exception:
        return False


def _build_chrome_command(
    chrome_binary: str,
    profile_dir: Path,
    port: int,
    *,
    headed: bool,
) -> list[str]:
    version = _browser_product_version(chrome_binary) or "145.0.0.0"
    user_agent = (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        f"Chrome/{version} Safari/537.36"
    )
    command = [
        chrome_binary,
        f"--user-data-dir={profile_dir}",
        f"--remote-debugging-port={port}",
        f"--user-agent={user_agent}",
        "--no-first-run",
        "--no-default-browser-check",
        "--start-maximized",
        "--disable-background-networking",
        "--disable-blink-features=AutomationControlled",
        "--disable-dev-shm-usage",
        "--disable-sync",
        "--lang=en-US,en",
        "--window-size=1366,768",
    ]
    if not headed:
        command.extend(
            [
                "--headless=new",
                "--disable-gpu",
            ]
        )
    command.extend(
        [
            "--no-sandbox",
            "--disable-setuid-sandbox",
        ]
    )
    command.append("about:blank")
    return command


def _browser_product_version(chrome_binary: str) -> str:
    try:
        output = subprocess.check_output([chrome_binary, "--product-version"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        output = ""
    if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", output):
        return ""
    return output


class _ProcessOutputTail:
    def __init__(
        self,
        stream: object,
        *,
        progress_label: str | None = None,
        log_path: Path | None = None,
    ) -> None:
        self._stream = stream
        self._progress_label = progress_label
        self._log_path = log_path
        self._head_lines: list[str] = []
        self._lines: deque[str] = deque(maxlen=40)
        self._signal_lines: deque[str] = deque(maxlen=12)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_path.write_text("", encoding="utf-8")
        if stream is not None:
            self._thread = threading.Thread(target=self._drain_stream, name="multishell-browser-log", daemon=True)
            self._thread.start()

    def _drain_stream(self) -> None:
        try:
            while True:
                raw_line = self._stream.readline()
                if not raw_line:
                    break
                line = _truncate_output(str(raw_line).strip(), limit=400)
                if not line:
                    continue
                with self._lock:
                    if len(self._head_lines) < 12:
                        self._head_lines.append(line)
                    self._lines.append(line)
                    if _is_browser_signal_line(line):
                        self._signal_lines.append(line)
                    if self._log_path is not None:
                        with self._log_path.open("a", encoding="utf-8") as handle:
                            handle.write(f"{line}\n")
                if self._progress_label is not None:
                    _log_progress(self._progress_label, f"browser output: {line}")
        except Exception as exc:
            line = f"<browser output reader error: {exc}>"
            with self._lock:
                self._lines.append(line)
                if self._log_path is not None:
                    with self._log_path.open("a", encoding="utf-8") as handle:
                        handle.write(f"{line}\n")
            if self._progress_label is not None:
                _log_progress(self._progress_label, f"browser output: {line}")

    def tail(self, *, limit: int = 8) -> str:
        with self._lock:
            lines = list(self._lines)[-limit:]
        return " | ".join(lines)

    def summary(self) -> str:
        with self._lock:
            parts: list[str] = []
            if self._head_lines:
                parts.append(f"browser output head: {' | '.join(self._head_lines[:6])}")
            if self._signal_lines:
                parts.append(f"browser fatal lines: {' | '.join(list(self._signal_lines)[-6:])}")
            if self._lines:
                parts.append(f"browser output tail: {' | '.join(list(self._lines)[-8:])}")
            if self._log_path is not None:
                parts.append(f"browser log: {self._log_path}")
        return "; ".join(parts)

    def join(self, timeout_seconds: float = 1.0) -> None:
        if self._thread is None:
            return
        self._thread.join(timeout=max(0.0, timeout_seconds))


def _is_browser_signal_line(line: str) -> bool:
    lowered = line.lower()
    return any(
        token in lowered
        for token in (
            "fatal:",
            "error:",
            "check failed",
            "received signal",
            "no usable sandbox",
            "trace/breakpoint trap",
            "illegal instruction",
            "segmentation fault",
            "stack trace",
        )
    )


def _stop_browser_process(process: subprocess.Popen[str] | None, output_tail: _ProcessOutputTail | None) -> None:
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            process.wait(timeout=10)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                pass
    if output_tail is not None:
        output_tail.join()


@contextmanager
def _isolated_chrome(playwright: object, agent_name: str, headed: bool, progress_label: str | None = None):
    chrome_binary = _resolve_browser_binary(playwright)
    if progress_label is not None:
        _log_progress(progress_label, f"launching browser binary {chrome_binary}")

    profile_dir = state_root() / "browser-profiles" / agent_name
    profile_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_chrome_profile(profile_dir)
    port = _reserve_port()
    command = _build_chrome_command(chrome_binary, profile_dir, port, headed=headed)
    browser_log_path = _browser_log_path(agent_name)
    process = None
    output_tail = None
    browser = None

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=child_env(os.environ.copy(), role="browser", agent=agent_name),
            start_new_session=True,
        )
        output_tail = _ProcessOutputTail(process.stdout, progress_label=progress_label, log_path=browser_log_path)
        if progress_label is not None:
            _log_progress(
                progress_label,
                f"browser launch: pid={process.pid} port={port} headed={'yes' if headed else 'no'} "
                f"profile={profile_dir} sandbox=off",
            )
            _log_progress(progress_label, f"browser command: {' '.join(shlex.quote(part) for part in command)}")
            _log_progress(progress_label, f"browser log file: {browser_log_path}")
            _log_progress(progress_label, f"waiting for Chrome DevTools endpoint on port {port}")
        endpoint = _wait_for_cdp_endpoint(
            port,
            timeout_seconds=30,
            progress_label=progress_label,
            process=process,
            output_tail=output_tail,
        )
        if progress_label is not None:
            _log_progress(progress_label, f"Chrome DevTools endpoint ready on port {port}")
        browser = playwright.chromium.connect_over_cdp(endpoint)
        context = browser.contexts[0]
        page = context.new_page()
        yield browser, context, page
    finally:
        try:
            if browser is not None:
                browser.close()
        except Exception:
            pass
        _stop_browser_process(process, output_tail)


def _browser_log_path(agent_name: str) -> Path:
    debug_dir = state_root() / "debug" / agent_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    return debug_dir / f"{timestamp}-browser.log"


def _resolve_browser_binary(playwright: object) -> str:
    override = os.environ.get("MULTISHELL_BROWSER_BINARY", "").strip()
    if override:
        if Path(override).exists():
            return override
        raise RuntimeError(f"configured browser binary does not exist: {override}")

    bundled = getattr(getattr(playwright, "chromium", None), "executable_path", "")
    if callable(bundled):
        bundled = bundled()
    bundled_path = str(bundled or "").strip()
    if bundled_path and Path(bundled_path).exists():
        return bundled_path

    system_chrome = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
    if system_chrome:
        return system_chrome
    raise RuntimeError("no browser binary is available; run `multishell install-browser` first")


def _cleanup_stale_chrome_profile(profile_dir: Path) -> None:
    token = f"--user-data-dir={profile_dir}"
    try:
        output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        output = ""

    stale_pids: list[int] = []
    for line in output.splitlines():
        if token not in line:
            continue
        pid_text, _, _args = line.strip().partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != os.getpid():
            stale_pids.append(pid)

    _terminate_pids(stale_pids, signal.SIGTERM, timeout_seconds=3.0)
    _terminate_pids(stale_pids, signal.SIGKILL, timeout_seconds=1.0)

    for name in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
        try:
            (profile_dir / name).unlink()
        except FileNotFoundError:
            continue
        except Exception:
            continue


def _terminate_pids(pids: list[int], sig: signal.Signals, timeout_seconds: float) -> None:
    pending: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, sig)
            pending.append(pid)
        except ProcessLookupError:
            continue
        except Exception:
            continue

    if not pending:
        return

    deadline = time.time() + max(0.1, timeout_seconds)
    while pending and time.time() < deadline:
        remaining: list[int] = []
        for pid in pending:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except Exception:
                continue
            remaining.append(pid)
        pending = remaining
        if pending:
            time.sleep(0.2)


def _wait_for_cdp_endpoint(
    port: int,
    timeout_seconds: int,
    progress_label: str | None = None,
    *,
    process: subprocess.Popen[str] | None = None,
    output_tail: _ProcessOutputTail | None = None,
) -> str:
    import json
    import urllib.request

    deadline = time.time() + timeout_seconds
    url = f"http://127.0.0.1:{port}/json/version"
    last_error = None
    last_progress = 0.0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                payload = json.load(response)
            web_socket_url = payload.get("webSocketDebuggerUrl")
            if web_socket_url:
                return f"http://127.0.0.1:{port}"
        except Exception as exc:
            last_error = exc
            if process is not None:
                return_code = process.poll()
                if return_code is not None:
                    detail = f"browser exited before opening DevTools endpoint on port {port} with exit code {return_code}"
                    summary = output_tail.summary() if output_tail is not None else ""
                    if summary:
                        detail = f"{detail}; {summary}"
                    raise RuntimeError(detail) from exc
            if progress_label is not None:
                process_state = ""
                if process is not None:
                    status = "alive" if process.poll() is None else f"exit_code={process.poll()}"
                    process_state = f" pid={process.pid} status={status}"
                last_progress = _periodic_progress(
                    progress_label,
                    last_progress,
                    f"still waiting for Chrome DevTools endpoint on port {port}{process_state}",
                )
        time.sleep(0.5)
    detail = f"timed out waiting for Chrome DevTools endpoint on port {port}"
    if process is not None:
        status = "alive" if process.poll() is None else f"exit_code={process.poll()}"
        detail = f"{detail} pid={process.pid} status={status}"
    if last_error is not None:
        detail = f"{detail}: {last_error}"
    summary = output_tail.summary() if output_tail is not None else ""
    if summary:
        detail = f"{detail}; {summary}"
    raise RuntimeError(detail)


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_live_login_surface(page: object, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        title = _page_title(page)
        body = _body_text(page)
        if _has_visible(page, ["button:has-text('Continue with Google')", "text=Continue with Google"], timeout_ms=1000):
            return
        if _has_visible(page, ["input[type='email']", "input[autocomplete='username']", "input[name='identifier']"], timeout_ms=1000):
            return
        if _has_visible(page, ["input[type='password']", "input[name='Passwd']", "input[autocomplete='current-password']"], timeout_ms=1000):
            return
        if _has_visible(page, ["input[name*=code]", "input[autocomplete='one-time-code']", "input[inputmode='numeric']"], timeout_ms=1000):
            return
        if "Sign in to Codex with ChatGPT" in body or "Select a workspace" in body:
            return
        if "Use your device code to grant access to Codex CLI" in body:
            return
        if "Oops, an error occurred!" in body:
            raise RuntimeError(f"OpenAI auth returned an error page: {body[:240]}")
        if title != "Just a moment..." and "Just a moment..." not in body:
            time.sleep(1)
            continue
        time.sleep(2)
    raise RuntimeError(f"timed out waiting for auth challenge to clear at {getattr(page, 'url', '<unknown>')}")


def _describe_auth_surface(page: object, *, body: str | None = None, title: str | None = None) -> str:
    body_text = body if body is not None else _body_text(page)
    title_text = title if title is not None else _page_title(page)
    location = getattr(page, "url", "") or "<unknown-url>"
    markers: list[str] = []

    if _visible_now(page, ["button:has-text('Continue with Google')", "text=Continue with Google"]):
        markers.append("google-button")
    if _visible_now(page, ["input[type='email']", "input[autocomplete='username']", "input[name='identifier']"]):
        markers.append("email-input")
    if _visible_now(page, ["input[type='password']", "input[name='Passwd']", "input[autocomplete='current-password']"]):
        markers.append("password-input")
    if _visible_now(page, ["input[name*=code]", "input[autocomplete='one-time-code']", "input[inputmode='numeric']"]):
        markers.append("code-input")
    if "Choose an account" in body_text:
        markers.append("choose-account")
    if "Use your device code to grant access to Codex CLI" in body_text:
        markers.append("device-code")
    if "Sign in to Codex with ChatGPT" in body_text or "Select a workspace" in body_text:
        markers.append("workspace-consent")
    if "This browser or app may not be secure" in body_text:
        markers.append("browser-not-secure")
    if "Couldn’t sign you in" in body_text or "Couldn't sign you in" in body_text:
        markers.append("signin-error")

    compact_body = " ".join(body_text.split())
    if not markers and compact_body:
        markers.append(compact_body[:120])

    marker_text = ", ".join(markers) if markers else "no-known-surface"
    return f"url={location} title={title_text!r} state={marker_text}"


def _wait_for_codex_auth(
    child: pexpect.spawn,
    agent_name: str,
    timeout_seconds: int,
    *,
    progress_label: str | None = None,
) -> None:
    deadline = time.time() + timeout_seconds
    last_progress = 0.0
    output_tail: deque[str] = deque(maxlen=12)
    while time.time() < deadline:
        _drain_child_output(child, output_tail=output_tail)
        if auth_path(agent_name).exists() or _codex_logged_in(agent_name):
            child.terminate(force=True)
            return
        if not child.isalive():
            _drain_child_output(child, output_tail=output_tail)
            if auth_path(agent_name).exists() or _codex_logged_in(agent_name):
                return
            raise RetryableLoginError(
                _codex_auth_wait_error(
                    f"codex login exited before authentication completed for {agent_name}",
                    output_tail,
                )
            )
        if progress_label is not None:
            last_progress = _periodic_progress(progress_label, last_progress, "still waiting for Codex auth.json")
        time.sleep(1)
    child.terminate(force=True)
    _drain_child_output(child, output_tail=output_tail)
    raise RetryableLoginError(
        _codex_auth_wait_error(
            f"timed out waiting for codex login to finish for {agent_name}",
            output_tail,
        )
    )


def _codex_logged_in(agent_name: str) -> bool:
    env = suppress_node_warnings(os.environ.copy())
    env["HOME"] = str(agent_home(agent_name))
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


def _drain_child_output(child: pexpect.spawn, *, output_tail: deque[str] | None = None) -> None:
    while True:
        try:
            chunk = child.read_nonblocking(size=4096, timeout=0)
        except (pexpect.TIMEOUT, pexpect.EOF):
            return
        except Exception:
            return
        if not chunk:
            return
        if output_tail is None:
            continue
        ansi_re = re.compile(r"\x1b\[[0-9;]*m")
        cleaned = ansi_re.sub("", chunk)
        for raw_line in cleaned.splitlines():
            line = _truncate_output(raw_line.strip(), limit=240)
            if line:
                output_tail.append(line)


def _codex_auth_wait_error(message: str, output_tail: deque[str]) -> str:
    if not output_tail:
        return message
    return f"{message}; codex output tail: {' | '.join(output_tail)}"


def _wait_for_claude_auth(
    child: pexpect.spawn,
    agent_name: str,
    timeout_seconds: int,
    *,
    progress_label: str | None = None,
) -> None:
    deadline = time.time() + timeout_seconds
    last_progress = 0.0
    while time.time() < deadline:
        if _claude_logged_in(agent_name):
            child.terminate(force=True)
            return
        if not child.isalive():
            if _claude_logged_in(agent_name):
                return
            raise RetryableLoginError(f"claude auth login exited before auth completed for {agent_name}")
        if progress_label is not None:
            last_progress = _periodic_progress(progress_label, last_progress, "still waiting for Claude auth status")
        time.sleep(1)
    child.terminate(force=True)
    raise RetryableLoginError(f"timed out waiting for Claude auth to finish for {agent_name}")


def _claude_logged_in(agent_name: str) -> bool:
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


def _selector(raw: str) -> str:
    return raw


def _fill_first(page: object, selectors: list[str], value: str) -> None:
    for selector in selectors:
        locator = page.locator(selector)
        try:
            count = locator.count()
        except Exception:
            count = 0
        candidates = [locator.nth(index) for index in range(count)] or [locator.first]
        for candidate in candidates:
            try:
                candidate.wait_for(state="visible", timeout=7000)
                candidate.fill(value)
                return
            except Exception:
                continue
    raise RuntimeError(
        f"unable to fill any selector from {selectors}; title={_page_title(page)!r}; url={page.url!r}; body={_body_text(page)[:240]!r}"
    )


def _type_first(page: object, selectors: list[str], value: str) -> None:
    for selector in selectors:
        locator = page.locator(selector)
        try:
            count = locator.count()
        except Exception:
            count = 0
        candidates = [locator.nth(index) for index in range(count)] or [locator.first]
        for candidate in candidates:
            try:
                candidate.wait_for(state="visible", timeout=7000)
                candidate.click()
                page.keyboard.press("Control+A")
                page.keyboard.type(value, delay=20)
                return
            except Exception:
                continue
    raise RuntimeError(
        f"unable to type into any selector from {selectors}; title={_page_title(page)!r}; url={page.url!r}; body={_body_text(page)[:240]!r}"
    )


def _click_first(page: object, selectors: list[str]) -> None:
    for selector in selectors:
        locator = page.locator(selector)
        try:
            count = locator.count()
        except Exception:
            count = 0
        candidates = [locator.nth(index) for index in range(count)] or [locator.first]
        for candidate in candidates:
            try:
                candidate.wait_for(state="visible", timeout=7000)
                candidate.click()
                return
            except Exception:
                continue
    raise RuntimeError(
        f"unable to click any selector from {selectors}; title={_page_title(page)!r}; url={page.url!r}; body={_body_text(page)[:240]!r}"
    )


def _click_optional(page: object, selectors: list[str]) -> None:
    for selector in selectors:
        locator = page.locator(selector)
        try:
            count = locator.count()
        except Exception:
            count = 0
        candidates = [locator.nth(index) for index in range(count)] or [locator.first]
        for candidate in candidates:
            try:
                candidate.wait_for(state="visible", timeout=2000)
                candidate.click()
                return
            except Exception:
                continue


def _has_visible(page: object, selectors: list[str], timeout_ms: int) -> bool:
    slice_timeout = max(250, timeout_ms // max(1, len(selectors)))
    for selector in selectors:
        locator = page.locator(selector)
        try:
            count = locator.count()
        except Exception:
            count = 0
        candidates = [locator.nth(index) for index in range(count)] or [locator.first]
        for candidate in candidates:
            try:
                candidate.wait_for(state="visible", timeout=slice_timeout)
                return True
            except Exception:
                continue
    return False


def _visible_now(page: object, selectors: list[str]) -> bool:
    for selector in selectors:
        locator = page.locator(selector)
        try:
            count = locator.count()
        except Exception:
            count = 0
        candidates = [locator.nth(index) for index in range(count)] or [locator.first]
        for candidate in candidates:
            try:
                if candidate.is_visible():
                    return True
            except Exception:
                continue
    return False


def _body_text(page: object) -> str:
    try:
        return page.locator("body").inner_text(timeout=1000)
    except Exception:
        return ""


def _page_title(page: object) -> str:
    try:
        return page.title()
    except Exception:
        return ""


def _enrich_login_error(exc: Exception, page: object | None, agent_name: str) -> RuntimeError:
    message = str(exc).strip() or exc.__class__.__name__
    if page is None:
        return _wrap_login_error(exc, message)
    debug_dir = _write_debug_artifacts(page, agent_name)
    _log_progress(agent_name, f"debug artifacts: {debug_dir}")
    return _wrap_login_error(exc, f"{message} (debug: {debug_dir})")


def _wrap_login_error(exc: Exception, message: str) -> RuntimeError:
    if isinstance(exc, NoAuthLoginError):
        return NoAuthLoginError(message)
    if isinstance(exc, RetryableLoginError):
        return RetryableLoginError(message)
    return RetryableLoginError(message)


def _write_debug_artifacts(page: object, agent_name: str) -> Path:
    debug_dir = state_root() / "debug" / agent_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    screenshot_path = debug_dir / f"{timestamp}.png"
    html_path = debug_dir / f"{timestamp}.html"
    meta_path = debug_dir / f"{timestamp}.txt"
    meta_lines: list[str] = []
    try:
        page.screenshot(path=str(screenshot_path), full_page=True)
        meta_lines.append(f"screenshot={screenshot_path.name}")
    except Exception as exc:
        meta_lines.append(f"screenshot_error={exc}")
    try:
        html_path.write_text(page.content(), encoding="utf-8")
        meta_lines.append(f"html={html_path.name}")
    except Exception as exc:
        meta_lines.append(f"html_error={exc}")
    try:
        meta_lines.append(f"url={page.url}")
    except Exception as exc:
        meta_lines.append(f"url_error={exc}")
    try:
        meta_lines.append(f"title={page.title()}")
    except Exception as exc:
        meta_lines.append(f"title_error={exc}")
    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    return debug_dir
