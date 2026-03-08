from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pexpect

from .config import AgentSpec, account_for_agent, all_agent_specs, credential_source_agent, state_root
from .homes import agent_home, auth_path, claude_home, ensure_agent_home, ensure_claude_home, has_claude_auth
from .runtime import apply_node_warning_suppression, child_env, suppress_node_warnings


DEVICE_URL = "https://auth.openai.com/codex/device"


@dataclass(frozen=True)
class AgentCredentials:
    spec: AgentSpec
    password: str


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


def run_auto_login(credentials: list[AgentCredentials], headed: bool = False, timeout_seconds: int = 180) -> None:
    apply_node_warning_suppression()
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run `python3 -m pip install playwright pexpect` "
            "and `python3 -m playwright install chromium` first."
        ) from exc

    with sync_playwright() as playwright:
        total = len(credentials)
        for index, credential in enumerate(credentials, start=1):
            print(
                f"auto-login {index}/{total}: {credential.spec.name} "
                f"({credential.spec.engine}, {credential.spec.account_email})"
            )
            if credential.spec.role == "claude-worker":
                _login_claude_one(playwright, credential, timeout_seconds, headed=headed)
            else:
                _login_codex_one(playwright, credential, timeout_seconds, headed=headed)


def _login_codex_one(playwright: object, credential: AgentCredentials, timeout_seconds: int, headed: bool) -> None:
    page = None
    child = None
    try:
        _ensure_home(credential.spec.name)
        if auth_path(credential.spec.name).exists():
            return

        env = suppress_node_warnings(os.environ.copy())
        env["HOME"] = str(agent_home(credential.spec.name))
        child = pexpect.spawn(
            "codex",
            ["login", "--device-auth"],
            env=env,
            encoding="utf-8",
            timeout=30,
        )
        try:
            output = child.read_nonblocking(size=4096, timeout=2)
        except Exception:
            output = ""

        url, device_code = _extract_device_flow(output)
        with _isolated_chrome(playwright, credential.spec.name, headed=headed) as (browser, context, page):
            _complete_openai_google_sign_in(page, url, device_code, credential)
            _wait_for_codex_auth(child, credential.spec.name, timeout_seconds)
    except Exception:
        if page is not None:
            _write_debug_artifacts(page, credential.spec.name)
        raise
    finally:
        if child is not None and child.isalive():
            child.terminate(force=True)


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
    page = None
    child = None
    try:
        ensure_claude_home(credential.spec.name)
        if _claude_logged_in(credential.spec.name):
            return

        env = suppress_node_warnings(os.environ.copy())
        env.pop("ANTHROPIC_API_KEY", None)
        env["HOME"] = str(claude_home(credential.spec.name))
        child = pexpect.spawn(
            "claude",
            ["auth", "login", "--email", credential.spec.account_email],
            env=env,
            encoding="utf-8",
            timeout=30,
        )
        output = _collect_child_output(child, seconds=5)
        manual_url = _extract_claude_login_url(output)
        auth_url = _wait_for_claude_local_callback_url(child.pid, manual_url, timeout_seconds=10) or manual_url
        with _isolated_chrome(playwright, credential_source_agent(credential.spec.name), headed=headed) as (browser, context, page):
            callback_code = _complete_claude_google_sign_in(page, auth_url, credential)
            if callback_code:
                child.sendline(callback_code)
            _wait_for_claude_auth(child, credential.spec.name, timeout_seconds)
    except Exception:
        if page is not None:
            _write_debug_artifacts(page, credential.spec.name)
        raise
    finally:
        if child is not None and child.isalive():
            child.terminate(force=True)


def _extract_device_flow(output: str) -> tuple[str, str]:
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    clean = ansi_re.sub("", output)
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


def _wait_for_claude_local_callback_url(child_pid: int, manual_url: str, timeout_seconds: int) -> str | None:
    deadline = time.time() + max(1, timeout_seconds)
    while time.time() < deadline:
        port = _claude_listen_port(child_pid)
        if port is not None:
            return _build_claude_local_callback_url(manual_url, port)
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


def _complete_openai_google_sign_in(page: object, url: str, device_code: str, credential: AgentCredentials) -> None:
    page.goto(url, wait_until="domcontentloaded")
    _normalize_openai_login_entry(page)
    flow_page = page
    if _has_visible(page, ["button:has-text('Continue with Google')", "text=Continue with Google"], timeout_ms=5000):
        flow_page = _click_google_and_capture_page(page)

    _advance_auth_flow(flow_page, credential, device_code)


def _complete_claude_google_sign_in(page: object, url: str, credential: AgentCredentials) -> str | None:
    page.goto(url, wait_until="domcontentloaded")
    flow_page = _preferred_claude_auth_page(page)
    if _has_visible(flow_page, ["button:has-text('Continue with Google')", "text=Continue with Google"], timeout_ms=5000):
        flow_page = _click_google_and_capture_page(flow_page)
        try:
            flow_page.wait_for_load_state("domcontentloaded", timeout=15000)
        except Exception:
            pass

    deadline = time.time() + 180
    while time.time() < deadline:
        flow_page = _preferred_claude_auth_page(flow_page)
        body = _body_text(flow_page)
        title = _page_title(flow_page)

        if _claude_logged_in(credential.spec.name):
            return None

        callback_code = _extract_claude_callback_code(body)
        if callback_code:
            return callback_code

        if _has_visible(
            flow_page,
            ["input[type='email']:visible", "input[name='identifier']:visible", "input[autocomplete='username']:visible"],
            timeout_ms=1000,
        ):
            _fill_first(
                flow_page,
                ["input[type='email']:visible", "input[name='identifier']:visible", "input[autocomplete='username']:visible"],
                credential.spec.account_email,
            )
            _click_first(flow_page, ["#identifierNext", "button:has-text('Next')"])
            time.sleep(1)
            continue

        if _has_visible(
            flow_page,
            ["input[type='password']:visible", "input[name='Passwd']:visible", "input[autocomplete='current-password']:visible"],
            timeout_ms=1000,
        ):
            _fill_first(
                flow_page,
                ["input[type='password']:visible", "input[name='Passwd']:visible", "input[autocomplete='current-password']:visible"],
                credential.password,
            )
            _click_first(flow_page, ["#passwordNext", "button:has-text('Next')"])
            time.sleep(1)
            continue

        if (
            "Choose an account" in body
            or _has_visible(flow_page, ["text=Use another account"], timeout_ms=1000)
        ) and _has_visible(
            flow_page,
            [f"text={credential.spec.account_email}", "text=Use another account"],
            timeout_ms=1000,
        ):
            account = flow_page.locator(f"text={credential.spec.account_email}").first
            try:
                account.click(timeout=3000)
                time.sleep(1)
                continue
            except Exception:
                pass

        if "Select organization" in body or "Logged in as" in body:
            if _click_claude_organization_option(flow_page):
                time.sleep(1)
                continue

        _click_optional(
            flow_page,
            [
                "button:has-text('Continue')",
                "button:has-text('Allow')",
                "button:has-text('Authorize')",
                "button:has-text('Accept')",
                "button:has-text('Open Claude')",
            ],
        )

        if "Choose an account" in body and credential.spec.account_email in body:
            time.sleep(1)
            continue

        if title == "Just a moment..." or "Just a moment..." in body:
            time.sleep(2)
            continue

        time.sleep(1)

    raise RuntimeError(f"timed out completing Claude auth flow at {flow_page.url!r} with title={_page_title(flow_page)!r}")


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
    _wait_for_live_login_surface(page, timeout_seconds=30)
    body = _body_text(page)
    if "Your session has ended" in body and _has_visible(page, ["text=Log in"], timeout_ms=5000):
        page.locator("text=Log in").first.click()
        _wait_for_live_login_surface(page, timeout_seconds=45)
    if "Oops, an error occurred!" in body and _has_visible(page, ["button:has-text('Try again')"], timeout_ms=5000):
        page.locator("button:has-text('Try again')").first.click()
        _wait_for_live_login_surface(page, timeout_seconds=30)


def _advance_auth_flow(page: object, credential: AgentCredentials, device_code: str) -> None:
    deadline = time.time() + 180
    while time.time() < deadline:
        body = _body_text(page)
        title = _page_title(page)

        if not body and not title:
            time.sleep(0.5)
            continue

        if _has_visible(page, ["button:has-text('Continue with Google')", "text=Continue with Google"], timeout_ms=1000):
            _click_google_and_capture_page(page)
            time.sleep(1)
            continue

        if _has_visible(page, ["input[type='email']:visible", "input[name='identifier']:visible", "input[autocomplete='username']:visible"], timeout_ms=1000):
            _fill_first(
                page,
                ["input[type='email']:visible", "input[name='identifier']:visible", "input[autocomplete='username']:visible"],
                credential.spec.account_email,
            )
            _click_first(page, ["#identifierNext", "button:has-text('Next')"])
            time.sleep(1)
            continue

        if _has_visible(page, ["input[type='password']:visible", "input[name='Passwd']:visible", "input[autocomplete='current-password']:visible"], timeout_ms=1000):
            _fill_first(
                page,
                ["input[type='password']:visible", "input[name='Passwd']:visible", "input[autocomplete='current-password']:visible"],
                credential.password,
            )
            _click_first(page, ["#passwordNext", "button:has-text('Next')"])
            time.sleep(1)
            continue

        if "Use your device code to grant access to Codex CLI" in body:
            _enter_device_code(page, device_code)
            _click_enabled_continue(page)
            return

        if "/deviceauth/callback" in page.url and page.locator("input:visible").count() >= 1:
            _enter_device_code(page, device_code)
            _click_enabled_continue(page)
            return

        if _has_visible(page, ["input[name*=code]", "input[autocomplete='one-time-code']", "input[inputmode='numeric']"], timeout_ms=1000):
            _enter_device_code(page, device_code)
            _click_enabled_continue(page)
            return

        if "Sign in to Codex with ChatGPT" in body or "Select a workspace" in body:
            _complete_workspace_consent(page)
            time.sleep(1)
            continue

        if "Just a moment..." in body or title == "Just a moment...":
            time.sleep(2)
            continue

        if not body:
            time.sleep(1)
            continue

        # Let navigations settle; OpenAI bounces through several pages after Google auth.
        time.sleep(1)

    raise RuntimeError(f"timed out completing auth flow at {page.url!r} with title={_page_title(page)!r}")


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


@contextmanager
def _isolated_chrome(playwright: object, agent_name: str, headed: bool):
    chrome_binary = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
    if chrome_binary is None:
        raise RuntimeError("google-chrome is not installed")

    profile_dir = state_root() / "browser-profiles" / agent_name
    profile_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_chrome_profile(profile_dir)
    port = _reserve_port()

    command = [
        chrome_binary,
        f"--user-data-dir={profile_dir}",
        f"--remote-debugging-port={port}",
        "--no-first-run",
        "--no-default-browser-check",
        "--start-maximized",
        "--disable-background-networking",
        "--disable-dev-shm-usage",
        "--disable-sync",
        "--window-size=1366,768",
        "about:blank",
    ]
    if not headed or not os.environ.get("DISPLAY"):
        command = ["xvfb-run", "-a", *command]

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

    try:
        endpoint = _wait_for_cdp_endpoint(port, timeout_seconds=30)
        browser = playwright.chromium.connect_over_cdp(endpoint)
        context = browser.contexts[0]
        page = context.new_page()
        yield browser, context, page
    finally:
        try:
            browser.close()
        except Exception:
            pass
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


def _wait_for_cdp_endpoint(port: int, timeout_seconds: int) -> str:
    import json
    import urllib.request

    deadline = time.time() + timeout_seconds
    url = f"http://127.0.0.1:{port}/json/version"
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                payload = json.load(response)
            web_socket_url = payload.get("webSocketDebuggerUrl")
            if web_socket_url:
                return f"http://127.0.0.1:{port}"
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for Chrome DevTools endpoint on port {port}: {last_error}")


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


def _wait_for_codex_auth(child: pexpect.spawn, agent_name: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if auth_path(agent_name).exists():
            child.terminate(force=True)
            return
        if not child.isalive():
            if auth_path(agent_name).exists():
                return
            raise RuntimeError(f"codex login exited before auth.json was created for {agent_name}")
        time.sleep(1)
    child.terminate(force=True)
    raise RuntimeError(f"timed out waiting for codex login to finish for {agent_name}")


def _wait_for_claude_auth(child: pexpect.spawn, agent_name: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _claude_logged_in(agent_name):
            child.terminate(force=True)
            return
        if not child.isalive():
            if _claude_logged_in(agent_name):
                return
            raise RuntimeError(f"claude auth login exited before auth completed for {agent_name}")
        time.sleep(1)
    child.terminate(force=True)
    raise RuntimeError(f"timed out waiting for Claude auth to finish for {agent_name}")


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


def _write_debug_artifacts(page: object, agent_name: str) -> None:
    debug_dir = state_root() / "debug" / agent_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    screenshot_path = debug_dir / f"{timestamp}.png"
    html_path = debug_dir / f"{timestamp}.html"
    meta_path = debug_dir / f"{timestamp}.txt"
    try:
        page.screenshot(path=str(screenshot_path), full_page=True)
    except Exception:
        pass
    try:
        html_path.write_text(page.content(), encoding="utf-8")
    except Exception:
        pass
    try:
        meta_path.write_text(f"url={page.url}\ntitle={page.title()}\n", encoding="utf-8")
    except Exception:
        pass
