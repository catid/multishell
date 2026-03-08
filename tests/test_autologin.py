import threading
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest
from playwright.sync_api import TimeoutError

from multishell.auth_flow_model import AuthModelFailure, AuthModelUnavailable
from multishell.autologin import (
    AgentCredentials,
    CodexDeviceAuthRateLimit,
    NoAuthLoginError,
    RetryableLoginError,
    run_auto_login,
    _complete_claude_google_sign_in,
    _auth_model_trace_logger,
    _advance_auth_flow,
    _normalize_openai_login_entry,
    _build_chrome_command,
    _build_claude_local_callback_url,
    _click_first,
    _click_google_and_capture_page,
    _enrich_login_error,
    _extract_claude_listen_port,
    _extract_device_flow,
    _login_codex_one,
    _resolve_browser_binary,
    _start_codex_device_auth,
    _wait_for_cdp_endpoint,
)
from multishell.config import AgentSpec


def test_extract_claude_listen_port_from_ss_output() -> None:
    ss_output = """
State  Recv-Q Send-Q Local Address:Port  Peer Address:PortProcess
LISTEN 0      511        127.0.0.1:40935      0.0.0.0:*    users:(("claude",pid=12345,fd=14))
"""
    assert _extract_claude_listen_port(ss_output, 12345) == 40935
    assert _extract_claude_listen_port(ss_output, 99999) is None


def test_build_claude_local_callback_url_rewrites_redirect_uri() -> None:
    manual_url = (
        "https://claude.ai/oauth/authorize"
        "?code=true"
        "&client_id=test-client"
        "&response_type=code"
        "&redirect_uri=https%3A%2F%2Fplatform.claude.com%2Foauth%2Fcode%2Fcallback"
        "&scope=user%3Aprofile"
        "&state=test-state"
    )
    rewritten = _build_claude_local_callback_url(manual_url, 40935)
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A40935%2Fcallback" in rewritten
    assert "client_id=test-client" in rewritten
    assert "state=test-state" in rewritten


class _NoPopupExpectation:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        raise TimeoutError("no popup")


class _SameTabGooglePage:
    def __init__(self) -> None:
        self.url = "https://chatgpt.com/"
        self.load_waits = 0

    def expect_popup(self, timeout: int):
        return _NoPopupExpectation()

    def wait_for_timeout(self, _timeout_ms: int) -> None:
        return None

    def wait_for_load_state(self, state: str, timeout: int) -> None:
        self.load_waits += 1


def test_click_google_and_capture_page_waits_for_same_tab_google_handoff(monkeypatch) -> None:
    page = _SameTabGooglePage()

    def fake_click_first(page_obj, selectors):
        page_obj.url = "https://accounts.google.com/v3/signin/identifier"

    monkeypatch.setattr("multishell.autologin._click_first", fake_click_first)
    monkeypatch.setattr("multishell.autologin._has_visible", lambda *_args, **_kwargs: False)

    returned = _click_google_and_capture_page(page)

    assert returned is page
    assert page.url.startswith("https://accounts.google.com/")
    assert page.load_waits == 1


class _Candidate:
    def __init__(self, visible: bool, clicked: list[int], index: int) -> None:
        self.visible = visible
        self.clicked = clicked
        self.index = index
        self.first = self

    def wait_for(self, state: str, timeout: int) -> None:
        if not self.visible:
            raise RuntimeError("hidden")

    def click(self) -> None:
        self.clicked.append(self.index)


class _LocatorList:
    def __init__(self, candidates: list[_Candidate]) -> None:
        self._candidates = candidates
        self.first = candidates[0]

    def count(self) -> int:
        return len(self._candidates)

    def nth(self, index: int) -> _Candidate:
        return self._candidates[index]


class _MultiMatchPage:
    def __init__(self, clicked: list[int]) -> None:
        self.url = "https://chatgpt.com/"
        self.clicked = clicked

    def locator(self, selector: str) -> _LocatorList:
        return _LocatorList([_Candidate(False, self.clicked, 0), _Candidate(True, self.clicked, 1)])


def test_click_first_skips_hidden_match_and_uses_visible_one() -> None:
    clicked: list[int] = []
    page = _MultiMatchPage(clicked)

    _click_first(page, ["button:has-text('Log in')"])

    assert clicked == [1]


class _FakePlaywrightContext:
    def __enter__(self):
        return SimpleNamespace()

    def __exit__(self, exc_type, exc, tb):
        return False


def test_run_auto_login_emits_progress_lines(monkeypatch, capsys) -> None:
    credential = AgentCredentials(
        spec=AgentSpec(
            name="worker-1",
            account_email="worker@example.com",
            role="worker",
            personality="Test worker.",
            accent_color=2,
            account_key="account-1",
        ),
        password="secret",
    )

    monkeypatch.setattr("multishell.autologin.apply_node_warning_suppression", lambda: None)
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: _FakePlaywrightContext())
    monkeypatch.setattr("multishell.autologin._login_codex_one", lambda *_args, **_kwargs: None)

    run_auto_login([credential])

    out = capsys.readouterr().out
    assert "auto-login 1/1: worker-1 (codex, worker@example.com)" in out


def test_run_auto_login_continues_after_lane_failure(monkeypatch, capsys) -> None:
    first = AgentCredentials(
        spec=AgentSpec(
            name="worker-1",
            account_email="worker1@example.com",
            role="worker",
            personality="Test worker.",
            accent_color=2,
            account_key="account-1",
        ),
        password="secret-1",
    )
    second = AgentCredentials(
        spec=AgentSpec(
            name="worker-2",
            account_email="worker2@example.com",
            role="worker",
            personality="Test worker.",
            accent_color=3,
            account_key="account-2",
        ),
        password="secret-2",
    )
    seen: list[str] = []

    monkeypatch.setattr("multishell.autologin.apply_node_warning_suppression", lambda: None)
    monkeypatch.setattr("multishell.autologin.apply_playwright_browser_path", lambda: None)
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: _FakePlaywrightContext())

    def fake_login(_playwright, credential, _timeout_seconds, headed):
        seen.append(credential.spec.name)
        if credential.spec.name == "worker-1":
            raise NoAuthLoginError("Wrong password")

    monkeypatch.setattr("multishell.autologin._login_codex_one", fake_login)

    with pytest.raises(RuntimeError, match="auto-login completed with failures:"):
        run_auto_login([first, second], max_parallel=1)

    out = capsys.readouterr().out
    assert seen == ["worker-1", "worker-2"]
    assert "[worker-1] login failed [no_auth]: Wrong password" in out
    assert "auto-login 2/2: worker-2 (codex, worker2@example.com)" in out


def test_run_auto_login_retries_retryable_failures(monkeypatch, capsys) -> None:
    credential = AgentCredentials(
        spec=AgentSpec(
            name="worker-1",
            account_email="worker@example.com",
            role="worker",
            personality="Test worker.",
            accent_color=2,
            account_key="account-1",
        ),
        password="secret",
    )
    attempts = {"count": 0}
    sleeps: list[int] = []
    resets: list[str] = []

    monkeypatch.setattr("multishell.autologin.LOGIN_ERROR_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr("multishell.autologin.LOGIN_ERROR_RETRY_DELAY_SECONDS", 15)
    monkeypatch.setattr("multishell.autologin.apply_node_warning_suppression", lambda: None)
    monkeypatch.setattr("multishell.autologin.apply_playwright_browser_path", lambda: None)
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: _FakePlaywrightContext())
    monkeypatch.setattr("multishell.autologin.time.sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr("multishell.autologin._reset_login_attempt_state", lambda agent_name: resets.append(agent_name))

    def fake_login(_playwright, _credential, _timeout_seconds, headed):
        attempts["count"] += 1
        raise RetryableLoginError("temporary browser error")

    monkeypatch.setattr("multishell.autologin._login_codex_one", fake_login)

    with pytest.raises(RuntimeError, match="worker-1 \\[error\\]: temporary browser error"):
        run_auto_login([credential], max_parallel=1)

    out = capsys.readouterr().out
    assert attempts["count"] == 3
    assert sleeps == [15, 15]
    assert resets == ["worker-1", "worker-1"]
    assert "retryable login error; retrying in 15s (attempt 2/3): temporary browser error" in out
    assert "retryable login error; retrying in 15s (attempt 3/3): temporary browser error" in out
    assert "[worker-1] login failed [error]: temporary browser error" in out


def test_run_auto_login_does_not_retry_no_auth_failures(monkeypatch, capsys) -> None:
    credential = AgentCredentials(
        spec=AgentSpec(
            name="worker-1",
            account_email="worker@example.com",
            role="worker",
            personality="Test worker.",
            accent_color=2,
            account_key="account-1",
        ),
        password="secret",
    )
    attempts = {"count": 0}
    sleeps: list[int] = []

    monkeypatch.setattr("multishell.autologin.LOGIN_ERROR_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr("multishell.autologin.apply_node_warning_suppression", lambda: None)
    monkeypatch.setattr("multishell.autologin.apply_playwright_browser_path", lambda: None)
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: _FakePlaywrightContext())
    monkeypatch.setattr("multishell.autologin.time.sleep", lambda seconds: sleeps.append(seconds))

    def fake_login(_playwright, _credential, _timeout_seconds, headed):
        attempts["count"] += 1
        raise NoAuthLoginError("Wrong password")

    monkeypatch.setattr("multishell.autologin._login_codex_one", fake_login)

    with pytest.raises(RuntimeError, match="worker-1 \\[no_auth\\]: Wrong password"):
        run_auto_login([credential], max_parallel=1)

    out = capsys.readouterr().out
    assert attempts["count"] == 1
    assert sleeps == []
    assert "retrying in" not in out
    assert "[worker-1] login failed [no_auth]: Wrong password" in out


def test_run_auto_login_limits_parallelism(monkeypatch) -> None:
    credentials = [
        AgentCredentials(
            spec=AgentSpec(
                name=f"claude-worker-{index}",
                account_email=f"worker{index}@example.com",
                role="claude-worker",
                personality="Claude worker.",
                accent_color=index,
                engine="claude",
                account_key=f"account-{index}",
            ),
            password=f"secret-{index}",
        )
        for index in range(1, 5)
    ]
    active = 0
    max_active = 0
    lock = threading.Lock()

    monkeypatch.setattr("multishell.autologin.apply_node_warning_suppression", lambda: None)
    monkeypatch.setattr("multishell.autologin.apply_playwright_browser_path", lambda: None)
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: _FakePlaywrightContext())

    def fake_login(_playwright, _credential, _timeout_seconds, headed):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1

    monkeypatch.setattr("multishell.autologin._login_claude_one", fake_login)

    run_auto_login(credentials, max_parallel=2)

    assert max_active == 2


def test_run_auto_login_limits_codex_parallelism_while_filling_other_slots(monkeypatch) -> None:
    credentials = [
        AgentCredentials(
            spec=AgentSpec(
                name="worker-1",
                account_email="worker1@example.com",
                role="worker",
                personality="Test worker.",
                accent_color=1,
                account_key="account-1",
            ),
            password="secret-1",
        ),
        AgentCredentials(
            spec=AgentSpec(
                name="worker-2",
                account_email="worker2@example.com",
                role="worker",
                personality="Test worker.",
                accent_color=2,
                account_key="account-2",
            ),
            password="secret-2",
        ),
        AgentCredentials(
            spec=AgentSpec(
                name="worker-3",
                account_email="worker3@example.com",
                role="worker",
                personality="Test worker.",
                accent_color=3,
                account_key="account-3",
            ),
            password="secret-3",
        ),
        AgentCredentials(
            spec=AgentSpec(
                name="claude-worker-1",
                account_email="claude1@example.com",
                role="claude-worker",
                personality="Claude worker.",
                accent_color=4,
                engine="claude",
                account_key="account-4",
            ),
            password="secret-4",
        ),
        AgentCredentials(
            spec=AgentSpec(
                name="claude-worker-2",
                account_email="claude2@example.com",
                role="claude-worker",
                personality="Claude worker.",
                accent_color=5,
                engine="claude",
                account_key="account-5",
            ),
            password="secret-5",
        ),
        AgentCredentials(
            spec=AgentSpec(
                name="claude-worker-3",
                account_email="claude3@example.com",
                role="claude-worker",
                personality="Claude worker.",
                accent_color=6,
                engine="claude",
                account_key="account-6",
            ),
            password="secret-6",
        ),
    ]
    active_total = 0
    active_by_engine = {"codex": 0, "claude": 0}
    max_total = 0
    max_codex = 0
    lock = threading.Lock()

    monkeypatch.setattr("multishell.autologin.apply_node_warning_suppression", lambda: None)
    monkeypatch.setattr("multishell.autologin.apply_playwright_browser_path", lambda: None)
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: _FakePlaywrightContext())

    def fake_login(engine: str):
        def inner(_playwright, _credential, _timeout_seconds, headed):
            nonlocal active_total, max_total, max_codex
            with lock:
                active_total += 1
                active_by_engine[engine] += 1
                max_total = max(max_total, active_total)
                max_codex = max(max_codex, active_by_engine["codex"])
            time.sleep(0.05)
            with lock:
                active_total -= 1
                active_by_engine[engine] -= 1

        return inner

    monkeypatch.setattr("multishell.autologin._login_codex_one", fake_login("codex"))
    monkeypatch.setattr("multishell.autologin._login_claude_one", fake_login("claude"))

    run_auto_login(credentials, max_parallel=4)

    assert max_total == 4
    assert max_codex == 1


def test_auth_model_trace_logger_writes_debug_log(monkeypatch, tmp_path) -> None:
    seen: list[tuple[str, str]] = []

    monkeypatch.setattr("multishell.autologin.state_root", lambda: tmp_path)
    monkeypatch.setattr("multishell.autologin.time.time", lambda: 1234567890)
    monkeypatch.setattr("multishell.autologin._log_progress", lambda agent, message: seen.append((agent, message)))

    logger, path = _auth_model_trace_logger("worker-1", "codex-google", "worker-1")
    logger("auth model raw response: {\"action\":\"done\"}")

    assert path == tmp_path / "debug" / "worker-1" / "1234567890-codex-google.auth-model.log"
    content = path.read_text(encoding="utf-8")
    assert "agent=worker-1" in content
    assert "flow=codex-google" in content
    assert "auth model raw response:" in content
    assert seen[0] == ("worker-1", f"auth model trace: {path}")
    assert seen[1] == ("worker-1", "auth model raw response: {\"action\":\"done\"}")


def test_auth_model_trace_logger_handles_concurrent_writes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("multishell.autologin.state_root", lambda: tmp_path)
    monkeypatch.setattr("multishell.autologin.time.time", lambda: 1234567890)
    logger, path = _auth_model_trace_logger("worker-1", "codex-google", None)

    def write_line(index: int) -> None:
        logger(f"line {index}")

    threads = [threading.Thread(target=write_line, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    content = path.read_text(encoding="utf-8")
    for index in range(6):
        assert f"line {index}" in content


class _DebugPage:
    url = "https://accounts.google.com/"

    def screenshot(self, path: str, full_page: bool) -> None:
        Path(path).write_bytes(b"png")

    def content(self) -> str:
        return "<html>debug</html>"

    def title(self) -> str:
        return "Debug Title"


def test_enrich_login_error_includes_debug_dir(monkeypatch, tmp_path) -> None:
    seen: list[tuple[str, str]] = []

    monkeypatch.setattr("multishell.autologin.state_root", lambda: tmp_path)
    monkeypatch.setattr("multishell.autologin.time.time", lambda: 1234567890)
    monkeypatch.setattr("multishell.autologin._log_progress", lambda agent, message: seen.append((agent, message)))

    wrapped = _enrich_login_error(RuntimeError("Wrong password"), _DebugPage(), "worker-1")

    assert isinstance(wrapped, RetryableLoginError)
    assert "Wrong password" in str(wrapped)
    assert f"(debug: {tmp_path / 'debug' / 'worker-1'})" in str(wrapped)
    assert seen == [("worker-1", f"debug artifacts: {tmp_path / 'debug' / 'worker-1'}")]
    assert (tmp_path / "debug" / "worker-1" / "1234567890.png").exists()
    assert (tmp_path / "debug" / "worker-1" / "1234567890.html").exists()
    assert (tmp_path / "debug" / "worker-1" / "1234567890.txt").exists()


class _ClosedDebugPage:
    @property
    def url(self) -> str:
        raise RuntimeError("page closed")

    def screenshot(self, path: str, full_page: bool) -> None:
        raise RuntimeError("page closed")

    def content(self) -> str:
        raise RuntimeError("page closed")

    def title(self) -> str:
        raise RuntimeError("page closed")


def test_enrich_login_error_writes_meta_when_page_is_closed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("multishell.autologin.state_root", lambda: tmp_path)
    monkeypatch.setattr("multishell.autologin.time.time", lambda: 1234567890)

    wrapped = _enrich_login_error(RuntimeError("timed out"), _ClosedDebugPage(), "worker-1")

    meta = (tmp_path / "debug" / "worker-1" / "1234567890.txt").read_text(encoding="utf-8")
    assert isinstance(wrapped, RetryableLoginError)
    assert "timed out" in str(wrapped)
    assert "screenshot_error=page closed" in meta
    assert "html_error=page closed" in meta
    assert "url_error=page closed" in meta
    assert "title_error=page closed" in meta


def test_enrich_login_error_preserves_no_auth_subclass(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("multishell.autologin.state_root", lambda: tmp_path)
    monkeypatch.setattr("multishell.autologin.time.time", lambda: 1234567890)

    wrapped = _enrich_login_error(NoAuthLoginError("Wrong password"), _DebugPage(), "worker-1")

    assert isinstance(wrapped, NoAuthLoginError)
    assert "Wrong password" in str(wrapped)


class _FakeCodexChild:
    pid = 12345

    def __init__(self, output: str = "https://auth.openai.com/codex/device K8OE-9GJ2U") -> None:
        self.terminated = False
        self.output = output

    def read_nonblocking(self, size: int, timeout: int) -> str:
        value, self.output = self.output, ""
        return value

    def isalive(self) -> bool:
        return not self.terminated

    def terminate(self, force: bool = False) -> None:
        self.terminated = True


class _LiveDebugPage(_DebugPage):
    pass


class _ClosingChromeContext:
    def __init__(self, page: _LiveDebugPage) -> None:
        self.page = page

    def __enter__(self):
        return None, None, self.page

    def __exit__(self, exc_type, exc, tb):
        self.page.url = "closed"
        return False


def test_login_codex_one_captures_debug_before_browser_context_closes(monkeypatch, tmp_path) -> None:
    credential = AgentCredentials(
        spec=AgentSpec(
            name="worker-1",
            account_email="worker@example.com",
            role="worker",
            personality="Test worker.",
            accent_color=2,
            account_key="account-1",
        ),
        password="secret",
    )
    page = _LiveDebugPage()

    monkeypatch.setattr("multishell.autologin.state_root", lambda: tmp_path)
    monkeypatch.setattr("multishell.autologin.time.time", lambda: 1234567890)
    monkeypatch.setattr("multishell.autologin._ensure_home", lambda _name: None)
    monkeypatch.setattr("multishell.autologin.auth_path", lambda _name: tmp_path / "missing-auth.json")
    monkeypatch.setattr(
        "multishell.autologin._start_codex_device_auth",
        lambda _agent_name, _env: (_FakeCodexChild(), "https://auth.openai.com/codex/device", "K8OE-9GJ2U"),
    )
    monkeypatch.setattr(
        "multishell.autologin._isolated_chrome",
        lambda *_args, **_kwargs: _ClosingChromeContext(page),
    )
    monkeypatch.setattr(
        "multishell.autologin._complete_openai_google_sign_in",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("browser flow failed")),
    )

    with pytest.raises(RuntimeError, match="browser flow failed"):
        _login_codex_one(SimpleNamespace(), credential, timeout_seconds=30, headed=False)

    assert (tmp_path / "debug" / "worker-1" / "1234567890.png").exists()
    assert (tmp_path / "debug" / "worker-1" / "1234567890.html").exists()
    assert (tmp_path / "debug" / "worker-1" / "1234567890.txt").exists()


def test_extract_device_flow_raises_rate_limit_error() -> None:
    with pytest.raises(CodexDeviceAuthRateLimit, match="429 Too Many Requests"):
        _extract_device_flow("Error logging in with device code: device code request failed with status 429 Too Many Requests")


def test_start_codex_device_auth_retries_rate_limited_output(monkeypatch) -> None:
    children = iter(
        [
            _FakeCodexChild("Error logging in with device code: device code request failed with status 429 Too Many Requests"),
            _FakeCodexChild("https://auth.openai.com/codex/device K8OE-9GJ2U"),
        ]
    )
    sleeps: list[int] = []
    logs: list[tuple[str, str]] = []

    monkeypatch.setattr("multishell.autologin.pexpect.spawn", lambda *args, **kwargs: next(children))
    monkeypatch.setattr("multishell.autologin._collect_child_output", lambda child, seconds: child.read_nonblocking(4096, 1))
    monkeypatch.setattr("multishell.autologin.time.sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr("multishell.autologin._log_progress", lambda agent, message: logs.append((agent, message)))

    child, url, device_code = _start_codex_device_auth("worker-4", {"HOME": "/tmp/home"}, max_attempts=2)

    assert url == "https://auth.openai.com/codex/device"
    assert device_code == "K8OE-9GJ2U"
    assert child.isalive() is True
    assert sleeps == [10]
    assert logs == [("worker-4", "codex device-auth hit a rate limit; retrying in 10s (attempt 2/2)")]


def test_start_codex_device_auth_surfaces_final_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr(
        "multishell.autologin.pexpect.spawn",
        lambda *args, **kwargs: _FakeCodexChild("Error logging in with device code: device code request failed with status 429 Too Many Requests"),
    )
    monkeypatch.setattr("multishell.autologin._collect_child_output", lambda child, seconds: child.read_nonblocking(4096, 1))
    monkeypatch.setattr("multishell.autologin.time.sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(RetryableLoginError, match="codex device-auth hit OpenAI rate limits after 2 attempts"):
        _start_codex_device_auth("worker-4", {"HOME": "/tmp/home"}, max_attempts=2)



def test_build_chrome_command_uses_real_headless_mode(tmp_path) -> None:
    command = _build_chrome_command("/usr/bin/google-chrome", tmp_path / "profile", 9222, headed=False)

    assert command[0] == "/usr/bin/google-chrome"
    assert "--headless=new" in command
    assert "--disable-gpu" in command
    assert "--disable-blink-features=AutomationControlled" in command
    assert any(arg.startswith("--user-agent=Mozilla/5.0") and " Chrome/" in arg and "HeadlessChrome" not in arg for arg in command)
    assert "about:blank" == command[-1]


def test_build_chrome_command_can_run_headed_when_requested(tmp_path) -> None:
    command = _build_chrome_command("/usr/bin/google-chrome", tmp_path / "profile", 9222, headed=True)

    assert "--headless=new" not in command


class _SecurityInterstitialBody:
    def inner_text(self, timeout: int = 1000) -> str:
        return "Performing security verification. This website uses a security service to protect against malicious bots."


class _SecurityInterstitialPage:
    url = "https://auth.openai.com/api/oauth/oauth2/auth"

    def locator(self, selector: str):
        assert selector == "body"
        return _SecurityInterstitialBody()

    def title(self) -> str:
        return "Just a moment..."


def test_normalize_openai_login_entry_allows_openai_security_interstitial(monkeypatch) -> None:
    monkeypatch.setattr(
        "multishell.autologin._wait_for_live_login_surface",
        lambda _page, timeout_seconds: (_ for _ in ()).throw(RuntimeError("timed out waiting for auth challenge to clear")),
    )

    _normalize_openai_login_entry(_SecurityInterstitialPage())


def test_resolve_browser_binary_prefers_playwright_bundled_binary(monkeypatch, tmp_path) -> None:
    bundled = tmp_path / "chrome"
    bundled.write_text("", encoding="utf-8")
    playwright = SimpleNamespace(chromium=SimpleNamespace(executable_path=str(bundled)))

    monkeypatch.delenv("MULTISHELL_BROWSER_BINARY", raising=False)
    monkeypatch.setattr("multishell.autologin.shutil.which", lambda _name: None)

    assert _resolve_browser_binary(playwright) == str(bundled)


def test_resolve_browser_binary_uses_override_when_configured(monkeypatch, tmp_path) -> None:
    override = tmp_path / "custom-browser"
    override.write_text("", encoding="utf-8")
    playwright = SimpleNamespace(chromium=SimpleNamespace(executable_path=""))

    monkeypatch.setenv("MULTISHELL_BROWSER_BINARY", str(override))

    assert _resolve_browser_binary(playwright) == str(override)


class _ExitedBrowserProcess:
    pid = 43210

    def poll(self) -> int:
        return 127


class _OutputTailStub:
    def tail(self, *, limit: int = 8) -> str:
        return "chrome: error while loading shared libraries: libgtk-3.so.0"


def test_wait_for_cdp_endpoint_reports_browser_exit_details(monkeypatch) -> None:
    def fake_urlopen(url: str, timeout: int):
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="browser exited before opening DevTools endpoint on port 9222 with exit code 127") as excinfo:
        _wait_for_cdp_endpoint(
            9222,
            timeout_seconds=30,
            progress_label="worker-1",
            process=_ExitedBrowserProcess(),
            output_tail=_OutputTailStub(),
        )
    assert "libgtk-3.so.0" in str(excinfo.value)


def test_advance_auth_flow_prefers_model_controller(monkeypatch) -> None:
    credential = AgentCredentials(
        spec=AgentSpec(
            name="worker-1",
            account_email="worker@example.com",
            role="worker",
            personality="Test worker.",
            accent_color=2,
            account_key="account-1",
        ),
        password="secret",
    )
    page = SimpleNamespace(url="https://accounts.google.com/v3/signin/identifier")
    seen: dict[str, object] = {"model": 0, "code": None, "continued": False}

    def fake_model(*_args, **_kwargs):
        seen["model"] += 1
        return True

    monkeypatch.setattr("multishell.autologin.drive_google_auth_with_model", fake_model)
    monkeypatch.setattr("multishell.autologin._enter_device_code", lambda _page, code: seen.__setitem__("code", code))
    monkeypatch.setattr("multishell.autologin._click_enabled_continue", lambda _page: seen.__setitem__("continued", True))

    _advance_auth_flow(page, credential, "ABCD-EFGHI")

    assert seen["model"] == 1
    assert seen["code"] is None
    assert seen["continued"] is False


def test_advance_auth_flow_raises_when_model_unavailable(monkeypatch) -> None:
    credential = AgentCredentials(
        spec=AgentSpec(
            name="worker-1",
            account_email="worker@example.com",
            role="worker",
            personality="Test worker.",
            accent_color=2,
            account_key="account-1",
        ),
        password="secret",
    )
    page = SimpleNamespace(url="https://accounts.google.com/v3/signin/identifier")
    monkeypatch.setattr(
        "multishell.autologin.drive_google_auth_with_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AuthModelUnavailable("offline")),
    )
    with pytest.raises(RetryableLoginError, match="auth model unavailable for Codex/Google flow: offline"):
        _advance_auth_flow(page, credential, "ABCD-EFGHI")


class _ClaudeFlowPage:
    def __init__(self) -> None:
        self.url = "about:blank"
        self.context = SimpleNamespace(pages=[self])

    def goto(self, url: str, wait_until: str = "domcontentloaded") -> None:
        self.url = url

    def wait_for_load_state(self, state: str, timeout: int) -> None:
        return None


def test_complete_claude_google_sign_in_reenters_model_on_claude_handoff(monkeypatch) -> None:
    credential = AgentCredentials(
        spec=AgentSpec(
            name="claude-worker-4",
            account_email="worker4@example.com",
            role="worker",
            personality="Test worker.",
            accent_color=5,
            account_key="account-4",
        ),
        password="secret",
    )
    page = _ClaudeFlowPage()
    drive_calls: list[str] = []

    monkeypatch.setattr("multishell.autologin._has_visible", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("multishell.autologin._auth_model_trace_logger", lambda *_args, **_kwargs: (lambda *_a, **_k: None, Path("/tmp/trace.log")))
    monkeypatch.setattr("multishell.autologin._body_text", lambda _page: "")
    monkeypatch.setattr("multishell.autologin._page_title", lambda _page: "Claude")
    monkeypatch.setattr("multishell.autologin.time.sleep", lambda *_args, **_kwargs: None)

    def fake_capture_snapshot(flow_page):
        return {
            "url": flow_page.url,
            "title": "Claude",
            "body_text": "Continue with Google" if "login" in flow_page.url else "Claude Code would like to connect",
            "elements": [{"id": "ms-auth-1", "tag": "button", "type": "", "text": "Continue"}],
        }

    def fake_drive(flow_page, **_kwargs):
        drive_calls.append(flow_page.url)
        if len(drive_calls) == 1:
            flow_page.url = "https://claude.ai/oauth/authorize?code=true"
        else:
            flow_page.url = "https://platform.claude.com/oauth/code/success"
        return True

    monkeypatch.setattr("multishell.autologin.capture_auth_snapshot", fake_capture_snapshot)
    monkeypatch.setattr("multishell.autologin.drive_google_auth_with_model", fake_drive)
    monkeypatch.setattr("multishell.autologin._claude_logged_in", lambda _name: len(drive_calls) >= 2)

    result = _complete_claude_google_sign_in(
        page,
        "https://claude.ai/login?returnTo=%2Foauth%2Fauthorize",
        credential,
    )

    assert result is None
    assert drive_calls == [
        "https://claude.ai/login?returnTo=%2Foauth%2Fauthorize",
        "https://claude.ai/oauth/authorize?code=true",
    ]


def test_advance_auth_flow_surfaces_model_failure(monkeypatch) -> None:
    credential = AgentCredentials(
        spec=AgentSpec(
            name="worker-1",
            account_email="worker@example.com",
            role="worker",
            personality="Test worker.",
            accent_color=2,
            account_key="account-1",
        ),
        password="secret",
    )
    page = SimpleNamespace(url="https://accounts.google.com/v3/signin/challenge/pwd")
    monkeypatch.setattr(
        "multishell.autologin.drive_google_auth_with_model",
        lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(AuthModelFailure("Google rejected the configured password or requested extra verification: Wrong password", failure_kind="no_auth")),
    )

    with pytest.raises(NoAuthLoginError, match="Google rejected the configured password"):
        _advance_auth_flow(page, credential, "ABCD-EFGHI")


def test_advance_auth_flow_surfaces_manual_verification_failure(monkeypatch) -> None:
    credential = AgentCredentials(
        spec=AgentSpec(
            name="worker-1",
            account_email="worker@example.com",
            role="worker",
            personality="Test worker.",
            accent_color=2,
            account_key="account-1",
        ),
        password="secret",
    )
    page = SimpleNamespace(url="https://accounts.google.com/v3/signin/challenge/selection")
    monkeypatch.setattr(
        "multishell.autologin.drive_google_auth_with_model",
        lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(AuthModelFailure("Google requires manual verification and cannot be automated here: Verify it's you", failure_kind="error")),
    )

    with pytest.raises(RetryableLoginError, match="Google requires manual verification"):
        _advance_auth_flow(page, credential, "ABCD-EFGHI")
