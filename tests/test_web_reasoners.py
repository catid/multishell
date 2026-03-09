from __future__ import annotations

import time
from threading import Event
from types import SimpleNamespace

import pytest

from multishell.web_reasoners import (
    CHATGPT_LOGIN_URL,
    _extract_last_assistant_message,
    WebReasonerEvent,
    WebNavigatorError,
    WebReasonerManager,
    _prepare_chatgpt_pro_surface,
    _prepare_gemini_deepthink_surface,
    _ensure_chatgpt_pro_workspace,
    _ensure_chatgpt_logged_in,
    _ensure_google_logged_in,
    _preferred_openai_flow_page,
    _wait_for_chatgpt_login_completion,
    _wait_for_stable_response,
    _web_profile_name,
    _submit_prompt,
)


class _SuccessManager(WebReasonerManager):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.started = Event()
        self.release = Event()

    def _execute_job(self, job) -> str:
        self.started.set()
        assert self.release.wait(timeout=2)
        return f"answer for {job.provider}"


class _CancelManager(WebReasonerManager):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.started = Event()

    def _execute_job(self, job) -> str:
        self.started.set()
        cancel_flag = self._cancel_flags[job.id]
        deadline = time.time() + 2
        while time.time() < deadline:
            if cancel_flag.wait(timeout=0.01):
                raise RuntimeError("canceled")
        return "unexpected"


def _wait_for_status(manager: WebReasonerManager, job_id: str, expected: str, timeout: float = 3.0) -> dict[str, object]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = manager.snapshot(job_id)
        if snapshot["status"] == expected:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {expected!r}; last snapshot={manager.snapshot(job_id)!r}")


def test_start_job_supports_agent_alias_and_completes() -> None:
    events: list[WebReasonerEvent] = []
    manager = _SuccessManager(callback=events.append)

    job = manager.start_job("chatgpt_pro", agent="manager", prompt="Plan it", label="planner", timeout_seconds=45)
    assert job["agent"] == "manager"
    assert job["label"] == "planner"
    assert job["timeout_seconds"] == 45

    assert manager.started.wait(timeout=1)
    manager.release.set()
    snapshot = _wait_for_status(manager, str(job["job_id"]), "completed")

    assert snapshot["result_text"] == "answer for chatgpt_pro"
    assert [event.kind for event in events] == ["job_started", "job_completed"]


def test_cancel_job_finishes_as_canceled_without_completed_event() -> None:
    events: list[WebReasonerEvent] = []
    manager = _CancelManager(callback=events.append)

    job = manager.start_job("gemini_deepthink", "manager", "Think slowly")
    assert manager.started.wait(timeout=1)

    cancel_snapshot = manager.cancel_job(str(job["job_id"]))
    assert cancel_snapshot is not None
    assert cancel_snapshot["status"] in {"canceling", "canceled"}

    snapshot = _wait_for_status(manager, str(job["job_id"]), "canceled")
    assert snapshot["error"] == "canceled"
    assert [event.kind for event in events] == ["job_started", "job_canceled"]


def test_start_job_validates_required_fields() -> None:
    manager = WebReasonerManager()

    with pytest.raises(Exception, match="unsupported web reasoner provider"):
        manager.start_job("unknown", "manager", "hello")

    with pytest.raises(Exception, match="account_agent is required"):
        manager.start_job("chatgpt_pro", "", "hello")

    with pytest.raises(Exception, match="prompt is required"):
        manager.start_job("chatgpt_pro", "manager", "   ")


def test_web_profile_name_is_stable_per_provider_and_account() -> None:
    assert _web_profile_name("chatgpt_pro", "manager") == "web-chatgpt_pro-manager"
    assert _web_profile_name("chatgpt_pro", "manager") == _web_profile_name("chatgpt_pro", "manager")
    assert _web_profile_name("chatgpt_pro", "manager", "12345678-aaaa-bbbb-cccc-ddddeeeeffff") == "web-chatgpt_pro-manager-12345678"


class _FakeLocator:
    def __init__(self, selector: str) -> None:
        self.selector = selector
        self.first = self

    def click(self, timeout: int | None = None) -> None:
        if self.selector.startswith("text="):
            raise AssertionError("password screen should not click the account picker row")


class _FakePage:
    def __init__(self, url: str) -> None:
        self.url = url
        self.gotos: list[str] = []
        self.context = SimpleNamespace(pages=[self])

    def locator(self, selector: str) -> _FakeLocator:
        return _FakeLocator(selector)

    def goto(self, url: str, wait_until: str | None = None, timeout: int | None = None) -> None:
        self.url = url
        self.gotos.append(url)

    def wait_for_timeout(self, _timeout_ms: int) -> None:
        return None

    def is_closed(self) -> bool:
        return False


class _FakePromptKeyboard:
    def __init__(self, page: "_FakePromptPage") -> None:
        self.page = page

    def press(self, key: str) -> None:
        self.page.keys.append(key)

    def type(self, text: str, delay: int | None = None) -> None:
        self.page.typed.append((text, delay))
        self.page.composer_value = text


class _FakePromptLocator:
    def __init__(self, page: "_FakePromptPage", kind: str) -> None:
        self.page = page
        self.kind = kind
        self.first = self

    def wait_for(self, state: str = "visible", timeout: int | None = None) -> None:
        if self.kind == "send" and not self.page.send_visible:
            raise RuntimeError("not visible")

    def click(self, timeout: int | None = None) -> None:
        if self.kind == "send":
            self.page.send_clicked += 1
            self.page.composer_value = ""
            return
        self.page.composer_clicked += 1

    def fill(self, value: str) -> None:
        self.page.composer_value = value

    def input_value(self, timeout: int | None = None) -> str:
        return self.page.composer_value

    def inner_text(self, timeout: int | None = None) -> str:
        return self.page.composer_value

    def text_content(self, timeout: int | None = None) -> str:
        return self.page.composer_value

    def count(self) -> int:
        if self.kind == "send":
            return 1 if self.page.send_visible else 0
        return 1

    def nth(self, index: int) -> "_FakePromptLocator":
        return self


class _FakePromptPage:
    def __init__(self) -> None:
        self.composer_value = ""
        self.send_visible = True
        self.send_clicked = 0
        self.composer_clicked = 0
        self.keys: list[str] = []
        self.typed: list[tuple[str, int | None]] = []
        self.keyboard = _FakePromptKeyboard(self)

    def locator(self, selector: str) -> _FakePromptLocator:
        if selector == "textarea:visible":
            return _FakePromptLocator(self, "textarea")
        if selector == "[contenteditable='true']:visible, [role='textbox']:visible":
            raise RuntimeError("contenteditable not used")
        return _FakePromptLocator(self, "send")

    def wait_for_timeout(self, _timeout_ms: int) -> None:
        return None


class _FakeStablePage:
    def wait_for_timeout(self, _timeout_ms: int) -> None:
        return None


def test_google_password_screen_prefers_password_input_over_account_picker(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[tuple[str, tuple[str, ...]]] = []
    page = _FakePage("https://accounts.google.com/v3/signin/challenge/pwd")

    monkeypatch.setattr(
        "multishell.web_reasoners._body_text",
        lambda _: (
            "Sign in with Google\n"
            "Hi bot\n"
            "bot@kuang2.ai\n"
            "Enter your password\n"
            "Next"
        ),
    )
    monkeypatch.setattr("multishell.web_reasoners._page_title", lambda _: "Hi bot")
    monkeypatch.setattr(
        "multishell.web_reasoners._has_visible",
        lambda _page, selectors, timeout_ms: any("password" in selector or "Passwd" in selector for selector in selectors),
    )

    def fake_fill_first(_page: object, selectors: list[str], value: str) -> None:
        recorded.append((value, tuple(selectors)))
        raise RuntimeError("stop after password fill")

    monkeypatch.setattr("multishell.web_reasoners._fill_first", fake_fill_first)
    monkeypatch.setattr("multishell.web_reasoners._click_first", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("multishell.web_reasoners._click_optional", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="stop after password fill"):
        _ensure_google_logged_in(page, "bot@kuang2.ai", "secret", Event())

    assert recorded == [
        (
            "secret",
            (
                "input[type='password']:visible",
                "input[name='Passwd']:visible",
                "input[autocomplete='current-password']:visible",
            ),
        )
    ]


def test_chatgpt_workspace_reuses_openai_consent_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage("https://auth.openai.com/authorize")
    consent_calls: list[str] = []
    body_state = {"value": "Sign in to ChatGPT with ChatGPT"}

    monkeypatch.setattr("multishell.web_reasoners._body_text", lambda _page: body_state["value"])
    monkeypatch.setattr("multishell.web_reasoners._has_visible", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "multishell.web_reasoners._complete_workspace_consent",
        lambda _page: consent_calls.append("consent") or body_state.__setitem__("value", "ChatGPT 5.4 Pro"),
    )
    monkeypatch.setattr("multishell.web_reasoners._click_optional", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("multishell.web_reasoners._check_cancel", lambda _flag: None)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)
    page.wait_for_timeout = lambda *_args, **_kwargs: None

    _ensure_chatgpt_pro_workspace(page, Event())

    assert consent_calls == ["consent"]


def test_chatgpt_workspace_accepts_ready_composer_without_exact_model_label(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage("https://chatgpt.com/")
    clicks: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        "multishell.web_reasoners._body_text",
        lambda _page: "Conversation with ChatGPT\nAsk anything",
    )
    monkeypatch.setattr(
        "multishell.web_reasoners._has_visible",
        lambda _page, selectors, timeout_ms: any(
            token in " ".join(selectors) for token in ("textarea", "contenteditable", "textbox")
        ),
    )
    monkeypatch.setattr(
        "multishell.web_reasoners._click_optional",
        lambda _page, selectors: clicks.append(tuple(selectors)),
    )
    monkeypatch.setattr("multishell.web_reasoners._dismiss_managed_profile_notice_in_context", lambda _page: None)
    monkeypatch.setattr("multishell.web_reasoners._preferred_openai_flow_page", lambda page_obj: page_obj)
    monkeypatch.setattr("multishell.web_reasoners._check_cancel", lambda _flag: None)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)
    page.wait_for_timeout = lambda *_args, **_kwargs: None

    _ensure_chatgpt_pro_workspace(page, Event())

    assert len(clicks) == 2


def test_chatgpt_login_starts_from_direct_openai_auth_url(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage("https://chatgpt.com/")
    normalize_calls: list[str] = []
    google_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "multishell.web_reasoners._body_text",
        lambda _page: "Log in to get answers based on saved chats, plus create images and upload files.",
    )
    monkeypatch.setattr("multishell.web_reasoners._has_visible", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        "multishell.web_reasoners._normalize_openai_login_entry",
        lambda _page: normalize_calls.append(_page.url),
    )
    monkeypatch.setattr(
        "multishell.web_reasoners._ensure_google_logged_in",
        lambda _page, email, password, _cancel: google_calls.append((email, password)),
    )
    monkeypatch.setattr("multishell.web_reasoners._wait_for_chatgpt_login_completion", lambda *_args, **_kwargs: None)

    _ensure_chatgpt_logged_in(page, "bot@kuang2.ai", "secret", Event())

    assert page.gotos == [CHATGPT_LOGIN_URL]
    assert normalize_calls == [CHATGPT_LOGIN_URL]
    assert google_calls == [("bot@kuang2.ai", "secret")]


def test_chatgpt_login_prefers_homepage_google_modal_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage("https://chatgpt.com/")
    clicks: list[tuple[str, ...]] = []
    google_calls: list[tuple[str, str]] = []

    monkeypatch.setattr("multishell.web_reasoners._body_text", lambda _page: "Log in")

    def fake_has_visible(_page: object, selectors: list[str], timeout_ms: int) -> bool:
        joined = " ".join(selectors)
        if "Continue with Google" in joined:
            return True
        if "button:has-text('Log in')" in joined or "a:has-text('Log in')" in joined:
            return True
        return False

    monkeypatch.setattr("multishell.web_reasoners._has_visible", fake_has_visible)
    monkeypatch.setattr(
        "multishell.web_reasoners._click_first",
        lambda _page, selectors: clicks.append(tuple(selectors)),
    )
    monkeypatch.setattr("multishell.web_reasoners._click_google_and_capture_page", lambda page_obj: page_obj)
    monkeypatch.setattr(
        "multishell.web_reasoners._ensure_google_logged_in",
        lambda _page, email, password, _cancel: google_calls.append((email, password)),
    )
    monkeypatch.setattr("multishell.web_reasoners._wait_for_chatgpt_login_completion", lambda *_args, **_kwargs: None)

    _ensure_chatgpt_logged_in(page, "bot@kuang2.ai", "secret", Event())

    assert page.gotos == []
    assert clicks == [("button:has-text('Log in')", "a:has-text('Log in')")]
    assert google_calls == [("bot@kuang2.ai", "secret")]


def test_submit_prompt_clicks_send_when_enter_leaves_text_in_composer() -> None:
    page = _FakePromptPage()

    _submit_prompt(page, "Reply with exactly OK.", Event())

    assert page.composer_clicked == 1
    assert page.send_clicked == 1
    assert page.composer_value == ""


def test_extract_last_assistant_message_skips_page_fallback_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoMessageLocator:
        def __init__(self, selector: str) -> None:
            self.selector = selector
            self.first = self

        def count(self) -> int:
            return 1 if self.selector == "main" else 0

        def nth(self, index: int) -> "_NoMessageLocator":
            return self

        def inner_text(self, timeout: int | None = None) -> str:
            if self.selector == "main":
                return "Gemini UI chrome"
            return ""

    class _NoMessagePage:
        def locator(self, selector: str) -> _NoMessageLocator:
            return _NoMessageLocator(selector)

    monkeypatch.setattr("multishell.web_reasoners._body_text", lambda _page: "Gemini body fallback")

    capture = _extract_last_assistant_message(_NoMessagePage(), allow_page_fallback=False)

    assert capture.text == ""
    assert capture.source == "none"


def test_wait_for_stable_response_rejects_page_level_capture_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakeStablePage()
    capture = SimpleNamespace(text="Gemini UI chrome", source="main_fallback")
    clock = {"value": 0.0}

    monkeypatch.setattr("multishell.web_reasoners._extract_last_assistant_message", lambda *_args, **_kwargs: capture)
    monkeypatch.setattr("multishell.web_reasoners._has_visible", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("multishell.web_reasoners.time.time", lambda: clock.__setitem__("value", clock["value"] + 0.4) or clock["value"])

    with pytest.raises(Exception, match="page-level UI text was visible via main_fallback"):
        _wait_for_stable_response(
            page,
            Event(),
            timeout_seconds=1,
            previous_text="",
            prompt_text="Reply with exactly OK.",
            stop_selectors=["button:has-text('Stop')"],
            allow_page_fallback=False,
            provider_label="Gemini",
        )


def test_wait_for_stable_response_marks_page_level_capture_as_weak_when_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakeStablePage()
    capture = SimpleNamespace(text="Answer text", source="main_fallback")

    monkeypatch.setattr("multishell.web_reasoners._extract_last_assistant_message", lambda *_args, **_kwargs: capture)
    monkeypatch.setattr("multishell.web_reasoners._has_visible", lambda *_args, **_kwargs: False)

    result = _wait_for_stable_response(
        page,
        Event(),
        timeout_seconds=5,
        previous_text="",
        prompt_text="Reply with exactly OK.",
        stop_selectors=["button:has-text('Stop')"],
        allow_page_fallback=True,
        provider_label="ChatGPT",
    )

    assert result.text == "Answer text"
    assert result.capture_source == "main_fallback"
    assert result.quality == "weak"
    assert "page-level text may include UI chrome" in result.validation_note


@pytest.mark.parametrize(
    ("provider", "prepare_attr", "resolve_attr", "run_attr", "wait_attr"),
    [
        (
            "chatgpt_pro",
            "_prepare_chatgpt_pro_surface",
            "_resolve_openai_account",
            "_run_chatgpt_pro",
            "_wait_for_chatgpt_response",
        ),
        (
            "gemini_deepthink",
            "_prepare_gemini_deepthink_surface",
            "_resolve_gemini_account",
            "_run_gemini_deepthink",
            "_wait_for_gemini_response",
        ),
    ],
)
def test_provider_run_uses_baseline_capture_text(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    prepare_attr: str,
    resolve_attr: str,
    run_attr: str,
    wait_attr: str,
) -> None:
    manager = WebReasonerManager()
    page = _FakePage("https://example.com/")
    job = SimpleNamespace(id="job-1", account_agent="manager", prompt="Reply with exactly OK.", timeout_seconds=30)
    observed: dict[str, object] = {}

    monkeypatch.setattr(manager, "_debug_target", lambda _job_id: "debug-target")
    monkeypatch.setattr(f"multishell.web_reasoners.{resolve_attr}", lambda _agent: ("user@example.com", "secret"))
    monkeypatch.setattr(f"multishell.web_reasoners.{prepare_attr}", lambda page_obj, **_kwargs: page_obj)
    monkeypatch.setattr(
        "multishell.web_reasoners._extract_last_assistant_message",
        lambda _page, **_kwargs: SimpleNamespace(text="baseline text", source="[data-testid='assistant-turn']"),
    )
    monkeypatch.setattr("multishell.web_reasoners._submit_prompt", lambda *_args, **_kwargs: None)

    def fake_wait(page_obj, cancel_flag, *, timeout_seconds: int, previous_text: str, prompt_text: str):
        observed["previous_text"] = previous_text
        observed["prompt_text"] = prompt_text
        return SimpleNamespace(text="OK", capture_source="main article", quality="ok", validation_note="")

    monkeypatch.setattr(f"multishell.web_reasoners.{wait_attr}", fake_wait)

    result = getattr(manager, run_attr)(page, job, Event())

    assert observed["previous_text"] == "baseline text"
    assert observed["prompt_text"] == "Reply with exactly OK."
    assert result.text == "OK"


def test_prepare_chatgpt_pro_surface_uses_web_navigator(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage("https://chatgpt.com/")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "multishell.web_reasoners.drive_web_navigation_with_cli",
        lambda page_obj, **kwargs: captured.update({"page": page_obj, **kwargs}) or page_obj,
    )

    result = _prepare_chatgpt_pro_surface(page, email="bot@kuang2.ai", password="secret", debug_label="debug-chatgpt")

    assert result is page
    assert captured["flow_label"] == "chatgpt_pro"
    assert captured["engine_order"] == ("codex", "claude")
    assert captured["debug_label"] == "debug-chatgpt"
    assert captured["secret_values"] == {"account_email": "bot@kuang2.ai", "password": "secret"}
    assert CHATGPT_LOGIN_URL in captured["allowed_urls"]


def test_prepare_chatgpt_pro_surface_falls_back_to_legacy_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage("https://chatgpt.com/")
    calls: list[str] = []

    monkeypatch.setattr(
        "multishell.web_reasoners.drive_web_navigation_with_cli",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(WebNavigatorError("navigator failed")),
    )
    monkeypatch.setattr(
        "multishell.web_reasoners._ensure_chatgpt_logged_in",
        lambda *_args, **_kwargs: calls.append("login"),
    )
    monkeypatch.setattr(
        "multishell.web_reasoners._ensure_chatgpt_pro_workspace",
        lambda *_args, **_kwargs: calls.append("workspace"),
    )

    result = _prepare_chatgpt_pro_surface(page, email="bot@kuang2.ai", password="secret")

    assert result is page
    assert calls == ["login", "workspace"]
    assert page.gotos == ["https://chatgpt.com/"]


def test_prepare_gemini_deepthink_surface_prefers_claude_first(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage("https://gemini.google.com/app")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "multishell.web_reasoners.drive_web_navigation_with_cli",
        lambda page_obj, **kwargs: captured.update({"page": page_obj, **kwargs}) or page_obj,
    )

    result = _prepare_gemini_deepthink_surface(
        page,
        email="bot@kuang2.ai",
        password="secret",
        debug_label="debug-gemini",
    )

    assert result is page
    assert captured["flow_label"] == "gemini_deepthink"
    assert captured["engine_order"] == ("claude", "codex")
    assert captured["debug_label"] == "debug-gemini"
    assert captured["secret_values"] == {"account_email": "bot@kuang2.ai", "password": "secret"}
    assert "https://gemini.google.com/app" in captured["allowed_urls"]


def test_preferred_openai_flow_page_prioritizes_auth_pages_over_managed_notice() -> None:
    current = _FakePage("about:blank")
    chatgpt = _FakePage("https://chatgpt.com/")
    notice = _FakePage("chrome://managed-user-profile-notice/")
    google = _FakePage("https://accounts.google.com/v3/signin/challenge/pwd")
    auth = _FakePage("https://auth.openai.com/workspace")
    pages = [current, chatgpt, notice, google, auth]
    for page in pages:
        page.context = SimpleNamespace(pages=pages)

    assert _preferred_openai_flow_page(current) is google

    google.url = "about:blank"
    assert _preferred_openai_flow_page(current) is auth

    auth.url = "about:blank"
    assert _preferred_openai_flow_page(current) is chatgpt


def test_wait_for_chatgpt_login_completion_retries_auth_from_public_home(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage("https://chatgpt.com/")
    state = {"body": "Log in to get answers based on saved chats."}
    normalized_urls: list[str] = []

    monkeypatch.setattr("multishell.web_reasoners._body_text", lambda _page: state["body"])
    monkeypatch.setattr("multishell.web_reasoners._page_title", lambda _page: "ChatGPT")
    monkeypatch.setattr("multishell.web_reasoners._check_cancel", lambda _flag: None)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    def fake_has_visible(_page: object, selectors: list[str], timeout_ms: int) -> bool:
        if any("textarea" in selector or "textbox" in selector or "contenteditable" in selector for selector in selectors):
            return "Log in" not in state["body"]
        return False

    monkeypatch.setattr("multishell.web_reasoners._has_visible", fake_has_visible)

    def fake_normalize(_page: object) -> None:
        normalized_urls.append(_page.url)
        _page.url = "https://chatgpt.com/"
        state["body"] = "ChatGPT 5.4 Pro"

    monkeypatch.setattr("multishell.web_reasoners._normalize_openai_login_entry", fake_normalize)

    _wait_for_chatgpt_login_completion(page, Event())

    assert page.gotos == [CHATGPT_LOGIN_URL]
    assert normalized_urls == [CHATGPT_LOGIN_URL]
