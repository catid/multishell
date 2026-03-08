import json
from types import SimpleNamespace

import pytest

from multishell.auth_flow_model import (
    AuthModelAction,
    AuthModelDecision,
    AuthModelProtocolError,
    DEFAULT_AUTH_MODEL_API_BASE,
    DEFAULT_AUTH_MODEL_NAME,
    _extract_json_object,
    _locate_tagged_element,
    _snapshot_fingerprint,
    _strip_thinking_markup,
    drive_google_auth_with_model,
    load_auth_model_settings,
    request_auth_model_decision,
    run_auth_model_smoke_test,
    snapshot_requires_model,
)


class _FakeHTTPResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body


def test_load_auth_model_settings_defaults(monkeypatch) -> None:
    monkeypatch.delenv("MULTISHELL_AUTH_MODEL_ENABLED", raising=False)
    monkeypatch.delenv("MULTISHELL_AUTH_MODEL_API_BASE", raising=False)
    monkeypatch.delenv("MULTISHELL_AUTH_MODEL", raising=False)

    settings = load_auth_model_settings()

    assert settings.enabled is True
    assert settings.api_base == DEFAULT_AUTH_MODEL_API_BASE
    assert settings.model == DEFAULT_AUTH_MODEL_NAME
    assert settings.timeout_seconds == 180
    assert settings.max_steps == 16
    assert settings.max_concurrency == 1


def test_request_auth_model_decision_disables_qwen_thinking(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "<think>hidden</think>{\"action\":\"done\",\"message\":\"complete\"}"
                            }
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    decision = request_auth_model_decision(
        load_auth_model_settings(),
        flow_label="codex/google",
        snapshot={"url": "https://accounts.google.com", "title": "Sign in", "body_text": "Enter email", "elements": []},
        history=[],
        carry_forward=[],
    )

    assert captured["payload"]["enable_thinking"] is False
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["top_p"] == 1.0
    assert captured["payload"]["seed"] == 0
    assert decision.action.action == "done"
    assert decision.action.message == "complete"


def test_request_auth_model_decision_parses_failure_kind(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return _FakeHTTPResponse(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "{\"action\":\"fail\",\"failure_kind\":\"no_auth\",\"message\":\"wrong password\"}"
                            }
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    decision = request_auth_model_decision(
        load_auth_model_settings(),
        flow_label="codex/google",
        snapshot={"url": "https://accounts.google.com", "title": "Sign in", "body_text": "Wrong password", "elements": []},
        history=[],
        carry_forward=[],
    )

    assert decision.action.action == "fail"
    assert decision.action.failure_kind == "no_auth"
    assert decision.action.message == "wrong password"


def test_request_auth_model_decision_compacts_snapshot_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(json.dumps({"choices": [{"message": {"content": "{\"action\":\"done\"}"}}]}))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    request_auth_model_decision(
        load_auth_model_settings(),
        flow_label="codex/google",
        snapshot={
            "url": "https://accounts.google.com/v3/signin/identifier?foo=bar&baz=qux",
            "title": "Sign in",
            "body_text": "x" * 5000,
            "headings": ["Sign in"],
            "labels": ["Email"],
            "alerts": [],
            "supporting": ["Use your Google Account"],
            "elements": [
                {
                    "id": "ms-auth-email",
                    "tag": "input",
                    "htmlId": "identifierId",
                    "type": "email",
                    "text": "Email",
                    "inputMode": "email",
                    "maxLength": "120",
                    "valueLength": 0,
                    "selected": False,
                }
            ]
            * 30,
        },
        history=[
            {"page": "older-1", "action": "fill"},
            {"page": "older-2", "action": "click"},
            {"page": "older-3", "action": "wait"},
        ],
        carry_forward=["remember device_code for later", "workspace not chosen yet"],
        available_keys=["account_email", "password", "device_code"],
    )

    messages = captured["payload"]["messages"]
    user_message = json.loads(messages[1]["content"])
    assert user_message["snapshot"]["url"] == "https://accounts.google.com/v3/signin/identifier"
    assert user_message["snapshot"]["digest"] == {
        "headings": ["Sign in"],
        "alerts": [],
        "labels": ["Email"],
        "supporting": ["Use your Google Account"],
    }
    assert len(user_message["snapshot"]["elements"]) == 20
    assert user_message["snapshot"]["elements"][0]["htmlId"] == "identifierId"
    assert user_message["snapshot"]["elements"][0]["inputMode"] == "email"
    assert user_message["snapshot"]["elements"][0]["maxLength"] == "120"
    assert "valueLength" not in user_message["snapshot"]["elements"][0]
    assert "selected" not in user_message["snapshot"]["elements"][0]
    assert user_message["last"] == "action=wait"
    assert user_message["memory"] == ["remember device_code for later", "workspace not chosen yet"]
    assert user_message["available_keys"] == ["account_email", "password", "device_code"]
    assert user_message["surface"]["interactive_count"] == 30
    assert user_message["surface"]["enabled_count"] == 30
    assert user_message["surface"]["input_count"] == 30
    assert user_message["surface"]["short_input_count"] == 0
    assert user_message["surface"]["all_short_inputs_filled"] is False
    assert user_message["keys"] == ["account_email", "password", "device_code"]
    assert user_message["reply"].startswith("JSON only:")


def test_request_auth_model_decision_logs_snapshot_and_raw_output(monkeypatch) -> None:
    logs: list[str] = []

    def fake_urlopen(request, timeout):
        return _FakeHTTPResponse(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "{\"action\":\"click\",\"target\":\"ms-auth-1\",\"message\":\"click next\"}"
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 123, "completion_tokens": 7, "total_tokens": 130},
                    "timings": {"prompt_per_second": 98.4, "predicted_per_second": 11.2},
                }
            )
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    decision = request_auth_model_decision(
        load_auth_model_settings(),
        flow_label="codex/google",
        snapshot={
            "url": "https://accounts.google.com",
            "title": "Sign in",
            "body_text": "Enter email",
            "elements": [{"id": "ms-auth-1", "tag": "button", "type": "", "text": "Next"}],
        },
        history=[],
        carry_forward=["use account_email first"],
        logger=logs.append,
    )

    assert decision.action.action == "click"
    assert any("auth model request:" in line for line in logs)
    assert any("auth model metrics:" in line and "decode_tps=11.20" in line for line in logs)
    assert any("auth model raw response:" in line for line in logs)
    assert any("auth model parsed action:" in line for line in logs)
    assert any("carry_forward=" in line for line in logs)


def test_request_auth_model_decision_includes_step_scoped_guidance(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(json.dumps({"choices": [{"message": {"content": "{\"action\":\"wait\",\"seconds\":1}"}}]}))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    request_auth_model_decision(
        load_auth_model_settings(),
        flow_label="codex/google",
        snapshot={
            "url": "https://auth.openai.com/codex/device",
            "title": "Device code",
            "body_text": "Use your device code to grant access to Codex CLI",
            "elements": [
                {
                    "id": "ms-auth-code-1",
                    "tag": "input",
                    "type": "text",
                    "name": "code-1",
                    "ariaLabel": "Character 1",
                    "inputMode": "text",
                    "maxLength": "1",
                    "filled": False,
                    "valueLength": 0,
                    "disabled": False,
                },
                {
                    "id": "ms-auth-code-2",
                    "tag": "input",
                    "type": "text",
                    "name": "code-2",
                    "ariaLabel": "Character 2",
                    "inputMode": "text",
                    "maxLength": "1",
                    "filled": False,
                    "valueLength": 0,
                    "disabled": False,
                },
            ],
        },
        history=[],
        carry_forward=["If the code form appears, use value_key=device_code on the first visible code input."],
        available_keys=["device_code"],
    )

    messages = captured["payload"]["messages"]
    system_message = messages[0]["content"]
    user_message = json.loads(messages[1]["content"])
    assert "Fresh context every step" in system_message
    assert "segmented token entry" in system_message
    assert "device_code handle" in system_message
    assert "URL tokens or callback codes" in system_message
    assert "available_keys" in system_message
    assert user_message["memory"] == ["If the code form appears, use value_key=device_code on the first visible code input."]
    assert user_message["available_keys"] == ["device_code"]
    assert user_message["last"] == ""
    assert user_message["surface"]["short_input_count"] == 2
    assert user_message["surface"]["all_short_inputs_filled"] is False


def test_request_auth_model_decision_marks_next_flight_payload_as_script_like(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(json.dumps({"choices": [{"message": {"content": "{\"action\":\"wait\",\"seconds\":1}"}}]}))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    request_auth_model_decision(
        load_auth_model_settings(),
        flow_label="claude/google",
        snapshot={
            "url": "https://claude.ai/oauth/authorize",
            "title": "Claude",
            "body_text": '(self.__next_f=self.__next_f||[]).push([0])self.__next_f.push([1,"1:\\\"$Sreact.fragment\\\""])',
            "elements": [],
        },
        history=[],
        carry_forward=[],
    )

    user_message = json.loads(captured["payload"]["messages"][1]["content"])
    system_message = captured["payload"]["messages"][0]["content"]
    assert user_message["surface"]["script_like_body"] is True
    assert "bootstrap or app script output" in system_message


def test_strip_thinking_markup_removes_think_blocks() -> None:
    assert _strip_thinking_markup("<think>internal</think>{\"action\":\"wait\"}") == "{\"action\":\"wait\"}"


def test_extract_json_object_repairs_malformed_action_payload() -> None:
    payload = _extract_json_object(
        '{ "action": "done", "target": "", "message": "complete", '
        '"action": "click", "target": "ms-auth-continue", "message": "Proceed", '
        '"carry_forward": ["await next step"] ] }'
    )

    assert payload == {
        "action": "click",
        "target": "ms-auth-continue",
        "message": "Proceed",
        "carry_forward": ["await next step"],
    }


def test_extract_json_object_repairs_broken_carry_forward_block() -> None:
    payload = _extract_json_object(
        '{ "action": "click", "target": "ms-auth-log-in", "message": "Session ended; initiating login flow.", '
        '"carry_forward": [ "carry_forward": [ "Login flow initiated via account_email", device_code: "Login flow initiated via account_email" ] }'
    )

    assert payload["action"] == "click"
    assert payload["target"] == "ms-auth-log-in"
    assert payload["message"] == "Session ended; initiating login flow."


def test_snapshot_fingerprint_ignores_focus_only_changes() -> None:
    base = {
        "url": "https://auth.openai.com/deviceauth/callback",
        "title": "Use your device code",
        "body_text": "Enter the 9-character code",
        "elements": [
            {
                "id": "ms-auth-character-1",
                "tag": "input",
                "type": "text",
                "filled": True,
                "focused": False,
                "disabled": False,
            }
        ],
    }
    focused = {
        **base,
        "elements": [
            {
                "id": "ms-auth-character-1",
                "tag": "input",
                "type": "text",
                "filled": True,
                "focused": True,
                "disabled": False,
            }
        ],
    }

    assert _snapshot_fingerprint(base) == _snapshot_fingerprint(focused)


class _FakeTaggedCandidate:
    def __init__(self, visible: bool, index: int, seen: list[int]) -> None:
        self.visible = visible
        self.index = index
        self.seen = seen

    def wait_for(self, state: str, timeout: int) -> None:
        self.seen.append(self.index)
        if not self.visible:
            raise RuntimeError("hidden")


class _FakeTaggedLocator:
    def __init__(self, visibilities: list[bool], seen: list[int]) -> None:
        self._visibilities = visibilities
        self._seen = seen

    def count(self) -> int:
        return len(self._visibilities)

    def nth(self, index: int) -> _FakeTaggedCandidate:
        return _FakeTaggedCandidate(self._visibilities[index], index, self._seen)


class _FakeTaggedPage:
    def __init__(self, visibilities: list[bool], seen: list[int]) -> None:
        self._visibilities = visibilities
        self._seen = seen

    def locator(self, selector: str) -> _FakeTaggedLocator:
        assert selector == '[data-multishell-auth-id="ms-auth-email"]'
        return _FakeTaggedLocator(self._visibilities, self._seen)


def test_locate_tagged_element_prefers_visible_match() -> None:
    seen: list[int] = []
    locator = _locate_tagged_element(_FakeTaggedPage([False, True], seen), "ms-auth-email")

    assert locator.index == 1
    assert seen == [0, 1]


def test_snapshot_requires_model_on_google_and_continue_with_google() -> None:
    assert snapshot_requires_model({"url": "https://accounts.google.com/v3/signin/identifier", "title": "", "body_text": "", "elements": []}) is True
    assert snapshot_requires_model({"url": "https://auth.openai.com/", "title": "", "body_text": "Continue with Google", "elements": []}) is True
    assert snapshot_requires_model({"url": "https://auth.openai.com/codex/device", "title": "", "body_text": "Use your device code to grant access to Codex CLI", "elements": []}) is True
    assert snapshot_requires_model({"url": "https://auth.openai.com/api/oauth/oauth2/auth", "title": "Log in", "body_text": "", "elements": []}) is True
    assert snapshot_requires_model({"url": "https://chatgpt.com/auth/login_with", "title": "ChatGPT", "body_text": "", "elements": []}) is True
    assert snapshot_requires_model({"url": "https://claude.ai/login?returnTo=%2Foauth%2Fauthorize", "title": "Claude", "body_text": "Continue with Google", "elements": []}) is True
    assert snapshot_requires_model(
        {
            "url": "https://claude.ai/oauth/authorize?code=true",
            "title": "Just a moment...",
            "body_text": "Performing security verification. This website uses a security service to protect against malicious bots.",
            "elements": [],
        }
    ) is True
    assert snapshot_requires_model(
        {
            "url": "https://platform.claude.com/oauth/code/success?app=claude-code",
            "title": "Claude Developer Platform",
            "body_text": "Build something great. You're all set up for Claude Code.",
            "elements": [],
        }
    ) is False
    assert snapshot_requires_model({"url": "https://chatgpt.com/", "title": "ChatGPT", "body_text": "Hello", "elements": []}) is False


def test_drive_google_auth_with_model_logs_when_snapshot_is_skipped(monkeypatch) -> None:
    logs: list[str] = []

    monkeypatch.setattr(
        "multishell.auth_flow_model.capture_auth_snapshot",
        lambda _page: {"url": "https://example.com/", "title": "Example", "body_text": "Hello", "elements": []},
    )

    assert drive_google_auth_with_model(object(), flow_label="codex/google", account_email="worker@example.com", password="secret", logger=logs.append) is False
    assert any("auth model skipped:" in line for line in logs)


class _FakePlaywrightContext:
    def __enter__(self):
        return SimpleNamespace()

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeBodyLocator:
    def inner_text(self) -> str:
        return "Smoke test passed"


class _FakePage:
    def goto(self, url: str, wait_until: str) -> None:
        return None

    def set_content(self, html: str, wait_until: str) -> None:
        assert "Continue with Google" in html

    def locator(self, selector: str):
        assert selector == "body"
        return _FakeBodyLocator()


class _FakeChromeContext:
    def __enter__(self):
        return None, None, _FakePage()

    def __exit__(self, exc_type, exc, tb):
        return False


def test_run_auth_model_smoke_test_uses_model_driver(monkeypatch, capsys) -> None:
    called: dict[str, object] = {"headed": None, "driver": False}

    monkeypatch.setattr("multishell.runtime.apply_node_warning_suppression", lambda: None)
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: _FakePlaywrightContext())
    monkeypatch.setattr(
        "multishell.autologin._isolated_chrome",
        lambda _playwright, _name, headed, progress_label=None: called.__setitem__("headed", headed) or _FakeChromeContext(),
    )
    monkeypatch.setattr(
        "multishell.auth_flow_model.drive_google_auth_with_model",
        lambda *_args, **_kwargs: called.__setitem__("driver", True) or True,
    )

    run_auth_model_smoke_test()

    out = capsys.readouterr().out
    assert called["headed"] is False
    assert called["driver"] is True
    assert "auth model smoke test passed" in out


def test_drive_google_auth_with_model_detects_repeated_no_progress(monkeypatch) -> None:
    logs: list[str] = []
    snapshot = {
        "url": "https://accounts.google.com/v3/signin/identifier",
        "title": "Sign in",
        "body_text": "Continue with Google",
        "elements": [{"id": "ms-auth-next", "tag": "button", "type": "button", "text": "Next"}],
    }

    monkeypatch.setattr(
        "multishell.auth_flow_model.capture_auth_snapshot",
        lambda _page: snapshot,
    )
    monkeypatch.setattr(
        "multishell.auth_flow_model.request_auth_model_decision",
        lambda *_args, **_kwargs: AuthModelDecision(action=AuthModelAction(action="click", target="ms-auth-next", message="click next")),
    )
    monkeypatch.setattr("multishell.auth_flow_model._apply_auth_action", lambda *_args, **_kwargs: None)

    with pytest.raises(AuthModelProtocolError, match="got stuck repeating the same action"):
        drive_google_auth_with_model(
            object(),
            flow_label="codex/google",
            account_email="worker@example.com",
            password="secret",
            logger=logs.append,
        )

    assert any("repeated the same action on the same page state" in line for line in logs)


def test_drive_google_auth_with_model_treats_closed_page_as_navigation_handoff(monkeypatch) -> None:
    logs: list[str] = []
    snapshots = iter(
        [
            {
                "url": "https://accounts.google.com/v3/signin/identifier",
                "title": "Sign in",
                "body_text": "Continue with Google",
                "elements": [{"id": "ms-auth-next", "tag": "button", "type": "button", "text": "Next"}],
            },
            RuntimeError("page closed"),
        ]
    )

    def fake_capture(_page):
        value = next(snapshots)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr("multishell.auth_flow_model.capture_auth_snapshot", fake_capture)
    monkeypatch.setattr(
        "multishell.auth_flow_model.request_auth_model_decision",
        lambda *_args, **_kwargs: AuthModelDecision(action=AuthModelAction(action="wait", seconds=0.1, message="waiting")),
    )

    assert drive_google_auth_with_model(
        object(),
        flow_label="claude/google",
        account_email="worker@example.com",
        password="secret",
        logger=logs.append,
    ) is True
    assert any("treating it as a navigation handoff" in line for line in logs)


def test_drive_google_auth_with_model_recovers_to_replacement_page(monkeypatch) -> None:
    logs: list[str] = []

    class _FakeContext:
        def __init__(self) -> None:
            self.pages: list[object] = []

    class _FakePage:
        def __init__(self, name: str, context: _FakeContext, closed: bool = False) -> None:
            self.name = name
            self.context = context
            self._closed = closed

        def is_closed(self) -> bool:
            return self._closed

    context = _FakeContext()
    closed_page = _FakePage("closed", context, closed=True)
    replacement_page = _FakePage("replacement", context, closed=False)
    context.pages[:] = [closed_page, replacement_page]
    seen: dict[str, int] = {"replacement": 0}

    def fake_capture(page):
        if getattr(page, "name", "") == "closed":
            raise RuntimeError("page closed")
        seen["replacement"] += 1
        if seen["replacement"] <= 2:
            return {
                "url": "https://accounts.google.com/v3/signin/identifier",
                "title": "Sign in",
                "body_text": "Continue with Google",
                "elements": [{"id": "ms-auth-next", "tag": "button", "type": "button", "text": "Next"}],
            }
        return {
            "url": "https://example.com/done",
            "title": "Done",
            "body_text": "Done",
            "elements": [],
        }

    monkeypatch.setattr("multishell.auth_flow_model.capture_auth_snapshot", fake_capture)
    monkeypatch.setattr(
        "multishell.auth_flow_model.request_auth_model_decision",
        lambda *_args, **_kwargs: AuthModelDecision(action=AuthModelAction(action="wait", seconds=0.1, message="waiting")),
    )
    monkeypatch.setattr("multishell.auth_flow_model._apply_auth_action", lambda *_args, **_kwargs: None)

    assert drive_google_auth_with_model(
        closed_page,
        flow_label="claude/google",
        account_email="worker@example.com",
        password="secret",
        logger=logs.append,
    ) is True
    assert any("switched to a replacement auth page" in line for line in logs)


def test_drive_google_auth_with_model_relays_model_carry_forward_between_steps(monkeypatch) -> None:
    logs: list[str] = []
    actions: list[AuthModelAction] = []
    memories_seen: list[list[str]] = []
    available_keys_seen: list[list[str]] = []
    snapshots = iter(
        [
            {
                "url": "https://auth.openai.com/codex/device",
                "title": "Device code",
                "body_text": "Use your device code to grant access to Codex CLI",
                "elements": [
                    {"id": "ms-auth-code-1", "tag": "input", "type": "text", "name": "code-1", "text": "", "placeholder": "", "ariaLabel": "Character 1", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": True, "disabled": False},
                    {"id": "ms-auth-code-2", "tag": "input", "type": "text", "name": "code-2", "text": "", "placeholder": "", "ariaLabel": "Character 2", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-continue", "tag": "button", "type": "button", "name": "", "text": "Continue", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": False, "disabled": True},
                ],
            },
            {
                "url": "https://auth.openai.com/codex/device",
                "title": "Device code",
                "body_text": "Use your device code to grant access to Codex CLI",
                "elements": [
                    {"id": "ms-auth-code-1", "tag": "input", "type": "text", "name": "code-1", "text": "", "placeholder": "", "ariaLabel": "Character 1", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": True, "disabled": False},
                    {"id": "ms-auth-code-2", "tag": "input", "type": "text", "name": "code-2", "text": "", "placeholder": "", "ariaLabel": "Character 2", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-continue", "tag": "button", "type": "button", "name": "", "text": "Continue", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": False, "disabled": True},
                ],
            },
            {
                "url": "https://auth.openai.com/codex/device",
                "title": "Device code",
                "body_text": "Use your device code to grant access to Codex CLI",
                "elements": [
                    {"id": "ms-auth-code-1", "tag": "input", "type": "text", "name": "code-1", "text": "", "placeholder": "", "ariaLabel": "Character 1", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": True, "valueLength": 1, "checked": False, "selected": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-code-2", "tag": "input", "type": "text", "name": "code-2", "text": "", "placeholder": "", "ariaLabel": "Character 2", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": True, "valueLength": 1, "checked": False, "selected": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-continue", "tag": "button", "type": "button", "name": "", "text": "Continue", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": False, "disabled": False},
                ],
            },
            {
                "url": "https://auth.openai.com/codex/device",
                "title": "Device code",
                "body_text": "Use your device code to grant access to Codex CLI",
                "elements": [
                    {"id": "ms-auth-code-1", "tag": "input", "type": "text", "name": "code-1", "text": "", "placeholder": "", "ariaLabel": "Character 1", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": True, "valueLength": 1, "checked": False, "selected": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-code-2", "tag": "input", "type": "text", "name": "code-2", "text": "", "placeholder": "", "ariaLabel": "Character 2", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": True, "valueLength": 1, "checked": False, "selected": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-continue", "tag": "button", "type": "button", "name": "", "text": "Continue", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": False, "disabled": False},
                ],
            },
            {
                "url": "https://chatgpt.com/",
                "title": "ChatGPT",
                "body_text": "Welcome back",
                "elements": [],
            },
        ]
    )

    monkeypatch.setattr("multishell.auth_flow_model.capture_auth_snapshot", lambda _page: next(snapshots))

    def fake_request(_settings, *, carry_forward, available_keys, **_kwargs):
        memories_seen.append(list(carry_forward))
        available_keys_seen.append(list(available_keys))
        if not carry_forward:
            return AuthModelDecision(
                action=AuthModelAction(action="fill", target="ms-auth-code-1", value_key="device_code", message="fill code"),
                carry_forward=("device code already entered; click Continue if it becomes enabled",),
            )
        return AuthModelDecision(
            action=AuthModelAction(action="click", target="ms-auth-continue", message="submit code"),
            carry_forward=("waiting for OpenAI redirect after device code submit",),
        )

    monkeypatch.setattr("multishell.auth_flow_model.request_auth_model_decision", fake_request)
    monkeypatch.setattr(
        "multishell.auth_flow_model._apply_auth_action",
        lambda _page, action, **_kwargs: actions.append(action),
    )
    monkeypatch.setattr("multishell.auth_flow_model._wait_for_surface_change", lambda *_args, **_kwargs: None)

    assert drive_google_auth_with_model(
        object(),
        flow_label="codex/google",
        account_email="worker@example.com",
        password="secret",
        device_code="ABCD-EFGHI",
        logger=logs.append,
    ) is True
    assert [action.action for action in actions] == ["fill", "click"]
    assert actions[0].value_key == "device_code"
    assert actions[1].target == "ms-auth-continue"
    assert memories_seen == [[], ["device code already entered; click Continue if it becomes enabled"]]
    assert available_keys_seen == [
        ["account_email", "password", "device_code"],
        ["account_email", "password", "device_code"],
    ]
    assert any("auth model carry-forward:" in line for line in logs)


def test_drive_google_auth_with_model_recovers_missing_device_code_failure(monkeypatch) -> None:
    logs: list[str] = []
    actions: list[AuthModelAction] = []
    memories_seen: list[list[str]] = []
    snapshots = iter(
        [
            {
                "url": "https://auth.openai.com/codex/device",
                "title": "Device code",
                "body_text": "Use your device code to grant access to Codex CLI",
                "elements": [
                    {"id": "ms-auth-code-1", "tag": "input", "type": "text", "name": "code-1", "text": "", "placeholder": "", "ariaLabel": "Character 1", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": True, "disabled": False},
                    {"id": "ms-auth-code-2", "tag": "input", "type": "text", "name": "code-2", "text": "", "placeholder": "", "ariaLabel": "Character 2", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-continue", "tag": "button", "type": "button", "name": "", "text": "Continue", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": False, "disabled": True},
                ],
            },
            {
                "url": "https://auth.openai.com/codex/device",
                "title": "Device code",
                "body_text": "Use your device code to grant access to Codex CLI",
                "elements": [
                    {"id": "ms-auth-code-1", "tag": "input", "type": "text", "name": "code-1", "text": "", "placeholder": "", "ariaLabel": "Character 1", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": True, "disabled": False},
                    {"id": "ms-auth-code-2", "tag": "input", "type": "text", "name": "code-2", "text": "", "placeholder": "", "ariaLabel": "Character 2", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-continue", "tag": "button", "type": "button", "name": "", "text": "Continue", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": False, "disabled": True},
                ],
            },
            {
                "url": "https://auth.openai.com/codex/device",
                "title": "Device code",
                "body_text": "Use your device code to grant access to Codex CLI",
                "elements": [
                    {"id": "ms-auth-code-1", "tag": "input", "type": "text", "name": "code-1", "text": "", "placeholder": "", "ariaLabel": "Character 1", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": True, "valueLength": 1, "checked": False, "selected": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-code-2", "tag": "input", "type": "text", "name": "code-2", "text": "", "placeholder": "", "ariaLabel": "Character 2", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": True, "valueLength": 1, "checked": False, "selected": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-continue", "tag": "button", "type": "button", "name": "", "text": "Continue", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": False, "disabled": False},
                ],
            },
            {
                "url": "https://auth.openai.com/codex/device",
                "title": "Device code",
                "body_text": "Use your device code to grant access to Codex CLI",
                "elements": [
                    {"id": "ms-auth-code-1", "tag": "input", "type": "text", "name": "code-1", "text": "", "placeholder": "", "ariaLabel": "Character 1", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": True, "valueLength": 1, "checked": False, "selected": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-code-2", "tag": "input", "type": "text", "name": "code-2", "text": "", "placeholder": "", "ariaLabel": "Character 2", "autocomplete": "", "inputMode": "text", "maxLength": "1", "filled": True, "valueLength": 1, "checked": False, "selected": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-continue", "tag": "button", "type": "button", "name": "", "text": "Continue", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "valueLength": 0, "checked": False, "selected": False, "focused": False, "disabled": False},
                ],
            },
            {
                "url": "https://chatgpt.com/",
                "title": "ChatGPT",
                "body_text": "Welcome back",
                "elements": [],
            },
        ]
    )

    monkeypatch.setattr("multishell.auth_flow_model.capture_auth_snapshot", lambda _page: next(snapshots))

    def fake_request(_settings, *, carry_forward, **_kwargs):
        memories_seen.append(list(carry_forward))
        if not carry_forward:
            return AuthModelDecision(
                action=AuthModelAction(
                    action="fail",
                    failure_kind="error",
                    message="Device code not provided in context; cannot proceed with token entry.",
                ),
                carry_forward=("waiting_for_device_code",),
            )
        return AuthModelDecision(
            action=AuthModelAction(action="click", target="ms-auth-continue", message="submit code"),
            carry_forward=("waiting for OpenAI redirect after device code submit",),
        )

    monkeypatch.setattr("multishell.auth_flow_model.request_auth_model_decision", fake_request)
    monkeypatch.setattr(
        "multishell.auth_flow_model._apply_auth_action",
        lambda _page, action, **_kwargs: actions.append(action),
    )
    monkeypatch.setattr("multishell.auth_flow_model._wait_for_surface_change", lambda *_args, **_kwargs: None)

    assert drive_google_auth_with_model(
        object(),
        flow_label="codex/google",
        account_email="worker@example.com",
        password="secret",
        device_code="ABCD-EFGHI",
        logger=logs.append,
    ) is True
    assert [action.action for action in actions] == ["fill", "click"]
    assert actions[0].value_key == "device_code"
    assert memories_seen == [
        [],
        [
            "device_code is available via value_key=device_code",
            "after the code is entered, continue when the button enables",
        ],
    ]
    assert any("auth model recovery: device_code handle is available" in line for line in logs)


def test_apply_auth_action_uses_visible_device_code_helper(monkeypatch) -> None:
    seen: dict[str, object] = {"value": None, "settled": False}

    monkeypatch.setattr(
        "multishell.auth_flow_model._fill_visible_device_code_inputs",
        lambda _page, value: seen.__setitem__("value", value),
    )
    monkeypatch.setattr(
        "multishell.auth_flow_model._settle_after_action",
        lambda _page: seen.__setitem__("settled", True),
    )

    _page = object()
    action = AuthModelAction(action="fill", target="ms-auth-code-1", value_key="device_code", message="fill code")

    from multishell.auth_flow_model import _apply_auth_action

    _apply_auth_action(_page, action, secret_values={"device_code": "ABCD-EFGHI"})

    assert seen["value"] == "ABCD-EFGHI"
    assert seen["settled"] is True


def test_drive_google_auth_with_model_uses_model_on_account_chooser_and_consent(monkeypatch) -> None:
    logs: list[str] = []
    actions: list[AuthModelAction] = []
    snapshots = iter(
        [
            {
                "url": "https://accounts.google.com/v3/signin/accountchooser",
                "title": "Sign in - Google Accounts",
                "body_text": "Sign in with Google Choose an account to continue to Claude two two two@kuang2.ai Use another account",
                "elements": [
                    {"id": "ms-auth-two-kuang2-ai-selected-switch-account", "tag": "div", "type": "", "name": "", "text": "two@kuang2.ai selected switch account", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "checked": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-two-two-two-kuang2-ai", "tag": "div", "type": "", "name": "", "text": "two two two@kuang2.ai", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "checked": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-use-another-account", "tag": "div", "type": "", "name": "", "text": "Use another account", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "checked": False, "focused": False, "disabled": False},
                ],
            },
            {
                "url": "https://accounts.google.com/v3/signin/accountchooser",
                "title": "Sign in - Google Accounts",
                "body_text": "Sign in with Google Choose an account to continue to Claude two two two@kuang2.ai Use another account",
                "elements": [
                    {"id": "ms-auth-two-kuang2-ai-selected-switch-account", "tag": "div", "type": "", "name": "", "text": "two@kuang2.ai selected switch account", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "checked": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-two-two-two-kuang2-ai", "tag": "div", "type": "", "name": "", "text": "two two two@kuang2.ai", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "checked": False, "focused": False, "disabled": False},
                    {"id": "ms-auth-use-another-account", "tag": "div", "type": "", "name": "", "text": "Use another account", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "checked": False, "focused": False, "disabled": False},
                ],
            },
            {
                "url": "https://claude.ai/oauth/authorize?code=true",
                "title": "Claude",
                "body_text": "Claude Code would like to connect to your Claude chat account",
                "elements": [
                    {"id": "ms-auth-authorize", "tag": "button", "type": "button", "name": "", "text": "Authorize", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "checked": False, "focused": False, "disabled": False},
                ],
            },
            {
                "url": "https://claude.ai/oauth/authorize?code=true",
                "title": "Claude",
                "body_text": "Claude Code would like to connect to your Claude chat account",
                "elements": [
                    {"id": "ms-auth-authorize", "tag": "button", "type": "button", "name": "", "text": "Authorize", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "checked": False, "focused": False, "disabled": False},
                ],
            },
            {
                "url": "https://example.com/done",
                "title": "Done",
                "body_text": "Done",
                "elements": [],
            },
        ]
    )

    monkeypatch.setattr("multishell.auth_flow_model.capture_auth_snapshot", lambda _page: next(snapshots))
    decisions = iter(
        [
            AuthModelDecision(
                action=AuthModelAction(action="click", target="ms-auth-two-two-two-kuang2-ai", message="pick configured account"),
                carry_forward=("selected configured account row",),
            ),
            AuthModelDecision(
                action=AuthModelAction(action="click", target="ms-auth-authorize", message="confirm Claude consent"),
                carry_forward=("waiting for Claude OAuth redirect",),
            ),
        ]
    )
    monkeypatch.setattr("multishell.auth_flow_model.request_auth_model_decision", lambda *_args, **_kwargs: next(decisions))
    monkeypatch.setattr(
        "multishell.auth_flow_model._apply_auth_action",
        lambda _page, action, **_kwargs: actions.append(action),
    )
    monkeypatch.setattr("multishell.auth_flow_model._wait_for_surface_change", lambda *_args, **_kwargs: None)

    assert drive_google_auth_with_model(
        object(),
        flow_label="claude/google",
        account_email="two@kuang2.ai",
        password="secret",
        logger=logs.append,
    ) is True
    assert [action.target for action in actions] == ["ms-auth-two-two-two-kuang2-ai", "ms-auth-authorize"]
    assert any("auth model carry-forward:" in line for line in logs)


def test_drive_google_auth_with_model_tolerates_target_churn_after_navigation(monkeypatch) -> None:
    logs: list[str] = []
    snapshots = iter(
        [
            {
                "url": "https://accounts.google.com/v3/signin/accountchooser",
                "title": "Sign in - Google Accounts",
                "body_text": "Sign in with Google Choose an account to continue to Claude two two two@kuang2.ai",
                "elements": [
                    {"id": "ms-auth-two-two-two-kuang2-ai", "tag": "div", "type": "", "name": "", "text": "two two two@kuang2.ai", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "checked": False, "focused": False, "disabled": False},
                ],
            },
            {
                "url": "https://accounts.google.com/v3/signin/accountchooser",
                "title": "Sign in - Google Accounts",
                "body_text": "Sign in with Google Choose an account to continue to Claude two two two@kuang2.ai",
                "elements": [
                    {"id": "ms-auth-two-two-two-kuang2-ai", "tag": "div", "type": "", "name": "", "text": "two two two@kuang2.ai", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "checked": False, "focused": False, "disabled": False},
                ],
            },
            {
                "url": "https://claude.ai/oauth/authorize?code=true",
                "title": "Claude",
                "body_text": "Claude Code would like to connect to your Claude chat account",
                "elements": [
                    {"id": "ms-auth-authorize", "tag": "button", "type": "button", "name": "", "text": "Authorize", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "checked": False, "focused": False, "disabled": False},
                ],
            },
            {
                "url": "https://claude.ai/oauth/authorize?code=true",
                "title": "Claude",
                "body_text": "Claude Code would like to connect to your Claude chat account",
                "elements": [
                    {"id": "ms-auth-authorize", "tag": "button", "type": "button", "name": "", "text": "Authorize", "placeholder": "", "ariaLabel": "", "autocomplete": "", "filled": False, "checked": False, "focused": False, "disabled": False},
                ],
            },
            {
                "url": "https://example.com/done",
                "title": "Done",
                "body_text": "Done",
                "elements": [],
            },
        ]
    )

    monkeypatch.setattr("multishell.auth_flow_model.capture_auth_snapshot", lambda _page: next(snapshots))
    decisions = iter(
        [
            AuthModelDecision(action=AuthModelAction(action="click", target="ms-auth-two-two-two-kuang2-ai", message="pick configured account")),
            AuthModelDecision(action=AuthModelAction(action="click", target="ms-auth-authorize", message="authorize Claude")),
        ]
    )
    monkeypatch.setattr("multishell.auth_flow_model.request_auth_model_decision", lambda *_args, **_kwargs: next(decisions))

    def fake_apply(_page, action, **_kwargs):
        if action.target == "ms-auth-two-two-two-kuang2-ai":
            raise AuthModelProtocolError("auth model targeted a missing element id: 'ms-auth-two-two-two-kuang2.ai'")

    monkeypatch.setattr("multishell.auth_flow_model._apply_auth_action", fake_apply)

    assert drive_google_auth_with_model(
        object(),
        flow_label="claude/google",
        account_email="two@kuang2.ai",
        password="secret",
        logger=logs.append,
    ) is True
    assert any("changed the page surface before the original target could be reused" in line for line in logs)


def test_drive_google_auth_with_model_tolerates_generic_action_failure_after_navigation(monkeypatch) -> None:
    logs: list[str] = []
    snapshots = iter(
        [
            {
                "url": "https://accounts.google.com/v3/signin/identifier",
                "title": "Sign in - Google Accounts",
                "body_text": "Sign in with Google Email or phone Next",
                "elements": [
                    {"id": "ms-auth-email", "tag": "input", "type": "email", "text": "", "placeholder": "Email or phone", "ariaLabel": "Email or phone", "autocomplete": "", "filled": False, "checked": False, "focused": True, "disabled": False},
                ],
            },
            {
                "url": "https://accounts.google.com/v3/signin/identifier",
                "title": "Sign in - Google Accounts",
                "body_text": "Sign in with Google Email or phone Next",
                "elements": [
                    {"id": "ms-auth-email", "tag": "input", "type": "email", "text": "", "placeholder": "Email or phone", "ariaLabel": "Email or phone", "autocomplete": "", "filled": False, "checked": False, "focused": True, "disabled": False},
                ],
            },
            {
                "url": "https://accounts.google.com/v3/signin/challenge/pwd",
                "title": "Welcome",
                "body_text": "Enter your password Next",
                "elements": [
                    {"id": "ms-auth-passwd", "tag": "input", "type": "password", "text": "", "placeholder": "", "ariaLabel": "Enter your password", "autocomplete": "", "filled": False, "checked": False, "focused": True, "disabled": False},
                ],
            },
            {
                "url": "https://example.com/done",
                "title": "Done",
                "body_text": "Done",
                "elements": [],
            },
        ]
    )

    monkeypatch.setattr("multishell.auth_flow_model.capture_auth_snapshot", lambda _page: next(snapshots))
    decisions = iter(
        [
            AuthModelDecision(action=AuthModelAction(action="fill", target="ms-auth-email", value_key="account_email", message="fill email")),
            AuthModelDecision(action=AuthModelAction(action="done", message="complete")),
        ]
    )
    monkeypatch.setattr("multishell.auth_flow_model.request_auth_model_decision", lambda *_args, **_kwargs: next(decisions))

    def fake_apply(_page, _action, **_kwargs):
        raise RuntimeError("Locator.fill: Timeout 5000ms exceeded.")

    monkeypatch.setattr("multishell.auth_flow_model._apply_auth_action", fake_apply)

    assert drive_google_auth_with_model(
        object(),
        flow_label="codex/google",
        account_email="worker@example.com",
        password="secret",
        logger=logs.append,
    ) is True
    assert any("changed the page surface before the original target could be reused" in line for line in logs)


def test_drive_google_auth_with_model_uses_model_on_empty_google_redirect(monkeypatch) -> None:
    logs: list[str] = []
    actions: list[AuthModelAction] = []
    snapshots = iter(
        [
            {
                "url": "https://accounts.google.com/signin/oauth/id?flowName=GeneralOAuthFlow",
                "title": "",
                "body_text": "",
                "elements": [],
            },
            {
                "url": "https://accounts.google.com/signin/oauth/id?flowName=GeneralOAuthFlow",
                "title": "",
                "body_text": "",
                "elements": [],
            },
            {
                "url": "https://example.com/done",
                "title": "Done",
                "body_text": "Done",
                "elements": [],
            },
        ]
    )

    monkeypatch.setattr("multishell.auth_flow_model.capture_auth_snapshot", lambda _page: next(snapshots))
    monkeypatch.setattr(
        "multishell.auth_flow_model.request_auth_model_decision",
        lambda *_args, **_kwargs: AuthModelDecision(action=AuthModelAction(action="wait", seconds=1.0, message="waiting for redirect render")),
    )
    monkeypatch.setattr(
        "multishell.auth_flow_model._apply_auth_action",
        lambda _page, action, **_kwargs: actions.append(action),
    )

    assert drive_google_auth_with_model(
        object(),
        flow_label="claude/google",
        account_email="three@kuang2.ai",
        password="secret",
        logger=logs.append,
    ) is True
    assert [action.action for action in actions] == ["wait"]
    assert any("auth model step 1: waiting for redirect render" in line for line in logs)
