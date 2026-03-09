from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from multishell.auth_flow_model import AuthModelAction, AuthModelDecision
from multishell.web_navigator import (
    _ready_shortcut,
    _shortcut_navigation_decision,
    _request_claude_navigation_decision,
    _request_codex_navigation_decision,
    drive_web_navigation_with_cli,
    request_navigation_decision,
)


def test_request_navigation_decision_falls_back_to_claude(monkeypatch) -> None:
    monkeypatch.setattr("multishell.web_navigator._navigation_engine_available", lambda _engine: True)
    monkeypatch.setattr(
        "multishell.web_navigator._request_codex_navigation_decision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("codex unavailable")),
    )
    monkeypatch.setattr(
        "multishell.web_navigator._request_claude_navigation_decision",
        lambda *_args, **_kwargs: AuthModelDecision(action=AuthModelAction(action="done")),
    )

    decision = request_navigation_decision(
        flow_label="chatgpt_pro",
        goal="prepare the page",
        done_when="the page is ready",
        snapshot={"url": "https://chatgpt.com/", "title": "ChatGPT", "body_text": "", "elements": []},
        history=[],
        carry_forward=[],
        allowed_urls=["https://chatgpt.com/"],
        available_keys=["account_email", "password"],
        engine_order=("codex", "claude"),
        timeout_seconds=30,
    )

    assert decision.action.action == "done"


def test_ready_shortcut_waits_for_script_only_chatgpt_workspace() -> None:
    decision = _ready_shortcut(
        "chatgpt_pro",
        {
            "url": "https://auth.openai.com/workspace",
            "body_text": "",
            "surface": {"script_like_body": True},
            "elements": [],
        },
    )

    assert decision is not None
    assert decision.action.action == "wait"
    assert decision.action.seconds == 2.0


def test_ready_shortcut_selects_non_personal_chatgpt_workspace() -> None:
    decision = _ready_shortcut(
        "chatgpt_pro",
        {
            "url": "https://auth.openai.com/workspace",
            "title": "Choose a workspace - OpenAI",
            "body_text": "Choose a workspace Workspace Kuang2 Personal account",
            "elements": [
                {"id": "ms-auth-workspace-id", "tag": "button", "name": "workspace_id", "text": "Kuang2"},
                {
                    "id": "ms-auth-workspace-id-2",
                    "tag": "button",
                    "name": "workspace_id",
                    "text": "Personal account",
                },
            ],
        },
    )

    assert decision is not None
    assert decision.action.action == "click"
    assert decision.action.target == "ms-auth-workspace-id"


def test_ready_shortcut_waits_for_disabled_chatgpt_workspace_handoff() -> None:
    decision = _ready_shortcut(
        "chatgpt_pro",
        {
            "url": "https://auth.openai.com/workspace",
            "title": "Choose a workspace - OpenAI",
            "body_text": "Choose a workspace Workspace Kuang2 Personal account",
            "elements": [
                {"id": "ms-auth-workspace-id", "tag": "button", "name": "workspace_id", "text": "Kuang2", "disabled": True},
                {
                    "id": "ms-auth-workspace-id-2",
                    "tag": "button",
                    "name": "workspace_id",
                    "text": "Personal account",
                    "disabled": True,
                },
            ],
        },
    )

    assert decision is not None
    assert decision.action.action == "wait"
    assert decision.action.seconds == 2.0


def test_ready_shortcut_waits_for_empty_gemini_app_hydration() -> None:
    decision = _ready_shortcut(
        "gemini_deepthink",
        {
            "url": "https://gemini.google.com/app",
            "title": "",
            "body_text": "",
            "elements": [],
        },
    )

    assert decision is not None
    assert decision.action.action == "wait"
    assert decision.action.seconds == 2.0


def test_shortcut_prefers_password_over_email_on_password_screen() -> None:
    decision = _shortcut_navigation_decision(
        flow_label="chatgpt_pro",
        snapshot={
            "url": "https://accounts.google.com/v3/signin/challenge/pwd",
            "title": "Welcome",
            "body_text": "Enter your password",
            "elements": [
                {"id": "ms-auth-identifierid", "tag": "input", "name": "identifier", "filled": False},
                {"id": "ms-auth-passwd", "tag": "input", "type": "password", "name": "Passwd", "filled": False},
            ],
        },
        available_keys=["account_email", "password"],
        done_when="ready",
    )

    assert decision is not None
    assert decision.action.action == "fill"
    assert decision.action.target == "ms-auth-passwd"
    assert decision.action.value_key == "password"


def test_request_codex_navigation_decision_reads_structured_output(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("multishell.web_navigator.agent_home", lambda _agent: tmp_path / "codex-home")
    monkeypatch.setattr("multishell.web_navigator.child_env", lambda *args, **kwargs: {})

    def fake_run(args, **kwargs):
        output_flag_index = args.index("-o")
        output_path = Path(args[output_flag_index + 1])
        output_path.write_text('{"action":"wait","seconds":1}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("multishell.web_navigator.subprocess.run", fake_run)

    decision = _request_codex_navigation_decision("decide the next action", timeout_seconds=30)

    assert decision.action.action == "wait"
    assert decision.action.seconds == 1.0


def test_request_claude_navigation_decision_parses_structured_output(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("multishell.web_navigator.claude_home", lambda _agent: tmp_path / "claude-home")
    monkeypatch.setattr("multishell.web_navigator.child_env", lambda *args, **kwargs: {})

    def fake_run(args, **kwargs):
        payload = {
            "type": "result",
            "subtype": "success",
            "structured_output": {"action": "done", "message": "ready"},
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("multishell.web_navigator.subprocess.run", fake_run)

    decision = _request_claude_navigation_decision("decide the next action", timeout_seconds=30)

    assert decision.action.action == "done"
    assert decision.action.message == "ready"


def test_request_claude_navigation_decision_writes_timeout_trace(monkeypatch, tmp_path: Path) -> None:
    trace_calls: list[dict[str, object]] = []

    monkeypatch.setattr("multishell.web_navigator.claude_home", lambda _agent: tmp_path / "claude-home")
    monkeypatch.setattr("multishell.web_navigator.child_env", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "multishell.web_navigator._write_navigation_trace",
        lambda **kwargs: trace_calls.append(kwargs),
    )

    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("claude", timeout=30, output='{"partial":true}', stderr="slow")

    monkeypatch.setattr("multishell.web_navigator.subprocess.run", fake_run)

    try:
        _request_claude_navigation_decision("decide the next action", timeout_seconds=30, debug_label="trace-label", step=2)
    except Exception as exc:
        assert "timed out after 30s" in str(exc)
    else:
        raise AssertionError("expected timeout")

    assert trace_calls == [
        {
            "debug_label": "trace-label",
            "engine": "claude",
            "step": 2,
            "prompt": "decide the next action",
            "stdout": '{"partial":true}',
            "stderr": "slow",
            "returncode": -1,
            "structured_output": '{"partial":true}',
        }
    ]


def test_drive_web_navigation_with_cli_applies_actions_until_done(monkeypatch) -> None:
    snapshots = iter(
        [
            {
                "url": "https://chatgpt.com/",
                "title": "ChatGPT",
                "body_text": "Login page",
                "elements": [{"id": "ms-auth-next", "tag": "button", "text": "Continue"}],
            },
            {
                "url": "https://chatgpt.com/",
                "title": "ChatGPT",
                "body_text": "Composer ready",
                "elements": [{"id": "ms-auth-composer", "tag": "textarea", "text": ""}],
            },
            {
                "url": "https://chatgpt.com/",
                "title": "ChatGPT",
                "body_text": "Composer ready",
                "elements": [{"id": "ms-auth-composer", "tag": "textarea", "text": ""}],
            },
        ]
    )
    decisions = iter(
        [
            AuthModelDecision(action=AuthModelAction(action="click", target="ms-auth-next")),
            AuthModelDecision(action=AuthModelAction(action="done")),
        ]
    )
    actions: list[tuple[str, str]] = []
    page = SimpleNamespace(context=SimpleNamespace(pages=[]), wait_for_timeout=lambda _ms: None)
    page.context.pages = [page]

    monkeypatch.setattr("multishell.web_navigator.capture_auth_snapshot", lambda _page: next(snapshots))
    monkeypatch.setattr("multishell.web_navigator.request_navigation_decision", lambda **_kwargs: next(decisions))
    monkeypatch.setattr(
        "multishell.web_navigator._apply_navigation_action",
        lambda _page, action, *, secret_values: actions.append((action.action, action.target)),
    )
    monkeypatch.setattr("multishell.web_navigator._wait_for_surface_change", lambda *_args, **_kwargs: None)

    result = drive_web_navigation_with_cli(
        page,
        flow_label="chatgpt_pro",
        goal="prepare the page",
        done_when="composer visible",
        secret_values={"account_email": "user@example.com", "password": "secret"},
        allowed_urls=["https://chatgpt.com/"],
        engine_order=("codex",),
    )

    assert result is page
    assert actions == [("click", "ms-auth-next")]
