from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

from .auth_flow_model import (
    AuthModelAction,
    AuthModelDecision,
    AuthModelProtocolError,
    _action_fingerprint,
    _apply_auth_action,
    _compact_last_step,
    _compact_snapshot_for_model,
    _device_code_input_elements,
    _normalize_auth_action,
    _snapshot_fingerprint,
    _surface_summary,
    _wait_for_surface_change,
    capture_auth_snapshot,
    summarize_snapshot,
)
from .config import CLAUDE_WORKER_SPECS, MANAGER_SPEC, state_root
from .homes import agent_home, claude_home, codex_logged_in, has_claude_auth
from .runtime import child_env


NAVIGATION_ALLOWED_ACTIONS = frozenset({"click", "fill", "press", "wait", "goto", "done", "fail"})
DEFAULT_NAVIGATION_MAX_STEPS = 18
DEFAULT_NAVIGATION_DECISION_TIMEOUT_SECONDS = 60
DEFAULT_CODEX_NAVIGATOR_AGENT = MANAGER_SPEC.name
DEFAULT_CLAUDE_NAVIGATOR_AGENT = CLAUDE_WORKER_SPECS[0].name if CLAUDE_WORKER_SPECS else ""
CLAUDE_OUTPUT_FORMAT = "json"
WEB_NAVIGATOR_CLAUDE_MODEL = "claude-sonnet-4-6"
WEB_NAVIGATOR_CLAUDE_EFFORT = "medium"
WEB_NAVIGATOR_CODEX_MODEL = "gpt-5.4"
WEB_NAVIGATOR_CODEX_REASONING_EFFORT = "medium"


class WebNavigatorError(RuntimeError):
    pass


def drive_web_navigation_with_cli(
    page: object,
    *,
    flow_label: str,
    goal: str,
    done_when: str,
    secret_values: dict[str, str],
    allowed_urls: Sequence[str],
    engine_order: Sequence[str],
    logger: Callable[[str], None] | None = None,
    debug_label: str | None = None,
    max_steps: int = DEFAULT_NAVIGATION_MAX_STEPS,
    decision_timeout_seconds: int = DEFAULT_NAVIGATION_DECISION_TIMEOUT_SECONDS,
) -> object:
    current_page = page
    history: list[dict[str, str]] = []
    carry_forward: list[str] = []
    last_repeat_signature = ""
    repeat_count = 0

    for step in range(1, max_steps + 1):
        current_page, snapshot = _capture_snapshot_with_recovery(
            current_page,
            logger=logger,
            recovery_label=f"before step {step}",
            preferred_urls=allowed_urls,
        )
        shortcut = _shortcut_navigation_decision(
            flow_label=flow_label,
            snapshot=snapshot,
            available_keys=[key for key, value in secret_values.items() if value],
            done_when=done_when,
        )
        if shortcut is not None:
            decision = shortcut
            if logger is not None:
                logger("web navigator decision source: shortcut")
            _write_navigation_trace(
                debug_label=debug_label,
                engine="shortcut",
                step=step,
                prompt=summarize_snapshot(snapshot),
                stdout="",
                stderr="",
                returncode=0,
                structured_output=json.dumps(
                    {
                        "action": decision.action.action,
                        "target": decision.action.target,
                        "value_key": decision.action.value_key,
                        "text": decision.action.text,
                        "message": decision.action.message,
                    },
                    ensure_ascii=True,
                ),
            )
        else:
            decision = request_navigation_decision(
                flow_label=flow_label,
                goal=goal,
                done_when=done_when,
                snapshot=snapshot,
                history=history,
                carry_forward=carry_forward,
                allowed_urls=allowed_urls,
                available_keys=[key for key, value in secret_values.items() if value],
                engine_order=engine_order,
                timeout_seconds=decision_timeout_seconds,
                step=step,
                debug_label=debug_label,
                logger=logger,
            )
        action = _normalize_navigation_action(snapshot, decision.action)
        carry_forward = list(decision.carry_forward[:6])
        if logger is not None:
            logger(
                f"web navigator step {step}: action={action.action} target={action.target!r} "
                f"value_key={action.value_key!r} text={action.text[:120]!r}"
            )
            if carry_forward:
                logger(f"web navigator carry-forward: {json.dumps(carry_forward, ensure_ascii=True)}")

        if action.action == "done":
            return current_page
        if action.action == "fail":
            raise WebNavigatorError(action.message or "web navigator reported an unrecoverable error")

        repeat_signature = f"{_snapshot_fingerprint(snapshot)}::{_action_fingerprint(action)}"
        if repeat_signature == last_repeat_signature:
            repeat_count += 1
        else:
            last_repeat_signature = repeat_signature
            repeat_count = 1
        max_repeat_count = 5 if action.action == "wait" else 3
        if repeat_count >= max_repeat_count:
            raise WebNavigatorError(
                "web navigator got stuck repeating "
                f"{_compact_action_for_history(action)} on {summarize_snapshot(snapshot)}"
            )

        previous_fingerprint = _snapshot_fingerprint(snapshot)
        try:
            _apply_navigation_action(current_page, action, secret_values=secret_values)
        except Exception:
            current_page, current_snapshot = _capture_snapshot_with_recovery(
                current_page,
                logger=logger,
                recovery_label=f"after failed step {step}",
                preferred_urls=allowed_urls,
            )
            if _snapshot_fingerprint(current_snapshot) != previous_fingerprint:
                history.append(
                    {
                        "page": summarize_snapshot(snapshot),
                        "action": _compact_action_for_history(action),
                        "post_page": summarize_snapshot(current_snapshot),
                        "changed": "true",
                        "carry_forward": "; ".join(carry_forward[:3]),
                    }
                )
                history[:] = history[-4:]
                continue
            raise

        if action.action in {"click", "press", "goto"}:
            _wait_for_surface_change(current_page, previous_fingerprint)

        current_page, post_snapshot = _capture_snapshot_with_recovery(
            current_page,
            logger=logger,
            recovery_label=f"after step {step}",
            preferred_urls=allowed_urls,
        )
        history.append(
            {
                "page": summarize_snapshot(snapshot),
                "action": _compact_action_for_history(action),
                "post_page": summarize_snapshot(post_snapshot),
                "changed": "true" if _snapshot_fingerprint(post_snapshot) != previous_fingerprint else "false",
                "carry_forward": "; ".join(carry_forward[:3]),
            }
        )
        history[:] = history[-4:]

    raise WebNavigatorError("web navigator exhausted its step budget before the page became ready")


def request_navigation_decision(
    *,
    flow_label: str,
    goal: str,
    done_when: str,
    snapshot: dict[str, object],
    history: list[dict[str, str]],
    carry_forward: list[str],
    allowed_urls: Sequence[str],
    available_keys: list[str],
    engine_order: Sequence[str],
    timeout_seconds: int,
    step: int = 0,
    debug_label: str | None = None,
    logger: Callable[[str], None] | None = None,
) -> AuthModelDecision:
    prompt = _navigation_prompt(
        flow_label=flow_label,
        goal=goal,
        done_when=done_when,
        snapshot=snapshot,
        history=history,
        carry_forward=carry_forward,
        allowed_urls=allowed_urls,
        available_keys=available_keys,
    )
    errors: list[str] = []
    for engine in engine_order:
        normalized = str(engine or "").strip().lower()
        if not _navigation_engine_available(normalized):
            errors.append(f"{normalized}: unavailable")
            if logger is not None:
                logger(f"web navigator skipped unavailable engine: {normalized}")
            continue
        if normalized == "codex":
            try:
                decision = _request_codex_navigation_decision(
                    prompt,
                    timeout_seconds=timeout_seconds,
                    debug_label=debug_label,
                    step=step,
                )
                if logger is not None:
                    logger("web navigator decision source: codex")
                return decision
            except Exception as exc:
                errors.append(f"codex: {exc}")
                if logger is not None:
                    logger(f"web navigator codex fallback: {exc}")
                continue
        if normalized == "claude":
            try:
                decision = _request_claude_navigation_decision(
                    prompt,
                    timeout_seconds=timeout_seconds,
                    debug_label=debug_label,
                    step=step,
                )
                if logger is not None:
                    logger("web navigator decision source: claude")
                return decision
            except Exception as exc:
                errors.append(f"claude: {exc}")
                if logger is not None:
                    logger(f"web navigator claude fallback: {exc}")
                continue
    detail = "; ".join(errors) if errors else "no navigation engines were configured"
    raise WebNavigatorError(f"unable to obtain a web-navigation decision: {detail}")


def _request_codex_navigation_decision(
    prompt: str,
    *,
    timeout_seconds: int,
    debug_label: str | None = None,
    step: int = 0,
) -> AuthModelDecision:
    with tempfile.TemporaryDirectory(prefix="multishell-web-nav-") as tmp_root:
        tmp_path = Path(tmp_root)
        schema_path = tmp_path / "schema.json"
        output_path = tmp_path / "decision.json"
        schema_path.write_text(json.dumps(_navigation_output_schema(), ensure_ascii=True), encoding="utf-8")
        codex_home = agent_home(DEFAULT_CODEX_NAVIGATOR_AGENT)
        env = child_env(os.environ.copy(), role="web-navigator", agent="codex", home=codex_home)
        args = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--cd",
            "/tmp",
            "--ephemeral",
            "--model",
            WEB_NAVIGATOR_CODEX_MODEL,
            "--config",
            f'model_reasoning_effort="{WEB_NAVIGATOR_CODEX_REASONING_EFFORT}"',
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
            prompt,
        ]
        try:
            result = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            _write_navigation_trace(
                debug_label=debug_label,
                engine="codex",
                step=step,
                prompt=prompt,
                stdout=_coerce_timeout_stream(exc.stdout),
                stderr=_coerce_timeout_stream(exc.stderr),
                returncode=-1,
                structured_output=output_path.read_text(encoding="utf-8") if output_path.exists() else "",
            )
            raise WebNavigatorError(f"codex timed out after {timeout_seconds}s") from exc
        _write_navigation_trace(
            debug_label=debug_label,
            engine="codex",
            step=step,
            prompt=prompt,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            structured_output=output_path.read_text(encoding="utf-8") if output_path.exists() else "",
        )
        if result.returncode != 0:
            raise WebNavigatorError(_format_cli_failure("codex", result))
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise WebNavigatorError(f"codex did not write valid structured output: {exc}") from exc
        return _coerce_navigation_decision(payload)


def _request_claude_navigation_decision(
    prompt: str,
    *,
    timeout_seconds: int,
    debug_label: str | None = None,
    step: int = 0,
) -> AuthModelDecision:
    env = child_env(
        os.environ.copy(),
        role="web-navigator",
        agent="claude",
        home=claude_home(DEFAULT_CLAUDE_NAVIGATOR_AGENT),
    )
    env.pop("ANTHROPIC_API_KEY", None)
    args = [
        "claude",
        "-p",
        "--output-format",
        CLAUDE_OUTPUT_FORMAT,
        "--json-schema",
        json.dumps(_navigation_output_schema(), ensure_ascii=True),
        "--permission-mode",
        "bypassPermissions",
        "--no-session-persistence",
        "--tools",
        "",
        "--model",
        WEB_NAVIGATOR_CLAUDE_MODEL,
        "--effort",
        WEB_NAVIGATOR_CLAUDE_EFFORT,
    ]
    try:
        result = subprocess.run(
            args,
            input=prompt,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds,
            cwd="/tmp",
        )
    except subprocess.TimeoutExpired as exc:
        _write_navigation_trace(
            debug_label=debug_label,
            engine="claude",
            step=step,
            prompt=prompt,
            stdout=_coerce_timeout_stream(exc.stdout),
            stderr=_coerce_timeout_stream(exc.stderr),
            returncode=-1,
            structured_output=_coerce_timeout_stream(exc.stdout),
        )
        raise WebNavigatorError(f"claude timed out after {timeout_seconds}s") from exc
    _write_navigation_trace(
        debug_label=debug_label,
        engine="claude",
        step=step,
        prompt=prompt,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        structured_output=result.stdout,
    )
    if result.returncode != 0:
        raise WebNavigatorError(_format_cli_failure("claude", result))
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise WebNavigatorError(f"claude did not return valid JSON output: {exc}") from exc
    structured = payload.get("structured_output")
    if not isinstance(structured, dict):
        raise WebNavigatorError("claude output did not include structured_output")
    return _coerce_navigation_decision(structured)


def _capture_snapshot_with_recovery(
    page: object,
    *,
    logger: Callable[[str], None] | None,
    recovery_label: str,
    preferred_urls: Sequence[str],
) -> tuple[object, dict[str, object]]:
    try:
        return page, capture_auth_snapshot(page)
    except Exception as exc:
        wait_fn = getattr(page, "wait_for_timeout", None)
        last_exc = exc
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if callable(wait_fn):
                try:
                    wait_fn(250)
                except Exception:
                    time.sleep(0.25)
            else:
                time.sleep(0.25)
            try:
                return page, capture_auth_snapshot(page)
            except Exception as retry_exc:
                last_exc = retry_exc
            recovered = _recover_alternate_page(page, preferred_urls=preferred_urls)
            if recovered is not None:
                recovered_page, snapshot = recovered
                if logger is not None:
                    logger(
                        "web navigator switched to a replacement page "
                        f"{summarize_snapshot(snapshot)} after {recovery_label} ({last_exc})"
                    )
                return recovered_page, snapshot
        recovered = _recover_alternate_page(page, preferred_urls=preferred_urls)
        if recovered is None:
            raise last_exc
        recovered_page, snapshot = recovered
        if logger is not None:
            logger(
                "web navigator switched to a replacement page "
                f"{summarize_snapshot(snapshot)} after {recovery_label} ({last_exc})"
            )
        return recovered_page, snapshot


def _recover_alternate_page(page: object, *, preferred_urls: Sequence[str]) -> tuple[object, dict[str, object]] | None:
    context = getattr(page, "context", None)
    pages = getattr(context, "pages", None)
    if not isinstance(pages, list):
        return None

    preferred_prefixes = tuple(str(url or "").strip() for url in preferred_urls if str(url or "").strip())
    fallback: tuple[object, dict[str, object]] | None = None
    for candidate in reversed(pages):
        if candidate is page:
            continue
        try:
            if candidate.is_closed():
                continue
        except Exception:
            continue
        try:
            snapshot = capture_auth_snapshot(candidate)
        except Exception:
            continue
        url = str(snapshot.get("url") or "")
        if preferred_prefixes and any(url.startswith(prefix) for prefix in preferred_prefixes):
            return candidate, snapshot
        if fallback is None and not _is_blank_snapshot(snapshot):
            fallback = (candidate, snapshot)
    return fallback


def _apply_navigation_action(page: object, action: AuthModelAction, *, secret_values: dict[str, str]) -> None:
    if action.action == "goto":
        destination = action.text.strip()
        if not destination:
            raise AuthModelProtocolError("web navigator goto action omitted the destination URL")
        page.goto(destination, wait_until="domcontentloaded", timeout=120000)
        try:
            page.wait_for_timeout(1000)
        except Exception:
            return
        return
    _apply_auth_action(page, action, secret_values=secret_values)


def _normalize_navigation_action(snapshot: dict[str, object], action: AuthModelAction) -> AuthModelAction:
    if action.action != "fill":
        return action
    return _normalize_auth_action(snapshot, action)


def _compact_action_for_history(action: AuthModelAction) -> str:
    parts = [f"action={action.action}"]
    if action.target:
        parts.append(f"target={action.target[:64]}")
    if action.value_key:
        parts.append(f"value_key={action.value_key[:32]}")
    if action.text:
        parts.append(f"text={action.text[:96]}")
    if action.key:
        parts.append(f"key={action.key[:24]}")
    return " ".join(parts)


@lru_cache(maxsize=8)
def _navigation_engine_available(engine: str) -> bool:
    if engine == "codex":
        return codex_logged_in(DEFAULT_CODEX_NAVIGATOR_AGENT)
    if engine == "claude":
        return has_claude_auth(DEFAULT_CLAUDE_NAVIGATOR_AGENT)
    return bool(engine)


def _navigation_prompt(
    *,
    flow_label: str,
    goal: str,
    done_when: str,
    snapshot: dict[str, object],
    history: list[dict[str, str]],
    carry_forward: list[str],
    allowed_urls: Sequence[str],
    available_keys: list[str],
) -> str:
    payload = {
        "flow": flow_label,
        "goal": goal,
        "done_when": done_when,
        "allowed_urls": [str(url) for url in allowed_urls][:8],
        "available_keys": available_keys[:6],
        "memory": carry_forward[-6:],
        "last": _compact_last_step(history),
        "snapshot": _compact_snapshot_for_navigation(snapshot),
        "surface": _surface_summary(snapshot),
        "reply": "JSON only: action,target,value_key,text,key,seconds,message,carry_forward",
    }
    return "\n".join(
        [
            "You are steering a live browser page from compact structural snapshots.",
            "Return one JSON action only, matching the schema exactly.",
            "Allowed actions: click, fill, press, wait, goto, done, fail.",
            "Use target IDs from snapshot.elements[].id.",
            "Use fill only with value_key from available_keys or a short literal text value.",
            "Use goto only for one of the allowed_urls.",
            "Use done only when the page already satisfies the done_when criteria.",
            "Use fail only for CAPTCHA, MFA/manual verification, hard auth rejection, or an unrecoverable page error.",
            "Keep carry_forward short, non-secret, and at most 6 items.",
            json.dumps(payload, ensure_ascii=True),
        ]
    )


def _compact_snapshot_for_navigation(snapshot: dict[str, object]) -> dict[str, object]:
    compact = _compact_snapshot_for_model(snapshot)
    digest = compact.get("digest")
    if isinstance(digest, dict):
        compact["digest"] = {
            "headings": list(digest.get("headings", []))[:2],
            "alerts": list(digest.get("alerts", []))[:2],
            "labels": list(digest.get("labels", []))[:4],
            "supporting": list(digest.get("supporting", []))[:3],
        }
    elements = compact.get("elements")
    if isinstance(elements, list):
        compact["elements"] = elements[:12]
    return compact


def _navigation_output_schema() -> dict[str, object]:
    properties = {
        "action": {"type": "string", "enum": sorted(NAVIGATION_ALLOWED_ACTIONS)},
        "target": {"type": "string"},
        "value_key": {"type": "string"},
        "text": {"type": "string"},
        "key": {"type": "string"},
        "seconds": {"type": "number"},
        "message": {"type": "string"},
        "carry_forward": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _coerce_navigation_decision(payload: dict[str, object]) -> AuthModelDecision:
    action = str(payload.get("action") or "").strip().lower()
    if action not in NAVIGATION_ALLOWED_ACTIONS:
        raise AuthModelProtocolError(f"web navigator returned unsupported action: {action!r}")
    target = str(payload.get("target") or "").strip()
    value_key = str(payload.get("value_key") or "").strip()
    text = str(payload.get("text") or "")
    key = str(payload.get("key") or "").strip()
    message = str(payload.get("message") or "").strip()
    seconds_raw = payload.get("seconds", 1.0)
    try:
        seconds = float(seconds_raw)
    except (TypeError, ValueError):
        seconds = 1.0
    carry_forward_raw = payload.get("carry_forward")
    carry_forward: list[str] = []
    if isinstance(carry_forward_raw, list):
        for item in carry_forward_raw[:6]:
            text_item = str(item or "").strip()
            if text_item:
                carry_forward.append(text_item[:160])
    return AuthModelDecision(
        action=AuthModelAction(
            action=action,
            target=target,
            value_key=value_key,
            text=text,
            key=key,
            seconds=max(0.0, min(10.0, seconds)),
            message=message,
        ),
        carry_forward=tuple(carry_forward),
    )


def _format_cli_failure(engine: str, result: subprocess.CompletedProcess[str]) -> str:
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    detail_parts = [f"{engine} exited with code {result.returncode}"]
    if stdout:
        detail_parts.append(f"stdout={stdout[:240]}")
    if stderr:
        detail_parts.append(f"stderr={stderr[:240]}")
    return "; ".join(detail_parts)


def _shortcut_navigation_decision(
    *,
    flow_label: str,
    snapshot: dict[str, object],
    available_keys: list[str],
    done_when: str,
) -> AuthModelDecision | None:
    ready = _ready_shortcut(flow_label, snapshot)
    if ready is not None:
        return ready

    if "password" in available_keys:
        password_target = _find_entry_target(snapshot, ("password", "passwd", "current-password"))
        if password_target:
            return AuthModelDecision(
                action=AuthModelAction(action="fill", target=password_target, value_key="password", message="shortcut password fill")
            )
    if "account_email" in available_keys:
        email_target = _find_entry_target(snapshot, ("email", "username", "identifier"))
        if email_target:
            return AuthModelDecision(
                action=AuthModelAction(action="fill", target=email_target, value_key="account_email", message="shortcut email fill")
            )
    if "device_code" in available_keys:
        device_code_elements = _device_code_input_elements(snapshot)
        if device_code_elements:
            return AuthModelDecision(
                action=AuthModelAction(
                    action="fill",
                    target=str(device_code_elements[0].get("id") or ""),
                    value_key="device_code",
                    message="shortcut device code fill",
                )
            )

    for texts in (
        ("continue with google", "sign in with google"),
        ("log in", "sign in", "try gemini", "continue"),
    ):
        target = _find_click_target(snapshot, texts)
        if target:
            return AuthModelDecision(
                action=AuthModelAction(action="click", target=target, message="shortcut button click")
            )

    advance_target = _find_advance_target(snapshot)
    if advance_target and _filled_entry_count(snapshot) > 0:
        return AuthModelDecision(
            action=AuthModelAction(action="click", target=advance_target, message="shortcut advance click")
        )
    if "smoke test passed" in done_when.lower() and "smoke test passed" in str(snapshot.get("body_text") or "").lower():
        return AuthModelDecision(action=AuthModelAction(action="done", message="shortcut completion"))
    return None


def _ready_shortcut(flow_label: str, snapshot: dict[str, object]) -> AuthModelDecision | None:
    body = str(snapshot.get("body_text") or "").lower()
    url = str(snapshot.get("url") or "").lower()
    surface = _surface_summary(snapshot)
    if flow_label == "chatgpt_pro":
        if _chatgpt_workspace_pending(snapshot):
            return AuthModelDecision(
                action=AuthModelAction(action="wait", seconds=2.0, message="shortcut wait for workspace handoff")
            )
        workspace_target = _chatgpt_workspace_target(snapshot)
        if workspace_target:
            return AuthModelDecision(
                action=AuthModelAction(action="click", target=workspace_target, message="shortcut workspace selection")
            )
    if (
        flow_label == "chatgpt_pro"
        and url.startswith("https://auth.openai.com/workspace")
        and (bool(surface.get("script_like_body")) or not body)
        and not _snapshot_elements(snapshot)
    ):
        return AuthModelDecision(
            action=AuthModelAction(
                action="wait",
                seconds=2.0,
                message="shortcut wait for workspace chooser",
            )
        )
    if flow_label == "chatgpt_pro":
        if _has_prompt_composer(snapshot) and "log in" not in body and "continue with google" not in body:
            return AuthModelDecision(action=AuthModelAction(action="done", message="shortcut chatgpt composer ready"))
    if flow_label == "gemini_deepthink":
        if url.startswith("https://gemini.google.com/app") and not body and not _snapshot_elements(snapshot):
            return AuthModelDecision(
                action=AuthModelAction(action="wait", seconds=2.0, message="shortcut wait for gemini app hydration")
            )
        if _has_prompt_composer(snapshot) and "sign in" not in body and "continue with google" not in body:
            deepthink_target = _find_click_target(snapshot, ("deep think", "2.5 pro"))
            if deepthink_target and not _label_already_selected(snapshot, ("deep think", "2.5 pro")):
                return AuthModelDecision(
                    action=AuthModelAction(action="click", target=deepthink_target, message="shortcut deep think selection")
                )
            return AuthModelDecision(action=AuthModelAction(action="done", message="shortcut gemini composer ready"))
    return None


def _chatgpt_workspace_target(snapshot: dict[str, object]) -> str:
    body = str(snapshot.get("body_text") or "").lower()
    headings = str(snapshot.get("title") or "").lower()
    if "choose a workspace" not in body and "choose a workspace" not in headings:
        return ""
    preferred = ""
    fallback = ""
    for element in _snapshot_elements(snapshot):
        if bool(element.get("disabled")):
            continue
        if str(element.get("name") or "").strip().lower() != "workspace_id":
            continue
        target = str(element.get("id") or "").strip()
        if not target:
            continue
        text = str(element.get("text") or "").strip().lower()
        if "personal account" not in text:
            preferred = target
            break
        if not fallback:
            fallback = target
    return preferred or fallback


def _chatgpt_workspace_pending(snapshot: dict[str, object]) -> bool:
    body = str(snapshot.get("body_text") or "").lower()
    title = str(snapshot.get("title") or "").lower()
    if "choose a workspace" not in body and "choose a workspace" not in title:
        return False
    workspace_buttons = [
        element
        for element in _snapshot_elements(snapshot)
        if str(element.get("name") or "").strip().lower() == "workspace_id"
    ]
    return bool(workspace_buttons) and all(bool(element.get("disabled")) for element in workspace_buttons)


def _find_entry_target(snapshot: dict[str, object], needles: Sequence[str]) -> str:
    for element in _snapshot_elements(snapshot):
        if bool(element.get("disabled")):
            continue
        if bool(element.get("filled")) or int(element.get("valueLength") or 0) > 0:
            continue
        tag = str(element.get("tag") or "").lower()
        if tag not in {"input", "textarea", "select"}:
            continue
        haystack = _element_haystack(element)
        if any(needle in haystack for needle in needles):
            return str(element.get("id") or "")
    return ""


def _find_click_target(snapshot: dict[str, object], labels: Sequence[str]) -> str:
    for element in _snapshot_elements(snapshot):
        if bool(element.get("disabled")):
            continue
        haystack = _element_haystack(element)
        if any(label in haystack for label in labels):
            target = str(element.get("id") or "")
            if target:
                return target
    return ""


def _find_advance_target(snapshot: dict[str, object]) -> str:
    return _find_click_target(snapshot, ("next", "continue", "allow", "authorize", "accept", "submit", "ok", "done", "proceed"))


def _filled_entry_count(snapshot: dict[str, object]) -> int:
    return sum(1 for element in _snapshot_elements(snapshot) if bool(element.get("filled")))


def _has_prompt_composer(snapshot: dict[str, object]) -> bool:
    for element in _snapshot_elements(snapshot):
        if bool(element.get("disabled")):
            continue
        tag = str(element.get("tag") or "").lower()
        role = str(element.get("role") or "").lower()
        haystack = _element_haystack(element)
        if tag == "textarea" or role == "textbox":
            return True
        if tag == "div" and any(token in haystack for token in ("message", "ask", "prompt")):
            return True
    return False


def _label_already_selected(snapshot: dict[str, object], labels: Sequence[str]) -> bool:
    for element in _snapshot_elements(snapshot):
        if not bool(element.get("selected")):
            continue
        haystack = _element_haystack(element)
        if any(label in haystack for label in labels):
            return True
    return False


def _snapshot_elements(snapshot: dict[str, object]) -> list[dict[str, object]]:
    raw = snapshot.get("elements")
    return [element for element in raw if isinstance(element, dict)] if isinstance(raw, list) else []


def _element_haystack(element: dict[str, object]) -> str:
    return " ".join(
        str(element.get(key) or "")
        for key in ("text", "ariaLabel", "placeholder", "name", "htmlId", "autocomplete", "type", "role")
    ).strip().lower()


def _write_navigation_trace(
    *,
    debug_label: str | None,
    engine: str,
    step: int,
    prompt: str,
    stdout: str,
    stderr: str,
    returncode: int,
    structured_output: str,
) -> None:
    if not debug_label:
        return
    trace_dir = state_root() / "debug" / debug_label
    trace_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    trace_path = trace_dir / f"{timestamp}-step{step:02d}-{engine}-decision.json"
    payload = {
        "engine": engine,
        "step": step,
        "returncode": returncode,
        "prompt": prompt,
        "stdout": stdout,
        "stderr": stderr,
        "structured_output": structured_output,
    }
    trace_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _coerce_timeout_stream(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _is_blank_snapshot(snapshot: dict[str, object]) -> bool:
    url = str(snapshot.get("url") or "").strip().lower()
    body = str(snapshot.get("body_text") or "").strip()
    elements = snapshot.get("elements")
    has_elements = isinstance(elements, list) and any(isinstance(element, dict) for element in elements)
    return url in {"", "about:blank"} and not body and not has_elements
