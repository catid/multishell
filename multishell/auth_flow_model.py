from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit


AUTH_MODEL_ENABLED_ENV_VAR = "MULTISHELL_AUTH_MODEL_ENABLED"
AUTH_MODEL_API_BASE_ENV_VAR = "MULTISHELL_AUTH_MODEL_API_BASE"
AUTH_MODEL_API_KEY_ENV_VAR = "MULTISHELL_AUTH_MODEL_API_KEY"
AUTH_MODEL_NAME_ENV_VAR = "MULTISHELL_AUTH_MODEL"
AUTH_MODEL_MAX_STEPS_ENV_VAR = "MULTISHELL_AUTH_MODEL_MAX_STEPS"
AUTH_MODEL_TIMEOUT_ENV_VAR = "MULTISHELL_AUTH_MODEL_TIMEOUT_SECONDS"

DEFAULT_AUTH_MODEL_API_BASE = "http://127.0.0.1:8080/v1"
DEFAULT_AUTH_MODEL_NAME = "Qwen/Qwen3.5-9B"
DEFAULT_AUTH_MODEL_API_KEY = "EMPTY"

_SMOKE_TEST_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Continue with Google</title>
    <style>
      body { font-family: sans-serif; max-width: 640px; margin: 40px auto; line-height: 1.4; }
      .hidden { display: none; }
      label { display: block; margin: 12px 0 6px; }
      input, button { font-size: 16px; padding: 10px; width: 100%; box-sizing: border-box; }
      button { margin-top: 16px; cursor: pointer; }
    </style>
  </head>
  <body>
    <h1>Continue with Google</h1>
    <p>This is a synthetic smoke test for the multishell auth model.</p>

    <section id="step-email">
      <label for="email">Email</label>
      <input id="email" name="identifier" type="email" autocomplete="username" placeholder="Email" />
      <button id="identifierNext" type="button">Next</button>
    </section>

    <section id="step-password" class="hidden">
      <label for="password">Password</label>
      <input id="password" name="Passwd" type="password" autocomplete="current-password" placeholder="Password" />
      <button id="passwordNext" type="button">Sign in</button>
    </section>

    <section id="step-device-code" class="hidden">
      <h2>Use your device code to grant access to Codex CLI</h2>
      <label for="device-code">Device code</label>
      <input id="device-code" name="device_code" type="text" autocomplete="one-time-code" placeholder="Enter code" />
      <button id="deviceCodeNext" type="button">Continue</button>
    </section>

    <section id="step-workspace" class="hidden">
      <h2>Sign in to Codex with ChatGPT</h2>
      <p>Select a workspace before continuing</p>
      <button id="workspaceOption" type="button">Personal Workspace</button>
      <button id="workspaceContinue" type="button" disabled>Continue</button>
    </section>

    <section id="step-success" class="hidden">
      <h2>Smoke test passed</h2>
      <p>Auth model reached the success screen.</p>
    </section>

    <script>
      const email = document.getElementById('email');
      const password = document.getElementById('password');
      const deviceCode = document.getElementById('device-code');
      const stepEmail = document.getElementById('step-email');
      const stepPassword = document.getElementById('step-password');
      const stepDeviceCode = document.getElementById('step-device-code');
      const stepWorkspace = document.getElementById('step-workspace');
      const stepSuccess = document.getElementById('step-success');
      const workspaceOption = document.getElementById('workspaceOption');
      const workspaceContinue = document.getElementById('workspaceContinue');

      const showPassword = () => {
        if (!email.value.trim()) return;
        stepEmail.classList.add('hidden');
        stepPassword.classList.remove('hidden');
        password.focus();
      };

      const showDeviceCode = () => {
        if (!password.value.trim()) return;
        stepPassword.classList.add('hidden');
        stepDeviceCode.classList.remove('hidden');
        deviceCode.focus();
      };

      const showWorkspace = () => {
        if (!deviceCode.value.trim()) return;
        stepDeviceCode.classList.add('hidden');
        stepWorkspace.classList.remove('hidden');
        workspaceOption.focus();
      };

      const showSuccess = () => {
        if (workspaceContinue.disabled) return;
        stepWorkspace.classList.add('hidden');
        stepSuccess.classList.remove('hidden');
        document.title = 'Smoke test passed';
      };

      document.getElementById('identifierNext').addEventListener('click', showPassword);
      document.getElementById('passwordNext').addEventListener('click', showDeviceCode);
      document.getElementById('deviceCodeNext').addEventListener('click', showWorkspace);
      workspaceOption.addEventListener('click', () => {
        workspaceOption.setAttribute('aria-pressed', 'true');
        workspaceContinue.disabled = false;
      });
      workspaceContinue.addEventListener('click', showSuccess);
      email.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          showPassword();
        }
      });
      password.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          showDeviceCode();
        }
      });
      deviceCode.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          showWorkspace();
        }
      });
      workspaceContinue.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          showSuccess();
        }
      });
    </script>
  </body>
</html>
"""

_MODEL_SNAPSHOT_JS = r"""
() => {
  const visible = (element) => {
    if (!element) return false;
    const style = window.getComputedStyle(element);
    if (!style || style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
    const rect = element.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };

  const textOf = (element) => {
    const text = (element.innerText || element.textContent || '').replace(/\s+/g, ' ').trim();
    return text.slice(0, 160);
  };

  const attr = (element, name) => (element.getAttribute(name) || '').trim().slice(0, 160);

  const slugify = (value) => {
    const slug = String(value || '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 48);
    return slug || 'node';
  };

  const stableKeyOf = (element) => {
    const pieces = [
      attr(element, 'id'),
      attr(element, 'name'),
      attr(element, 'autocomplete'),
      attr(element, 'aria-label'),
      attr(element, 'placeholder'),
      textOf(element),
      attr(element, 'type'),
      (element.tagName || '').toLowerCase(),
    ];
    for (const piece of pieces) {
      if (piece) return slugify(piece);
    }
    return 'node';
  };

  const mark = (element, suffix) => {
    const id = `ms-auth-${suffix}`;
    element.setAttribute('data-multishell-auth-id', id);
    return id;
  };

  for (const node of document.querySelectorAll('[data-multishell-auth-id]')) {
    node.removeAttribute('data-multishell-auth-id');
  }

  const elements = [];
  const seenKeys = new Map();
  const selectors = [
    'button',
    'input',
    'textarea',
    'select',
    'a',
    '[role="button"]',
    '[role="link"]',
    '[role="option"]',
    '[role="menuitem"]',
    '[contenteditable="true"]',
  ];
  const nodes = Array.from(document.querySelectorAll(selectors.join(',')));
  for (const node of nodes) {
    if (!visible(node)) continue;
    const stableKey = stableKeyOf(node);
    const seen = (seenKeys.get(stableKey) || 0) + 1;
    seenKeys.set(stableKey, seen);
    const id = mark(node, seen === 1 ? stableKey : `${stableKey}-${seen}`);
    const tag = (node.tagName || '').toLowerCase();
    const isTextualField = ['input', 'textarea', 'select'].includes(tag);
    const currentValue = isTextualField ? String(node.value || '').trim() : '';
    const selected = attr(node, 'aria-selected') === 'true' || attr(node, 'aria-current') === 'true' || attr(node, 'aria-pressed') === 'true';
    elements.push({
      id,
      tag,
      htmlId: attr(node, 'id'),
      type: attr(node, 'type'),
      role: attr(node, 'role'),
      name: attr(node, 'name'),
      autocomplete: attr(node, 'autocomplete'),
      inputMode: attr(node, 'inputmode'),
      maxLength: attr(node, 'maxlength'),
      placeholder: attr(node, 'placeholder'),
      ariaLabel: attr(node, 'aria-label'),
      text: textOf(node),
      filled: currentValue.length > 0,
      valueLength: currentValue.length,
      checked: !!node.checked,
      selected,
      focused: document.activeElement === node,
      disabled: !!node.disabled || attr(node, 'aria-disabled') === 'true',
    });
    if (elements.length >= 40) break;
  }

  const title = document.title || '';
  const bodyText = ((document.body && (document.body.innerText || document.body.textContent)) || '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 1600);

  return {
    url: window.location.href,
    title,
    body_text: bodyText,
    elements,
  };
}
"""


class AuthModelUnavailable(RuntimeError):
    pass


class AuthModelFailure(RuntimeError):
    def __init__(self, message: str, *, failure_kind: str = "error") -> None:
        super().__init__(message)
        self.failure_kind = failure_kind if failure_kind in {"no_auth", "error"} else "error"


class AuthModelProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class AuthModelSettings:
    enabled: bool
    api_base: str
    api_key: str
    model: str
    timeout_seconds: int
    max_steps: int


@dataclass(frozen=True)
class AuthModelAction:
    action: str
    target: str = ""
    value_key: str = ""
    text: str = ""
    key: str = ""
    seconds: float = 1.0
    message: str = ""
    failure_kind: str = ""


@dataclass(frozen=True)
class AuthModelDecision:
    action: AuthModelAction
    carry_forward: tuple[str, ...] = ()


def run_auth_model_smoke_test(*, headed: bool = False) -> None:
    settings = load_auth_model_settings()
    if not settings.enabled:
        raise RuntimeError("auth model is disabled; unset MULTISHELL_AUTH_MODEL_ENABLED=0 before running the smoke test")

    from .runtime import apply_node_warning_suppression, apply_playwright_browser_path

    apply_node_warning_suppression()
    apply_playwright_browser_path()
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run `multishell install-browser` first."
        ) from exc

    from .autologin import _isolated_chrome

    print(f"auth model smoke test: api_base={settings.api_base} model={settings.model}")
    with sync_playwright() as playwright:
        with _isolated_chrome(playwright, "auth-model-smoke-test", headed=headed, progress_label="auth-model-smoke-test") as (
            _browser,
            _context,
            page,
        ):
            page.goto("about:blank", wait_until="domcontentloaded")
            page.set_content(_SMOKE_TEST_HTML, wait_until="domcontentloaded")
            print("auth model smoke test: loaded synthetic login page")
            drive_google_auth_with_model(
                page,
                flow_label="smoke-test/google",
                account_email="smoke@example.com",
                password="smoke-password",
                device_code="ABCD-EFGHI",
                logger=lambda message: print(f"[auth-model-smoke-test] {message}", flush=True),
            )
            body = " ".join(page.locator("body").inner_text().split())
            if "Smoke test passed" not in body:
                raise RuntimeError(f"auth model smoke test did not reach success state: {body[:240]!r}")
    print("auth model smoke test passed")


def load_auth_model_settings() -> AuthModelSettings:
    enabled = os.environ.get(AUTH_MODEL_ENABLED_ENV_VAR, "1").strip().lower() not in {"0", "false", "no", "off"}
    api_base = os.environ.get(AUTH_MODEL_API_BASE_ENV_VAR, DEFAULT_AUTH_MODEL_API_BASE).strip() or DEFAULT_AUTH_MODEL_API_BASE
    api_key = os.environ.get(AUTH_MODEL_API_KEY_ENV_VAR, DEFAULT_AUTH_MODEL_API_KEY).strip() or DEFAULT_AUTH_MODEL_API_KEY
    model = os.environ.get(AUTH_MODEL_NAME_ENV_VAR, DEFAULT_AUTH_MODEL_NAME).strip() or DEFAULT_AUTH_MODEL_NAME
    timeout_seconds = _safe_int(os.environ.get(AUTH_MODEL_TIMEOUT_ENV_VAR), default=60, minimum=5, maximum=300)
    max_steps = _safe_int(os.environ.get(AUTH_MODEL_MAX_STEPS_ENV_VAR), default=16, minimum=1, maximum=40)
    return AuthModelSettings(
        enabled=enabled,
        api_base=api_base.rstrip("/"),
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        max_steps=max_steps,
    )


def drive_google_auth_with_model(
    page: object,
    *,
    flow_label: str,
    account_email: str,
    password: str,
    device_code: str = "",
    logger: Callable[[str], None] | None = None,
) -> bool:
    settings = load_auth_model_settings()
    if not settings.enabled:
        raise AuthModelUnavailable("auth model is disabled")

    current_page = page
    try:
        current_page, snapshot = _capture_snapshot_with_recovery(
            current_page,
            logger=logger,
            recovery_label="before acting",
        )
    except Exception as exc:
        if logger is not None:
            logger(f"auth model page became unavailable before acting; treating it as a navigation handoff ({exc})")
        return True
    if not snapshot_requires_model(snapshot):
        if logger is not None:
            logger(f"auth model skipped: snapshot is outside model-controlled auth surfaces ({summarize_snapshot(snapshot)})")
        return False

    history: list[dict[str, str]] = []
    carry_forward: list[str] = []
    last_repeat_signature = ""
    repeat_count = 0
    last_snapshot_fingerprint = _snapshot_fingerprint(snapshot)
    for step in range(1, settings.max_steps + 1):
        try:
            current_page, snapshot = _capture_snapshot_with_recovery(
                current_page,
                logger=logger,
                recovery_label=f"at step {step}",
            )
        except Exception as exc:
            if logger is not None:
                logger(f"auth model page became unavailable after step {step - 1}; treating it as a navigation handoff ({exc})")
            return True
        if not snapshot_requires_model(snapshot):
            if logger is not None:
                logger(f"auth model transitioned out of Google auth after step {step - 1}")
            return True

        decision = request_auth_model_decision(
            settings,
            flow_label=flow_label,
            snapshot=snapshot,
            history=history,
            carry_forward=carry_forward,
            logger=logger,
        )
        action = decision.action
        carry_forward = list(decision.carry_forward[:6])
        if logger is not None:
            logger(f"auth model step {step}: {summarize_action(action)}")
            if carry_forward:
                logger(f"auth model carry-forward: {json.dumps(carry_forward, ensure_ascii=True)}")

        if action.action == "done":
            return True
        if action.action == "fail":
            raise AuthModelFailure(
                action.message or "auth model reported an unrecoverable auth error",
                failure_kind=action.failure_kind or "error",
            )

        repeat_signature = f"{_snapshot_fingerprint(snapshot)}::{_action_fingerprint(action)}"
        if repeat_signature == last_repeat_signature:
            repeat_count += 1
        else:
            last_repeat_signature = repeat_signature
            repeat_count = 1
        if repeat_count >= 3:
            if logger is not None:
                logger("auth model repeated the same action on the same page state; aborting model control")
            raise AuthModelProtocolError("auth model got stuck repeating the same action on the same page")

        previous_fingerprint = _snapshot_fingerprint(snapshot)
        last_snapshot_fingerprint = previous_fingerprint
        try:
            _apply_auth_action(
                current_page,
                action,
                secret_values={
                    "account_email": account_email,
                    "password": password,
                    "device_code": device_code,
                },
            )
        except AuthModelProtocolError:
            try:
                current_page, current_snapshot = _capture_snapshot_with_recovery(
                    current_page,
                    logger=logger,
                    recovery_label=f"after step {step}",
                )
            except Exception as exc:
                if logger is not None:
                    logger(f"auth model action triggered a navigation handoff while the prior target disappeared ({exc})")
                return True
            if _snapshot_fingerprint(current_snapshot) != previous_fingerprint:
                if logger is not None:
                    logger("auth model action changed the page surface before the original target could be reused; continuing")
                history.append(
                    {
                        "page": summarize_snapshot(snapshot),
                        "action": summarize_action(action),
                        "post_page": summarize_snapshot(current_snapshot),
                        "changed": "true",
                        "carry_forward": "; ".join(carry_forward[:3]),
                    }
                )
                history[:] = history[-4:]
                last_snapshot_fingerprint = _snapshot_fingerprint(current_snapshot)
                continue
            raise
        if action.action in {"click", "press"}:
            _wait_for_surface_change(current_page, previous_fingerprint)
        try:
            current_page, post_snapshot = _capture_snapshot_with_recovery(
                current_page,
                logger=logger,
                recovery_label=f"after step {step}",
            )
        except Exception as exc:
            if logger is not None:
                logger(f"auth model post-action snapshot unavailable ({exc})")
            return True
        post_changed = "true" if _snapshot_fingerprint(post_snapshot) != previous_fingerprint else "false"
        post_summary = summarize_snapshot(post_snapshot)
        last_snapshot_fingerprint = _snapshot_fingerprint(post_snapshot)
        if logger is not None:
            logger(
                "auth model post-action snapshot: "
                f"{post_summary} "
                f"elements={_summarize_elements(post_snapshot.get('elements'))} "
                f"changed={post_changed}"
            )
        history.append(
            {
                "page": summarize_snapshot(snapshot),
                "action": summarize_action(action),
                "post_page": post_summary,
                "changed": post_changed,
                "carry_forward": "; ".join(carry_forward[:3]),
            }
        )
        history[:] = history[-4:]

    if _wait_for_auth_surface_exit(
        current_page,
        last_snapshot_fingerprint,
        logger=logger,
    ):
        return True
    raise AuthModelProtocolError("auth model exhausted its step budget before the page advanced")


def _capture_snapshot_with_recovery(
    page: object,
    *,
    logger: Callable[[str], None] | None = None,
    recovery_label: str,
) -> tuple[object, dict[str, object]]:
    try:
        return page, capture_auth_snapshot(page)
    except Exception as exc:
        recovered = _recover_alternate_auth_page(page)
        if recovered is None:
            raise exc
        recovered_page, snapshot = recovered
        if logger is not None:
            logger(
                "auth model switched to a replacement auth page "
                f"{summarize_snapshot(snapshot)} after {recovery_label} ({exc})"
            )
        return recovered_page, snapshot


def _recover_alternate_auth_page(page: object) -> tuple[object, dict[str, object]] | None:
    context = getattr(page, "context", None)
    pages = getattr(context, "pages", None)
    if not isinstance(pages, list):
        return None

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
        if snapshot_requires_model(snapshot):
            return candidate, snapshot
        if fallback is None:
            fallback = (candidate, snapshot)
    return fallback


def _wait_for_auth_surface_exit(
    page: object,
    previous_fingerprint: str,
    *,
    logger: Callable[[str], None] | None = None,
    timeout_seconds: float = 6.0,
) -> bool:
    deadline = time.time() + max(0.5, timeout_seconds)
    while time.time() < deadline:
        try:
            current_page, snapshot = _capture_snapshot_with_recovery(
                page,
                logger=logger,
                recovery_label="during post-budget grace wait",
            )
        except Exception:
            return True
        if not snapshot_requires_model(snapshot):
            if logger is not None:
                logger("auth model exited model-controlled auth during the post-budget grace wait")
            return True
        if _snapshot_fingerprint(snapshot) != previous_fingerprint:
            page = current_page
            previous_fingerprint = _snapshot_fingerprint(snapshot)
        if hasattr(current_page, "wait_for_timeout"):
            current_page.wait_for_timeout(250)
        else:
            time.sleep(0.25)
    return False


def capture_auth_snapshot(page: object) -> dict[str, object]:
    snapshot = page.evaluate(_MODEL_SNAPSHOT_JS)
    if not isinstance(snapshot, dict):
        raise AuthModelProtocolError("auth snapshot did not return an object")
    return snapshot


def snapshot_requires_model(snapshot: dict[str, object]) -> bool:
    url = str(snapshot.get("url") or "")
    title = str(snapshot.get("title") or "")
    body = str(snapshot.get("body_text") or "")
    body_lower = body.lower()
    title_lower = title.lower()
    if "platform.claude.com/oauth/code/success" in url:
        return False
    if "you’re all set up for claude code" in body_lower or "you're all set up for claude code" in body_lower:
        return False
    if "accounts.google.com" in url:
        return True
    if "auth.openai.com" in url:
        return True
    if "chatgpt.com/auth/login" in url:
        return True
    if "claude.ai/login" in url:
        return True
    if "claude.ai/oauth/authorize" in url or "platform.claude.com/oauth" in url:
        return True
    if "Continue with Google" in body:
        return True
    if "Continue with Google" in title:
        return True
    if "Choose an account" in body:
        return True
    if "Use your device code to grant access to Codex CLI" in body:
        return True
    if "Sign in to Codex with ChatGPT" in body:
        return True
    if "Select a workspace" in body:
        return True
    if "Select organization" in body:
        return True
    if "Logged in as" in body:
        return True
    if "This browser or app may not be secure" in body:
        return True
    if "Couldn’t sign you in" in body or "Couldn't sign you in" in body:
        return True
    if "just a moment..." in title_lower or "just a moment..." in body_lower:
        return True
    if "security verification" in body_lower:
        return True
    if "website uses a security service to protect against malicious bots" in body_lower:
        return True
    return False


def request_auth_model_decision(
    settings: AuthModelSettings,
    *,
    flow_label: str,
    snapshot: dict[str, object],
    history: list[dict[str, str]],
    carry_forward: list[str],
    logger: Callable[[str], None] | None = None,
) -> AuthModelDecision:
    messages = _auth_messages(
        flow_label=flow_label,
        snapshot=snapshot,
        history=history,
        carry_forward=carry_forward,
    )
    if logger is not None:
        logger(f"auth model system prompt: {_truncate_for_log(str(messages[0].get('content') or ''), limit=2200)}")
        logger(f"auth model user payload: {_truncate_for_log(str(messages[1].get('content') or ''), limit=3200)}")
    if logger is not None:
        logger(
            "auth model request: "
            f"api_base={settings.api_base} model={settings.model} "
            f"snapshot={summarize_snapshot(snapshot)} "
            f"elements={_summarize_elements(snapshot.get('elements'))} "
            f"carry_forward={json.dumps(carry_forward[-6:], ensure_ascii=True)}"
        )
    payload = {
        "model": settings.model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": 220,
        "stream": False,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{settings.api_base}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise AuthModelUnavailable(f"unable to reach auth model at {settings.api_base}: {exc.reason}") from exc
    except Exception as exc:
        raise AuthModelUnavailable(f"auth model request failed: {exc}") from exc

    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AuthModelProtocolError(f"auth model returned invalid JSON: {raw[:240]!r}") from exc

    content = _extract_response_content(body)
    metrics = _summarize_response_metrics(body)
    if logger is not None and metrics:
        logger(f"auth model metrics: {metrics}")
    if logger is not None:
        logger(f"auth model raw response: {_truncate_for_log(content, limit=600)}")
    action_payload = _extract_json_object(content)
    if not isinstance(action_payload, dict):
        raise AuthModelProtocolError(f"auth model did not return a JSON object: {content[:240]!r}")

    decision = _coerce_auth_decision(action_payload)
    if logger is not None:
        logger(f"auth model parsed action: {summarize_action(decision.action)}")
        if decision.carry_forward:
            logger(f"auth model parsed carry-forward: {json.dumps(list(decision.carry_forward), ensure_ascii=True)}")
    return decision


def _auth_messages(
    *,
    flow_label: str,
    snapshot: dict[str, object],
    history: list[dict[str, str]],
    carry_forward: list[str],
) -> list[dict[str, str]]:
    model_snapshot = _compact_snapshot_for_model(snapshot)
    system = _auth_system_prompt(flow_label=flow_label, snapshot=snapshot)
    user_payload = {
        "flow": flow_label,
        "allowed_value_keys": ["account_email", "password", "device_code"],
        "carry_forward_contract": {
            "description": (
                "You start fresh on every step. Only the strings you put into carry_forward survive to the next request. "
                "Use carry_forward for short non-secret working notes, references to secret handles, or page-derived facts "
                "that future steps should remember."
            ),
            "rules": [
                "Do not include chain-of-thought.",
                "Do not copy literal passwords or other literal secrets.",
                "Do use handles like account_email, password, and device_code when you want future steps to use those values.",
                "Keep each note under 160 characters and keep at most 6 notes total.",
                "If you want the next step to remember a device code or a pending action, write that into carry_forward explicitly.",
            ],
            "examples": [
                "A segmented token entry is visible; use the right value_key on the first box, then preserve whether the token entry is complete.",
                "A choice already appears selected; reuse that state and advance only when needed.",
                "A click did not materially change the page, so choose a different action on the next step.",
                "A callback page has no controls and mostly script/bootstrap text; wait for the next visible auth surface instead of returning done.",
                "surface_summary says all_short_inputs_filled=true, so advance instead of filling again even if older carry_forward says more input is needed.",
                "Do not copy opaque query parameters or OAuth callback codes into carry_forward as the terminal device code; use the device_code handle instead.",
            ],
        },
        "carry_forward": carry_forward[-6:],
        "history": history[-2:],
        "snapshot": model_snapshot,
        "surface_summary": _surface_summary(snapshot),
        "response_schema": {
            "action": "click|fill|press|wait|done|fail",
            "target": "element id for click/fill, else empty string",
            "value_key": "account_email|password|device_code|''",
            "text": "only for non-secret literal text when necessary, else empty string",
            "key": "keyboard key such as Enter, else empty string",
            "seconds": "wait duration for wait, else 0 or omitted",
            "message": "short status or failure reason",
            "failure_kind": "for fail only: no_auth|error|''",
            "carry_forward": ["full replacement list of short notes for the next step"],
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
    ]


def _compact_snapshot_for_model(snapshot: dict[str, object]) -> dict[str, object]:
    raw_elements = snapshot.get("elements")
    compact_elements: list[dict[str, object]] = []
    if isinstance(raw_elements, list):
        for element in raw_elements[:20]:
            if not isinstance(element, dict):
                continue
            compact_elements.append(
                {
                    "id": str(element.get("id") or "")[:64],
                    "tag": str(element.get("tag") or "")[:16],
                    "htmlId": str(element.get("htmlId") or "")[:48],
                    "type": str(element.get("type") or "")[:24],
                    "role": str(element.get("role") or "")[:24],
                    "name": str(element.get("name") or "")[:40],
                    "autocomplete": str(element.get("autocomplete") or "")[:40],
                    "inputMode": str(element.get("inputMode") or "")[:24],
                    "maxLength": str(element.get("maxLength") or "")[:12],
                    "placeholder": str(element.get("placeholder") or "")[:80],
                    "ariaLabel": str(element.get("ariaLabel") or "")[:80],
                    "text": str(element.get("text") or "")[:80],
                    "filled": bool(element.get("filled")),
                    "valueLength": int(element.get("valueLength") or 0),
                    "checked": bool(element.get("checked")),
                    "selected": bool(element.get("selected")),
                    "focused": bool(element.get("focused")),
                    "disabled": bool(element.get("disabled")),
                }
            )
    return {
        "url": _compact_url_for_model(snapshot.get("url")),
        "title": str(snapshot.get("title") or "")[:120],
        "body_text": str(snapshot.get("body_text") or "")[:900],
        "elements": compact_elements,
    }


def _compact_url_for_model(raw_url: object) -> str:
    text = str(raw_url or "").strip()
    if not text:
        return ""
    base, _, _query = text.partition("?")
    return base[:200]


def _extract_response_content(body: dict[str, object]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AuthModelProtocolError(f"auth model response missing choices: {body!r}")
    first = choices[0]
    if not isinstance(first, dict):
        raise AuthModelProtocolError(f"auth model response choice malformed: {first!r}")
    message = first.get("message")
    if not isinstance(message, dict):
        raise AuthModelProtocolError(f"auth model response missing message: {first!r}")
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text") or ""))
        content = "".join(text_parts)
    return _strip_thinking_markup(str(content or ""))


def _strip_thinking_markup(text: str) -> str:
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", text)
    return cleaned.strip()


def _extract_json_object(text: str) -> object:
    stripped = text.strip()
    if not stripped:
        raise AuthModelProtocolError("auth model returned empty content")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise AuthModelProtocolError(f"auth model did not return JSON content: {stripped[:240]!r}")
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = _repair_auth_payload(candidate)
            if repaired is None:
                raise
            return repaired


def _repair_auth_payload(text: str) -> dict[str, object] | None:
    repaired: dict[str, object] = {}
    for field in ("action", "target", "value_key", "text", "key", "message", "failure_kind"):
        matches = re.findall(rf'"{field}"\s*:\s*"((?:\\.|[^"])*)"', text)
        if not matches:
            continue
        raw_value = matches[-1]
        try:
            repaired[field] = json.loads(f'"{raw_value}"')
        except json.JSONDecodeError:
            repaired[field] = raw_value

    seconds_matches = re.findall(r'"seconds"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if seconds_matches:
        try:
            repaired["seconds"] = float(seconds_matches[-1])
        except ValueError:
            pass

    carry_forward = _extract_json_array_after_last_key(text, "carry_forward")
    if carry_forward is not None:
        repaired["carry_forward"] = carry_forward

    return repaired if "action" in repaired else None


def _extract_json_array_after_last_key(text: str, key: str) -> list[object] | None:
    marker = f'"{key}"'
    start = text.rfind(marker)
    if start < 0:
        return None
    bracket = text.find("[", start)
    if bracket < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(bracket, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "[":
            depth += 1
            continue
        if char == "]":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[bracket : index + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, list) else None
    return None


def _coerce_auth_decision(payload: dict[str, object]) -> AuthModelDecision:
    action = str(payload.get("action") or "").strip().lower()
    if action not in {"click", "fill", "press", "wait", "done", "fail"}:
        raise AuthModelProtocolError(f"auth model returned unsupported action: {action!r}")
    target = str(payload.get("target") or "").strip()
    value_key = str(payload.get("value_key") or "").strip()
    text = str(payload.get("text") or "")
    key = str(payload.get("key") or "").strip()
    message = str(payload.get("message") or "").strip()
    failure_kind = str(payload.get("failure_kind") or "").strip().lower()
    if failure_kind not in {"", "no_auth", "error"}:
        failure_kind = ""
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
            failure_kind=failure_kind,
        ),
        carry_forward=tuple(carry_forward),
    )


def _apply_auth_action(page: object, action: AuthModelAction, *, secret_values: dict[str, str]) -> None:
    if action.action == "click":
        _locate_tagged_element(page, action.target).click(timeout=5000)
        _settle_after_action(page)
        return
    if action.action == "fill":
        value = secret_values.get(action.value_key, "")
        if not value and action.text:
            value = action.text
        if not value:
            raise AuthModelProtocolError(f"auth model fill action is missing a usable value: {action!r}")
        if action.value_key == "device_code":
            _fill_visible_device_code_inputs(page, value)
            _settle_after_action(page)
            return
        locator = _locate_tagged_element(page, action.target)
        try:
            locator.click(timeout=5000)
            page.keyboard.press("Control+A")
            page.keyboard.type(value, delay=20)
        except Exception:
            locator.fill(value, timeout=5000)
        _settle_after_action(page)
        return
    if action.action == "press":
        key = action.key or "Enter"
        page.keyboard.press(key)
        _settle_after_action(page)
        return
    if action.action == "wait":
        try:
            page.wait_for_timeout(int(action.seconds * 1000))
        except Exception:
            return
        return
    raise AuthModelProtocolError(f"unsupported executable auth action: {action.action!r}")


def _locate_tagged_element(page: object, target: str):
    if not target:
        raise AuthModelProtocolError("auth model action omitted the target element id")
    locator = page.locator(f'[data-multishell-auth-id="{target}"]')
    try:
        count = locator.count()
    except Exception as exc:
        raise AuthModelProtocolError(f"auth model targeted a missing element id: {target!r}") from exc
    if count <= 0:
        raise AuthModelProtocolError(f"auth model targeted a missing element id: {target!r}")
    for index in range(count):
        candidate = locator.nth(index)
        try:
            candidate.wait_for(state="visible", timeout=750)
            return candidate
        except Exception:
            continue
    raise AuthModelProtocolError(f"auth model targeted a missing element id: {target!r}")


def _settle_after_action(page: object) -> None:
    try:
        page.wait_for_load_state("domcontentloaded", timeout=5000)
    except Exception:
        pass


def _device_code_input_elements(snapshot: dict[str, object]) -> list[dict[str, object]]:
    if not _is_device_code_surface(snapshot):
        return []
    raw_elements = snapshot.get("elements")
    if not isinstance(raw_elements, list):
        return []
    matches: list[dict[str, object]] = []
    for element in raw_elements:
        if not isinstance(element, dict):
            continue
        if str(element.get("tag") or "").lower() != "input":
            continue
        haystack = " ".join(
            [
                str(element.get("type") or ""),
                str(element.get("name") or ""),
                str(element.get("autocomplete") or ""),
                str(element.get("placeholder") or ""),
                str(element.get("ariaLabel") or ""),
                str(element.get("text") or ""),
            ]
        ).lower()
        if "password" in haystack or "email" in haystack:
            continue
        if any(token in haystack for token in ("code", "otp", "one-time", "character")):
            matches.append(element)
            continue
        if str(element.get("type") or "").lower() in {"text", "tel", "number", ""}:
            matches.append(element)
    return matches


def _is_device_code_surface(snapshot: dict[str, object]) -> bool:
    url = str(snapshot.get("url") or "").lower()
    title = str(snapshot.get("title") or "").lower()
    body = str(snapshot.get("body_text") or "").lower()
    if "use your device code to grant access to codex cli" in body:
        return True
    if "/deviceauth/" in url or "/codex/device" in url:
        return True
    if "device code" in title and "codex" in body:
        return True
    return False


def _auth_system_prompt(*, flow_label: str, snapshot: dict[str, object]) -> str:
    hints = _surface_prompt_hints(snapshot)
    hint_text = " ".join(hints)
    flow_detail = "OpenAI/Codex sign-in" if flow_label.startswith("codex/") else "Claude sign-in"
    base = (
        f"You control a headless browser for {flow_detail}. "
        "Return a single JSON object only. "
        "Do not output chain-of-thought, explanations, markdown, or prose outside the JSON object. "
        "You start from a fresh context every step. If you want future steps to remember something, put it into carry_forward. "
        "Allowed actions are click, fill, press, wait, done, fail. "
        "For fill, prefer value_key account_email, password, or device_code instead of literal secrets. "
        "Never reveal or request the literal password. "
        "Use the provided element ids exactly as shown. "
        "Those ids are temporary harness handles for the current step only, not site-stable ids. "
        "Use text, ariaLabel, placeholder, autocomplete, inputMode, maxLength, selected, valueLength, filled, checked, focused, and disabled to infer the form state. "
        "Trust the current snapshot and surface_summary over older carry_forward notes when they disagree. "
        "Never click a disabled element. "
        "If a relevant field is already filled, prefer advancing instead of filling again. "
        "If surface_summary.all_short_inputs_filled is true, treat the segmented token entry as complete and prefer click or press to advance instead of fill. "
        "If the previous action did not materially change the page, do not repeat it blindly; choose a different strategy. "
        "If the page text says choose or select and actionable options are present, make the required selection before advancing with Continue or Next. "
        "Do not claim an account, workspace, or option is selected unless the current snapshot shows selected state or the page clearly advanced after your action. "
        "Do not treat opaque query parameters, OAuth callback codes, or long URL tokens as the human-entered device code. "
        "When a device-code surface appears, use the provided device_code handle instead of inferring a code from the URL. "
        "Do not return done merely because a callback or redirect page has no visible controls. "
        "If the page has zero interactive elements and the body looks like bootstrap or script text, prefer wait for the next auth surface. "
        "If a page is still redirecting or running a security check and there is no clear interactive control, prefer wait. "
        "If credentials are clearly wrong, the account does not exist, or access is denied because the sign-in details are invalid, return fail with failure_kind no_auth and a short reason. "
        "If the page shows a transient site problem, expired session, retry-later situation, security interstitial you cannot get through, browser error, or any other non-credential failure, return fail with failure_kind error and a short reason. "
        "If the page asks for captcha, phone verification, recovery email, 2-step verification, or any unknown manual challenge, return fail with failure_kind error and a short reason. "
        "If the login flow is complete or the page is no longer asking for auth input, return done."
    )
    if hint_text:
        base = f"{base} Surface guidance: {hint_text}"
    return base


def _surface_prompt_hints(snapshot: dict[str, object]) -> list[str]:
    hints: list[str] = []
    body = str(snapshot.get("body_text") or "")
    body_lower = body.lower()
    title_lower = str(snapshot.get("title") or "").lower()
    raw_elements = snapshot.get("elements")
    elements = [element for element in raw_elements if isinstance(element, dict)] if isinstance(raw_elements, list) else []
    enabled_elements = [element for element in elements if not bool(element.get("disabled"))]
    short_text_inputs = []
    for element in elements:
        if str(element.get("tag") or "").lower() != "input":
            continue
        if bool(element.get("disabled")):
            continue
        max_length = str(element.get("maxLength") or "").strip()
        if max_length in {"1", "2"}:
            short_text_inputs.append(element)

    if short_text_inputs:
        hints.append(
            f"There are {len(short_text_inputs)} short input boxes. If they behave like a segmented token entry, "
            "you may target the first visible box with the appropriate value_key and preserve that plan in carry_forward."
        )

    selected_count = sum(1 for element in elements if bool(element.get("selected")))
    if selected_count:
        hints.append(
            f"{selected_count} element(s) already appear selected or pressed. Reuse that state instead of re-selecting unless the page suggests otherwise."
        )

    if not enabled_elements and (not body_lower or "just a moment..." in title_lower or "security verification" in body_lower):
        hints.append("No enabled interactive controls are visible right now; waiting is often safer than guessing.")

    return hints


def _surface_summary(snapshot: dict[str, object]) -> dict[str, int | bool]:
    raw_elements = snapshot.get("elements")
    elements = [element for element in raw_elements if isinstance(element, dict)] if isinstance(raw_elements, list) else []
    body_text = str(snapshot.get("body_text") or "")
    body_lower = body_text.lower()
    interactive_count = len(elements)
    enabled_count = sum(1 for element in elements if not bool(element.get("disabled")))
    input_count = sum(1 for element in elements if str(element.get("tag") or "").lower() in {"input", "textarea", "select"})
    short_input_count = 0
    filled_short_input_count = 0
    selected_count = 0
    for element in elements:
        if bool(element.get("selected")):
            selected_count += 1
        if str(element.get("tag") or "").lower() != "input":
            continue
        max_length = str(element.get("maxLength") or "").strip()
        if max_length not in {"1", "2"}:
            continue
        short_input_count += 1
        if bool(element.get("filled")):
            filled_short_input_count += 1
    return {
        "interactive_count": interactive_count,
        "enabled_count": enabled_count,
        "input_count": input_count,
        "short_input_count": short_input_count,
        "filled_short_input_count": filled_short_input_count,
        "all_short_inputs_filled": short_input_count > 0 and filled_short_input_count == short_input_count,
        "selected_count": selected_count,
        "script_like_body": (
            "window." in body_text
            or "__reactroutercontext" in body_lower
            or "sessionstorage" in body_lower
            or "history.replaceState".lower() in body_lower
        ),
    }


def _fill_visible_device_code_inputs(page: object, device_code: str) -> None:
    visible_inputs = page.locator("input:visible")
    input_count = visible_inputs.count()
    if input_count <= 0:
        raise AuthModelProtocolError(f"device code page has no visible inputs at {getattr(page, 'url', '')!r}")

    normalized = device_code.replace("-", "")
    if input_count == 1:
        input_field = visible_inputs.first
        try:
            input_field.click(timeout=5000)
            page.keyboard.press("Control+A")
            page.keyboard.type(device_code, delay=20)
            return
        except Exception:
            input_field.fill(device_code, timeout=5000)
            return

    max_lengths: list[str | None] = []
    for index in range(input_count):
        try:
            max_lengths.append(visible_inputs.nth(index).get_attribute("maxlength"))
        except Exception:
            max_lengths.append(None)

    if all(length in {None, "1", "2"} for length in max_lengths):
        try:
            visible_inputs.first.click(timeout=5000)
            page.keyboard.type(normalized, delay=35)
            return
        except Exception:
            pass
        for index, char in enumerate(normalized[:input_count]):
            field = visible_inputs.nth(index)
            try:
                field.click(timeout=5000)
                page.keyboard.press("Control+A")
                page.keyboard.type(char, delay=20)
            except Exception:
                field.fill(char, timeout=5000)
        return

    try:
        visible_inputs.first.click(timeout=5000)
        page.keyboard.press("Control+A")
        page.keyboard.type(device_code, delay=20)
    except Exception:
        visible_inputs.first.fill(device_code, timeout=5000)
    try:
        page.wait_for_timeout(500)
    except Exception:
        pass


def _wait_for_surface_change(page: object, previous_fingerprint: str, *, timeout_seconds: float = 6.0) -> None:
    deadline = time.time() + max(0.5, timeout_seconds)
    while time.time() < deadline:
        try:
            snapshot = capture_auth_snapshot(page)
        except Exception:
            return
        if _snapshot_fingerprint(snapshot) != previous_fingerprint:
            return
        if hasattr(page, "wait_for_timeout"):
            page.wait_for_timeout(250)
        else:
            time.sleep(0.25)


def summarize_snapshot(snapshot: dict[str, object]) -> str:
    url = str(snapshot.get("url") or "")
    title = str(snapshot.get("title") or "")
    body = str(snapshot.get("body_text") or "")
    compact = " ".join(body.split())[:120]
    return f"url={url} title={title!r} body={compact!r}"


def summarize_action(action: AuthModelAction) -> str:
    detail = action.message or action.action
    if action.action == "fail" and action.failure_kind:
        detail = f"{detail} failure_kind={action.failure_kind}"
    if action.target:
        detail = f"{detail} target={action.target}"
    if action.value_key:
        detail = f"{detail} value_key={action.value_key}"
    if action.key:
        detail = f"{detail} key={action.key}"
    return detail


def _summarize_elements(raw_elements: object) -> str:
    if not isinstance(raw_elements, list):
        return "[]"
    parts: list[str] = []
    for element in raw_elements[:8]:
        if not isinstance(element, dict):
            continue
        elem_id = str(element.get("id") or "")
        tag = str(element.get("tag") or "")
        label = str(element.get("text") or element.get("placeholder") or element.get("ariaLabel") or element.get("name") or "")
        elem_type = str(element.get("type") or "")
        disabled = " disabled" if element.get("disabled") else ""
        filled = " filled" if element.get("filled") else ""
        focused = " focused" if element.get("focused") else ""
        parts.append(f"{elem_id}:{tag}:{elem_type}:{label[:40]}{disabled}{filled}{focused}")
    return "[" + ", ".join(parts) + "]"


def _summarize_response_metrics(body: dict[str, object]) -> str:
    parts: list[str] = []
    usage = body.get("usage")
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if isinstance(prompt_tokens, int):
            parts.append(f"prompt_tokens={prompt_tokens}")
        if isinstance(completion_tokens, int):
            parts.append(f"completion_tokens={completion_tokens}")
        if isinstance(total_tokens, int):
            parts.append(f"total_tokens={total_tokens}")
    timings = body.get("timings")
    if isinstance(timings, dict):
        prompt_tps = _coerce_float(timings.get("prompt_per_second"))
        decode_tps = _coerce_float(timings.get("predicted_per_second"))
        if prompt_tps is not None:
            parts.append(f"prompt_tps={prompt_tps:.2f}")
        if decode_tps is not None:
            parts.append(f"decode_tps={decode_tps:.2f}")
    return " ".join(parts)


def _truncate_for_log(text: str, *, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _snapshot_fingerprint(snapshot: dict[str, object]) -> str:
    normalized_elements: list[object] = []
    raw_elements = snapshot.get("elements")
    if isinstance(raw_elements, list):
        for element in raw_elements:
            if not isinstance(element, dict):
                normalized_elements.append(element)
                continue
            normalized = dict(element)
            normalized.pop("focused", None)
            normalized_elements.append(normalized)
    payload = {
        "url": snapshot.get("url"),
        "title": snapshot.get("title"),
        "body_text": snapshot.get("body_text"),
        "elements": normalized_elements,
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _action_fingerprint(action: AuthModelAction) -> str:
    payload = {
        "action": action.action,
        "target": action.target,
        "value_key": action.value_key,
        "text": action.text,
        "key": action.key,
        "seconds": round(action.seconds, 3),
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _safe_int(raw: str | None, *, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(raw).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _coerce_float(raw: object) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
