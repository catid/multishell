from __future__ import annotations

import json
import os
import re
import threading
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
AUTH_MODEL_MAX_CONCURRENCY_ENV_VAR = "MULTISHELL_AUTH_MODEL_MAX_CONCURRENCY"

DEFAULT_AUTH_MODEL_API_BASE = "http://127.0.0.1:8080/v1"
DEFAULT_AUTH_MODEL_NAME = "Qwen/Qwen3.5-9B"
DEFAULT_AUTH_MODEL_API_KEY = "EMPTY"

_AUTH_MODEL_SEMAPHORE_LOCK = threading.Lock()
_AUTH_MODEL_SEMAPHORE_CACHE: dict[int, threading.Semaphore] = {}

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

  const collectTexts = (selectors, limit) => {
    const values = [];
    const seen = new Set();
    const nodes = Array.from(document.querySelectorAll(selectors.join(',')));
    for (const node of nodes) {
      if (!visible(node)) continue;
      const text = textOf(node);
      if (!text) continue;
      const key = text.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      values.push(text);
      if (values.length >= limit) break;
    }
    return values;
  };

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
    headings: collectTexts(['h1', 'h2', 'h3', 'h4', '[role="heading"]'], 4),
    labels: collectTexts(['label', 'legend'], 6),
    alerts: collectTexts(['[role="alert"]', '[aria-live="assertive"]', '[aria-live="polite"]'], 4),
    supporting: collectTexts(['p', '[role="note"]', '[role="status"]'], 6),
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
    max_concurrency: int


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
    timeout_seconds = _safe_int(os.environ.get(AUTH_MODEL_TIMEOUT_ENV_VAR), default=180, minimum=5, maximum=900)
    max_steps = _safe_int(os.environ.get(AUTH_MODEL_MAX_STEPS_ENV_VAR), default=16, minimum=1, maximum=40)
    max_concurrency = _safe_int(os.environ.get(AUTH_MODEL_MAX_CONCURRENCY_ENV_VAR), default=1, minimum=1, maximum=16)
    return AuthModelSettings(
        enabled=enabled,
        api_base=api_base.rstrip("/"),
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        max_steps=max_steps,
        max_concurrency=max_concurrency,
    )


def _auth_model_semaphore(limit: int) -> threading.Semaphore:
    normalized = max(1, int(limit))
    with _AUTH_MODEL_SEMAPHORE_LOCK:
        semaphore = _AUTH_MODEL_SEMAPHORE_CACHE.get(normalized)
        if semaphore is None:
            semaphore = threading.Semaphore(normalized)
            _AUTH_MODEL_SEMAPHORE_CACHE[normalized] = semaphore
        return semaphore


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

    available_keys = [
        key
        for key, value in (
            ("account_email", account_email),
            ("password", password),
            ("device_code", device_code),
        )
        if value
    ]
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

        shortcut_action = _shortcut_auth_action(snapshot, device_code=device_code)
        if shortcut_action is not None:
            action = shortcut_action
            carry_forward = []
            if logger is not None:
                logger(f"auth model shortcut: {summarize_action(action)}")
                logger(f"auth model step {step}: {summarize_action(action)}")
        else:
            decision = request_auth_model_decision(
                settings,
                flow_label=flow_label,
                snapshot=snapshot,
                history=history,
                carry_forward=carry_forward,
                available_keys=available_keys,
                logger=logger,
            )
            recovered_decision = _recover_missing_device_code_decision(
                snapshot,
                decision,
                device_code=device_code,
                logger=logger,
            )
            if recovered_decision is not None:
                decision = recovered_decision
            action = decision.action
            carry_forward = list(decision.carry_forward[:6])
            if logger is not None:
                logger(f"auth model step {step}: {summarize_action(action)}")
                if carry_forward:
                    logger(f"auth model carry-forward: {json.dumps(carry_forward, ensure_ascii=True)}")
        normalized_action = _normalize_auth_action(snapshot, action)
        if normalized_action != action:
            if logger is not None:
                logger(
                    "auth model normalized action: "
                    f"{summarize_action(action)} -> {summarize_action(normalized_action)}"
                )
            action = normalized_action

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
        except Exception:
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
            recovered = _recover_alternate_auth_page(page)
            if recovered is not None:
                recovered_page, snapshot = recovered
                if logger is not None:
                    logger(
                        "auth model switched to a replacement auth page "
                        f"{summarize_snapshot(snapshot)} after {recovery_label} ({last_exc})"
                    )
                return recovered_page, snapshot
        recovered = _recover_alternate_auth_page(page)
        if recovered is None:
            raise last_exc
        recovered_page, snapshot = recovered
        if logger is not None:
            logger(
                "auth model switched to a replacement auth page "
                f"{summarize_snapshot(snapshot)} after {recovery_label} ({last_exc})"
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
        if fallback is None and not _is_blank_snapshot(snapshot):
            fallback = (candidate, snapshot)
    return fallback


def _is_blank_snapshot(snapshot: dict[str, object]) -> bool:
    url = str(snapshot.get("url") or "").strip().lower()
    body = str(snapshot.get("body_text") or "").strip()
    elements = snapshot.get("elements")
    has_elements = isinstance(elements, list) and any(isinstance(element, dict) for element in elements)
    return url in {"", "about:blank"} and not body and not has_elements


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
    available_keys: list[str] | None = None,
    logger: Callable[[str], None] | None = None,
) -> AuthModelDecision:
    messages = _auth_messages(
        flow_label=flow_label,
        snapshot=snapshot,
        history=history,
        carry_forward=carry_forward,
        available_keys=available_keys or [],
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
            f"available_keys={json.dumps((available_keys or [])[-6:], ensure_ascii=True)} "
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
        semaphore = _auth_model_semaphore(settings.max_concurrency)
        queue_started_at = time.time()
        acquired_immediately = semaphore.acquire(blocking=False)
        if not acquired_immediately:
            if logger is not None:
                logger(f"auth model queue: waiting for slot (max_concurrency={settings.max_concurrency})")
            semaphore.acquire()
        try:
            waited_seconds = time.time() - queue_started_at
            if logger is not None and waited_seconds >= 0.05:
                logger(f"auth model queue: acquired slot after {waited_seconds:.2f}s")
            with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        finally:
            semaphore.release()
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
    available_keys: list[str],
) -> list[dict[str, str]]:
    model_snapshot = _compact_snapshot_for_model(snapshot)
    system = _auth_system_prompt(flow_label=flow_label, snapshot=snapshot)
    user_payload = {
        "flow": flow_label,
        "keys": ["account_email", "password", "device_code"],
        "available_keys": available_keys[:6],
        "memory": carry_forward[-6:],
        "last": _compact_last_step(history),
        "snapshot": model_snapshot,
        "surface": _surface_summary(snapshot),
        "reply": "JSON only: action,target,value_key,text,key,seconds,message,failure_kind,carry_forward",
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
            compact = {
                "id": str(element.get("id") or "")[:64],
                "tag": str(element.get("tag") or "")[:16],
            }
            for src, dst, limit in (
                ("htmlId", "htmlId", 48),
                ("type", "type", 24),
                ("role", "role", 24),
                ("name", "name", 40),
                ("autocomplete", "autocomplete", 40),
                ("inputMode", "inputMode", 24),
                ("maxLength", "maxLength", 12),
                ("placeholder", "placeholder", 80),
                ("ariaLabel", "ariaLabel", 80),
                ("text", "text", 80),
            ):
                value = str(element.get(src) or "")[:limit]
                if value:
                    compact[dst] = value
            value_length = int(element.get("valueLength") or 0)
            if value_length > 0:
                compact["valueLength"] = value_length
            for key in ("filled", "checked", "selected", "focused", "disabled"):
                if bool(element.get(key)):
                    compact[key] = True
            compact_elements.append(compact)
    return {
        "url": _compact_url_for_model(snapshot.get("url")),
        "title": str(snapshot.get("title") or "")[:120],
        "digest": _structural_digest_for_model(snapshot),
        "elements": compact_elements,
    }


def _compact_last_step(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    last = history[-1]
    parts: list[str] = []
    action = str(last.get("action") or "").strip()
    if action:
        parts.append(f"action={action[:96]}")
    changed = str(last.get("changed") or "").strip()
    if changed:
        parts.append(f"changed={changed[:8]}")
    post_page = str(last.get("post_page") or "").strip()
    if post_page:
        parts.append(f"post={post_page[:160]}")
    carry = str(last.get("carry_forward") or "").strip()
    if carry:
        parts.append(f"mem={carry[:120]}")
    return " ".join(parts)


def _structural_digest_for_model(snapshot: dict[str, object]) -> dict[str, list[str]]:
    return {
        "headings": _compact_text_list(snapshot.get("headings"), limit=4, width=120),
        "alerts": _compact_text_list(snapshot.get("alerts"), limit=4, width=160),
        "labels": _compact_text_list(snapshot.get("labels"), limit=6, width=80),
        "supporting": _compact_supporting_text(snapshot),
    }


def _compact_text_list(raw: object, *, limit: int, width: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for item in raw[:limit]:
        text = " ".join(str(item or "").split())[:width]
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(text)
    return values


def _compact_supporting_text(snapshot: dict[str, object]) -> list[str]:
    collected = _compact_text_list(snapshot.get("supporting"), limit=6, width=140)
    if collected:
        return collected[:4]
    body_text = " ".join(str(snapshot.get("body_text") or "").split())
    if not body_text:
        return []
    chunks: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", body_text):
        text = sentence.strip()[:140]
        if text:
            chunks.append(text)
        if len(chunks) >= 3:
            break
    return chunks


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


def _normalize_auth_action(snapshot: dict[str, object], action: AuthModelAction) -> AuthModelAction:
    if action.action != "fill":
        return action
    target_element = _snapshot_element_by_id(snapshot, action.target)
    if target_element is None or not _is_text_entry_element(target_element):
        return action
    if not bool(target_element.get("filled")):
        return action

    advance_target = _best_advance_target(snapshot)
    if advance_target:
        return AuthModelAction(
            action="click",
            target=advance_target,
            message="advancing from an already-filled field",
        )

    if bool(target_element.get("focused")):
        return AuthModelAction(
            action="press",
            key="Enter",
            message="submitting an already-filled field with Enter",
        )

    return AuthModelAction(
        action="wait",
        seconds=0.5,
        message="waiting after a stale refill request on an already-filled field",
    )


def _snapshot_element_by_id(snapshot: dict[str, object], target: str) -> dict[str, object] | None:
    if not target:
        return None
    raw_elements = snapshot.get("elements")
    if not isinstance(raw_elements, list):
        return None
    for element in raw_elements:
        if not isinstance(element, dict):
            continue
        if str(element.get("id") or "").strip() == target:
            return element
    return None


def _is_text_entry_element(element: dict[str, object]) -> bool:
    tag = str(element.get("tag") or "").lower()
    if tag not in {"input", "textarea", "select"}:
        return False
    element_type = str(element.get("type") or "").lower()
    return element_type not in {
        "button",
        "checkbox",
        "color",
        "file",
        "hidden",
        "image",
        "radio",
        "range",
        "reset",
        "submit",
    }


def _best_advance_target(snapshot: dict[str, object]) -> str:
    raw_elements = snapshot.get("elements")
    if not isinstance(raw_elements, list):
        return ""

    preferred_labels = (
        "next",
        "continue",
        "sign in",
        "log in",
        "submit",
        "authorize",
        "allow",
        "verify",
        "confirm",
        "proceed",
        "finish",
        "done",
        "ok",
    )
    best_target = ""
    best_score: tuple[int, int] | None = None
    for index, element in enumerate(raw_elements):
        if not isinstance(element, dict) or bool(element.get("disabled")):
            continue
        target = str(element.get("id") or "").strip()
        if not target:
            continue
        tag = str(element.get("tag") or "").lower()
        role = str(element.get("role") or "").lower()
        element_type = str(element.get("type") or "").lower()
        if tag not in {"button", "input", "a", "div", "span"} and role not in {"button", "link", "option", "menuitem"}:
            continue
        label = " ".join(
            str(element.get(key) or "")
            for key in ("text", "ariaLabel", "placeholder", "name")
        ).strip().lower()
        if not label and element_type != "submit":
            continue

        rank = next((position for position, needle in enumerate(preferred_labels) if needle in label), None)
        if rank is None and element_type == "submit":
            rank = len(preferred_labels)
        if rank is None:
            continue

        score = (rank, index)
        if best_score is None or score < best_score:
            best_score = score
            best_target = target
    return best_target


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


def _shortcut_auth_action(snapshot: dict[str, object], *, device_code: str) -> AuthModelAction | None:
    recovery_action = _openai_recovery_action(snapshot)
    if recovery_action is not None:
        return recovery_action

    if not device_code or not _is_device_code_surface(snapshot):
        return None
    inputs = _device_code_input_elements(snapshot)
    if not inputs:
        return None

    if not all(bool(element.get("filled")) for element in inputs):
        target = str(inputs[0].get("id") or "").strip()
        if not target:
            return None
        return AuthModelAction(
            action="fill",
            target=target,
            value_key="device_code",
            message="filling the Codex device code directly",
        )

    submit_target = _device_code_submit_target(snapshot)
    if submit_target:
        return AuthModelAction(
            action="click",
            target=submit_target,
            message="submitting the filled Codex device code directly",
        )

    return AuthModelAction(
        action="wait",
        seconds=0.5,
        message="waiting for the Codex device code form to enable Continue",
    )


def _device_code_submit_target(snapshot: dict[str, object]) -> str:
    raw_elements = snapshot.get("elements")
    if not isinstance(raw_elements, list):
        return ""
    for element in raw_elements:
        if not isinstance(element, dict):
            continue
        if bool(element.get("disabled")):
            continue
        tag = str(element.get("tag") or "").lower()
        if tag != "button":
            continue
        text = " ".join(str(element.get("text") or "").split()).lower()
        if any(label in text for label in ("continue", "submit", "authorize")):
            target = str(element.get("id") or "").strip()
            if target:
                return target
    return ""


def _openai_recovery_action(snapshot: dict[str, object]) -> AuthModelAction | None:
    url = str(snapshot.get("url") or "")
    title = str(snapshot.get("title") or "")
    body = str(snapshot.get("body_text") or "")
    if "auth.openai.com" not in url:
        return None

    if "Your session has ended" in title or "Your session has ended" in body:
        target = _find_enabled_element_by_text(snapshot, "log in")
        if target:
            return AuthModelAction(
                action="click",
                target=target,
                message="restarting the OpenAI sign-in flow directly",
            )

    if "Oops, an error occurred!" in title or "Oops, an error occurred!" in body:
        target = _find_enabled_element_by_text(snapshot, "try again")
        if target:
            return AuthModelAction(
                action="click",
                target=target,
                message="retrying the OpenAI sign-in flow directly",
            )

    return None


def _find_enabled_element_by_text(snapshot: dict[str, object], text_fragment: str) -> str:
    raw_elements = snapshot.get("elements")
    if not isinstance(raw_elements, list):
        return ""
    needle = " ".join(text_fragment.split()).lower()
    for element in raw_elements:
        if not isinstance(element, dict):
            continue
        if bool(element.get("disabled")):
            continue
        target = str(element.get("id") or "").strip()
        if not target:
            continue
        text = " ".join(str(element.get("text") or "").split()).lower()
        if needle and needle in text:
            return target
    return ""


def _auth_system_prompt(*, flow_label: str, snapshot: dict[str, object]) -> str:
    hints = _surface_prompt_hints(snapshot)
    hint_text = " ".join(hints)
    flow_detail = "OpenAI/Codex sign-in" if flow_label.startswith("codex/") else "Claude sign-in"
    base = (
        f"You control a headless browser for {flow_detail}. "
        "Reply with one JSON object only. No prose or chain-of-thought. "
        "Fresh context every step: only carry_forward survives. Keep carry_forward short, non-secret, at most 6 items. "
        "Actions: click, fill, press, wait, done, fail. "
        "Use current snapshot over old memory. Element ids are per-step handles only. "
        "For secrets use value_key account_email, password, or device_code; never output literal passwords. "
        "The user payload's available_keys list tells you which secret handles are actually available right now. "
        "If a needed handle is listed there, use that value_key directly instead of claiming the secret is missing. "
        "Never click disabled controls. If a field is already filled, advance instead of refilling. "
        "If all short inputs are filled, treat token entry as complete and advance. "
        "If the last action did not change the page, try something else. "
        "If a page asks you to choose or select and enabled options are present, make the needed selection before Continue or Next. "
        "Do not claim selection unless the snapshot shows it or the page clearly advanced. "
        "Do not treat URL tokens or callback codes as the device code. Use the device_code handle on device-code pages. "
        "If there are no useful controls and the page looks like redirect/bootstrap/security-check text, wait. "
        "If credentials are wrong, return fail with failure_kind no_auth. "
        "If there is captcha, MFA, recovery, security interstitial, transient site/browser/session trouble, or another non-credential blocker, return fail with failure_kind error. "
        "If auth is complete or no auth input is needed anymore, return done."
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

    if _looks_like_script_body(body) and not enabled_elements:
        hints.append("The page body looks like bootstrap or app script output rather than a usable auth form. Wait for the real UI to render.")

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
        "script_like_body": _looks_like_script_body(body_text),
    }


def _looks_like_script_body(body_text: str) -> bool:
    body_lower = body_text.lower()
    return (
        "window." in body_text
        or "__reactroutercontext" in body_lower
        or "sessionstorage" in body_lower
        or "history.replacestate" in body_lower
        or "self.__next_f" in body_lower
        or "__next_f.push" in body_lower
        or "__next_s" in body_lower
    )


def _recover_missing_device_code_decision(
    snapshot: dict[str, object],
    decision: AuthModelDecision,
    *,
    device_code: str,
    logger: Callable[[str], None] | None = None,
) -> AuthModelDecision | None:
    if not device_code:
        return None
    action = decision.action
    if action.action != "fail":
        return None
    if not _is_device_code_surface(snapshot):
        return None
    detail = " ".join(
        part
        for part in [
            action.message,
            *decision.carry_forward,
        ]
        if part
    ).lower()
    if "device code" not in detail:
        return None
    if not any(token in detail for token in ("not provided", "missing", "waiting_for_device_code", "waiting for device code")):
        return None
    inputs = _device_code_input_elements(snapshot)
    if not inputs:
        return None
    target = str(inputs[0].get("id") or "").strip()
    if not target:
        return None
    if logger is not None:
        logger("auth model recovery: device_code handle is available; filling the visible device code inputs")
    return AuthModelDecision(
        action=AuthModelAction(
            action="fill",
            target=target,
            value_key="device_code",
            message="fill the device code from the available handle",
        ),
        carry_forward=(
            "device_code is available via value_key=device_code",
            "after the code is entered, continue when the button enables",
        ),
    )


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
    url = _compact_url_for_model(snapshot.get("url"))
    title = str(snapshot.get("title") or "")
    digest = _structural_digest_for_model(snapshot)
    summary_parts = digest.get("headings", [])[:1] + digest.get("alerts", [])[:1] + digest.get("supporting", [])[:1]
    compact = " | ".join(summary_parts)[:120]
    return f"url={url} title={title!r} digest={compact!r}"


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
