from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from .autologin import (
    _body_text,
    _click_first,
    _click_google_and_capture_page,
    _click_optional,
    _complete_workspace_consent,
    _fill_first,
    _has_visible,
    _isolated_chrome,
    _normalize_openai_login_entry,
    _page_title,
    _write_debug_artifacts,
)
from .config import email_env_var, gemini_email_env_var, gemini_password_env_var, password_env_var


SUPPORTED_PROVIDERS = frozenset({"chatgpt_pro", "gemini_deepthink"})
CHATGPT_LOGIN_URL = "https://auth.openai.com/log-in?redirect_url=https%3A%2F%2Fchatgpt.com%2F"


def _now() -> float:
    return time.time()


class WebReasonerError(RuntimeError):
    pass


class _JobCanceled(WebReasonerError):
    pass


@dataclass(frozen=True)
class WebReasonerEvent:
    ts: float
    kind: str
    provider: str
    job_id: str
    account_agent: str
    message: str
    data: dict[str, object] = field(default_factory=dict)


@dataclass
class WebReasonerJob:
    id: str
    provider: str
    account_agent: str
    prompt: str
    label: str
    timeout_seconds: int
    status: str = "queued"
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    started_at: float | None = None
    finished_at: float | None = None
    result: str = ""
    error: str | None = None
    cancel_requested: bool = False

    def snapshot(self) -> dict[str, object]:
        preview = self.prompt if len(self.prompt) <= 160 else f"{self.prompt[:157]}..."
        return {
            "id": self.id,
            "job_id": self.id,
            "provider": self.provider,
            "account_agent": self.account_agent,
            "agent": self.account_agent,
            "label": self.label,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "timeout_seconds": self.timeout_seconds,
            "prompt_preview": preview,
            "cancel_requested": self.cancel_requested,
            "result": self.result,
            "result_text": self.result,
            "error": self.error,
        }


class WebReasonerManager:
    def __init__(
        self,
        event_callback: Callable[[WebReasonerEvent], None] | None = None,
        callback: Callable[[WebReasonerEvent], None] | None = None,
    ) -> None:
        self._jobs: dict[str, WebReasonerJob] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel_flags: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._event_callback = event_callback or callback

    def start_job(
        self,
        provider: str,
        account_agent: str | None = None,
        prompt: str | None = None,
        *,
        agent: str | None = None,
        label: str | None = None,
        timeout_seconds: int = 900,
    ) -> dict[str, object]:
        if provider not in SUPPORTED_PROVIDERS:
            raise WebReasonerError(f"unsupported web reasoner provider: {provider}")

        resolved_agent = str(account_agent or agent or "").strip()
        resolved_prompt = str(prompt or "").strip()
        if not resolved_agent:
            raise WebReasonerError("account_agent is required")
        if not resolved_prompt:
            raise WebReasonerError("prompt is required")

        try:
            normalized_timeout = int(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise WebReasonerError("timeout_seconds must be an integer") from exc

        resolved_label = str(label or "").strip() or resolved_prompt[:72].strip() or provider

        job = WebReasonerJob(
            id=str(uuid.uuid4()),
            provider=provider,
            account_agent=resolved_agent,
            prompt=resolved_prompt,
            label=resolved_label,
            timeout_seconds=max(30, min(7200, normalized_timeout)),
        )
        cancel_flag = threading.Event()
        thread = threading.Thread(
            target=self._run_job,
            args=(job.id,),
            name=f"web-reasoner-{provider}-{job.id[:8]}",
            daemon=True,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._threads[job.id] = thread
            self._cancel_flags[job.id] = cancel_flag
        thread.start()
        return job.snapshot()

    def job_snapshot(self, job_id: str) -> dict[str, object] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.snapshot() if job is not None else None

    def snapshot(self, job_id: str) -> dict[str, object]:
        snapshot = self.job_snapshot(job_id)
        if snapshot is None:
            raise WebReasonerError(f"unknown job: {job_id}")
        return snapshot

    def list_jobs(self, provider: str | None = None) -> list[dict[str, object]]:
        with self._lock:
            jobs = list(self._jobs.values())
        if provider:
            jobs = [job for job in jobs if job.provider == provider]
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return [job.snapshot() for job in jobs]

    def cancel_job(self, job_id: str) -> dict[str, object] | None:
        emit_final_cancel = False
        with self._lock:
            job = self._jobs.get(job_id)
            flag = self._cancel_flags.get(job_id)
            if job is None or flag is None:
                return None
            if job.status in {"completed", "failed", "canceled"}:
                return job.snapshot()
            job.cancel_requested = True
            job.updated_at = _now()
            flag.set()
            if job.status == "queued":
                job.status = "canceled"
                job.error = "canceled before start"
                job.finished_at = job.updated_at
                emit_final_cancel = True
            elif job.status != "canceling":
                job.status = "canceling"
            snapshot = job.snapshot()
        if emit_final_cancel:
            self._emit_event(job, "job_canceled", "cancel requested")
        return snapshot

    def stop(self) -> None:
        with self._lock:
            job_ids = list(self._cancel_flags)
            threads = list(self._threads.values())
        for job_id in job_ids:
            self.cancel_job(job_id)
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=5.0)

    def _run_job(self, job_id: str) -> None:
        page = None
        try:
            job = self._mark_running(job_id)
            if job is None:
                return
            self._emit_event(job, "job_started", f"{job.provider} job started")
            result = self._execute_job(job)
            self._finalize_success(job_id, result)
        except _JobCanceled:
            self._finalize_canceled(job_id, "canceled")
        except Exception as exc:
            page = getattr(exc, "_web_reasoner_page", None)
            if page is None:
                page = self._page_for_debug(job_id)
            if page is not None:
                debug_target = self._debug_target(job_id)
                _write_debug_artifacts(page, debug_target)
            self._finalize_failure(job_id, str(exc))

    def _mark_running(self, job_id: str) -> WebReasonerJob | None:
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_requested or job.status == "canceled":
                return None
            job.status = "running"
            job.started_at = _now()
            job.updated_at = job.started_at
            return job

    def _execute_job(self, job: WebReasonerJob) -> str:
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            raise WebReasonerError(
                "Playwright is not installed. Run `python3 -m pip install playwright` and "
                "`python3 -m playwright install chromium` first."
            ) from exc

        cancel_flag = self._cancel_flags[job.id]
        page = None
        try:
            with sync_playwright() as playwright:
                profile_name = _web_profile_name(job.provider, job.account_agent, job.id)
                with _isolated_chrome(playwright, profile_name, headed=False) as (_browser, _context, page):
                    _check_cancel(cancel_flag)
                    if job.provider == "chatgpt_pro":
                        return self._run_chatgpt_pro(page, job, cancel_flag)
                    if job.provider == "gemini_deepthink":
                        return self._run_gemini_deepthink(page, job, cancel_flag)
                    raise WebReasonerError(f"unsupported web reasoner provider: {job.provider}")
        except Exception as exc:
            if page is not None:
                setattr(exc, "_web_reasoner_page", page)
            raise

    def _finalize_success(self, job_id: str, result: str) -> None:
        job = None
        canceled = False
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_requested or self._cancel_flags[job_id].is_set():
                canceled = True
                job.status = "canceled"
                job.error = job.error or "canceled"
            else:
                job.status = "completed"
                job.result = result.strip()
                job.error = None
            job.finished_at = _now()
            job.updated_at = job.finished_at
        if job is None:
            return
        if canceled:
            self._emit_event(job, "job_canceled", "cancel requested")
        else:
            self._emit_event(job, "job_completed", f"{job.provider} job completed")

    def _finalize_failure(self, job_id: str, error: str) -> None:
        job = None
        canceled = False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            canceled = job.cancel_requested or self._cancel_flags[job_id].is_set()
            job.status = "canceled" if canceled else "failed"
            job.error = "canceled" if canceled else error
            job.finished_at = _now()
            job.updated_at = job.finished_at
        if canceled:
            self._emit_event(job, "job_canceled", "cancel requested")
        else:
            self._emit_event(job, "job_failed", error)

    def _finalize_canceled(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.status == "canceled" and job.finished_at is not None:
                return
            job.status = "canceled"
            job.error = message
            job.finished_at = _now()
            job.updated_at = job.finished_at
        self._emit_event(job, "job_canceled", message)

    def _page_for_debug(self, job_id: str) -> object | None:
        return None

    def _debug_target(self, job_id: str) -> str:
        snapshot = self.snapshot(job_id)
        return f"web-{snapshot['provider']}-{snapshot['account_agent']}-{str(snapshot['job_id'])[:8]}"

    def _run_chatgpt_pro(self, page: object, job: WebReasonerJob, cancel_flag: threading.Event) -> str:
        email, password = _resolve_openai_account(job.account_agent)
        _check_cancel(cancel_flag)
        page.goto("https://chatgpt.com/", wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(2000)
        _ensure_chatgpt_logged_in(page, email, password, cancel_flag)
        _check_cancel(cancel_flag)
        page.goto("https://chatgpt.com/", wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(3000)
        _ensure_chatgpt_pro_workspace(page, cancel_flag)
        baseline = _extract_last_assistant_message(page)
        _submit_prompt(page, job.prompt, cancel_flag)
        return _wait_for_chatgpt_response(
            page,
            cancel_flag,
            timeout_seconds=job.timeout_seconds,
            previous_text=baseline,
            prompt_text=job.prompt,
        )

    def _run_gemini_deepthink(self, page: object, job: WebReasonerJob, cancel_flag: threading.Event) -> str:
        email = os.environ.get(gemini_email_env_var(), "")
        password = os.environ.get(gemini_password_env_var(), "")
        if not email or not password:
            raise WebReasonerError(
                f"missing Gemini OAuth credentials in env: {gemini_email_env_var()} / {gemini_password_env_var()}"
            )
        _check_cancel(cancel_flag)
        page.goto("https://gemini.google.com/app", wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(3000)
        _ensure_google_logged_in(page, email, password, cancel_flag)
        _ensure_gemini_deepthink_ready(page, cancel_flag)
        baseline = _extract_last_assistant_message(page)
        _submit_prompt(page, job.prompt, cancel_flag)
        return _wait_for_gemini_response(
            page,
            cancel_flag,
            timeout_seconds=job.timeout_seconds,
            previous_text=baseline,
            prompt_text=job.prompt,
        )

    def _emit_event(self, job: WebReasonerJob, kind: str, message: str) -> None:
        if self._event_callback is None:
            return
        event = WebReasonerEvent(
            ts=_now(),
            kind=kind,
            provider=job.provider,
            job_id=job.id,
            account_agent=job.account_agent,
            message=message,
            data=job.snapshot(),
        )
        try:
            self._event_callback(event)
        except Exception:
            return


def _resolve_openai_account(account_agent: str) -> tuple[str, str]:
    email = os.environ.get(email_env_var(account_agent), "")
    password = os.environ.get(password_env_var(account_agent), "")
    if not email or not password:
        raise WebReasonerError(f"missing OpenAI OAuth credentials for {account_agent}")
    return email, password


def _check_cancel(cancel_flag: threading.Event) -> None:
    if cancel_flag.is_set():
        raise _JobCanceled("canceled")


def _ensure_chatgpt_logged_in(page: object, email: str, password: str, cancel_flag: threading.Event) -> None:
    body = _body_text(page)
    if "Continue with Google" not in body and "Log in" not in body and _has_visible(
        page,
        ["textarea", "[contenteditable='true']", "[role='textbox']"],
        timeout_ms=1000,
    ):
        return

    if _has_visible(page, ["button:has-text('Log in')", "a:has-text('Log in')"], timeout_ms=3000):
        _click_first(page, ["button:has-text('Log in')", "a:has-text('Log in')"])
        page.wait_for_timeout(1000)

    flow_page = page
    if not _has_visible(page, ["button:has-text('Continue with Google')", "text=Continue with Google"], timeout_ms=5000):
        page.goto(CHATGPT_LOGIN_URL, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(1000)
        _normalize_openai_login_entry(page)
        flow_page = page

    if _has_visible(flow_page, ["button:has-text('Continue with Google')", "text=Continue with Google"], timeout_ms=5000):
        flow_page = _click_google_and_capture_page(flow_page)
        try:
            flow_page.wait_for_load_state("domcontentloaded", timeout=15000)
        except Exception:
            pass
    _ensure_google_logged_in(flow_page, email, password, cancel_flag)
    _wait_for_chatgpt_login_completion(flow_page, cancel_flag)


def _ensure_google_logged_in(page: object, email: str, password: str, cancel_flag: threading.Event) -> None:
    deadline = time.time() + 180
    while time.time() < deadline:
        _dismiss_managed_profile_notice_in_context(page)
        page = _preferred_openai_flow_page(page)
        _check_cancel(cancel_flag)
        body = _body_text(page)
        title = _page_title(page)
        current_url = getattr(page, "url", "")

        if _is_managed_profile_notice(current_url, title, body):
            if _dismiss_managed_profile_notice(page):
                time.sleep(1)
                continue

        if _is_google_account_picker(body) and email in body:
            try:
                _click_first(page, [f"text={email}"])
                time.sleep(1)
                continue
            except Exception:
                pass

        if _has_visible(
            page,
            ["input[type='email']:visible", "input[name='identifier']:visible", "input[autocomplete='username']:visible"],
            timeout_ms=1000,
        ):
            _fill_first(
                page,
                ["input[type='email']:visible", "input[name='identifier']:visible", "input[autocomplete='username']:visible"],
                email,
            )
            _click_first(page, ["#identifierNext", "button:has-text('Next')"])
            time.sleep(1)
            continue

        if _has_visible(
            page,
            ["input[type='password']:visible", "input[name='Passwd']:visible", "input[autocomplete='current-password']:visible"],
            timeout_ms=1000,
        ):
            _fill_first(
                page,
                ["input[type='password']:visible", "input[name='Passwd']:visible", "input[autocomplete='current-password']:visible"],
                password,
            )
            _click_first(page, ["#passwordNext", "button:has-text('Next')"])
            time.sleep(1)
            continue

        _click_optional(
            page,
            [
                "button:has-text('Continue')",
                "button:has-text('Allow')",
                "button:has-text('Authorize')",
                "button:has-text('Accept')",
                "button:has-text('I agree')",
                "button:has-text('OK')",
            ],
        )

        if title == "Just a moment..." or "Just a moment..." in body:
            time.sleep(2)
            continue

        if "accounts.google.com" not in current_url and body:
            return
        time.sleep(1)

    raise WebReasonerError(
        "timed out completing Google sign-in at "
        f"{getattr(page, 'url', '<unknown>')}"
        f" title={title!r} body={body[:240]!r}"
    )


def _web_profile_name(provider: str, account_agent: str, job_id: str | None = None) -> str:
    base = f"web-{provider}-{account_agent}"
    if not job_id:
        return base
    return f"{base}-{job_id[:8]}"


def _preferred_openai_flow_page(page: object) -> object:
    context = getattr(page, "context", None)
    if context is None:
        return page

    google_page = None
    auth_page = None
    chatgpt_page = None
    notice_page = None
    fallback = page
    for candidate in getattr(context, "pages", []):
        try:
            if candidate.is_closed():
                continue
            url = candidate.url
        except Exception:
            continue
        if url.startswith("chrome://managed-user-profile-notice/"):
            notice_page = candidate
            continue
        if "accounts.google.com" in url:
            google_page = candidate
            continue
        if "auth.openai.com" in url:
            auth_page = candidate
            continue
        if "chatgpt.com" in url:
            chatgpt_page = candidate
            continue
        if url and url != "about:blank":
            fallback = candidate
    return google_page or auth_page or chatgpt_page or notice_page or fallback


def _is_google_account_picker(body: str) -> bool:
    if "Choose an account" in body:
        return True
    return "Use another account" in body and "Enter your password" not in body


def _is_managed_profile_notice(url: str, title: str, body: str) -> bool:
    if url.startswith("chrome://managed-user-profile-notice/"):
        return True
    notice_text = "Your organization will manage this profile"
    return notice_text in title or notice_text in body


def _dismiss_managed_profile_notice(page: object) -> bool:
    for selector in (
        "[role='button']:has-text('Continue as')",
        "text=/^Continue as /",
        "text=Continue as bot",
    ):
        locator = page.locator(selector).first
        try:
            locator.wait_for(state="visible", timeout=2000)
            try:
                locator.click(timeout=3000)
            except Exception:
                locator.evaluate("(el) => el.click()")
            page.wait_for_timeout(1500)
            return True
        except Exception:
            continue

    try:
        page.keyboard.press("Tab")
        page.keyboard.press("Enter")
        page.wait_for_timeout(1500)
        return True
    except Exception:
        return False


def _dismiss_managed_profile_notice_in_context(page: object) -> bool:
    context = getattr(page, "context", None)
    if context is None:
        return False
    dismissed = False
    for candidate in getattr(context, "pages", []):
        try:
            if candidate.is_closed():
                continue
            url = candidate.url
            title = _page_title(candidate)
            body = _body_text(candidate)
        except Exception:
            continue
        if _is_managed_profile_notice(url, title, body):
            dismissed = _dismiss_managed_profile_notice(candidate) or dismissed
    return dismissed


def _wait_for_chatgpt_login_completion(page: object, cancel_flag: threading.Event) -> None:
    deadline = time.time() + 180
    retry_to_auth = False
    while time.time() < deadline:
        _dismiss_managed_profile_notice_in_context(page)
        page = _preferred_openai_flow_page(page)
        _check_cancel(cancel_flag)
        body = _body_text(page)
        title = _page_title(page)
        current_url = getattr(page, "url", "")

        if _is_managed_profile_notice(current_url, title, body):
            if _dismiss_managed_profile_notice(page):
                time.sleep(1)
                continue

        if "Sign in to ChatGPT" in body or "Select a workspace" in body:
            _complete_workspace_consent(page)
            page.wait_for_timeout(1500)
            continue

        if "Choose a workspace" in body:
            if _has_visible(page, ["text=Kuang2"], timeout_ms=2000):
                _click_first(page, ["text=Kuang2"])
                page.wait_for_timeout(3000)
                continue

        if "chatgpt.com" in current_url:
            if "Log in" not in body and _has_visible(
                page,
                ["textarea", "[contenteditable='true']", "[role='textbox']"],
                timeout_ms=1000,
            ):
                return
            if "Log in" in body and not retry_to_auth:
                retry_to_auth = True
                page.goto(CHATGPT_LOGIN_URL, wait_until="domcontentloaded", timeout=120000)
                page.wait_for_timeout(1000)
                _normalize_openai_login_entry(page)
                continue

        time.sleep(1)

    raise WebReasonerError(
        "timed out waiting for ChatGPT login completion at "
        f"{getattr(page, 'url', '<unknown>')} title={_page_title(page)!r} body={_body_text(page)[:240]!r}"
    )


def _ensure_chatgpt_pro_workspace(page: object, cancel_flag: threading.Event) -> None:
    deadline = time.time() + 180
    while time.time() < deadline:
        _dismiss_managed_profile_notice_in_context(page)
        page = _preferred_openai_flow_page(page)
        _check_cancel(cancel_flag)
        body = _body_text(page)
        title = _page_title(page)
        current_url = getattr(page, "url", "")

        if _is_managed_profile_notice(current_url, title, body):
            if _dismiss_managed_profile_notice(page):
                time.sleep(1)
                continue

        if "Sign in to ChatGPT" in body or "Select a workspace" in body:
            _complete_workspace_consent(page)
            page.wait_for_timeout(1500)
            continue

        if "Choose a workspace" in body:
            if _has_visible(page, ["text=Kuang2"], timeout_ms=2000):
                _click_first(page, ["text=Kuang2"])
                page.wait_for_timeout(4000)
                continue
            raise WebReasonerError("ChatGPT workspace chooser did not offer the Kuang2 workspace")

        if "ChatGPT 5.4 Pro" in body and _has_visible(
            page,
            ["textarea", "[contenteditable='true']", "[role='textbox']"],
            timeout_ms=2000,
        ):
            return

        if _has_visible(
            page,
            [
                "[data-testid='model-switcher-dropdown-button']",
                "button[aria-label*='Model selector']",
                "button:has-text('ChatGPT 5.4 Pro')",
                "button:has-text('Pro')",
            ],
            timeout_ms=1200,
        ):
            _click_optional(
                page,
                [
                    "[data-testid='model-switcher-dropdown-button']",
                    "button[aria-label*='Model selector']",
                ],
            )
            page.wait_for_timeout(1000)
            _click_optional(
                page,
                [
                    "button:has-text('ChatGPT 5.4 Pro')",
                    "button:has-text('Pro')",
                    "text=ChatGPT 5.4 Pro",
                    "text=Pro",
                ],
            )
            page.wait_for_timeout(1200)

        _click_optional(
            page,
            [
                "button:has-text('Not now')",
                "button:has-text('Got it')",
                "button:has-text('Close')",
            ],
        )
        time.sleep(1)

    raise WebReasonerError(f"timed out waiting for ChatGPT 5.4 Pro workspace at {getattr(page, 'url', '<unknown>')!r}")


def _ensure_gemini_deepthink_ready(page: object, cancel_flag: threading.Event) -> None:
    deadline = time.time() + 180
    while time.time() < deadline:
        _check_cancel(cancel_flag)
        body = _body_text(page)

        if any(token in body for token in ("Gemini", "Deep Think", "2.5 Pro")) and _has_visible(
            page,
            ["textarea", "[contenteditable='true']", "[role='textbox']"],
            timeout_ms=2000,
        ):
            if "Deep Think" in body or "2.5 Pro" in body:
                return
            _click_optional(
                page,
                [
                    "button:has-text('Deep Think')",
                    "button:has-text('2.5 Pro')",
                    "text=Deep Think",
                    "text=2.5 Pro",
                    "button:has-text('Pro')",
                ],
            )
            page.wait_for_timeout(1500)
            if "Deep Think" in _body_text(page) or "2.5 Pro" in _body_text(page):
                return

        _click_optional(
            page,
            [
                "button:has-text('Sign in')",
                "text=Continue with Google",
                "button:has-text('Continue with Google')",
                "button:has-text('Try Gemini')",
            ],
        )
        time.sleep(1)

    raise WebReasonerError(
        "Gemini Deep Think UI was not ready or selectable at "
        f"{getattr(page, 'url', '<unknown>')!r}; title={_page_title(page)!r}; body={_body_text(page)[:240]!r}"
    )


def _submit_prompt(page: object, prompt: str, cancel_flag: threading.Event) -> None:
    deadline = time.time() + 30
    while time.time() < deadline:
        _check_cancel(cancel_flag)
        try:
            area = page.locator("textarea:visible").first
            area.wait_for(state="visible", timeout=1000)
            area.click()
            try:
                area.fill(prompt)
            except Exception:
                page.keyboard.press("Control+A")
                page.keyboard.type(prompt, delay=10)
            page.keyboard.press("Enter")
            return
        except Exception:
            pass

        try:
            area = page.locator("[contenteditable='true']:visible, [role='textbox']:visible").first
            area.wait_for(state="visible", timeout=1000)
            area.click()
            page.keyboard.press("Control+A")
            page.keyboard.type(prompt, delay=10)
            page.keyboard.press("Enter")
            return
        except Exception:
            time.sleep(1)

    raise WebReasonerError("unable to find a visible prompt composer")


def _wait_for_chatgpt_response(
    page: object,
    cancel_flag: threading.Event,
    *,
    timeout_seconds: int,
    previous_text: str,
    prompt_text: str,
) -> str:
    return _wait_for_stable_response(
        page,
        cancel_flag,
        timeout_seconds=timeout_seconds,
        previous_text=previous_text,
        prompt_text=prompt_text,
        stop_selectors=["button:has-text('Stop')", "button[aria-label*='Stop']", "text=Stop generating"],
    )


def _wait_for_gemini_response(
    page: object,
    cancel_flag: threading.Event,
    *,
    timeout_seconds: int,
    previous_text: str,
    prompt_text: str,
) -> str:
    return _wait_for_stable_response(
        page,
        cancel_flag,
        timeout_seconds=timeout_seconds,
        previous_text=previous_text,
        prompt_text=prompt_text,
        stop_selectors=["button:has-text('Stop')", "button:has-text('Cancel')"],
    )


def _wait_for_stable_response(
    page: object,
    cancel_flag: threading.Event,
    *,
    timeout_seconds: int,
    previous_text: str,
    prompt_text: str,
    stop_selectors: list[str],
) -> str:
    deadline = time.time() + timeout_seconds
    stable_text = ""
    stable_polls = 0
    normalized_previous = previous_text.strip()
    normalized_prompt = prompt_text.strip()

    while time.time() < deadline:
        _check_cancel(cancel_flag)
        page.wait_for_timeout(2000)
        candidate = _extract_last_assistant_message(page).strip()
        if not candidate:
            continue
        if candidate == normalized_previous or candidate == normalized_prompt:
            continue

        if candidate == stable_text:
            stable_polls += 1
        else:
            stable_text = candidate
            stable_polls = 1

        if stable_text and stable_polls >= 3 and not _has_visible(page, stop_selectors, timeout_ms=500):
            return stable_text

    raise WebReasonerError("timed out waiting for web reasoner response")


def _extract_last_assistant_message(page: object) -> str:
    selectors = [
        "[data-message-author-role='assistant']",
        "[data-testid='assistant-turn']",
        "main [role='article']",
        "main article",
        "main .markdown",
    ]
    for selector in selectors:
        locator = page.locator(selector)
        try:
            count = locator.count()
        except Exception:
            continue
        if count == 0:
            continue
        for index in range(count - 1, -1, -1):
            try:
                text = locator.nth(index).inner_text(timeout=1000).strip()
            except Exception:
                continue
            if text:
                return text
    try:
        return page.locator("main").inner_text(timeout=1000).strip()
    except Exception:
        return _body_text(page).strip()
