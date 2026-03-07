from __future__ import annotations

from dataclasses import dataclass
import signal

import multishell.tui as tui


def test_message_color_uses_live_accent_map(monkeypatch) -> None:
    monkeypatch.setattr(tui.curses, "color_pair", lambda pair: pair)

    color = tui._message_color(
        "claude-worker-3",
        "info",
        {
            "worker-1": 2,
            "claude-worker-3": 4,
        },
    )

    assert color == 4


def test_message_color_keeps_error_priority(monkeypatch) -> None:
    monkeypatch.setattr(tui.curses, "color_pair", lambda pair: pair)

    color = tui._message_color("claude-worker-5", "error", {"claude-worker-5": 6})

    assert color == 6


def test_wrap_display_text_handles_wide_characters() -> None:
    lines = tui._wrap_display_text("alpha 🪶 beta gamma", 10)

    assert all(tui._display_width(line) <= 10 for line in lines)
    assert "alpha" in lines[0]


def test_compose_buffer_tracks_cursor_with_wide_characters() -> None:
    buffer = tui._compose_buffer("hello🪶world", len("hello🪶world"), 8)

    assert all(tui._display_width(line) <= 8 for line in buffer.lines)
    assert buffer.cursor_col <= 8


@dataclass
class _Message:
    ts: float


class _Controller:
    def session_rows(self):
        return [
            {"updated_at": 10.0},
            {"updated_at": 25.0},
        ]

    def recent_messages(self, limit=80):
        return [_Message(ts=17.0)] if limit else []


def test_last_activity_timestamp_uses_latest_session_or_message() -> None:
    assert tui._last_activity_timestamp(_Controller()) == 25.0


def test_single_line_snippet_collapses_and_truncates() -> None:
    snippet = tui._single_line_snippet("line one\nline two with more words", 12)

    assert snippet.endswith("...")
    assert "\n" not in snippet
    assert tui._display_width(snippet) <= 12


def test_monitor_chip_formats_compact_status() -> None:
    chip = tui._monitor_chip(
        {
            "label": "codex-2",
            "status": "running",
            "pending_tasks": 1,
            "failed_turns": 0,
            "completed_turns": 0,
        }
    )

    assert "codex-2" in chip
    assert "RUN1" in chip


class _RedrawWindow:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def redrawwin(self) -> None:
        self.calls.append("redrawwin")

    def clearok(self, flag: bool) -> None:
        self.calls.append(("clearok", flag))


def test_maybe_force_full_redraw_repaints_when_due() -> None:
    window = _RedrawWindow()

    next_due = tui._maybe_force_full_redraw(window, next_redraw_at=5.0, now=5.0)

    assert window.calls == ["redrawwin", ("clearok", True)]
    assert next_due == 5.0 + tui.FULL_REDRAW_INTERVAL_SECONDS


def test_maybe_force_full_redraw_skips_when_not_due() -> None:
    window = _RedrawWindow()

    next_due = tui._maybe_force_full_redraw(window, next_redraw_at=5.0, now=4.5)

    assert window.calls == []
    assert next_due == 5.0


def test_capture_sigint_sets_flag_and_restores_previous_handler() -> None:
    previous = signal.getsignal(signal.SIGINT)

    with tui._capture_sigint() as interrupt_state:
        handler = signal.getsignal(signal.SIGINT)
        assert handler is not previous
        handler(signal.SIGINT, None)
        assert interrupt_state.requested is True

    assert signal.getsignal(signal.SIGINT) is previous


class _TerminalWindow:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def keypad(self, enabled: bool) -> None:
        self.calls.append(("keypad", enabled))

    def timeout(self, value: int) -> None:
        self.calls.append(("timeout", value))


def test_restore_terminal_resets_modes(monkeypatch) -> None:
    window = _TerminalWindow()
    calls: list[str] = []

    monkeypatch.setattr(tui.curses, "echo", lambda: calls.append("echo"))
    monkeypatch.setattr(tui.curses, "nocbreak", lambda: calls.append("nocbreak"))
    monkeypatch.setattr(tui.curses, "nl", lambda: calls.append("nl"))
    monkeypatch.setattr(tui.curses, "qiflush", lambda: calls.append("qiflush"))
    monkeypatch.setattr(tui.curses, "endwin", lambda: calls.append("endwin"))

    tui._restore_terminal(window)

    assert window.calls == [("keypad", False), ("timeout", -1)]
    assert calls == ["echo", "nocbreak", "nl", "qiflush", "endwin"]
