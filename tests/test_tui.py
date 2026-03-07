from __future__ import annotations

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


def test_visible_input_window_tracks_cursor_with_wide_characters() -> None:
    visible, cursor = tui._visible_input_window("hello🪶world", len("hello🪶world"), 8)

    assert tui._display_width(visible) <= 8
    assert cursor <= 8
