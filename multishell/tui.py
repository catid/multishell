from __future__ import annotations

import curses
import math
import time
import unicodedata
from dataclasses import dataclass, field

from .control import format_timestamp
from .orchestrator import MultiShellController


MIN_HEIGHT = 12
MIN_WIDTH = 48


@dataclass
class DebugState:
    selected_index: int = 0
    scroll_offsets: dict[str, int] = field(default_factory=dict)


@dataclass
class InputState:
    text: str = ""
    cursor: int = 0

    def clear(self) -> None:
        self.text = ""
        self.cursor = 0


def run_tui(controller: MultiShellController) -> None:
    curses.wrapper(lambda stdscr: _main(stdscr, controller))


def _main(stdscr: curses.window, controller: MultiShellController) -> None:
    try:
        curses.curs_set(1)
    except curses.error:
        pass
    stdscr.timeout(100)
    stdscr.keypad(True)
    _init_colors()

    debug_mode = False
    debug_state = DebugState()
    input_state = InputState()

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        session_names = [row["name"] for row in controller.session_rows()]

        if height < MIN_HEIGHT or width < MIN_WIDTH:
            _draw_too_small(stdscr, height, width)
        else:
            content_top = _draw_header(stdscr, controller, width)
            content_bottom = height - 4
            if debug_mode:
                _draw_debug_view(stdscr, controller, debug_state, content_top, content_bottom, width)
            else:
                _draw_chat_view(stdscr, controller, content_top, content_bottom, width)
            _draw_footer(stdscr, controller, debug_mode, debug_state, height, width)
            _draw_input(stdscr, input_state, height, width, debug_mode)

        try:
            stdscr.refresh()
        except curses.error:
            continue

        try:
            key = stdscr.get_wch()
        except curses.error:
            continue

        if key == "\x03":
            return
        if key == "\t":
            debug_mode = not debug_mode
            continue
        if key == curses.KEY_RESIZE:
            continue

        if debug_mode and _handle_debug_key(key, debug_state, session_names, _debug_column_count(width)):
            continue

        if key in ("\n", "\r"):
            text = input_state.text.strip()
            if text:
                controller.send_user_message(text)
            input_state.clear()
            continue
        if key in ("\x08", "\x7f") or key == curses.KEY_BACKSPACE:
            _delete_backwards(input_state)
            continue
        if key == curses.KEY_DC:
            _delete_forwards(input_state)
            continue
        if not debug_mode and key == curses.KEY_LEFT:
            input_state.cursor = max(0, input_state.cursor - 1)
            continue
        if not debug_mode and key == curses.KEY_RIGHT:
            input_state.cursor = min(len(input_state.text), input_state.cursor + 1)
            continue
        if not debug_mode and key == curses.KEY_HOME:
            input_state.cursor = 0
            continue
        if not debug_mode and key == curses.KEY_END:
            input_state.cursor = len(input_state.text)
            continue
        if key == "\x01":
            input_state.cursor = 0
            continue
        if key == "\x05":
            input_state.cursor = len(input_state.text)
            continue
        if isinstance(key, str) and key.isprintable():
            _insert_text(input_state, key)


def _init_colors() -> None:
    try:
        curses.start_color()
        curses.use_default_colors()
    except curses.error:
        return
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_MAGENTA, -1)
    curses.init_pair(5, curses.COLOR_BLUE, -1)
    curses.init_pair(6, curses.COLOR_RED, -1)
    curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_CYAN)


def _draw_too_small(stdscr: curses.window, height: int, width: int) -> None:
    lines = [
        "Multishell",
        f"Terminal too small: need at least {MIN_WIDTH}x{MIN_HEIGHT}",
        f"Current size: {width}x{height}",
        "Resize the terminal or reduce the SSH font size.",
        "Ctrl+C quits.",
    ]
    for row, line in enumerate(lines[: max(0, height - 1)]):
        _safe_addnstr(stdscr, row, 0, line, width - 1)


def _draw_header(stdscr: curses.window, controller: MultiShellController, width: int) -> int:
    sessions = controller.session_rows()
    worker_count = max(0, len(sessions) - 1)
    title = f" Multishell  manager + {worker_count} workers  Tab=debug  Ctrl+C=quit "
    _safe_fill_line(stdscr, 0, width, " ", curses.color_pair(7))
    _safe_add_display_text(stdscr, 0, 0, title, width - 1, curses.color_pair(7))

    row = 1
    col = 0
    _safe_fill_line(stdscr, row, width, " ")
    for session in sessions:
        label = _session_chip(session)
        if col and col + _display_width(label) >= width:
            row += 1
            col = 0
            _safe_fill_line(stdscr, row, width, " ")
        attr = curses.color_pair(_status_color(session)) | curses.A_BOLD
        _safe_add_display_text(stdscr, row, col, label, max(0, width - col - 1), attr)
        col += _display_width(label)

    reasoners = controller.active_reasoner_counts()
    summary = (
        f" chat={len(controller.recent_messages(200))}"
        f"  web_run={reasoners['running']}"
        f"  web_q={reasoners['queued']}"
        f"  updated={format_timestamp(time.time())} "
    )
    _safe_fill_line(stdscr, row + 1, width, " ")
    _safe_add_display_text(stdscr, row + 1, 0, summary, width - 1, curses.color_pair(1))
    return row + 2


def _draw_chat_view(
    stdscr: curses.window,
    controller: MultiShellController,
    content_top: int,
    content_bottom: int,
    width: int,
) -> None:
    available_rows = max(1, content_bottom - content_top + 1)
    messages = controller.recent_messages(160)
    accent_by_source = {str(row["name"]): int(row.get("accent_color", 1)) for row in controller.session_rows()}
    rendered: list[tuple[str, int, int]] = []

    for message in messages:
        prefix = f"[{format_timestamp(message.ts)}] {message.source}: "
        prefix_width = _display_width(prefix)
        wrapped = _wrap_display_text(message.text, max(12, width - prefix_width - 1)) or [""]
        color = _message_color(message.source, message.level, accent_by_source)
        rendered.append((prefix + wrapped[0], 0, color))
        for continuation in wrapped[1:]:
            rendered.append((continuation, prefix_width, 0))

    if not rendered:
        rendered = [("No chat messages yet. Type a task for the manager below.", 0, curses.color_pair(3))]

    visible = rendered[-available_rows:]
    start_row = content_bottom - len(visible) + 1
    for row in range(content_top, content_bottom + 1):
        _safe_fill_line(stdscr, row, width, " ")
    for index, (line, indent, color) in enumerate(visible):
        row = start_row + index
        attr = color if isinstance(color, int) else 0
        _safe_add_display_text(stdscr, row, indent, line, max(0, width - indent - 1), attr)


def _draw_debug_view(
    stdscr: curses.window,
    controller: MultiShellController,
    debug_state: DebugState,
    content_top: int,
    content_bottom: int,
    width: int,
) -> None:
    overviews = {row["name"]: row for row in controller.session_rows()}
    transcripts = controller.transcripts_for_debug()
    names = list(overviews)
    if not names:
        return

    debug_state.selected_index = max(0, min(debug_state.selected_index, len(names) - 1))
    columns = _debug_column_count(width)
    panel_rows = math.ceil(len(names) / columns)
    available_rows = max(1, content_bottom - content_top + 1)
    panel_height = max(6, available_rows // panel_rows)
    panel_width = max(24, width // columns)

    for index, name in enumerate(names):
        grid_row = index // columns
        grid_col = index % columns
        top = content_top + grid_row * panel_height
        left = grid_col * panel_width
        if top > content_bottom:
            break
        bottom = min(content_bottom, top + panel_height - 1)
        right = min(width - 1, left + panel_width - 1)
        selected = index == debug_state.selected_index
        _draw_panel(
            stdscr,
            top,
            left,
            bottom,
            right,
            overviews[name],
            transcripts.get(name, []),
            debug_state,
            selected,
        )


def _draw_panel(
    stdscr: curses.window,
    top: int,
    left: int,
    bottom: int,
    right: int,
    overview: dict[str, object],
    entries: list,
    debug_state: DebugState,
    selected: bool,
) -> None:
    if bottom - top < 4 or right - left < 16:
        return

    name = str(overview["name"])
    inner_width = max(8, right - left - 1)
    body_rows = max(1, bottom - top - 1)
    border_attr = curses.color_pair(8 if selected else _status_color(overview)) | (curses.A_BOLD if selected else 0)
    title = f" {name} {overview['status']} q={overview['pending_tasks']} "
    if selected:
        title += f"scroll={debug_state.scroll_offsets.get(name, 0)} "

    _safe_addnstr(stdscr, top, left, "+" + "-" * max(0, right - left - 1) + "+", right - left + 1, border_attr)
    _safe_addnstr(stdscr, top, left + 2, title, max(0, right - left - 3), border_attr)
    for row in range(top + 1, bottom):
        _safe_addnstr(stdscr, row, left, "|", 1, border_attr)
        _safe_addnstr(stdscr, row, right, "|", 1, border_attr)
    _safe_addnstr(stdscr, bottom, left, "+" + "-" * max(0, right - left - 1) + "+", right - left + 1, border_attr)

    lines = _panel_lines(overview, entries, inner_width - 1)
    max_visible = min(body_rows, len(lines)) if lines else body_rows
    max_offset = max(0, len(lines) - max_visible)
    offset = max(0, min(debug_state.scroll_offsets.get(name, 0), max_offset))
    debug_state.scroll_offsets[name] = offset
    start = max(0, len(lines) - max_visible - offset)
    visible = lines[start : start + max_visible]

    for index, (line, attr) in enumerate(visible):
        row = top + 1 + index
        _safe_addnstr(stdscr, row, left + 1, line.ljust(inner_width - 1), inner_width - 1, attr)


def _panel_lines(overview: dict[str, object], entries: list, width: int) -> list[tuple[str, int]]:
    lines: list[tuple[str, int]] = []
    last_duration = overview.get("last_turn_duration")
    duration_text = f"{last_duration:.1f}s" if isinstance(last_duration, float) else "-"
    running_for = overview.get("running_for_seconds")
    running_text = f"{running_for:.1f}s" if isinstance(running_for, float) else "-"
    summary = (
        f"ok={overview['completed_turns']} fail={overview['failed_turns']} "
        f"run={running_text} last={duration_text}"
    )
    lines.append((summary[:width], curses.color_pair(1)))
    engine = str(overview.get("engine") or "")
    model = str(overview.get("model") or "")
    if engine or model:
        for wrapped in _wrap_display_text(f"{engine}: {model}".strip(": "), max(8, width))[:1]:
            lines.append((wrapped, curses.color_pair(2)))
    persona_label = str(overview.get("persona_label") or "")
    if persona_label:
        for wrapped in _wrap_display_text(f"persona: {persona_label}", max(8, width))[:1]:
            lines.append((wrapped, curses.color_pair(4)))
    cwd = str(overview.get("cwd") or "")
    if cwd:
        for wrapped in _wrap_display_text(f"cwd: {cwd}", max(8, width))[:2]:
            lines.append((wrapped, curses.color_pair(5)))
    last_error = str(overview.get("last_error") or "").strip()
    if last_error:
        for wrapped in _wrap_display_text(f"error: {last_error}", max(8, width))[:2]:
            lines.append((wrapped, curses.color_pair(6)))

    for entry in entries:
        prefix = f"{format_timestamp(entry.ts)} {entry.source[:4]} "
        prefix_width = _display_width(prefix)
        wrapped = _wrap_display_text(entry.text, max(8, width - prefix_width)) or [""]
        lines.append((prefix + wrapped[0], _entry_color(entry.source)))
        for continuation in wrapped[1:]:
            lines.append((" " * prefix_width + continuation, 0))
    return lines or [("(no transcript)", curses.color_pair(3))]


def _draw_footer(
    stdscr: curses.window,
    controller: MultiShellController,
    debug_mode: bool,
    debug_state: DebugState,
    height: int,
    width: int,
) -> None:
    sessions = controller.session_rows()
    focus = sessions[debug_state.selected_index if sessions else 0] if sessions else None
    if debug_mode and focus is not None:
        running_for = focus.get("running_for_seconds")
        running_text = f"{running_for:.1f}s" if isinstance(running_for, float) else "-"
        footer = (
            f"focus={focus['name']} status={focus['status']} q={focus['pending_tasks']} "
            f"ok={focus['completed_turns']} fail={focus['failed_turns']} run={running_text}  "
            "Arrows=focus  PgUp/PgDn=scroll  Tab=chat"
        )
    else:
        footer = "Enter=send  Tab=debug  Ctrl+C=quit"
    _safe_fill_line(stdscr, height - 3, width, " ")
    _safe_addnstr(stdscr, height - 3, 0, footer, width - 1, curses.color_pair(3))


def _draw_input(
    stdscr: curses.window,
    input_state: InputState,
    height: int,
    width: int,
    debug_mode: bool,
) -> None:
    prompt = "chat(debug)> " if debug_mode else "chat> "
    _safe_hline(stdscr, height - 2, width, "-")
    available = max(1, width - _display_width(prompt) - 1)
    visible_input, cursor_column = _visible_input_window(input_state.text, input_state.cursor, available)
    _safe_fill_line(stdscr, height - 1, width, " ")
    _safe_add_display_text(stdscr, height - 1, 0, prompt + visible_input, width - 1)
    cursor_x = min(width - 1, _display_width(prompt) + cursor_column)
    try:
        stdscr.move(height - 1, cursor_x)
    except curses.error:
        pass


def _insert_text(input_state: InputState, text: str) -> None:
    left = input_state.text[: input_state.cursor]
    right = input_state.text[input_state.cursor :]
    input_state.text = left + text + right
    input_state.cursor += len(text)


def _delete_backwards(input_state: InputState) -> None:
    if input_state.cursor <= 0:
        return
    left = input_state.text[: input_state.cursor - 1]
    right = input_state.text[input_state.cursor :]
    input_state.text = left + right
    input_state.cursor -= 1


def _delete_forwards(input_state: InputState) -> None:
    if input_state.cursor >= len(input_state.text):
        return
    input_state.text = input_state.text[: input_state.cursor] + input_state.text[input_state.cursor + 1 :]


def _visible_input_window(text: str, cursor: int, available: int) -> tuple[str, int]:
    bounded_cursor = max(0, min(cursor, len(text)))
    if available <= 0:
        return "", 0
    if _display_width(text) <= available:
        return text, _display_width(text[:bounded_cursor])

    if available == 1:
        return "<", 0

    body_width = max(1, available - 2)
    cursor_left = _display_width(text[:bounded_cursor])
    start = 0
    max_start = len(text)
    while start < max_start and cursor_left - _display_width(text[:start]) > body_width // 2:
        start += 1
    visible_body = _truncate_display(text[start:], body_width)
    visible_width = _display_width(visible_body)
    while start > 0 and cursor_left - _display_width(text[:start]) < 0:
        start -= 1
        visible_body = _truncate_display(text[start:], body_width)
        visible_width = _display_width(visible_body)
    prefix = "<" if start > 0 else ""
    suffix = ">" if start + len(visible_body) < len(text) else ""
    cursor_column = len(prefix) + max(0, cursor_left - _display_width(text[:start]))
    visible = prefix + visible_body
    if suffix and visible_width < body_width:
        visible += suffix
    elif suffix:
        visible = _truncate_display(visible, max(0, available - 1)) + suffix
    return visible, min(_display_width(visible), cursor_column)


def _handle_debug_key(key: object, debug_state: DebugState, session_names: list[str], columns: int) -> bool:
    session_count = len(session_names)
    if session_count <= 0:
        return False
    if key in (curses.KEY_LEFT, "h"):
        debug_state.selected_index = max(0, debug_state.selected_index - 1)
        return True
    if key in (curses.KEY_RIGHT, "l"):
        debug_state.selected_index = min(session_count - 1, debug_state.selected_index + 1)
        return True
    if key in (curses.KEY_UP, "k"):
        debug_state.selected_index = max(0, debug_state.selected_index - columns)
        return True
    if key in (curses.KEY_DOWN, "j"):
        debug_state.selected_index = min(session_count - 1, debug_state.selected_index + columns)
        return True
    if key == curses.KEY_PPAGE:
        _shift_scroll(debug_state, session_names, +5)
        return True
    if key == curses.KEY_NPAGE:
        _shift_scroll(debug_state, session_names, -5)
        return True
    if key == "g":
        _shift_scroll(debug_state, session_names, +10_000)
        return True
    if key == "G":
        _shift_scroll(debug_state, session_names, -10_000)
        return True
    return False


def _shift_scroll(debug_state: DebugState, session_names: list[str], delta: int) -> None:
    if not session_names:
        return
    name = session_names[max(0, min(debug_state.selected_index, len(session_names) - 1))]
    current = debug_state.scroll_offsets.get(name, 0)
    debug_state.scroll_offsets[name] = max(0, current + delta)


def _session_chip(session: dict[str, object]) -> str:
    return (
        f" {session['name']}:{session['status']} "
        f"q={session['pending_tasks']} ok={session['completed_turns']} fail={session['failed_turns']} "
    )


def _status_color(session: dict[str, object]) -> int:
    status = session["status"]
    if status == "error":
        return 6
    if status == "running":
        return int(session.get("accent_color", 1))
    if status == "stopped":
        return 4
    if session["pending_tasks"]:
        return 3
    return 2


def _message_color(source: str, level: str, accent_by_source: dict[str, int] | None = None) -> int:
    if level == "error":
        return curses.color_pair(6)
    if level == "warn":
        return curses.color_pair(3)
    if accent_by_source is not None and source in accent_by_source:
        return curses.color_pair(accent_by_source[source])
    if source == "manager":
        return curses.color_pair(1)
    if source == "user":
        return curses.color_pair(2)
    if source == "system":
        return curses.color_pair(4)
    return 0


def _entry_color(source: str) -> int:
    if source in {"assistant", "manager"}:
        return curses.color_pair(1)
    if source in {"system", "user"}:
        return curses.color_pair(2)
    if source in {"error", "warn"}:
        return curses.color_pair(6)
    if source in {"meta", "event"}:
        return curses.color_pair(3)
    return 0


def _debug_column_count(width: int) -> int:
    return 1 if width < 110 else 2


def _cell_width(char: str) -> int:
    if not char:
        return 0
    if unicodedata.combining(char):
        return 0
    if unicodedata.category(char).startswith("C"):
        return 0
    if unicodedata.east_asian_width(char) in {"W", "F"}:
        return 2
    return 1


def _display_width(text: str) -> int:
    return sum(_cell_width(char) for char in text)


def _truncate_display(text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""
    width = 0
    result: list[str] = []
    for char in text:
        cell_width = _cell_width(char)
        if cell_width and width + cell_width > max_width:
            break
        result.append(char)
        width += cell_width
    return "".join(result)


def _wrap_display_text(text: str, width: int) -> list[str]:
    if width <= 0:
        return [""]
    paragraphs = text.splitlines() or [text]
    lines: list[str] = []
    for paragraph in paragraphs:
        remaining = paragraph.strip()
        if not remaining:
            lines.append("")
            continue
        while remaining:
            if _display_width(remaining) <= width:
                lines.append(remaining)
                break
            candidate = _truncate_display(remaining, width)
            split_at = candidate.rfind(" ")
            if split_at > 0:
                line = candidate[:split_at].rstrip()
                remaining = remaining[split_at + 1 :].lstrip()
            else:
                line = candidate
                remaining = remaining[len(candidate) :].lstrip()
            lines.append(line or candidate)
    return lines or [""]


def _safe_add_display_text(stdscr: curses.window, row: int, col: int, text: str, max_cells: int, attr: int = 0) -> None:
    _safe_addnstr(stdscr, row, col, _truncate_display(text, max_cells), max_cells, attr)


def _safe_addnstr(stdscr: curses.window, row: int, col: int, text: str, max_chars: int, attr: int = 0) -> None:
    if max_chars <= 0:
        return
    try:
        if attr:
            stdscr.attron(attr)
        stdscr.addnstr(row, col, text, max_chars)
    except curses.error:
        return
    finally:
        if attr:
            try:
                stdscr.attroff(attr)
            except curses.error:
                pass


def _safe_fill_line(stdscr: curses.window, row: int, width: int, char: str, attr: int = 0) -> None:
    _safe_addnstr(stdscr, row, 0, char * max(0, width - 1), width - 1, attr)


def _safe_hline(stdscr: curses.window, row: int, width: int, char: str) -> None:
    try:
        stdscr.hline(row, 0, char, max(0, width - 1))
    except curses.error:
        pass
