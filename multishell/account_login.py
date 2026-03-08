from __future__ import annotations

import curses
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from .accounts import (
    PROVIDER_ANTHROPIC,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    AccountRecord,
    load_accounts_from_dotenv,
    next_account_key,
    save_accounts_to_dotenv,
)
from .config import dotenv_path, dotenv_template_text
from .envfile import DotenvFile


MIN_HEIGHT = 16
MIN_WIDTH = 92
EDITABLE_COLUMNS = ("openai", "anthropic", "gemini", "email", "password")
CHECKBOX_PROVIDERS = {
    0: PROVIDER_OPENAI,
    1: PROVIDER_ANTHROPIC,
    2: PROVIDER_GEMINI,
}


@dataclass
class AccountState:
    env_file: DotenvFile
    accounts: list[AccountRecord]
    selected_row: int = 0
    selected_col: int = 0
    status: str = ""


@dataclass
class InterruptState:
    requested: bool = False


def run_login_editor(path: Path | None = None) -> int:
    env_path = Path(path or dotenv_path()).expanduser()
    env_file = DotenvFile.load(env_path, dotenv_template_text())
    state = AccountState(
        env_file=env_file,
        accounts=load_accounts_from_dotenv(env_file),
        status=f"editing {env_path}",
    )
    with _capture_sigint() as interrupt_state:
        curses.wrapper(lambda stdscr: _main(stdscr, state, interrupt_state))
    return 0


def mask_secret(value: str) -> str:
    if not value:
        return "(missing)"
    return "*" * min(12, max(8, len(value)))


def _main(stdscr: curses.window, state: AccountState, interrupt_state: InterruptState) -> None:
    try:
        try:
            curses.noecho()
            curses.cbreak()
            curses.nonl()
            curses.noqiflush()
        except curses.error:
            pass
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.timeout(100)
        stdscr.keypad(True)
        _init_colors()

        while True:
            if interrupt_state.requested:
                return
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            if height < MIN_HEIGHT or width < MIN_WIDTH:
                _draw_too_small(stdscr, height, width)
            else:
                _draw_screen(stdscr, state, width)
            try:
                stdscr.refresh()
            except curses.error:
                continue

            try:
                key = stdscr.get_wch()
            except KeyboardInterrupt:
                interrupt_state.requested = True
                continue
            except curses.error:
                continue

            if interrupt_state.requested or key in ("\x03", "q", "Q"):
                return
            if key in ("\t", curses.KEY_RIGHT, "l"):
                state.selected_col = (state.selected_col + 1) % len(EDITABLE_COLUMNS)
                continue
            if key in (curses.KEY_LEFT, "h"):
                state.selected_col = (state.selected_col - 1) % len(EDITABLE_COLUMNS)
                continue
            if key in (curses.KEY_UP, "k"):
                state.selected_row = max(0, state.selected_row - 1)
                continue
            if key in (curses.KEY_DOWN, "j"):
                state.selected_row = min(max(0, len(state.accounts) - 1), state.selected_row + 1)
                continue
            if key in ("a", "A"):
                _add_account(state)
                continue
            if key in ("x", "X", curses.KEY_DC):
                _remove_selected(state)
                continue
            if key in ("c", "C"):
                _clear_selected(state)
                continue
            if key == " ":
                _toggle_selected_checkbox(state)
                continue
            if key in ("\n", "\r", "e", "E"):
                _activate_selected(stdscr, state)
                continue
    finally:
        _restore_terminal(stdscr)


def _init_colors() -> None:
    try:
        curses.start_color()
        curses.use_default_colors()
    except curses.error:
        return
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_CYAN)


def _draw_too_small(stdscr: curses.window, height: int, width: int) -> None:
    lines = [
        "Multishell Login",
        f"Terminal too small: need at least {MIN_WIDTH}x{MIN_HEIGHT}",
        f"Current size: {width}x{height}",
        "Resize the terminal and rerun `multishell login`.",
    ]
    for row, line in enumerate(lines[: max(0, height - 1)]):
        _safe_addnstr(stdscr, row, 0, line, max(0, width - 1))


def _draw_screen(stdscr: curses.window, state: AccountState, width: int) -> None:
    _fill_line(stdscr, 0, width, " ", curses.color_pair(4))
    _safe_addnstr(stdscr, 0, 0, " Multishell Google Account Setup ", width - 1, curses.color_pair(4))
    _safe_addnstr(
        stdscr,
        1,
        0,
        "Add Google accounts, then toggle which ones can be used for OpenAI, Anthropic, and Gemini AI Ultra.",
        width - 1,
        curses.color_pair(1),
    )

    header_row = 3
    column_widths = _column_widths(width)
    _draw_table_border(stdscr, header_row, width)
    _draw_row(
        stdscr,
        header_row + 1,
        ["Account", "OpenAI", "Anthropic", "Gemini", "Email", "Password"],
        column_widths,
        header=True,
    )
    _draw_table_border(stdscr, header_row + 2, width)

    if state.accounts:
        for index, account in enumerate(state.accounts):
            row = header_row + 3 + index
            values = [
                f"#{index + 1}",
                _checkbox(account.uses(PROVIDER_OPENAI)),
                _checkbox(account.uses(PROVIDER_ANTHROPIC)),
                _checkbox(account.uses(PROVIDER_GEMINI)),
                account.email or "(missing)",
                mask_secret(account.password),
            ]
            _draw_row(stdscr, row, values, column_widths, selected_row=index == state.selected_row, selected_col=state.selected_col)
    else:
        row = header_row + 3
        _safe_addnstr(stdscr, row, 0, "|", 1)
        _safe_addnstr(
            stdscr,
            row,
            2,
            "No accounts configured. Press A to add your first Google account.",
            max(0, width - 4),
            curses.color_pair(3),
        )
        _safe_addnstr(stdscr, row, width - 2, "|", 1)

    footer_row = header_row + 4 + max(1, len(state.accounts))
    _draw_table_border(stdscr, footer_row - 1, width)
    _safe_addnstr(
        stdscr,
        footer_row,
        0,
        "Arrows move  Tab next field  Space toggles checkbox  Enter edits  A adds  X removes  C clears field  Q quits",
        width - 1,
        curses.color_pair(3),
    )
    _safe_addnstr(
        stdscr,
        footer_row + 1,
        0,
        state.status,
        width - 1,
        curses.color_pair(2),
    )


def _column_widths(width: int) -> list[int]:
    usable = max(MIN_WIDTH, width) - 7
    account = 8
    checkbox = 10
    password = 14
    email = max(22, usable - account - checkbox - checkbox - checkbox - password)
    return [account, checkbox, checkbox, checkbox, email, password]


def _checkbox(enabled: bool) -> str:
    return "[x]" if enabled else "[ ]"


def _draw_table_border(stdscr: curses.window, row: int, width: int) -> None:
    _safe_addnstr(stdscr, row, 0, "+" + "-" * max(0, width - 3) + "+", width - 1)


def _draw_row(
    stdscr: curses.window,
    row: int,
    values: list[str],
    column_widths: list[int],
    *,
    header: bool = False,
    selected_row: bool = False,
    selected_col: int = 0,
) -> None:
    col = 0
    _safe_addnstr(stdscr, row, col, "|", 1)
    col += 1
    for index, (value, size) in enumerate(zip(values, column_widths, strict=True)):
        attr = curses.A_BOLD if header else 0
        if selected_row and index > 0 and (index - 1) == selected_col:
            attr |= curses.color_pair(5) | curses.A_BOLD
        elif header:
            attr |= curses.color_pair(1)
        text = _fit_cell(value, size)
        _safe_addnstr(stdscr, row, col, text.ljust(size), size, attr)
        col += size
        _safe_addnstr(stdscr, row, col, "|", 1)
        col += 1


def _fit_cell(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def _add_account(state: AccountState) -> None:
    state.accounts.append(AccountRecord(key=next_account_key(state.accounts), email="", password="", providers=()))
    state.selected_row = len(state.accounts) - 1
    state.selected_col = 3
    _save_accounts(state)
    state.status = f"added account #{len(state.accounts)}"


def _remove_selected(state: AccountState) -> None:
    if not state.accounts:
        state.status = "no account to remove"
        return
    removed = state.accounts.pop(state.selected_row)
    state.selected_row = min(state.selected_row, max(0, len(state.accounts) - 1))
    _save_accounts(state)
    state.status = f"removed {removed.email or removed.key}"


def _clear_selected(state: AccountState) -> None:
    if not state.accounts:
        state.status = "no account to clear"
        return
    account = state.accounts[state.selected_row]
    if state.selected_col in CHECKBOX_PROVIDERS:
        state.status = "use Space to toggle provider checkboxes"
        return
    if state.selected_col == 3:
        updated = AccountRecord(key=account.key, email="", password=account.password, providers=account.providers)
        state.accounts[state.selected_row] = updated
        _save_accounts(state)
        state.status = f"cleared email for account #{state.selected_row + 1}"
        return
    updated = AccountRecord(key=account.key, email=account.email, password="", providers=account.providers)
    state.accounts[state.selected_row] = updated
    _save_accounts(state)
    state.status = f"cleared password for account #{state.selected_row + 1}"


def _toggle_selected_checkbox(state: AccountState) -> None:
    if not state.accounts:
        state.status = "press A to add an account first"
        return
    provider = CHECKBOX_PROVIDERS.get(state.selected_col)
    if provider is None:
        state.status = "press Enter to edit email or password fields"
        return
    account = state.accounts[state.selected_row]
    providers = [item for item in account.providers if item != provider]
    enabled = provider not in account.providers
    if enabled:
        providers.append(provider)
    ordered = tuple(provider_name for provider_name in (PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_GEMINI) if provider_name in providers)
    state.accounts[state.selected_row] = AccountRecord(
        key=account.key,
        email=account.email,
        password=account.password,
        providers=ordered,
    )
    _save_accounts(state)
    state.status = f"{'enabled' if enabled else 'disabled'} {provider} for account #{state.selected_row + 1}"


def _activate_selected(stdscr: curses.window, state: AccountState) -> None:
    if not state.accounts:
        _add_account(state)
        return
    if state.selected_col in CHECKBOX_PROVIDERS:
        _toggle_selected_checkbox(state)
        return
    account = state.accounts[state.selected_row]
    is_password = state.selected_col == 4
    label = f"account #{state.selected_row + 1} {'password' if is_password else 'email'}"
    initial = account.password if is_password else account.email
    updated = _edit_value(stdscr, label, initial, secret=is_password)
    if updated is None:
        state.status = f"canceled edit for {label}"
        return
    state.accounts[state.selected_row] = AccountRecord(
        key=account.key,
        email=account.email if is_password else updated.strip(),
        password=updated if is_password else account.password,
        providers=account.providers,
    )
    _save_accounts(state)
    state.status = f"saved {label} to {state.env_file.path}"


def _save_accounts(state: AccountState) -> None:
    save_accounts_to_dotenv(state.env_file, state.accounts)
    state.env_file.save()


def _edit_value(stdscr: curses.window, label: str, initial: str, *, secret: bool) -> str | None:
    buffer = list(initial)
    cursor = len(buffer)
    while True:
        height, width = stdscr.getmaxyx()
        prompt_row = max(0, height - 3)
        value_row = max(0, height - 2)
        help_row = max(0, height - 1)
        _fill_line(stdscr, prompt_row, width, " ", curses.color_pair(4))
        _fill_line(stdscr, value_row, width, " ")
        _fill_line(stdscr, help_row, width, " ", curses.color_pair(3))
        _safe_addnstr(stdscr, prompt_row, 0, f" Editing {label} ", width - 1, curses.color_pair(4))
        _safe_addnstr(stdscr, help_row, 0, "Enter save  Esc cancel  Ctrl+U clear  Left/Right move cursor", width - 1, curses.color_pair(3))

        text = "".join(buffer)
        shown = "*" * len(text) if secret and text else text
        display, display_cursor = _visible_segment(shown, cursor, max(1, width - 3))
        _safe_addnstr(stdscr, value_row, 0, "> " + display, width - 1)
        try:
            stdscr.move(value_row, min(width - 2, 2 + display_cursor))
        except curses.error:
            pass
        try:
            stdscr.refresh()
        except curses.error:
            continue

        try:
            key = stdscr.get_wch()
        except curses.error:
            continue

        if key in ("\n", "\r"):
            return "".join(buffer)
        if key in ("\x1b",):
            return None
        if key == "\x15":
            buffer = []
            cursor = 0
            continue
        if key in ("\x08", "\x7f") or key == curses.KEY_BACKSPACE:
            if cursor > 0:
                del buffer[cursor - 1]
                cursor -= 1
            continue
        if key == curses.KEY_DC:
            if cursor < len(buffer):
                del buffer[cursor]
            continue
        if key == curses.KEY_LEFT:
            cursor = max(0, cursor - 1)
            continue
        if key == curses.KEY_RIGHT:
            cursor = min(len(buffer), cursor + 1)
            continue
        if key == curses.KEY_HOME:
            cursor = 0
            continue
        if key == curses.KEY_END:
            cursor = len(buffer)
            continue
        if isinstance(key, str) and key.isprintable():
            buffer[cursor:cursor] = [key]
            cursor += 1


def _visible_segment(text: str, cursor: int, width: int) -> tuple[str, int]:
    if len(text) <= width:
        return text, cursor
    start = max(0, cursor - width + 1)
    end = start + width
    return text[start:end], cursor - start


def _safe_addnstr(stdscr: curses.window, row: int, col: int, text: str, max_chars: int, attr: int = 0) -> None:
    if max_chars <= 0:
        return
    try:
        stdscr.addnstr(row, col, text, max_chars, attr)
    except curses.error:
        pass


def _fill_line(stdscr: curses.window, row: int, width: int, char: str, attr: int = 0) -> None:
    _safe_addnstr(stdscr, row, 0, char * max(0, width - 1), max(0, width - 1), attr)


@contextmanager
def _capture_sigint():
    interrupt_state = InterruptState()
    previous = signal.getsignal(signal.SIGINT)

    def _handle_sigint(_signum: int, _frame: FrameType | None) -> None:
        interrupt_state.requested = True

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        yield interrupt_state
    finally:
        signal.signal(signal.SIGINT, previous)


def _restore_terminal(stdscr: curses.window) -> None:
    try:
        stdscr.keypad(False)
    except (AttributeError, curses.error):
        pass
    try:
        stdscr.timeout(-1)
    except (AttributeError, curses.error):
        pass
    for reset in (curses.echo, curses.nocbreak, curses.nl, curses.qiflush):
        try:
            reset()
        except curses.error:
            pass
    try:
        curses.endwin()
    except curses.error:
        pass
