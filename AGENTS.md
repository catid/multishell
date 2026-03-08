## Local Notes

- Any subprocess that can invoke Node.js under the hood, including `codex`, `claude`, and `python -m playwright ...`, must inherit `NODE_NO_WARNINGS=1` via `suppress_node_warnings()` or `child_env()`. Do not launch those commands with a raw environment.
- When a TUI is run under `curses.wrapper(...)`, do not call `curses.endwin()` manually from the wrapped code path. `wrapper()` already handles shutdown, and double-ending curses can raise `_curses.error: endwin() returned ERR`.
- Long-running install or auth automation steps must print explicit progress messages before they begin so browser downloads and login automation do not look hung.
