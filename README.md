# Multishell

`multishell` runs one Codex manager behind a terminal TUI and fans work out to persistent Codex and Claude workers.

## Quick start

1. Copy the env template and fill in the Google OAuth accounts locally:

```bash
cp .env.example .env
```

2. Install the project and browser automation dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install playwright pexpect
.venv/bin/python -m playwright install chromium
```

3. If you want headed browser automation over SSH, make sure `google-chrome` and `xvfb` are installed on the machine.

4. Log in all lanes automatically:

```bash
python3 -m multishell auto-login --all --headed
```

5. Start multishell:

```bash
python3 -m multishell run
```

## Notes

- `.env` is ignored by git. Do not commit real credentials.
- On startup, `multishell` cleans up stale old sessions from the same `.multishell` state root before starting the fresh controller.
- `Tab` toggles the debug view and `Ctrl+C` exits.

For everything else, use:

```bash
python3 -m multishell --help
```
