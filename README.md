# Multishell

`multishell` runs one Codex manager behind a terminal TUI and fans work out to a mixed pool of Codex, Claude, Spark, and slow web-reasoner helpers.

## Current layout

- `manager`: the only manager, hosted via `codex-cli`
- `worker-1` .. `worker-4`: persistent Codex workers
- `claude-worker-1` .. `claude-worker-4`: Claude workers sharing the same account lanes as the matching Codex workers
- `claude-worker-5`: a fifth Claude worker that reuses the manager/bot OAuth lane
- `worker-*-spark`: paired `gpt-5.3-spark` delegates used by the Codex workers
- `gpt_5_4_pro` and `gemini_deepthink`: slow parallel browser reasoners launched by the manager

Each account lane gets its own generated `HOME` under `.multishell/homes/...`. Codex auth, Claude auth, and agent-local state stay isolated per lane.

## Models

- Codex manager and Codex workers: `gpt-5.4` with `medium` reasoning
- Spark delegates: `gpt-5.3-spark` with `xhigh` reasoning
- Claude workers: `claude-opus-4-6` with `high` effort

## Configure accounts

Copy the template and fill in the real Google OAuth credentials locally:

```bash
cp .env.example .env
```

Required OpenAI/Codex lanes in `.env`:

```dotenv
MULTISHELL_MANAGER_EMAIL=bot@example.com
MULTISHELL_MANAGER_PASSWORD='replace-me'

MULTISHELL_WORKER_1_EMAIL=worker1@example.com
MULTISHELL_WORKER_1_PASSWORD='replace-me'
MULTISHELL_WORKER_2_EMAIL=worker2@example.com
MULTISHELL_WORKER_2_PASSWORD='replace-me'
MULTISHELL_WORKER_3_EMAIL=worker3@example.com
MULTISHELL_WORKER_3_PASSWORD='replace-me'
MULTISHELL_WORKER_4_EMAIL=worker4@example.com
MULTISHELL_WORKER_4_PASSWORD='replace-me'
```

Gemini Deep Think uses a separate Google OAuth mapping:

```dotenv
MULTISHELL_GEMINI_EMAIL=bot@example.com
MULTISHELL_GEMINI_PASSWORD='replace-me'
```

`.env` is intentionally ignored by git. Do not commit real credentials.

## Install

Typical local setup:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install playwright pexpect pytest
.venv/bin/python -m playwright install chromium
```

If you want headed browser automation over SSH, make sure `google-chrome` and `xvfb` are installed on the host.

## Login

Manual login:

```bash
python3 -m multishell login manager
python3 -m multishell login worker-1
python3 -m multishell login worker-2
python3 -m multishell login worker-3
python3 -m multishell login worker-4
python3 -m multishell login claude-worker-1
python3 -m multishell login claude-worker-2
python3 -m multishell login claude-worker-3
python3 -m multishell login claude-worker-4
python3 -m multishell login claude-worker-5
```

Automatic login with Playwright:

```bash
python3 -m multishell auto-login --all --headed
```

## Run

Check what is configured and logged in:

```bash
python3 -m multishell status
```

Start the TUI:

```bash
python3 -m multishell run
```

On startup, `multishell` now scans for stale prior sessions under the same `.multishell` state root, terminates them, and then starts the fresh controller. On shutdown, the controller stops the manager, workers, Spark delegates, and browser reasoners instead of leaving old subprocesses behind.

## TUI keys

- `Enter`: send the current chat line to the manager
- `Tab`: toggle the debug panel view
- `Arrow keys` or `h/j/k/l`: move focus in debug mode
- `PgUp` / `PgDn`: scroll the focused transcript in debug mode
- `g` / `G`: jump to older or newer transcript content
- `Ctrl+C`: quit

## Notes

- The manager talks to the user through the MCP bridge tool layer, not raw assistant text.
- Restarting or stopping a worker session intentionally clears that worker's active memory.
- Claude workers are for creative exploration, code review, and different-model pressure testing; Codex workers remain the more reliable execution path.
- Spark output is draft material only. Existing-file edits should be reviewed by the owning Codex worker before they hit disk.
