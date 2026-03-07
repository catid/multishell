# Multishell

`multishell` runs one Codex manager behind a terminal TUI and fans work out to persistent Codex and Claude workers.

## Install

Prerequisites:

- Python 3.11+
- `google-chrome`
- `xvfb-run` when installing over SSH or any shell without `DISPLAY`

One-command install and first-time setup:

```bash
curl -fsSL https://raw.githubusercontent.com/catid/multishell/main/scripts/install.sh | bash
```

The installer:

- creates a dedicated Python venv under `~/.local/share/multishell`
- installs a `multishell` wrapper into `~/.local/bin`
- launches `multishell login` so you can enter the Google account emails and passwords in a curses TUI
- runs `multishell install-browser`
- runs `multishell auto-login --all` and uses `--headed` automatically when `DISPLAY` is available

## Run

After install, start `multishell` from the repository or project directory you want the workers to use:

```bash
cd /path/to/project
multishell
```

The worker sessions default to the directory where you launch `multishell`, so you can use one install across different repos.

## Setup Commands

Edit the account list again later:

```bash
multishell login
```

## Uninstall

Remove the installed wrapper, runtime, and saved state:

```bash
multishell uninstall
```

Remove the install but keep the saved account data, auth state, browser profiles, and other runtime files under `~/.multishell`:

```bash
multishell uninstall --keep-state
```

## Notes

- Runtime state lives under `~/.multishell` by default.
- Set `MULTISHELL_STATE_ROOT` to move the runtime state elsewhere.
- Set `MULTISHELL_WORKSPACE_ROOT` if you want to override the default worker cwd instead of using the current directory.
- The legacy repo-local `.env` file is still read as a fallback for older setups.
- On startup, `multishell` cleans up stale old sessions from the same state root before starting the fresh controller.
- `Tab` toggles the debug view and `Ctrl+C` exits.

For everything else, use:

```bash
multishell --help
```
