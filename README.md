# Multishell

`multishell` runs one Codex manager behind a terminal TUI and fans work out to persistent Codex and Claude workers.

## Install

Prerequisites:

- Python 3.11+
- On Ubuntu, the installer uses `sudo` to install `python3-venv`, `cmake`, `build-essential`, and Playwright browser dependencies if they are missing.
- On other Linux distributions, install `cmake` and a C++ compiler yourself before running the setup script.

One-command install and first-time setup:

```bash
curl -fsSL https://raw.githubusercontent.com/catid/multishell/main/scripts/install.sh | bash
```

The installer:

- creates a dedicated Python venv under `~/.local/share/multishell`
- installs a `multishell` wrapper into `~/.local/bin`
- installs missing Ubuntu system packages with `sudo` when needed
- launches `multishell login` so you can add Google accounts in a curses TUI, mask/edit passwords, and toggle OpenAI, Anthropic, and Gemini AI Ultra access per account
- runs `multishell install-browser --with-deps` on Ubuntu, otherwise `multishell install-browser`
- runs `multishell install-auth-model`
- runs `multishell auto-login --all` in headless mode

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

The login TUI starts with an empty account list. Add as many Google accounts as you want and toggle provider access per row:

- OpenAI-enabled accounts determine the manager account plus the available Codex worker pool.
- Anthropic-enabled accounts determine the available Claude worker pool.
- Gemini-enabled accounts become the Gemini AI Ultra account pool for Deep Think jobs.

## Headless Auth Model

`multishell auto-login` now uses a headless Playwright-managed Chromium by default, even on machines with `DISPLAY` set. Pass `--headed` only when you explicitly want visible browser windows for debugging.

`multishell auto-login` now manages the local auth-model server for you. If the local model runtime is installed, it starts the `llama.cpp` server on demand, uses a text-only auth controller first for the brittle Google sign-in steps, then falls back to the older selector flow if the model returns unusable actions.

If Google returns a hard error like wrong password, account not found, browser not secure, or a manual verification challenge, `multishell` now fails that lane immediately with a specific message instead of sitting on the page until timeout.

If one account fails during `multishell auto-login`, the remaining accounts still run. The command prints a per-lane failure line as it goes and raises one summary at the end listing every failed account, including the debug artifact directory when a browser page dump was captured.

The default local model settings target a `llama.cpp`-style server:

```bash
export MULTISHELL_AUTH_MODEL_API_BASE=http://127.0.0.1:8080/v1
export MULTISHELL_AUTH_MODEL=Qwen/Qwen3.5-9B
```

Install or refresh the full local auth-model runtime again later with:

```bash
multishell install-auth-model
```

If you want to force a rebuild of `llama.cpp` and a fresh model download, use:

```bash
multishell install-auth-model --force
```

The GGUF download is resumable, so rerunning the command after an interrupted download continues from the existing `.part` file instead of restarting from zero. The local `llama.cpp` build and Qwen GGUF cache live under `~/.multishell/cache/auth-model` by default.

The auth controller sends `enable_thinking=false` for Qwen and strips any stray `<think>...</think>` output before parsing actions.

The Playwright browser runtime is also kept under the `multishell` install root instead of the default global Playwright cache, so the install stays self-contained.

If you are setting up on Ubuntu outside the one-shot installer and need Playwright’s shared browser libraries too, run:

```bash
multishell install-browser --with-deps
```

Each auth-model run also writes a per-agent trace log under `~/.multishell/debug/<agent>/` with the page snapshot summary, visible elements, model metrics, raw model output, and chosen action. On failures, those traces sit next to the saved screenshot and HTML dump.

To verify the local auth-model server and headless browser loop without touching Google, run:

```bash
multishell auth-model-smoke-test
```

To benchmark local CPU speed for the auth model:

```bash
multishell auth-model-bench
```

The lower-level `llama.cpp` build step is still available separately if you only want the binaries:

```bash
multishell install-llama-cpp
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
- On startup, `multishell` cleans up stale old sessions from the same state root before starting the fresh controller.
- `Tab` toggles the debug view and `Ctrl+C` exits.

For everything else, use:

```bash
multishell --help
```
