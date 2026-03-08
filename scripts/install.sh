#!/usr/bin/env bash
set -euo pipefail

REPO="${MULTISHELL_GITHUB_REPO:-catid/multishell}"
REF="${MULTISHELL_GITHUB_REF:-main}"
INSTALL_ROOT="${MULTISHELL_INSTALL_ROOT:-$HOME/.local/share/multishell}"
BIN_DIR="${MULTISHELL_BIN_DIR:-$HOME/.local/bin}"
APP_DIR="$INSTALL_ROOT/app"
VENV_DIR="$INSTALL_ROOT/venv"

log_step() {
  printf '\n==> %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

is_ubuntu() {
  if [[ ! -r /etc/os-release ]]; then
    return 1
  fi
  # shellcheck disable=SC1091
  . /etc/os-release
  [[ "${ID:-}" == "ubuntu" ]]
}

require_sudo() {
  if ! have_cmd sudo; then
    echo "sudo is required for automatic Ubuntu dependency installation" >&2
    exit 1
  fi
}

apt_updated=0
ubuntu_apt_install() {
  require_sudo
  if [[ "$apt_updated" -eq 0 ]]; then
    sudo apt-get update
    apt_updated=1
  fi
  sudo apt-get install -y "$@"
}

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

install_ubuntu_dependencies() {
  local packages=()

  if ! "$PYTHON_BIN" -m venv --help >/dev/null 2>&1; then
    packages+=(python3-venv)
  fi
  if ! have_cmd cmake; then
    packages+=(cmake)
  fi
  if ! have_cmd c++; then
    packages+=(build-essential)
  fi
  if ((${#packages[@]} > 0)); then
    ubuntu_apt_install "${packages[@]}"
  fi
}

require_cmd curl
require_cmd tar

PYTHON_BIN="${MULTISHELL_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if have_cmd python3; then
    PYTHON_BIN="python3"
  elif have_cmd python; then
    PYTHON_BIN="python"
  else
    echo "Python 3.11+ is required" >&2
    exit 1
  fi
fi

python_ok="$("$PYTHON_BIN" - <<'PY'
import sys
print("yes" if sys.version_info >= (3, 11) else "no")
PY
)"
if [[ "$python_ok" != "yes" ]]; then
  echo "Python 3.11+ is required" >&2
  exit 1
fi

if is_ubuntu; then
  install_ubuntu_dependencies
fi

if ! "$PYTHON_BIN" -m venv --help >/dev/null 2>&1; then
  echo "python venv support is required; install python3-venv and rerun the installer" >&2
  exit 1
fi

tarball="$tmp_dir/multishell.tar.gz"
log_step "Downloading multishell source from GitHub"
curl -fsSL "https://codeload.github.com/$REPO/tar.gz/$REF" -o "$tarball"
tar -xzf "$tarball" -C "$tmp_dir"
src_dir="$(find "$tmp_dir" -mindepth 1 -maxdepth 1 -type d -name 'multishell-*' | head -n 1)"
if [[ -z "$src_dir" ]]; then
  echo "failed to unpack multishell source" >&2
  exit 1
fi

mkdir -p "$INSTALL_ROOT" "$BIN_DIR"
rm -rf "$APP_DIR"
cp -R "$src_dir" "$APP_DIR"

log_step "Creating Python virtual environment"
"$PYTHON_BIN" -m venv "$VENV_DIR"
log_step "Installing multishell into $VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install "$APP_DIR"

cat > "$BIN_DIR/multishell" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec "$VENV_DIR/bin/python" -m multishell "\$@"
EOF
chmod +x "$BIN_DIR/multishell"

export PATH="$BIN_DIR:$PATH"

log_step "Preparing config file"
"$BIN_DIR/multishell" init-config

needs_login="$("$VENV_DIR/bin/python" - <<'PY'
from multishell.install_state import needs_account_login
print("yes" if needs_account_login() else "no")
PY
)"

if [[ "$needs_login" == "yes" ]]; then
  if [[ -r /dev/tty ]]; then
    setup_tty=/dev/tty
  else
    echo "interactive account setup requires /dev/tty because no saved credentials were detected" >&2
    exit 1
  fi
  log_step "Opening account login TUI"
  "$BIN_DIR/multishell" login <"$setup_tty" >"$setup_tty" 2>"$setup_tty"
else
  log_step "Detected saved account credentials; skipping account login TUI"
fi

browser_args=(install-browser)
if is_ubuntu; then
  browser_args+=(--with-deps)
fi
log_step "Installing Playwright browser runtime (and Linux browser deps on Ubuntu)"
"$BIN_DIR/multishell" "${browser_args[@]}"
log_step "Installing the local auth model runtime and model file (this can take a while)"
"$BIN_DIR/multishell" install-auth-model

missing_auth="$("$VENV_DIR/bin/python" - <<'PY'
from multishell.install_state import missing_auth_agents
print("yes" if missing_auth_agents() else "no")
PY
)"

if [[ "$missing_auth" == "yes" ]]; then
  auto_args=(auto-login --all)
  log_step "Running headless browser auth automation for all configured accounts (press Ctrl+C to skip)"
  "$BIN_DIR/multishell" "${auto_args[@]}"
else
  log_step "Detected existing auth state for all configured agents; skipping headless auth automation"
fi

echo
echo "multishell installed"
echo "binary: $BIN_DIR/multishell"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  echo "add this to your shell config:"
  echo "  export PATH=\"$BIN_DIR:\$PATH\""
fi
echo "next:"
echo "  cd /path/to/project"
echo "  multishell"
