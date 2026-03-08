#!/usr/bin/env bash
set -euo pipefail

REPO="${MULTISHELL_GITHUB_REPO:-catid/multishell}"
REF="${MULTISHELL_GITHUB_REF:-main}"
INSTALL_ROOT="${MULTISHELL_INSTALL_ROOT:-$HOME/.local/share/multishell}"
BIN_DIR="${MULTISHELL_BIN_DIR:-$HOME/.local/bin}"
APP_DIR="$INSTALL_ROOT/app"
VENV_DIR="$INSTALL_ROOT/venv"

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

install_ubuntu_chrome() {
  if have_cmd google-chrome || have_cmd google-chrome-stable; then
    return 0
  fi

  local arch deb
  arch="$(dpkg --print-architecture 2>/dev/null || true)"
  if [[ "$arch" != "amd64" ]]; then
    echo "automatic google-chrome install currently supports Ubuntu amd64 only" >&2
    exit 1
  fi

  deb="$tmp_dir/google-chrome-stable_current_amd64.deb"
  ubuntu_apt_install ca-certificates
  curl -fsSL https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -o "$deb"
  sudo apt-get install -y "$deb"
}

install_ubuntu_dependencies() {
  local packages=()

  if ! "$PYTHON_BIN" -m venv --help >/dev/null 2>&1; then
    packages+=(python3-venv)
  fi
  if [[ -z "${DISPLAY:-}" ]] && ! have_cmd xvfb-run; then
    packages+=(xvfb)
  fi
  if ((${#packages[@]} > 0)); then
    ubuntu_apt_install "${packages[@]}"
  fi

  install_ubuntu_chrome
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

if ! have_cmd google-chrome && ! have_cmd google-chrome-stable; then
  echo "google-chrome is required for the automated Google login flow" >&2
  if ! is_ubuntu; then
    echo "install it manually, or run the installer on Ubuntu so it can install it with sudo" >&2
  fi
  exit 1
fi

if [[ -z "${DISPLAY:-}" ]] && ! have_cmd xvfb-run; then
  echo "xvfb-run is required when DISPLAY is not set" >&2
  if ! is_ubuntu; then
    echo "install xvfb manually, or run the installer on Ubuntu so it can install it with sudo" >&2
  fi
  exit 1
fi

tarball="$tmp_dir/multishell.tar.gz"
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

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install "$APP_DIR"

cat > "$BIN_DIR/multishell" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec "$VENV_DIR/bin/python" -m multishell "\$@"
EOF
chmod +x "$BIN_DIR/multishell"

if [[ -r /dev/tty ]]; then
  setup_tty=/dev/tty
else
  echo "interactive setup requires /dev/tty" >&2
  exit 1
fi

export PATH="$BIN_DIR:$PATH"

"$BIN_DIR/multishell" init-config <"$setup_tty" >"$setup_tty" 2>"$setup_tty"
"$BIN_DIR/multishell" login <"$setup_tty" >"$setup_tty" 2>"$setup_tty"
"$BIN_DIR/multishell" install-browser <"$setup_tty" >"$setup_tty" 2>"$setup_tty"

auto_args=(auto-login --all)
if [[ -n "${DISPLAY:-}" ]]; then
  auto_args+=(--headed)
fi
"$BIN_DIR/multishell" "${auto_args[@]}" <"$setup_tty" >"$setup_tty" 2>"$setup_tty"

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
