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

require_cmd curl
require_cmd tar

PYTHON_BIN="${MULTISHELL_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
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

if ! command -v google-chrome >/dev/null 2>&1 && ! command -v google-chrome-stable >/dev/null 2>&1; then
  echo "google-chrome is required for the automated Google login flow" >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" ]] && ! command -v xvfb-run >/dev/null 2>&1; then
  echo "xvfb-run is required when DISPLAY is not set" >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

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
