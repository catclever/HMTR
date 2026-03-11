#!/bin/bash
set -euo pipefail

export JULIA_PKG_SERVER="${JULIA_PKG_SERVER:-https://mirrors.bfsu.edu.cn/julia}"

echo "=== HMTR Julia Setup Script ==="

JULIA_VERSION="${JULIA_VERSION:-1.12.3}"
JULIA_INSTALL_DIR="${JULIA_INSTALL_DIR:-$HOME/julia-$JULIA_VERSION}"
JULIA_BIN="$JULIA_INSTALL_DIR/bin/julia"

if [[ -x "$JULIA_BIN" ]]; then
  echo "Julia found: $($JULIA_BIN --version)"
else
  echo "Julia not found at $JULIA_INSTALL_DIR. Downloading Julia $JULIA_VERSION..."
  mkdir -p "$JULIA_INSTALL_DIR"

  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64|amd64)
      JULIA_ARCH_DIR="x64"
      JULIA_ARCH="x86_64"
      ;;
    aarch64|arm64)
      JULIA_ARCH_DIR="aarch64"
      JULIA_ARCH="aarch64"
      ;;
    *) echo "Unsupported arch: $ARCH" ; exit 1 ;;
  esac

  TARBALL="julia-${JULIA_VERSION}-linux-${JULIA_ARCH}.tar.gz"
  URL="https://julialang-s3.julialang.org/bin/linux/${JULIA_ARCH_DIR}/${JULIA_VERSION%.*}/${TARBALL}"

  LOCAL_TARBALL="${JULIA_TARBALL:-}"

  if [[ -n "${LOCAL_TARBALL}" && -f "${LOCAL_TARBALL}" ]]; then
    echo "Using local Julia tarball: ${LOCAL_TARBALL}"
  else
    echo "Downloading Julia from ${URL} ..."

    TMPDIR="$(mktemp -d)"
    cleanup() { rm -rf "$TMPDIR"; }
    trap cleanup EXIT

    if command -v curl >/dev/null 2>&1; then
      curl -fL "$URL" -o "$TMPDIR/$TARBALL"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$TMPDIR/$TARBALL" "$URL"
    else
      echo "Need curl or wget to download Julia." ; exit 1
    fi

    LOCAL_TARBALL="$TMPDIR/$TARBALL"
  fi

  tar -xzf "$LOCAL_TARBALL" -C "$JULIA_INSTALL_DIR" --strip-components=1

  echo "Julia installed: $($JULIA_BIN --version)"
fi

echo "Instantiating Julia project dependencies..."
"$JULIA_BIN" --project=. -e 'using Pkg; Pkg.instantiate()'

LINK_DIR="/usr/local/bin"
if [[ -w "$LINK_DIR" || ( -d "$LINK_DIR" && -w "$LINK_DIR" ) ]]; then
  ln -sf "$JULIA_BIN" "$LINK_DIR/julia"
  echo "Julia added to PATH via symlink: $LINK_DIR/julia"
else
  LINK_DIR="$HOME/.local/bin"
  mkdir -p "$LINK_DIR"
  ln -sf "$JULIA_BIN" "$LINK_DIR/julia"
  case ":$PATH:" in
    *":$LINK_DIR:"*) ;;
    *)
      export PATH="$LINK_DIR:$PATH"
      BASHRC="$HOME/.bashrc"
      if [[ -f "$BASHRC" ]]; then
        if command -v grep >/dev/null 2>&1; then
          if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$BASHRC"; then
            printf '\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$BASHRC"
          fi
        else
          printf '\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$BASHRC"
        fi
      else
        printf 'export PATH="$HOME/.local/bin:$PATH"\n' > "$BASHRC"
      fi
      ;;
  esac
  echo "Julia added to PATH via symlink: $LINK_DIR/julia"
fi

echo "=== Environment Ready ==="
echo "Run training with: $JULIA_BIN --project=. src/train_stage1.jl"
