#!/usr/bin/env bash
# Builds conscious and puts the `conscious` command on this machine.
#
#   ./install.sh               build and install to ~/.local/bin
#   PREFIX=/some/dir ./install.sh   install to /some/dir/bin instead
#   ./install.sh --uninstall   remove the command (your data is kept)
#
# It never uses sudo and never installs anything else for you: when
# something is missing it says what, and how to get it. It finishes by
# running `conscious check`, which reports what the machine has and which
# model would be used.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin_dir="${PREFIX:-$HOME/.local}/bin"
target="$bin_dir/conscious"

say()  { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

if [ "${1:-}" = "--uninstall" ]; then
    if [ -e "$target" ]; then
        # The background process would otherwise outlive the command.
        "$target" stop --force >/dev/null 2>&1 || true
        rm -f "$target"
        say "Removed $target."
    else
        say "Nothing to remove: $target does not exist."
    fi
    say "Your data (memory, skills, timeline) is kept in ${XDG_DATA_HOME:-$HOME/.local/share}/conscious."
    exit 0
fi
[ $# -eq 0 ] || fail "unknown argument \`$1\`; use ./install.sh or ./install.sh --uninstall"

step "Checking what is needed to build"
case "$(uname -s)" in
    Linux)  say "ok    Linux" ;;
    Darwin) say "ok    macOS (the shell sandbox is Linux-only, so shell commands will ask first)" ;;
    *)      fail "unsupported system: $(uname -s). conscious runs on Linux and macOS." ;;
esac
[ -f "$here/Cargo.toml" ] || fail "run this from the conscious source folder (no Cargo.toml in $here)"
if ! command -v cargo >/dev/null 2>&1; then
    fail "Rust is not installed (no \`cargo\` on PATH).
       Install it from https://rustup.rs, open a new terminal, and run this again."
fi
say "ok    $(cargo --version)"
if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
    fail "no C compiler found, which Rust needs to link.
       On Debian/Ubuntu: sudo apt install build-essential"
fi
say "ok    C compiler"

# The first build also downloads the ONNX Runtime library that the local
# embedding model runs on, so it needs the network once.
step "Building (the first build takes a few minutes and needs the network)"
(cd "$here" && cargo build --release --locked)
built="$here/target/release/conscious"
[ -x "$built" ] || fail "the build finished but $built is missing"

step "Installing"
mkdir -p "$bin_dir"
# Copy beside the destination, then rename: a running copy is never
# half-overwritten.
cp "$built" "$target.new"
chmod 755 "$target.new"
mv -f "$target.new" "$target"
say "ok    $target"
case ":$PATH:" in
    *":$bin_dir:"*) say "ok    $bin_dir is on your PATH" ;;
    *) say "warn  $bin_dir is not on your PATH. Add this line to your shell profile:"
       say "          export PATH=\"$bin_dir:\$PATH\"" ;;
esac

step "Checking this machine"
# The same checks conscious runs every time it starts.
if "$target" check; then
    say ""
    say "Installed. Go to a project folder and run: conscious"
else
    say ""
    say "Installed, but no model can be used yet. Fix what is marked FAIL above,"
    say "then run \`conscious check\` again. \`conscious --demo\` works without a model."
    exit 1
fi
