#!/bin/bash
# Run from any directory. Packages live in this copy of Capri's .venv.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash start.sh [--check | --help]

Start Capri on this Mac. First launch creates a private Python environment and
installs its packages; later launches reuse it. Keep Terminal open while chatting.
Press Control-C to stop. No browser is opened automatically.

  --check   Check the Mac, Python version and free storage.
  --help    Show these instructions.

Python 3.13 for macOS: https://www.python.org/downloads/release/python-31317/
HELP
}

fail() {
    printf '\nCapri: %s\n' "$*" >&2
    exit 1
}

CHECK_ONLY=0
if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi
case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --check) CHECK_ONLY=1 ;;
    '') ;;
    *) usage >&2; exit 2 ;;
esac

CAPRI_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$CAPRI_DIR"
[[ -f requirements.txt && -f server.py ]] || fail 'This download is incomplete. Download and unzip Capri again.'

[[ "$(uname -s)" == Darwin ]] || fail 'This build needs macOS on an Apple Silicon Mac.'
[[ "$(uname -m)" == arm64 ]] || fail 'This build needs Apple Silicon running natively. Intel Macs and Terminal running under Rosetta are unsupported.'

MAC_VERSION="$(sw_vers -productVersion)"
IFS=. read -r MAC_MAJOR MAC_MINOR MAC_PATCH <<< "$MAC_VERSION"
MAC_MINOR="${MAC_MINOR:-0}"
[[ "$MAC_MAJOR" =~ ^[0-9]+$ && "$MAC_MINOR" =~ ^[0-9]+$ ]] || fail 'Could not read the macOS version.'
if (( MAC_MAJOR < 26 || (MAC_MAJOR == 26 && MAC_MINOR < 2) )); then
    fail "This build needs macOS 26.2 or later; this Mac is running $MAC_VERSION."
fi

MEMORY_BYTES="$(sysctl -n hw.memsize 2>/dev/null)" || fail "Could not read this Mac's memory size."
[[ "$MEMORY_BYTES" =~ ^[0-9]+$ ]] || fail "Could not read this Mac's memory size."
if (( MEMORY_BYTES < 24 * 1024 * 1024 * 1024 )); then
    fail 'This build needs at least 24 GB of unified memory. A Mac with 32 GB or more is recommended.'
fi

# Ignore unrelated Python configuration left by another project or application.
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

is_python_313() {
    "$1" -c 'import platform, sys; sys.exit(0 if sys.version_info[:2] == (3, 13) and platform.machine() == "arm64" else 1)' >/dev/null 2>&1
}

PYTHON=""
PYTHON_CANDIDATES=()
if [[ -n "${CAPRI_PYTHON:-}" ]]; then
    PYTHON_CANDIDATES+=("$CAPRI_PYTHON")
else
    PYTHON_CANDIDATES+=(/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13)
    for PYTHON_NAME in python3.13 python3; do
        PYTHON_PATH="$(command -v "$PYTHON_NAME" 2>/dev/null || true)"
        [[ -z "$PYTHON_PATH" ]] || PYTHON_CANDIDATES+=("$PYTHON_PATH")
    done
fi
for PYTHON_PATH in "${PYTHON_CANDIDATES[@]}"; do
    if [[ -x "$PYTHON_PATH" ]] && is_python_313 "$PYTHON_PATH"; then
        PYTHON="$PYTHON_PATH"
        break
    fi
done
[[ -n "$PYTHON" ]] || fail 'Native Python 3.13 was not found. Install the Python 3.13 macOS installer from https://www.python.org/downloads/release/python-31317/ and run this command again.'

printf 'Mac: Apple Silicon, macOS %s, %s GB unified memory\n' "$MAC_VERSION" "$(( MEMORY_BYTES / 1024 / 1024 / 1024 ))"
printf 'Python: %s\n' "$PYTHON"
printf 'Capri needs about 20 GB of available memory; close other memory-heavy apps.\n'
if (( CHECK_ONLY )); then
    "$PYTHON" - <<'PY'
import os
from pathlib import Path
from shutil import disk_usage
import sys


def free_storage(label, path):
    try:
        path = path.resolve()
        volume_path = path
        while not volume_path.exists() and volume_path != volume_path.parent:
            volume_path = volume_path.parent
        available = disk_usage(volume_path).free
    except (OSError, RuntimeError) as error:
        print(f"Capri: Could not measure free storage for {path}. {error}", file=sys.stderr)
        return None
    print(f"{label}: {available / 1e9:.1f} GB free ({available:,} bytes) at {path}")
    return available


def expand_path(value):
    return Path(os.path.expandvars(os.path.expanduser(value)))


free_storage("Capri folder volume", Path.cwd())
local_model = os.environ.get("CAPRI_MODEL_PATH")
if local_model:
    free_storage("Local model volume", Path(local_model))
else:
    cache_home = os.environ.get("XDG_CACHE_HOME", "~/.cache")
    hf_home = expand_path(os.environ.get("HF_HOME", os.path.join(cache_home, "huggingface")))
    legacy_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    hub = expand_path(os.environ.get("HF_HUB_CACHE", legacy_cache))
    available = free_storage("Model cache volume", hub)
    if available is not None and available < 18_000_000_000:
        print("Capri: A first model download may need 18 GB of free storage on the model cache volume.", file=sys.stderr)
PY
    printf 'Checks passed. No packages installed and no model started.\n'
    exit 0
fi

if [[ ! -d .venv ]]; then
    printf "\nCreating Capri's private Python environment...\n"
    "$PYTHON" -m venv "$CAPRI_DIR/.venv" || fail 'Could not create .venv. Check that the Python 3.13 installation completed.'
fi
VENV_PYTHON="$CAPRI_DIR/.venv/bin/python"
[[ -x "$VENV_PYTHON" ]] && is_python_313 "$VENV_PYTHON" || fail 'This folder has an incomplete or incompatible .venv. Rename that .venv folder and run setup again, or unzip a fresh copy of Capri.'
"$VENV_PYTHON" -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)' || fail "Python did not enter Capri's private environment. No packages were installed."

REQUIREMENTS_KEY="$("$VENV_PYTHON" -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$CAPRI_DIR/requirements.txt")"
STAMP="$CAPRI_DIR/.venv/.capri-requirements.sha256"
INSTALLED_KEY="$(cat "$STAMP" 2>/dev/null || true)"

runtime_ok() {
    "$VENV_PYTHON" - "$CAPRI_DIR/requirements.txt" <<'PY'
import importlib
import importlib.metadata
import pathlib
import sys

for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        name, expected = line.split("==", 1)
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"{name} must be {expected}")
for module in ("mlx.core", "mlx_lm", "numpy", "safetensors", "huggingface_hub"):
    importlib.import_module(module)
PY
}

if [[ "$INSTALLED_KEY" != "$REQUIREMENTS_KEY" ]] || ! runtime_ok >/dev/null 2>&1; then
    printf "\nInstalling Capri's packages into .venv. This can take several minutes...\n"
    "$VENV_PYTHON" -m pip --isolated install --disable-pip-version-check --only-binary=:all: -r "$CAPRI_DIR/requirements.txt" \
        || fail 'Package installation did not finish. Check your internet connection and run the same command again.'
    runtime_ok || fail 'The installed packages could not load. Check the error above; no model was started.'
    "$VENV_PYTHON" -m pip --isolated check || fail 'Package dependencies do not match. No model was started.'
    printf '%s\n' "$REQUIREMENTS_KEY" > "$STAMP"
else
    printf "\nUsing Capri's installed packages.\n"
fi

printf '\nStarting Capri. The first model download is about 16.1 GB.\n'
printf 'Open https://maxwen.ca/capri, click Connect local Capri, and allow local-network access if your browser asks.\n'
printf 'If your browser cannot connect, open http://127.0.0.1:8777 for the same 3D chat.\n'
printf 'Wait for the model to finish downloading and loading before sending a message.\n'
printf 'Keep this Terminal open. Press Control-C to stop; run the same command to start again.\n\n'
exec "$VENV_PYTHON" -u server.py
