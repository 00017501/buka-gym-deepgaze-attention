#!/usr/bin/env bash
# Recreate the environment used for the results (macOS / Linux, CPU).
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3.13}
DEEPGAZE_COMMIT=c7db17e2d1d7ea6468ffdee2cfaddf141095dcff

"$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

if [ ! -d third_party/DeepGaze ]; then
  git clone https://github.com/matthias-k/DeepGaze.git third_party/DeepGaze
fi
git -C third_party/DeepGaze checkout "$DEEPGAZE_COMMIT"
git -C third_party/DeepGaze apply ../../patches/deepgaze_optional_msdb_import.patch || true
.venv/bin/pip install ./third_party/DeepGaze

if [ ! -f centerbias_mit1003.npy ]; then
  curl -L -o centerbias_mit1003.npy \
    https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
fi

echo "Setup done. Next: .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pytest -m slow"
