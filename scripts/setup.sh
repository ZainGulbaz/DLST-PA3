#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs .cache .tools data/raw data/processed checkpoints figures results report
export PIP_CACHE_DIR="$PWD/.cache/pip" UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_PYTHON_BIN_DIR="$PWD/.python/bin" MPLCONFIGDIR="$PWD/.cache/matplotlib" XDG_CACHE_HOME="$PWD/.cache"
export OMP_NUM_THREADS=$(( $(nproc) / 2 ))
python3 -m venv .venv
nohup bash -c '.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install -r requirements.txt' > logs/setup.log 2>&1 < /dev/null
.venv/bin/uv python install 3.12
.venv/bin/uv venv --python 3.12 .kaggle-venv
.venv/bin/uv pip install --python .kaggle-venv/bin/python -r requirements-kaggle.txt
curl -L --fail https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz -o .tools/tectonic-musl.tar.gz
tar -xzf .tools/tectonic-musl.tar.gz -C .tools
