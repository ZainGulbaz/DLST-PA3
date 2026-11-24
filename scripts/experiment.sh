#!/usr/bin/env bash
# One persistent, sequential execution of all required timing tests and final fits.
set -euo pipefail
export OMP_NUM_THREADS=$(( $(nproc) / 2 ))
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export XDG_CACHE_HOME="$PWD/.cache" MPLCONFIGDIR="$PWD/.cache/matplotlib"
nohup nice -n 10 .venv/bin/python -u -m src.train > logs/training.log 2>&1 < /dev/null
