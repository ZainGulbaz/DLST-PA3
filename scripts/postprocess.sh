#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=$(( $(nproc) / 2 ))
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export XDG_CACHE_HOME="$PWD/.cache" MPLCONFIGDIR="$PWD/.cache/matplotlib"
nohup .venv/bin/python -u -m scripts.analyze > logs/analysis.log 2>&1 < /dev/null
.venv/bin/python scripts/build_report.py
nohup .tools/tectonic report/PA3_Zain_Gulbaz_25280005.tex > logs/report_build.log 2>&1 < /dev/null
.venv/bin/python -m scripts.audit > logs/audit.log 2>&1
.venv/bin/python scripts/finalize_docs.py
printf 'POSTPROCESS COMPLETE\n'
