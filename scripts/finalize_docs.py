"""Write README from measured outputs. DONE is written only after final audit/commit preparation."""
import json,pathlib,csv
r=json.load(open('results/results.json')); c=json.load(open('results/config.json')); d=json.load(open('results/data_summary.json'))
rows=['| Model | Test MAE | Test RMSE | Training seconds | Epochs (best) |','|---|---:|---:|---:|---:|']
for k,v in r.items(): rows.append(f'| {k} | {v["test"]["mae"]:.3f} | {v["test"]["rmse"]:.3f} | {v["training_seconds"]:.3f} | {v["epochs"]} ({v["best_epoch"]}) |')
table='\n'.join(rows)
readme=f'''# AI651 Assignment 3 — LargeST San Diego

Zain Gulbaz · 25280005 · Fall 2026. CPU only; seed 42.

The report is [report/PA3_Zain_Gulbaz_25280005.pdf](report/PA3_Zain_Gulbaz_25280005.pdf). It contains Q1–Q7, measured results, derivations, all required visualizations and the AI disclosure with the full supplied task prompt. See [TASK.md](TASK.md) for the original instructions and [DONE.md](DONE.md) for completion details.

## Measured results

Errors are vehicles per **15-minute interval**, obtained by summing three PeMS five-minute counts. All models use the same April–December 2021 test period. Times include training and per-epoch validation, excluding final evaluation, and exclude timing trials.

{table}

Hybrid alpha: {r['hybrid']['alpha']:.8f}. One-epoch timing trials total {c['timing_trial_seconds']:.3f} seconds. Full metrics, including validation/test per-horizon MAE and RMSE, are in [results/results.json](results/results.json); the compact table is [results/results_table.csv](results/results_table.csv).

## Setup

Work from this folder. System packages are unnecessary. `scripts/setup.sh` creates a local `.venv`, installs the CPU PyTorch wheel and pinned packages, then installs local Python 3.12 and latest-at-execution Kaggle 2.2.5 in `.kaggle-venv`. The original system Python is 3.10, whose compatible Kaggle 1.7 does not support the supplied new-format access token. An extra `six` dependency is pinned for the latest CLI. Local Tectonic 0.17.0 builds LaTeX without system TeX. Network access is needed for installation, Kaggle files, and first-use TeX packages.

```bash
nohup bash scripts/setup.sh > logs/setup_orchestrator.log 2>&1 < /dev/null
```

On a fresh checkout, create `logs/` first. Configure your own Kaggle token by Kaggle's instructions. The CLI reads the user's existing credential store; no secret belongs in this repository. Do not print or copy credentials. Data are CC BY-NC 4.0 and excluded from Git.

## Exact reproduction commands

Run in order from the repository root. Each long operation uses nohup with a local log. Commands below run in the foreground, allowing straightforward sequential execution while ignoring hangup signals. Add `&` only in a persistent shell if backgrounding is desired, then poll its log until completion before the next dependency.

```bash
export OMP_NUM_THREADS=$(( $(nproc) / 2 ))
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export XDG_CACHE_HOME="$PWD/.cache" MPLCONFIGDIR="$PWD/.cache/matplotlib"
nohup bash scripts/download.sh > logs/download.log 2>&1 < /dev/null
nohup .venv/bin/python -u scripts/preprocess.py > logs/preprocess.log 2>&1 < /dev/null
.venv/bin/python q1_q2/verify.py > logs/q1_q2.log 2>&1
.venv/bin/python -m scripts.check_models > logs/model_checks.log 2>&1
nohup nice -n 10 .venv/bin/python -u -m src.train > logs/training.log 2>&1 < /dev/null
nohup .venv/bin/python -u -m scripts.analyze > logs/analysis.log 2>&1 < /dev/null
.venv/bin/python scripts/build_report.py
nohup .tools/tectonic report/PA3_Zain_Gulbaz_25280005.tex > logs/report_build.log 2>&1 < /dev/null
.venv/bin/python -m scripts.audit > logs/audit.log 2>&1
.venv/bin/python scripts/finalize_docs.py
```

`src.train` sequentially runs a **complete one-epoch trial for temporal, fixed, adaptive and hybrid**, resets seeds/optimizers, and then runs each final experiment. Every final experiment uses all 2020 training windows and the same validation rule; all four use maximum {c['max_epochs']} epochs, patience 5, batch 64, Adam 0.001, normalized MAE and hidden 16. Trial-based wall-clock budgeting caps at 30 epochs; host load can affect the automatically selected cap on another machine. The actual cap and selected epochs are saved in JSON. Checkpoints select minimum validation MAE, never test error.

## Data and leakage controls

The download script first lists Kaggle files and inspects actual filenames, downloading only 2020/2021 HDFs, metadata and the road graph. District 11 selects exactly 716 San Diego sensors. Data columns follow metadata ID; adjacency axes follow the same rows' ID2, with assertions. Nonfinite raw entries: {d['missing_total']:,} ({d['missing_percent']:.6f}%). Missingness per sensor/split is in [results/missing_per_sensor.csv](results/missing_per_sensor.csv). Causal forward filling is confined to each split; leading training gaps use zero, validation/test leading gaps use training sensor means. Zeros stay valid. Imputed targets remain in evaluation: 19 sensors are entirely missing in training and 43 are entirely missing in each held-out split. Entirely missing held-out sensors become constant training-mean targets, so the required aggregate metrics cannot assess actual traffic accuracy at them. This limitation is discussed explicitly in the report. Summation differs intentionally from the official LargeST notebook's averaging/rounding.

Chronological arrays: train all 2020; validation Jan–Mar 2021; test Apr–Dec 2021. Inputs 8 steps, outputs 4, stride one; no window crosses a boundary. Global z-score mean/std come only from training. Both embeddings and learned alpha update only on training loss. All models share the temporal encoder, forecast head and last-reading residual. Graph models share a graph projection and differ only in adjacency. Directed road normalization is row-normalized A+I. Adaptive top-20 selection occurs before softmax; gradients flow through selected scores, not discrete indices. Hybrid alpha is learned from an initial 0.5.

## Repository map

- `q1_q2/verify.py`: exhaustive 164-multiset checks and ring derivations' numerical verification/heatmaps.
- `scripts/download.sh`, `scripts/preprocess.py`: reproducible raw acquisition and SD preparation.
- `scripts/reference/`: official LargeST source notebooks, README and paper used for conventions, with attribution.
- `src/models.py`, `src/common.py`, `src/train.py`: four controlled models, windows, metrics and training.
- `scripts/check_models.py`: shape/normalization/gradient checks.
- `scripts/analyze.py`: three test-sensor predictions, learned graphs, overlap and geography.
- `scripts/build_report.py`: LaTeX generated from measured artifacts; full TASK.md included verbatim.
- `scripts/audit.py`: normalization, windows, graph/checkpoint/metric consistency and final PDF audit.
- `results/`: complete metrics, histories, timing trials, hardware, graph summaries and audit.
- `figures/`, `report/`: final report visuals, source, PDF and extracted text.

`.gitignore` excludes raw/processed data, environments, caches, checkpoints and logs. The local run retains these for inspection; reproduction regenerates them. No remote is configured or push performed. Source data: [LargeST](https://github.com/liuxu77/LargeST), [Kaggle](https://www.kaggle.com/datasets/liuxu77/largest), [PeMS units](https://pems.dot.ca.gov/?dnode=pass).

## AI disclosure and student actions

OpenAI Codex generated and executed code, analysis, figures and report text. Claude (Anthropic) helped write TASK.md. The report includes the full prompt. Before submission, review the work, replace `GITHUB_LINK_HERE`, fill in the **Edits made by the student** paragraph truthfully, push your repository, set it public as required, rebuild the PDF after editing and submit it yourself. The agent did not submit to the LMS.
'''
pathlib.Path('README.md').write_text(readme)
# Detailed metric CSV: one row per model/split/horizon, alongside JSON.
with open('results/metrics_by_horizon.csv','w') as f:
    w=csv.writer(f);w.writerow(['model','split','horizon_minutes','mae','rmse'])
    for k,v in r.items():
        for split in ['val','test']:
            for h in range(4): w.writerow([k,split,15*(h+1),v[split]['horizon_mae'][h],v[split]['horizon_rmse'][h]])
print(table)
