# DONE

Completed all TASK.md steps. Final audit: **PASS**.

Report: [PA3_Zain_Gulbaz_25280005.pdf](report/PA3_Zain_Gulbaz_25280005.pdf) — 25 pages, 15 figures, 13 tables. The extracted PDF text is in `report/extracted_report.txt`. Every nonempty original TASK.md line was verified in the PDF after whitespace normalization; Unicode glyphs are preserved.

| Model | Test MAE | Test RMSE | Training time (s) | Epochs / best |
|---|---:|---:|---:|---:|
| temporal | 71.198 | 112.918 | 208.541 | 18 / 13 |
| fixed | 69.308 | 109.646 | 593.163 | 30 / 29 |
| adaptive | 59.392 | 95.945 | 743.329 | 30 / 29 |
| hybrid | 59.946 | 97.013 | 678.964 | 24 / 19 |

Errors are vehicles per 15-minute interval. All models use the same April–December 2021 test windows. Training times include per-epoch validation and exclude final scoring and timing trials.

Total final training time: 2223.998 seconds. Four one-epoch timing trials: 93.870 seconds. Hybrid alpha: 0.00255803.

## Completed evidence

- Q1–Q2: complete derivations, nine aggregation cases with explicit witnesses/tests, all 164 count triples checked, ring least squares/history/directional numerical checks and heatmaps. History coefficients also work for consistent forward motion.
- Q3: Kaggle file listing followed by only the four requested downloads; official District-11/ID2 extraction, 716 aligned sensors, 15-minute sums, per-sensor/split missingness, causal split-local filling, graph statistics/heatmap/degrees, one-/two-hop ID lists/map, and processed arrays.
- Q4–Q5: full specified chronological windows, train-only normalization, shared lightweight encoder/head/residual, fixed/adaptive/hybrid graphs implemented directly, CPU seed 42, Adam/MAE, identical protocol, complete timing tests, best-validation checkpoints, original-unit overall/per-horizon val/test metrics and hardware metadata.
- Q6: all required forecasts, learning curves, learned graph heatmaps, per-node top-10 road overlap, geographic distance comparisons, non-road link examples and all discussion questions.
- Q7: LaTeX source and compiled PDF, README and exact reproduction commands, pinned environments, full AI disclosure and verbatim TASK.md, gitignore, and one local commit. No remote created or push performed.
- Audit evidence: `results/audit.json`, `results/model_checks.json`, `results/q1_q2.json`, metrics JSON/CSVs and measured timing histories. Local checkpoints/logs/raw and processed data remain available but are excluded from Git.

## Decisions, deviations and limitations

- Used hidden dimension 16, top-20-before-softmax sparsification and a 30-epoch maximum for CPU feasibility; these are allowed by the assignment. Early stopping selected 18/30/30/24 final epochs. Training stayed below the two-hour budget.
- Summed PeMS interval vehicle counts instead of the official notebook’s mean/round operation, preserving vehicles per 15-minute interval as the assignment requires. The directed road graph uses row normalization.
- System Python 3.10 cannot run the latest token-compatible Kaggle package. Installed local Python 3.12 and Kaggle 2.2.5 in `.kaggle-venv`; training uses `.venv`. The latest CLI also required an explicit `six` dependency. No system packages/services were changed.
- Used local Tectonic for LaTeX because a system TeX engine was unavailable. Switched to Unicode OpenType fonts after detecting missing punctuation; the final build has no missing glyphs or LaTeX errors.
- Initial sensor IDs were stored as object arrays. Migrated them to Unicode strings and fixed preprocessing; all numeric arrays were verified byte-identical (`results/id_dtype_migration.json`), so trained results were unchanged.
- An initial Git status inherited an ancestor repository and listed filenames outside this folder. A dedicated local repository was initialized immediately; subsequent Git work is local. No other application or service was changed.
- 19 sensors are entirely missing in training and 43 in each held-out split. Filled targets are included in the required metrics; constant fallback targets cannot establish real accuracy for wholly missing sensors. This limitation is explicit in the report. Results use one seed and do not establish causal graph connectivity or statistical significance.

## Student actions before submission

1. Review and understand the generated code, derivations, results and disclosure.
2. Replace `GITHUB_LINK_HERE` on the report title page with your actual public repository URL.
3. Complete **Edits made by the student** truthfully in the report LaTeX source.
4. Recompile your edited source with `XDG_CACHE_HOME="$PWD/.cache" .tools/tectonic report/PA3_Zain_Gulbaz_25280005.tex`. Do not rerun the report generator unless you also update its text, because it regenerates the source.
5. Create your remote, push the repository, make it public as required, and submit the final PDF through LMS yourself.

No results, plots or timings were invented. All empirical values come from executed code; full details are in the report and results artifacts.
