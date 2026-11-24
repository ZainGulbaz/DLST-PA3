# AI651 PA3 – full autonomous task

You are completing my university assignment end to end, fully autonomously. I will NOT be watching. Do not stop to ask questions, and do not end your turn or hand back to me until EVERYTHING below is finished. Make reasonable decisions, record each one in the report, and keep going. The assignment is in ./DL4STG_PA3.pdf (extract it with pdftotext, or pypdf if that's missing). Read the whole PDF first and follow every "Required" item in it.

## Hard constraints
- Working folder: /home/etn/WJL_RUNING_APPS_PRODUCTION/pa3. This is a shared PRODUCTION server. Work ONLY inside this folder. Do not read, modify or delete anything else in WJL_RUNING_APPS_PRODUCTION or elsewhere. Do not touch other processes, pm2 apps, services or system packages.
- Use at most half the CPU cores: set OMP_NUM_THREADS and torch.set_num_threads to nproc/2. Run all training with `nice -n 10`.
- The late deadline is TODAY. Finish everything (code, results, PDF report) within about 6 hours. Budget about 2 hours of total model training. If training is slow, reduce epochs and record that in the report.
- NEVER invent, estimate or "fill in" any number, result, timing or plot. Every reported value must come from code that actually ran. If something fails, fix it. If it truly cannot be done, say so honestly in the report.
- CPU only. Use a fixed random seed (42) everywhere: python, numpy, torch.
- Use no validation or test data for normalization statistics, learned graphs or any training-dependent feature.
- Run every long job (download, preprocessing, each training run) with nohup, writing to logs/<name>.log, so it survives disconnection. Poll the logs until each job finishes.
- Use a virtual environment in this folder (python3 -m venv .venv). Install the CPU build of torch (pip install torch --index-url https://download.pytorch.org/whl/cpu) plus numpy, pandas, matplotlib, h5py, tables, scipy, pypdf, and the LATEST kaggle package (pip install -U kaggle). My Kaggle token is already saved in the new format at ~/.kaggle/access_token. NEVER print, copy or commit that token or any other credential.

## Report details
- Name: Zain Gulbaz. Roll number: 25280005. Course: AI651 – Deep Learning for Space, Time and Graphs, Fall 2026, Assignment 3.
- First page: the placeholder text GITHUB_LINK_HERE for the repo link (I will push the repo myself).

## Step 1 – Questions 1 and 2 (math + NumPy, no dataset)
Answer every part fully, with derivations, in the report.
- Q1(a): show the aggregate is w·S_v + c·n_v, explain why equal (n_v, S_v) gives identical outputs for any w, c, rho, and give two neighbourhoods with the same n_v and S_v but different numbers of steady neighbours (e.g. {-1,+1,0} vs {0,0,0}).
- Q1(b): construct phi(-1), phi(0), phi(1) as place values in base K+1 (e.g. 1, K+1, (K+1)^2), explain decoding by repeated division/mod, and verify in NumPy for K=8 over EVERY triple (a,b,c) with 1 <= a+b+c <= 8 that the decoding is unique and correct. Print the number of triples checked. Also give phi for counting non-steady neighbours (phi(x) = x^2 or |x|).
- Q1(c): fill the full 9-row table (sum/mean/max × 3 tasks). For each Yes, give the coordinate(s) and the test. For each No, give two allowed neighbourhoods with the same aggregated vector but different answers, writing out the vector and both answers. Check every row with a brute-force NumPy script over all multisets of size 1–8.
- Q2(a)(i): derive the least-squares a, b analytically, showing predicted vs true values at sensors 19, 20, 21. Confirm numerically.
- Q2(a)(ii): roll out the fitted layer for 20 steps from x_i(0) = exp(-(i-20)^2/8), and separately the true shift (np.roll(x,-1)). Plot two heatmaps side by side (sensor on x-axis, t=0..20 on y-axis, SAME colour scale). Add a 1–2 sentence comparison.
- Q2(b): find a, w-, w+ giving the exact next snapshot; verify by substitution and numerically. Explain why directional weights fix (a). Give the 1D conv kernel in the order [x_{i-1}, x_i, x_{i+1}] and circular padding for the ring.
- Q2(c): derive a, b, c using history (rewrite x_i(t-1) in terms of a current reading under backward motion). Verify numerically. Check the forward-motion case by substitution and numerically, show whether it fails, and explain how history breaks the symmetry.
Save scripts in q1_q2/ and figures in figures/.

## Step 2 – Data (Q3)
- Use the Kaggle CLI on dataset liuxu77/largest. FIRST list its files (kaggle datasets files liuxu77/largest) and download only what is needed: the 2020 and 2021 history files, the metadata CSV and the road-network adjacency (.npy). Do not assume file names; inspect them. Put raw files in data/raw/. If GitHub is reachable, check the official repo https://github.com/liuxu77/LargeST data-processing scripts and follow its San Diego extraction convention.
- Select the San Diego sensors from the metadata (County == San Diego, or the repo's method). Assert exactly 716 sensors. Keep the sensor order identical between the data tensor and the adjacency submatrix (index the full adjacency with the same indices). Assert this in code.
- Resample 5-min to 15-min. Find out whether the values are 5-min vehicle counts (PeMS Total Flow) or rates, choose sum or mean, and justify the choice and the physical units in the report. Note what the LargeST repo does.
- Report missing values (count, % per sensor, overall). Choose and document a handling strategy that leaks no information across splits (e.g. forward-fill then backward-fill within each split, or masked loss). Explain it.
- Plot one week of 15-min flow for 3 sensors.
- Q3(b): number of nodes, number of nonzero edges (with and without the diagonal), symmetry check (max |A - A^T|), adjacency heatmap, degree distribution histogram, and an explanation of what A_ij means.
- Q3(c): pick one sensor, list its 1-hop and 2-hop neighbours, plot them on lat/lon from the metadata if available, and answer the three discussion bullets (including over-smoothing).
- Save processed arrays to data/processed/ (.npz) so training scripts load fast.

## Step 3 – Models (Q4, Q5)
- Windows: input 8 steps, output 4 steps. Chronological splits: train = all of 2020, val = Jan–Mar 2021, test = Apr–Dec 2021. Windows must never cross split boundaries. Z-score normalise with train statistics only.
- One shared codebase in src/. Models:
  1. Temporal-only: temporal conv encoder (e.g. 2 Conv1d/gated TCN layers over time per sensor, hidden d=32) -> linear/MLP head predicting 4 steps. No graph.
  2. STGNN-Fixed: same encoder -> GCN H = ReLU(Â Z W_g) -> same head. Â from the road adjacency. If A is symmetric use D^-1/2 (A+I) D^-1/2; if not, use a documented directed normalisation (e.g. row-normalised D^-1 (A+I)). It stays unchanged during training.
  3. STGNN-Adaptive: trainable E1, E2 in R^{N x de} (de=10), A_adp = row-softmax(ReLU(E1 E2^T)). Implement this yourself and explain it. Optional top-k sparsification (k=20): document whether it happens before/after normalisation and how gradients flow.
  4. STGNN-Hybrid: A = alpha·Â_fixed + (1-alpha)·A_adp, with both row-normalised consistently; learnable alpha via sigmoid (init 0.5). Report the learned alpha.
  Any residual/skip connection must be IDENTICAL across all models. The only difference between models 2–4 is the graph.
- Same training protocol for all: Adam, lr 1e-3, batch size 32–64, MAE loss on normalised data, early stopping on val MAE (patience ~5). Log per-epoch train/val loss and epoch time.
- Run a 1-epoch timing test per model first, then set max epochs so total training fits ~2 hours.
- Evaluate val and test MAE and RMSE in ORIGINAL flow units, overall and per horizon (15/30/45/60 min), on the same test period for all models.
- Measure wall-clock training time per model. Record hardware: CPU model, cores used, RAM, OS, torch version (lscpu, free -h).
- Save checkpoints in checkpoints/, and results in results/results.json and results/results_table.csv.

## Step 4 – Analysis (Q6)
- Final table: Model | Adjacency | MAE | RMSE | Training time (test set), plus an extra table with val metrics and per-horizon test metrics.
- Predicted vs actual plots for at least 3 sensors over a test-period window (e.g. 3 days), all 4 models on each plot.
- Graph analysis: heatmaps of the learned adaptive and hybrid adjacency. Quantify: the fraction of each node's top-10 learned neighbours that are also road-graph edges; the haversine distance of learned top-k neighbours vs road neighbours vs random pairs (histogram/boxplot); examples of strong learned links absent from the road graph. Answer all 4 Q6(b) questions and all 3 Q6(c) bullets with reference to YOUR numbers. Interpret honestly even if the graph did not help.

## Step 5 – Report and repo (Q7)
- Write the report in LaTeX (or Markdown -> PDF via pandoc) and build report/PA3_Zain_Gulbaz_25280005.pdf. Structure: title page (name, roll number, GITHUB_LINK_HERE), Q1–Q7 in order, every figure captioned, the comparison tables, hyperparameters, epochs, seed, hardware, training times, reproduction instructions. Extract the final PDF's text and check every figure and table appears and no placeholder other than GITHUB_LINK_HERE and the student-edits paragraph remains.
- The final section "AI Usage Disclosure" (required by the course policy) must state that OpenAI Codex was used as the coding agent and that Claude (Anthropic) helped write this task prompt, include this TASK.md verbatim as the prompt, summarise what the AI produced (code, analysis, report text), and include a clearly marked placeholder paragraph titled "Edits made by the student" for me to fill in.
- Repo layout: README.md (setup, data download, exact commands to reproduce each experiment, results table), requirements.txt, src/, q1_q2/, scripts/, figures/, results/, report/. Add a .gitignore excluding data/raw, data/processed, .venv, checkpoints and logs. Run git init and make one clean commit. Do NOT create a remote or push.

## When done
Write DONE.md and print a summary: the final results table, the PDF path, any deviations from the assignment and why, and what I must do by hand (replace GITHUB_LINK_HERE, fill in student edits, push the repo).
