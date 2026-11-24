"""Generate LaTeX report directly from measured artifacts, then compile separately."""
import json,pathlib,pandas as pd,numpy as np
P=pathlib.Path('.')
def load(name): return json.load(open('results/'+name+'.json'))
def esc(s):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','_':r'\_','#':r'\#','$':r'\$','{':r'\{','}':r'\}','^':r'\textasciicircum{}','~':r'\textasciitilde{}','→':r'\ensuremath{\to}','²':r'\textsuperscript{2}'}
    return ''.join(mapping.get(c,c) for c in str(s))

def fmt(v): return f'{v:.3f}'
q=load('q1_q2'); d=load('data_summary'); r=load('results'); g=load('graph_analysis'); cfg=load('config'); trials=load('timing_trials')
out=[]
def add(s): out.append(s)
def para(s): add(esc(s)+'\n\n')
def fig(file,caption,width='.91'):
    add(r'\begin{figure}[htbp]\centering\includegraphics[width='+width+r'\linewidth]{../figures/'+file+r'}\caption{'+esc(caption)+r'}\end{figure}'+'\n')
def table(headers,rows,caption,align=None):
    align=align or ('l'*len(headers))
    add(r'\begin{table}[htbp]\centering\small\caption{'+esc(caption)+r'}\begin{tabular}{'+align+r'}\toprule'+'\n')
    add(' & '.join(esc(x) for x in headers)+r'\\\midrule'+'\n')
    for row in rows: add(' & '.join(esc(x) for x in row)+r'\\'+'\n')
    add(r'\bottomrule\end{tabular}\end{table}'+'\n')
add(r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=23mm]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,array,float}
\usepackage{fontspec}
\setmainfont{lmroman10-regular.otf}[BoldFont=lmroman10-bold.otf,ItalicFont=lmroman10-italic.otf,BoldItalicFont=lmroman10-bolditalic.otf]
\setmonofont{lmmono10-regular.otf}
\usepackage{fvextra}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}
\setlength{\emergencystretch}{3em}
\setcounter{tocdepth}{1}
\begin{document}
\begin{titlepage}\centering
{\Large Lahore University of Management Sciences\par}\vspace{15mm}
{\huge AI651 -- Deep Learning for Space, Time and Graphs\par}\vspace{8mm}
{\Large Fall 2026, Assignment 3\par}\vspace{15mm}
{\LARGE Zain Gulbaz\par}\vspace{4mm}
{\Large Roll number: 25280005\par}\vspace{15mm}
{\large Repository: \url{https://github.com/ZainGulbaz/DLST-PA3}\par}\vspace{8mm}
{\large CPU experiments, seed 42\par}\vfill
{\large 24 November 2026\par}
\end{titlepage}
\tableofcontents\newpage
\section{Q1 -- What does a sum remember?}
\subsection{(a) Affine messages}
Let $n_v=|\mathcal N(v)|$ and $S_v=\sum_{u\in\mathcal N(v)}x_u$. Then
\[
\sum_u\phi(x_u)=\sum_u(wx_u+c)=wS_v+cn_v.
\]
This also holds coordinatewise for vector parameters. Equal $(n_v,S_v)$ forces the same input to $\rho$ for every $w,c$, hence the same output for every $\rho$. The neighbourhoods $\{-1,+1,0\}$ and $\{0,0,0\}$ both have $(n_v,S_v)=(3,0)$ and aggregate $3c$, but have respectively one and three steady neighbours. No subsequent activation can recover information already discarded.
\subsection{(b) A scalar nonlinear encoding}
Set $B=K+1$, $\phi(-1)=1$, $\phi(0)=B$, $\phi(+1)=B^2$. If counts are $(a,b,c)$, the aggregate is $m=a+Bb+B^2c$. Each count is at most $K<B$, so no carries occur. Decode by $a=m\bmod B$, replace $m$ by $\lfloor m/B\rfloor$, recover $b=m\bmod B$, divide again, and obtain $c$. This is unique base-$B$ representation; it relies on exact arithmetic (large $K$ may require integers beyond floating-point precision). For counting only non-steady neighbours, $\phi(x)=x^2=|x|$ on this alphabet gives $a+c$ directly.
''')
para(f'For K=8, the NumPy script checks all {q["triples_checked"]} triples with total size 1 through 8 (binomial(11,3)-1 = 164). Every code is unique and every decoded triple equals its input. The exact stdout and structured outputs are in logs/q1_q2.log and results/q1_q2.json.')
add(r'\subsection{(c) Sum, mean and max of one-hot messages}'+'\n')
para('Write the aggregate as (z1,z2,z3) in slowing, steady, speeding-up order. The nine decisions below were checked against every multiset of size 1–8; exact rational keys avoid floating-point collisions for means.')
reasons={ (1,'sum'):'Yes: steady count = z2.',(1,'mean'):'No: proportions lose size.',(1,'max'):'No: presence loses multiplicity.',(2,'sum'):'Yes: 2 z1 > z1+z2+z3.',(2,'mean'):'Yes: z1 > 1/2.',(2,'max'):'No: presence loses proportions.',(3,'sum'):'Yes: z1 > 0.',(3,'mean'):'Yes: z1 > 0.',(3,'max'):'Yes: z1 = 1.' }
table(['Task','Method','Decision and calculation'],[(row['task'],row['method'],reasons[(row['task'],row['method'])]) for row in q['rows']],'All nine aggregation decisions. Task 1 counts steady; 2 tests slowing majority; 3 tests any slowing.','llp{10cm}')
for row in q['rows']:
    if row['possible']: continue
    w=row['witness']; counts=lambda t:'{' + ', '.join(['-1']*t[0]+['0']*t[1]+['+1']*t[2])+'}'
    para(f'Task {row["task"]}, {row["method"]}: neighbourhoods {counts(w["counts1"])} and {counts(w["counts2"])} both give vector ({", ".join(w["aggregate"])}), but the required answers are {w["answer1"]} and {w["answer2"]}. Both are nonempty and have at most eight members. Thus no function of the aggregate can distinguish them.')
add(r'''\section{Q2 -- Which way is the slowdown moving?}
\subsection{(a) Symmetric spatial layer}
For the isolated spike, predictions at sensors $(19,20,21)$ are $(b,a,b)$ and targets are $(1,0,0)$. All other errors vanish, hence
\[
L(a,b)=(b-1)^2+a^2+b^2,\quad \partial_aL=2a=0,\quad \partial_bL=4b-2=0.
\]
Thus $a=0,b=1/2$, uniquely, since the Hessian is $\operatorname{diag}(2,4)$ and positive definite. The fitted readings are $(1/2,0,1/2)$ versus $(1,0,0)$, giving minimum squared error $1/2$. NumPy least squares confirms these coefficients.
''')
fig('q2_rollout.png','Twenty recursive steps from the specified Gaussian bump, with a shared 0–1 colour scale and time rows 0–20.')
para('The fitted layer averages its two neighbours, producing symmetric diffusion: the bump broadens and its peak falls rather than translating against traffic. The true layer translates one sensor per step while preserving the profile and peak, including circular wraparound.')
add(r'''\subsection{(b) Directional neighbours}
Choose $a=0,w_-=0,w_+=1$. Substitution gives $\hat x_i(t+1)=x_{i+1}(t)$ identically for every snapshot. The cross-correlation kernel multiplying $[x_{i-1},x_i,x_{i+1}]$ is $[0,0,1]$; use circular padding of one sample on either side (a mathematical convolution convention reverses the stored kernel). Direction labels distinguish the ahead and behind neighbours, breaking the reflection ambiguity. Identical symmetric layers and pointwise nonlinearities alone preserve the isolated spike's reflection symmetry at sensors 19 and 21.
\subsection{(c) History with equal spatial weights}
Under backward motion, $x_{i-1}(t)=x_i(t-1)$. Therefore
\[
\hat x_i(t+1)=a x_i(t)+(b+c)x_{i-1}(t)+b x_{i+1}(t).
\]
For equality to $x_{i+1}(t)$ for every snapshot, equate coefficients: $a=0,b=1,c=-1$. The two spatial neighbours sum, and history subtracts the neighbour from which the current reading has already moved.
For forward motion $x_i(t)=x_{i-1}(t-1)$, so $x_i(t-1)=x_{i+1}(t)$ and the same coefficients yield
\[
\hat x_i(t+1)=x_{i-1}(t)+x_{i+1}(t)-x_{i+1}(t)=x_{i-1}(t).
\]
The same coefficients therefore \emph{also work for forward motion}; they do not fail for a consistent forward history. History breaks the snapshot reflection ambiguity by exposing the direction of travel across time; the subtraction adapts to whichever neighbour matches the previous local value. Constant or otherwise indistinguishable patterns can remain direction-ambiguous, but the required forecasts then agree.
''')
table(['Numerical check (seed 42)','Max absolute error'],[['Directional next snapshot',f'{q["directional_error"]:.3e}'],['History, backward motion',f'{q["history_backward_error"]:.3e}'],['History, forward motion',f'{q["history_forward_error"]:.3e}']],'Checks over a random 40-sensor ring; errors are floating-point roundoff.')
add(r'\section{Q3 -- Dataset preparation and graph construction}\subsection{(a) Extraction, units and missingness}'+'\n')
para('The Kaggle CLI first listed all seven files. Only ca_his_raw_2020.h5, ca_his_raw_2021.h5, ca_meta.csv and ca_rn_adj.npy were downloaded. The official San Diego notebook filters District == 11; here all 716 selected entries also have County == San Diego. Metadata ID strings select HDF columns, while the same rows’ ID2 values select both axes of the full graph. Assertions verify unique IDs, 716 sensors, column order and the ID2-to-ID mapping. The graph order convention comes from the official notebook; no raw graph ID labels are supplied independently.')
add(r'''PeMS Total Flow is the vehicle count across lanes during a five-minute interval, normalized upstream for good controller samples \cite{pems}. We therefore sum three counts, yielding vehicles per 15 minutes. The LargeST repository's CA preprocessing instead averages three values and rounds, retaining a five-minute-equivalent average volume \cite{repo}. Our sum preserves interval volume and follows the assignment's explicit interval-count instruction; metric magnitudes should not be compared directly with publications using the repository's averaging convention.

''')
para(f'There are {d["missing_total"]:,} nonfinite/missing raw SD readings out of the two-year complete grid ({d["missing_percent"]:.6f}%). Zeros are retained as valid readings, not reclassified as missing. Each year is reindexed to a complete five-minute grid, separately split at the prescribed boundaries, then causally forward-filled inside that split. Leading gaps in training use zero; leading gaps in validation/test use each sensor’s training-only mean after training imputation. No validation/test reading contributes to those fallback means or normalization. No backward fill or cross-boundary fill is used. Filling precedes summation so missing subintervals cannot silently shrink a quarter-hour count. Long outages can yield stale artificial plateaus; evaluation includes imputed targets and may understate error on genuinely observed traffic. PeMS may already impute upstream, and these files do not expose observation flags.')
table(['Split','15-min steps','Windows','Raw missing','Missing %'],[(s,v['timestamps'],v['windows'],f'{v["raw_missing"]:,}',fmt(v['missing_percent'])) for s,v in d['splits'].items()],'Chronological split sizes and missing raw 5-minute readings.')
missing=pd.read_csv('results/missing_per_sensor.csv')
missing_dist=[(split,fmt(part.missing_percent.min()),fmt(part.missing_percent.median()),fmt(part.missing_percent.max()),int((part.missing_percent==100).sum())) for split,part in missing.groupby('split',sort=False)]
table(['Split','Min missing %','Median missing %','Max missing %','All-missing sensors'],missing_dist,'Distribution of per-sensor raw missingness, including complete outages.')
para('There are 19 sensors with no finite raw readings throughout training and 43 throughout each of validation and test. For an entirely missing held-out sensor the chosen strategy supplies its training fallback mean at every step, not observed traffic. The primary required metrics include these synthetic constant targets and cannot establish real forecasting accuracy at those sensors. This is a material limitation when interpreting either the absolute errors or the benefit of spatial aggregation.')
para('Per-sensor missing counts and percentages for each split are in results/missing_per_sensor.csv; Appendix A additionally reports each sensor’s combined two-year missing count and percentage. results/raw_quality.csv separately records each year’s missing and zero counts.')
fig('week_flow.png','One complete week of 15-minute interval vehicle counts at sensor indices 0, 357 and 715; selection was fixed before fitting.')
add(r'\subsection{(b) Road graph}'+'\n')
para(f'The graph has {d["nodes"]} nodes, {d["edges_with_diagonal"]:,} nonzero matrix entries including {d["diagonal_nonzeros"]} diagonal entries, and {d["edges_without_diagonal"]:,} nonself directed edges. These are ordered matrix entries, not halved undirected counts. max |A-A^T| = {d["symmetry_max"]:.9f}, so the graph is directed. Nonself row degree has mean {d["mean_degree"]:.3f}, minimum {d["min_degree"]}, and maximum {d["max_degree"]}.')
add(r'''The provided $A_{ij}$ is a thresholded Gaussian strength based on road driving distance, not a learned flow correlation. The LargeST paper describes querying driving distances only for candidates within a 4 km geodesic radius and pruning Gaussian strengths below 0.01 \cite{largest}. We preserve the supplied matrix. Operationally, row $i$ receives features from column $j$ through $(AZ)_i=\sum_j A_{ij}Z_j$. This aggregation convention is distinct from asserting a causal direction of traffic influence.
''')
fig('road_adjacency.png','Provided San Diego adjacency, ordered by the same metadata rows as the data tensor.')
fig('degrees.png','Number of nonzero nonself row neighbours for each of the 716 nodes.')
add(r'\subsection{(c) Neighbourhoods}'+'\n')
nb=d['neighborhood']; para(f'Selected sensor ID {nb["id"]}, tensor index {nb["index"]}, is the median-degree node (degree-based selection uses no traffic data). We exclude self-loops when listing neighbours. Its {len(nb["one_hop_ids"])} one-hop sensor IDs are: '+', '.join(nb['one_hop_ids'])+'.')
para(f'Its {len(nb["within_two_hop_ids"])} distinct nonself sensors reachable through at most two row-neighbour steps (including one-hop nodes) are: '+', '.join(nb['within_two_hop_ids'])+'.')
para('The provided matrix has a nonzero self-loop at the selected sensor. Under the assignment’s literal N(i) = {j : Aij != 0} definition, also include sensor '+nb['id']+' itself in each of the above sets (21 one-hop entries and 39 within-two-hop entries). The lists above isolate the 20 and 38 nonself spatial neighbours, respectively; the selected sensor is plotted separately.')
para('Nearby road sensors can reveal arriving queues, shared commuting demand and upstream or downstream congestion before these appear fully at the target. One graph layer mixes direct neighbours; two stacked layers introduce neighbours of neighbours and multiply path weights. More layers may erase local differences through over-smoothing, propagate irrelevant or missing-data signals, and increase optimization difficulty. Directed reachability follows the supplied rows and is not symmetrized.')
fig('neighborhood.png','Selected sensor and its one-hop and within-two-hop sets on metadata latitude/longitude; grey points show all remaining sensors.')
add(r'\section{Q4 -- Forecasting architecture and protocol}\subsection{(a) Windows and leakage controls}'+'\n')
para(f'Every window has input shape (8,716,1) and target shape (4,716,1). Train uses all of 2020, validation January–March 2021, and test April–December 2021. Windows are constructed independently inside each array, with len(split)-11 samples; the first input starts at the split boundary and predicts steps 8–11. No input or target crosses a split. Training-only global mean is {d["train_mean"]:.9f} and population standard deviation is {d["train_std"]:.9f} vehicles per 15 minutes. Both inputs and targets use these statistics. Graph embeddings learn only through training losses; validation selects the checkpoint, and the untouched common test period is used only for final evaluation and descriptive plots/analysis.')
add(r'\subsection{(b) Architecture and training}'+'\n')
para('The shared temporal encoder applies two Conv1d layers per sensor: channels 1→16→16, kernel size 3, padding 1, ReLU after each. It retains the last temporal position, giving Z of shape (batch,716,16). All inputs are historical, so symmetric convolution padding within the eight-step history does not expose targets. The graph variants compute H = ReLU(A Z Wg), with one bias-free 16×16 graph projection. A shared linear 16→4 head predicts four direct horizons. Every model adds the last normalized local reading to all four outputs as an identical persistence residual. Temporal-only feeds Z directly to the same head; the three STGNNs differ only in adjacency. Negative forecasts are not clipped.')
para(f'All runs use CPU, seed 42 for Python/NumPy/PyTorch, Adam lr 0.001, batch size 64, shuffled full training windows, normalized MAE loss, patience 5 on full validation MAE, and the lowest-validation checkpoint. Hidden dimension 16 is within the assignment’s 16–32 CPU guidance. PyTorch deterministic algorithms are enabled, with no GPU operations or extra DataLoader workers. Each model has a fresh optimizer and a reset seed; shared encoder/head initialization matches. There is one graph layer, no dropout, and no weight decay. A complete one-epoch trial for each model precedes fresh final fits; trials are discarded and do not initialize final fits. The measured trials determine a common maximum of {cfg["max_epochs"]} epochs, capped at 30, reserving time inside a 7,200-second training budget. Per-epoch normalized train/validation MAE and times are recorded in JSON and logs.')
table(['Model','Trial train (s)','Trial incl. val (s)','Final epochs','Best epoch','Parameters'],[(k,fmt(trials[k]['history'][0]['training_epoch_seconds']),fmt(trials[k]['training_seconds']),r[k]['epochs'],r[k]['best_epoch'],r[k]['parameters']) for k in r],'Measured full-epoch timing tests and final model sizes/checkpoint selection.')
para('Training wall time includes epoch training, validation and in-memory best-state snapshots; final disk checkpoint writing, final val/test scoring, data loading and analysis are excluded. Timing-trial costs are reported separately. Different early-stopping epochs are allowed under the same rule; single-seed, time-limited results cannot establish statistical significance.')
add(r'\subsection{(c) Temporal baseline comparison}'+'\n')
best=min(r,key=lambda k:r[k]['test']['mae']); base=r['temporal']['test']['mae']; fixed=r['fixed']['test']['mae']
para(f'Temporal-only test MAE is {base:.3f}; fixed-road STGNN test MAE is {fixed:.3f}, a signed change of {fixed-base:+.3f} vehicles per 15 minutes. The lowest measured test MAE belongs to {best} ({r[best]["test"]["mae"]:.3f}). These results compare spatial aggregation with the same local temporal representation and persistence residual; they do not imply that graphs always help.')
add(r'\section{Q5 -- Fixed, adaptive and hybrid graphs}\subsection{(a) Fixed}'+'\n')
add(r'''Because $A$ is asymmetric we use $\widehat A_{ij}=(A+I)_{ij}/\sum_k(A+I)_{ik}$ rather than symmetric degree normalization. The original diagonal is retained and $I$ is added as specified, increasing self mass. All row sums equal one, including isolated nodes after adding identity. This tensor is a frozen buffer, not a trainable parameter. Nonzero $\widehat A_{ij}$ weights the feature of source $j$ in the update of receiver $i$.
\subsection{(b) Adaptive}
Learn $E_1,E_2\in\mathbb R^{716\times10}$, initialized from $\mathcal N(0,0.1^2)$, with $S=\operatorname{ReLU}(E_1E_2^\top)$. For CPU efficiency, retain the 20 largest scores per row \emph{before} normalization and apply softmax only to those scores:
\[
A^{\rm adp}_{ij}=\begin{cases}\exp(S_{ij})/\sum_{k\in T_i}\exp(S_{ik})&j\in T_i,\\0&\text{otherwise.}\end{cases}
\]
This is a sparsified version of row-softmax(ReLU($E_1E_2^\top$)); no road graph is used to select its support. Backpropagation flows through selected logits, softmax values and embeddings (and positive ReLU inputs), not through discrete top-k indices; excluded entries have zero direct gradient on that step, although their embeddings may change through other entries. There is no self-edge requirement. A sparse COO matrix multiplies node features while preserving value gradients. Nonnegative rows sum to one; the learned graph is generally directed.
\subsection{(c) Hybrid}
Both components are row-normalized. Let $\alpha=\operatorname{sigmoid}(\beta)$ with $\beta=0$ initially; then $A^{\rm hybrid}=\alpha\widehat A+(1-\alpha)A^{\rm adp}$, whose rows also sum to one. The scalar and embeddings receive gradients from training MAE. No extra sparsification follows mixing, so the support is the union of road and adaptive supports.
''')
para(f'The selected hybrid checkpoint has alpha = {r["hybrid"]["alpha"]:.8f}. All models 2–4 share precisely the same temporal, graph-projection and forecast-head layers; graph construction is their only architectural difference. The results in Q6 include their measured error and training cost.')
fig('adaptive_adjacency.png','Learned standalone adaptive adjacency at its best-validation checkpoint; top-20 support per row.')
fig('hybrid_adjacency.png','Hybrid adjacency at its best-validation checkpoint, including both fixed and learned components.')
fig('hybrid_adaptive_component_adjacency.png','Adaptive component inside the trained hybrid model; distinct from the standalone adaptive model.')
add(r'\section{Q6 -- Experimental analysis and interpretation}\subsection{(a) Forecasting comparison}'+'\n')
para('For errors e in original interval-count units, MAE = sum |e| / number of predictions and RMSE = sqrt(sum e² / number of predictions). Overall metrics pool every test window, horizon and sensor, with each predicted entry equally weighted. Per-horizon metrics pool windows and sensors; overall RMSE is not an arithmetic mean of horizon RMSE. Overlapping targets therefore occur at different forecast origins, as intended. Validation metrics use the same definition on its own period.')
names={'temporal':'Temporal-only','fixed':'STGNN-Fixed','adaptive':'STGNN-Adaptive','hybrid':'STGNN-Hybrid'}
table(['Model','Adjacency','Test MAE','Test RMSE','Train time (s)'],[(names[k],'None' if k=='temporal' else k,fmt(v['test']['mae']),fmt(v['test']['rmse']),fmt(v['training_seconds'])) for k,v in r.items()],'Final comparison on the common April–December 2021 test period; errors in vehicles per 15 minutes.')
table(['Model','Val MAE','Val RMSE','15-min MAE/RMSE','30-min MAE/RMSE'],[(names[k],fmt(v['val']['mae']),fmt(v['val']['rmse']),f'{v["test"]["horizon_mae"][0]:.3f} / {v["test"]["horizon_rmse"][0]:.3f}',f'{v["test"]["horizon_mae"][1]:.3f} / {v["test"]["horizon_rmse"][1]:.3f}') for k,v in r.items()],'Validation totals and first two test horizons.')
table(['Model','45-min MAE/RMSE','60-min MAE/RMSE'],[(names[k],f'{v["test"]["horizon_mae"][2]:.3f} / {v["test"]["horizon_rmse"][2]:.3f}',f'{v["test"]["horizon_mae"][3]:.3f} / {v["test"]["horizon_rmse"][3]:.3f}') for k,v in r.items()],'Remaining test horizons, in original units.')
table(['Model','15-min MAE/RMSE','30-min MAE/RMSE'],[(names[k],f'{v["val"]["horizon_mae"][0]:.3f} / {v["val"]["horizon_rmse"][0]:.3f}',f'{v["val"]["horizon_mae"][1]:.3f} / {v["val"]["horizon_rmse"][1]:.3f}') for k,v in r.items()],'Validation horizons 15 and 30 minutes.')
table(['Model','45-min MAE/RMSE','60-min MAE/RMSE'],[(names[k],f'{v["val"]["horizon_mae"][2]:.3f} / {v["val"]["horizon_rmse"][2]:.3f}',f'{v["val"]["horizon_mae"][3]:.3f} / {v["val"]["horizon_rmse"][3]:.3f}') for k,v in r.items()],'Validation horizons 45 and 60 minutes.')
for s in [0,357,715]:
    sensor=pd.read_csv('results/sd_meta.csv').ID.iloc[s]
    fig(f'prediction_{sensor}.png',f'Actual and all four model predictions at sensor {sensor}, first 288 test forecast origins, 15-minute horizon. Each prediction uses its own preceding eight true readings; this is rolling-origin forecasting, not a three-day recursive rollout.')
fig('learning_curves.png','Full final-run training and validation normalized MAE histories; selected checkpoints minimize validation MAE.')
add(r'\subsection{(b) Graph structure analysis}'+'\n')
para('For each node, exclude its diagonal, rank learned weights, take its ten strongest neighbours and compute the fraction also present as nonzero nonself entries in the original road graph. Report the mean of these 716 fractions. Haversine distances use metadata coordinates with Earth radius 6,371 km. Road distances here are geodesic distances of directed nonzero road pairs, not the OSRM distances that generated the weights. The comparison uses 7,160 random ordered nonself pairs sampled with seed 42; they can repeat. Top-10 analysis uses the effective hybrid graph and separately its adaptive component, so physical mixing is not mistaken for discovered overlap.')
table(['Graph','Road overlap %','Top-10 mean km','Road mean km','Random mean km'],[(k.replace('hybrid_adaptive_component','Hybrid adaptive part'),fmt(100*v['mean_top10_road_overlap']),fmt(v['top10_mean_km']),fmt(v['road_mean_km']),fmt(v['random_mean_km'])) for k,v in g.items()],'Learned top-10 overlap and geographic-distance comparison. Road overlap excludes diagonal.','p{4cm}rrrr')
for k in ['adaptive','hybrid','hybrid_adaptive_component']:
    fig(k+'_distances.png',f'{k.capitalize()} top-10 nonself neighbour distances versus road neighbours and random pairs. Boxplots show medians and interquartile ranges; fliers are hidden.')
para(f'1. Strongest versus physical links: standalone adaptive overlap is {100*g["adaptive"]["mean_top10_road_overlap"]:.3f}%; hybrid effective overlap is {100*g["hybrid"]["mean_top10_road_overlap"]:.3f}%; the hybrid adaptive part is {100*g["hybrid_adaptive_component"]["mean_top10_road_overlap"]:.3f}%. The hybrid effective statistic partly reflects its built-in road component, not learned recovery of connectivity.')
para(f'2. Geographic proximity: standalone adaptive top-10 links have mean {g["adaptive"]["top10_mean_km"]:.3f} km and median {g["adaptive"]["top10_median_km"]:.3f} km, compared with road mean {g["adaptive"]["road_mean_km"]:.3f} km and random mean {g["adaptive"]["random_mean_km"]:.3f} km. Thus the observed learned distances must be judged against these distributions rather than assuming that learning recovers local roads.')
nearby=g['adaptive']['top10_mean_km'] < g['adaptive']['random_mean_km']
para('The learned neighbours are '+('closer on average than the sampled random pairs' if nearby else 'not closer on average than the sampled random pairs')+', while the road edges provide the local-distance reference. This is descriptive evidence about this learned graph, not a significance test.')
para('3. Dependencies absent from the physical graph: the following highest-weight positive standalone adaptive links have Aij=0 in the road matrix. They show that the parameterization represents nonlocal links, but do not prove causal influence or independently validate those dependencies. Shared daily demand, imputation patterns and finite-data overfitting can all create them.')
table(['Receiver ID','Source ID','Learned weight','Distance km'],[(x['sensor_i'],x['sensor_j'],f'{x["weight"]:.6f}',fmt(x['km'])) for x in g['adaptive']['examples_absent_road'][:5]],'Strong positive adaptive links absent from the road graph.')
para(f'4. Why learning can help or hurt: fixed/adaptive/hybrid test MAEs are {r["fixed"]["test"]["mae"]:.3f}, {r["adaptive"]["test"]["mae"]:.3f}, {r["hybrid"]["test"]["mae"]:.3f}. Adaptive links can capture synchronized routes outside the road support, but embeddings add parameters, support selection is discrete, static averages miss changing congestion, and short single-seed fits may remain poorly optimized. Road aggregation can also dilute a useful local signal. The common persistence residual reduces the need for graph mixing. Training times in the comparison table quantify computational cost rather than assuming graph learning is free. The outcome supports the measured ranking for this protocol, not a general claim of superiority.')
add(r'\subsection{(c) Critical discussion}'+'\n')
para(f'Lower forecast error does not establish actual road connectivity: optimizing prediction can exploit shared schedules or missing-data artifacts, and the measured standalone top-10 road overlap is only {100*g["adaptive"]["mean_top10_road_overlap"]:.3f}%. Forecasting and physical graph recovery are different targets; no connectivity labels supervise the embeddings.')
para(f'A hybrid may outperform either source by retaining local physical propagation while learning complementary global demand signals, with alpha regulating that tradeoff. Here alpha = {r["hybrid"]["alpha"]:.6f}; its test MAE is {r["hybrid"]["test"]["mae"]:.3f} against fixed {r["fixed"]["test"]["mae"]:.3f} and adaptive {r["adaptive"]["test"]["mae"]:.3f}. The observed comparison, including any lack of improvement, is reported without claiming that mixing guarantees a gain.')
para(f'Here the learned fixed-graph mixing mass is only {100*r["hybrid"]["alpha"]:.6f}%, so the selected hybrid is effectively adaptive. Its MAE exceeds standalone adaptive by {r["hybrid"]["test"]["mae"]-r["adaptive"]["test"]["mae"]:.3f}; this run does not show a gain from combining the two sources.')
para('A static graph averages morning/evening direction changes, weekday versus weekend demand, incidents, closures and evolving queues. A single scalar alpha cannot express such variation, and road-distance edges do not encode travel-time delays. A future time-conditioned graph could address this, but it would require leakage-safe training, stronger regularization and additional compute. The 2020-to-2021 split also faces pandemic-era demand shift; imputed targets and a single seed limit external validity.')
add(r'\section{Q7 -- Reproducibility, hardware and limitations}'+'\n')
para(f'Hardware: Intel Xeon Gold 5218R at 2.10 GHz; 80 logical CPUs reported, 40 threads configured (at most half); approximately 376 GiB RAM from free -h. OS and exact CPU details are retained verbatim in results/hardware.txt. PyTorch {cfg["torch"]}, Python {cfg["python"]}; CPU only, OMP_NUM_THREADS=40 and torch.set_num_threads(40), interop threads 1. All training is launched under nice -n 10; long setup/download/preprocessing/training/analysis jobs use nohup and per-job logs. No system packages, services or other applications are modified.')
para(f'Measured total timing-trial cost is {cfg["timing_trial_seconds"]:.3f} seconds; final model training costs sum to {sum(v["training_seconds"] for v in r.values()):.3f} seconds. Maximum epochs {cfg["max_epochs"]}, hidden size 16 and top-k 20 were chosen for CPU feasibility. Training time includes validation, with the same timing boundaries for all variants. This is one seed with no uncertainty intervals, and early stopping can produce unequal amounts of optimization.')
para('Environment deviation: the system Python 3.10 supports Kaggle 1.7.4.5, which fails with the new access-token format. A local Python 3.12 and .kaggle-venv run the latest Kaggle 2.2.5 (plus its missing six dependency), while .venv runs training. Both environments and their caches live in this folder; no credential contents were printed or copied. The report uses local Tectonic because no system TeX engine was available. No unrun metrics are included.')
para('An initial Git status query inherited an ancestor repository and listed filenames outside the working folder. A dedicated local repository was initialized immediately; subsequent Git operations are scoped to this folder. No other application or service was changed.')
add(r'''\subsection{Exact reproduction sequence}
Run from the repository root. Install requirements in a local virtual environment; install CPU PyTorch from its CPU wheel index. If Python is below 3.11, reproduce the separate Kaggle environment using \texttt{scripts/setup.sh}. A configured Kaggle access token is required, but must never be committed. The commands below intentionally use nohup in the foreground: they survive a hangup and can also be launched in a persistent terminal.
\begin{Verbatim}[fontsize=\small,breaklines=true]
bash scripts/setup.sh
nohup bash scripts/download.sh > logs/download.log 2>&1 < /dev/null
OMP_NUM_THREADS=40 nohup .venv/bin/python -u scripts/preprocess.py > logs/preprocess.log 2>&1 < /dev/null
OMP_NUM_THREADS=40 .venv/bin/python q1_q2/verify.py
OMP_NUM_THREADS=40 .venv/bin/python -m scripts.check_models
OMP_NUM_THREADS=40 MKL_NUM_THREADS=40 OPENBLAS_NUM_THREADS=40 nohup nice -n 10 .venv/bin/python -u -m src.train > logs/training.log 2>&1 < /dev/null
OMP_NUM_THREADS=40 nohup .venv/bin/python -u -m scripts.analyze > logs/analysis.log 2>&1 < /dev/null
.venv/bin/python scripts/build_report.py
XDG_CACHE_HOME="$PWD/.cache" .tools/tectonic report/PA3_Zain_Gulbaz_25280005.tex
.venv/bin/python -m scripts.audit
\end{Verbatim}
''')
para('src/train.py runs timing trials and all four experiments sequentially with the common protocol; results/config.json records the selected maximum and measured metadata. Checkpoints contain the best validation states. Processed SD arrays include IDs, graph indices, split timestamps and training statistics. results/results.json and results/results_table.csv hold the final metrics. requirements.txt and requirements-kaggle.txt pin the actually installed packages. Raw and processed data, environments, checkpoints and logs are excluded from Git; the repository includes scripts, figures, summary CSV/JSON files and this report. The repository is initialized and committed locally; no remote is created and nothing is pushed.')
add(r'''\begin{thebibliography}{9}
\bibitem{largest} X. Liu et al. LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting. NeurIPS Datasets and Benchmarks, 2023. \url{https://arxiv.org/abs/2306.08259}.
\bibitem{repo} Official LargeST repository, San Diego extraction and CA processing notebooks. \url{https://github.com/liuxu77/LargeST}. Accessed 24 November 2026; downloaded notebook snapshots are in scripts/reference.
\bibitem{pems} Caltrans PeMS. Traffic data downloads, five-minute field specifications, Total Flow (vehicles / 5 min). \url{https://pems.dot.ca.gov/?dnode=pass}. Accessed 24 November 2026.
\bibitem{kaggle} LargeST dataset: \url{https://www.kaggle.com/datasets/liuxu77/largest}. CC BY-NC 4.0. Only required files downloaded; raw data are not redistributed in Git.
\end{thebibliography}
\clearpage\appendix\section{Per-sensor missingness}
Missingness below refers to the combined complete five-minute grids of 2020 and 2021, before any filling; the denominator is 210,528 readings per sensor. The machine-readable split-specific counts and percentages are in results/missing\_per\_sensor.csv. Zeros are not missing. Each row below packs three independent sensors to keep the appendix compact.
\small
\begin{longtable}{rrr|rrr|rrr}
\caption{Every sensor's raw missing count and percentage, two years combined.}\\
\toprule ID & Count & \% & ID & Count & \% & ID & Count & \%\\\midrule
\endfirsthead
\toprule ID & Count & \% & ID & Count & \% & ID & Count & \%\\\midrule\endhead
''')
quality=pd.read_csv('results/raw_quality.csv'); den=210528
for i in range(0,len(quality),3):
    vals=[]
    for j in range(3):
        if i+j<len(quality):
            row=quality.iloc[i+j]; count=int(row.missing_2020+row.missing_2021); vals.extend([str(int(row.sensor_id)),str(count),f'{100*count/den:.3f}'])
        else: vals.extend(['','',''])
    add(' & '.join(vals)+r'\\'+'\n')
add(r'''\bottomrule\end{longtable}\normalsize
\clearpage\section*{AI Usage Disclosure}\addcontentsline{toc}{section}{AI Usage Disclosure}
OpenAI Codex was used as the coding agent. Claude (Anthropic) helped write the task prompt supplied by the student. Codex produced the Python and shell code, data extraction and leakage controls, exhaustive NumPy checks, model implementations, executed training and evaluation, figures, numerical analysis, this report text and reproduction documentation. The measured outputs are preserved in results artifacts; the code and report are AI-produced and require the student's review and understanding. This disclosure does not claim independent student authorship of the generated work.

\subsection*{Edits made by the student}
\textbf{PLACEHOLDER -- to be completed by the student:} Describe the parts you reviewed, corrected, rewrote or independently verified after receiving the AI output, and explain your own contributions. Do not submit this paragraph unchanged.

\subsection*{Exact supplied prompt}
The following is TASK.md verbatim, including its original wording. The student's accompanying instruction was: ``Read TASK.md and complete every step in it fully and autonomously. Do not stop until DONE.md is written''.
\VerbatimInput[fontsize=\scriptsize,breaklines=true,breakanywhere=true]{../TASK.md}
\end{document}
''')
pathlib.Path('report/PA3_Zain_Gulbaz_25280005.tex').write_text(''.join(out))
print('REPORT SOURCE COMPLETE')
