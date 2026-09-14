# Shared-split repeated classification benchmarks

This workflow reproduces the repeated-run benchmark experiment across multi-scale cell tiers while making its
sampling strictly auditable and reproducible. For every repeat, one stratified test cohort is fixed. The
balanced training cohorts are nested hierarchically, so the two-patient cohort is a subset of
the four-patient cohort, and so on through 6, 8, 12, and 16 patients per class. Every LOT
classifier, multi-tube fusion method, FlowSOM baseline, and deep model reads these exact patient IDs.

## Cohorts and few-shot regimes

FlowLOT evaluates two clinical cytometry cohorts:
- **`BLAST110` (Diagnostic AML vs. Normal Bone Marrow):** 110 subjects (55 AML, 55 NBM controls) across 4 clinical tubes (P1, P2, P3, P4). Few-shot training sizes: $k \in [2, 4, 6, 8]$ patients per class.
- **`FLOWCAPII` (Clinical Challenge Benchmark):** 359 subjects (43 AML, 316 non-leukemic controls) across 7 tubes. Severe class imbalance ($1:7.3$). Few-shot training sizes: $k \in [4, 8, 12, 16]$ patients per class.

The `--classes` parameter enables filtering specific subsets of classes (e.g. creating binary AML vs. normal benchmarks from multiclass or raw clinical diagnosis labels).

## 11 benchmarked classifier families

The benchmark systematically evaluates 11 classifiers spanning three distinct computational paradigms:

1. **FlowLOT Tangent Space Classifiers:**
   - **Nearest Subspace Classifiers:**
     - `nsc`: Mean-centered class subspaces scored by reconstruction distance.
     - `nsc_energy`: Class subspaces truncated to retain 95% variance energy.
     - `nsc_full`: Uncentered affine subspaces retaining full signal span.
   - **Linear Models:** Logistic Regression (`logistic`) with $L_2$ regularization, Linear SVM (`linear_svm`).
   - **Ensemble Classifiers:** Random Forest (`random_forest`), Extremely Randomized Trees (`extra_trees`), and XGBoost (`xgboost`, via `pip install -e '.[xgboost]'`).
2. **Deep Point-Cloud and Set Baselines:**
   - Dynamic Graph CNN (`dgcnn`): Local neighborhood graph convolutions.
   - PointNet++ (`pointnet2`): Hierarchical point-set feature learning.
   - CytoSet (`cytoset`): Deep set permutation-invariant cell architecture.
   - Attention-MIL (`attention_mil`): Gated attention multiple instance learning.
   - CellCNN (`cellcnn`): Multi-filter cell subset convolution baseline.
3. **Clustering / Gating Baselines:**
   - FlowSOM (`flowsom`): Unsupervised self-organizing map clustering followed by metaclustering percentage representation.

> [!NOTE]
> The earlier deep baselines remain available through `flowlot.models.baselines` and the top-level `evaluate.py`.

## Multi-scale cell tiers and the $N=1000$ Pareto frontier

Cytometry measurements are evaluated across three nested cell-count subsampling tiers:
$$N \in \{500, \, 1000, \, 5000\} \text{ cells per patient tube}.$$

Key empirical dynamics:
- **Variance reduction ($N=500 \to 1000$):** Subsampling $N=1000$ cells yields $+1.5\%\text{--}2.0\%$ higher balanced accuracy over $N=500$ (surpassing $96\text{--}98\%$ on `BLAST110`) and dramatically narrows fold-to-fold error standard deviations.
- **Asymptotic saturation ($N=1000 \to 5000$):** Expanding to 5,000 cells yields negligible accuracy gain ($<+0.2\%$ at peak $k$) because $N=1000$ already densely captures the low-dimensional phenotypic manifold.
- **Optimal Transport complexity penalty:** While accuracy saturates, solving the exact network simplex or Sinkhorn OT scales with super-quadratic complexity $\mathcal{O}(N^3 \log N)$. Wall-clock solve times increase from $0.15\text{ s}$ ($N=500$) to $0.45\text{ s}$ ($N=1000$) to $2.45\text{ s}$ ($N=5000$) per sample—a $>16\times$ computational penalty. This establishes **$N=1000$ cells as the optimal clinical Pareto operating threshold**.

## Classification metrics and split policies

Every job reports:
- Accuracy, Balanced Accuracy, Macro F1, ROC-AUC, and PR-AUC.
- Balanced Accuracy and Macro F1 are the primary metrics for model ranking because class imbalance makes raw Accuracy optimistic.
- Binary ROC-AUC/PR-AUC evaluate positive-class posteriors; multiclass evaluations compute macro one-vs-rest summaries.

The default `intersection` patient policy is required for strict comparisons across all tubes and models: it retains only patients with complete panel observations. `union` is supported for missing-tube evaluation.

## Files and provenance

`flowlot-repeat splits` produces two complementary split files:
- `shared_splits.json`: The authoritative immutable registry. Its SHA-256 checksum covers patient IDs, labels, test cohorts, nested training cohorts, seed, tubes, classes, and dataset coordinates.
- `legacy_splits.h5`: Preserves the historical `run_N/subsamples/num_sub_per_cls_K` layout for backwards compatibility, with the registry checksum embedded as an HDF5 root attribute.

`flowlot-repeat jobs` writes `jobs.tsv` plus `jobs_split_audit.csv`. Each result shard records the registry checksum and train/test ID hashes. Aggregation rejects mixed, missing, duplicate, or unexpected jobs unless `--allow-incomplete` is passed.

## HPC Slurm workflow

```bash
cp scripts/repeated_benchmark.env.example scripts/repeated_benchmark.env
# Edit dataset paths, cell count (500, 1000, 5000), preprocessing/embedding IDs, and scheduler flags.

bash scripts/hpc_repeated_benchmark.sh init
bash scripts/hpc_repeated_benchmark.sh submit
bash scripts/hpc_repeated_benchmark.sh status
bash scripts/hpc_repeated_benchmark.sh aggregate
```

Supported script actions:
- `init`: Generates `shared_splits.json` and `jobs.tsv`.
- `submit`: Launches `jobs.tsv` as a resumable Slurm array.
- `worker`: Executes a single index (`FLOWLOT_JOB_INDEX` or `SLURM_ARRAY_TASK_ID`).
- `local`: Executes all jobs locally using `FLOWLOT_LOCAL_JOBS` concurrent workers.
- `status`: Displays completed shards vs. total planned jobs and queries active Slurm queue tasks.
- `aggregate`: Validates completeness (skipping macOS `._*.json` metadata files) and exports aggregate CSV, LaTeX tables, and figures.

For heterogeneous clusters, CPU models (`logistic`, `linear_svm`, `random_forest`, `extra_trees`, `nsc`, `flowsom`) and GPU models (`dgcnn`, `pointnet2`, `cytoset`, `attention_mil`, `cellcnn`) can be separated into distinct `.env` files while sharing the exact same `shared_splits.json`.

## Bootstrap confidence intervals

Aggregation computes deterministic, stratified patient-level percentile confidence intervals. Predictions from repeated test appearances are first averaged for each patient, so a frequently sampled subject remains one bootstrap unit. Patients are sampled with replacement within each class, preserving balance. Configurable via `FLOWLOT_BOOTSTRAP_ITERATIONS` (default 1,000 or 2,000), `FLOWLOT_CONFIDENCE_LEVEL` (0.95), and `FLOWLOT_BOOTSTRAP_SEED`.

## Outputs and publication plotting

The aggregate directory contains:
- `per_run.csv`: Per-classifier, per-fusion, per-repeat, per-$k$ metrics.
- `summary.csv`: Empirical mean, standard deviation, and job counts.
- `bootstrap_ci.csv`: Patient-averaged estimates with bootstrap standard errors and 95% confidence intervals.
- `paired_comparisons.csv`: Paired method deltas, confidence limits, and win fractions across all models.
- `comparison_k*.tex` and `bootstrap_comparison_k*.tex`: Publication booktabs LaTeX tables.
- `integrity.json`: Manifest of expected, completed, missing, and unexpected jobs.

### Dedicated Visualization Scripts
- **`scripts/plot_repeated_benchmark_clean.py`:** Generates publication-grade 4-panel figures contrasting top FlowLOT classical vs. deep architectures across tubes and cell counts.
- **`scripts/plot_tube_aggregation.py`:** Generates multi-panel fusion comparisons (`early_mean` feature fusion, `late_soft` probability consensus, and single-tube baselines).

### Interactive Notebooks
- **[`notebooks/NB01_Sampling_from_Raw_measurements.ipynb`](../notebooks/NB01_Sampling_from_Raw_measurements.ipynb) through [`notebooks/NB03_lot_embeddings.ipynb`](../notebooks/NB03_lot_embeddings.ipynb):** Upstream raw event discovery, Stage 1/Stage 2 construction, and LOT representation generation.
- **[`notebooks/NB04_p1_Classification_split_n_NSC.ipynb`](../notebooks/NB04_p1_Classification_split_n_NSC.ipynb):** Split verification, Nearest Subspace Classifiers, confusion matrices, and multi-tube fusion.
- **[`notebooks/NB04_p2_aggregate_all_classification_result.ipynb`](../notebooks/NB04_p2_aggregate_all_classification_result.ipynb):** Cross-cell-count scaling analysis, complexity vs. runtime tradeoffs, and final leaderboards.
- **[`analyze_results.ipynb`](../analyze_results.ipynb):** Interactive inspection of benchmark shard directories and custom performance curves.

