# End-to-end tutorial

## 1. Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[fcs,notebook]'
```

## 2. Build Stage 1 from a raw-data folder

The guided route is [`notebooks/NB01_Sampling_from_Raw_measurements.ipynb`](../notebooks/NB01_Sampling_from_Raw_measurements.ipynb).
Its configuration cells discover raw files, clinical label tables,
filename patterns, output manifests, marker mappings, and requested cell counts.
It previews discovery across cohorts (`BLAST110`, `LAIP29`, `FLOWCAPII`) and parses
real files before enabling any write.

FlowLOT accepts FCS, CSV/TSV/TXT, NPY, and NPZ inputs. FCS events are read with
FlowIO and marker names use `PnS` then `PnN`; clinical labels are joined from
a dedicated table rather than guessed from FCS metadata. Text tables use rows as
cells and columns as markers. NPY matrices accept an explicit marker mapping; NPZ
files may include a `markers` array. The Stage 1 `counts` attribute always records
the original event count before subsampling.

A typical input layout is:

```text
raw/MyCohort/FCS/P001_T1.fcs
raw/MyCohort/FCS/P001_T2.fcs
raw/MyCohort/labels/P001_T1.csv  # event_ID,WBC,Blast[,LAIP]
metadata/mycohort_labels.csv  # columns: patient_id,label
```

Generate an explicit manifest from that folder:

```python
from flowlot.io import create_manifest_from_folder

create_manifest_from_folder(
    raw_root="raw/MyCohort/FCS",
    output="manifests/mycohort.csv",
    filename_pattern=r"(?P<patient_id>[^/]+)_(?P<tube_id>T[0-9]+)\.(?:fcs|csv|npy)$",
    labels="metadata/mycohort_labels.csv",
    markers_by_tube={"T1": ["FSC-A", "SSC-A", "CD45"]},
    event_labels_root="raw/MyCohort/labels",
    event_id_column="event_ID",
    population_columns=["WBC", "Blast", "LAIP"],
)
```

The regular expression is matched against paths relative to `raw_root` and must expose
named `patient_id` and `tube_id` groups. The generated manifest has one row per
patient/tube with `patient_id,tube_id,path,label,markers`. Conflicting labels,
duplicate patient/tube files, missing labels, and unmatched supported files are
rejected by default.

BLAST110 and LAIP29 use two label layers. `sample_info.csv` provides the
sample-level class, while `labels/<raw-file-stem>.csv` provides `event_ID` and
binary/numeric WBC, Blast, and LAIP indicators. For the original BLAST110 table,
set `labels_id_column="BLAST110_ID"`, `labels_label_column="sample_type"`, and
`labels_match="file_id"` (use `LAIP29_ID` for LAIP29). FlowLOT joins the event
CSV to the raw event-ID channel, then records sampled and original counts plus
`100 * population_count / WBC_count`. The aligned per-event indicators are also
stored, so every reported target remains auditable.

To build several levels, call the builder for each count with the same seed and
append mode. Sampling uses a seed-specific permutation, so the levels are nested
hierarchically (for example, 500 cells $\subset$ 1,000 cells $\subset$ 5,000 cells):

```bash
flowlot-build stage1 --manifest manifests/mycohort.csv --output data/stage1/stage1_raw_data.h5 \
  --dataset MyCohort --cells 1000 --seed 42
```

Python API equivalent:

```python
from flowlot.io import build_stage1_from_manifest

build_stage1_from_manifest("manifests/mycohort.csv", "data/stage1/stage1_raw_data.h5", "MyCohort", 1000, seed=42)
```

## 3. Build and preprocess Stage 2

The guided organization workflow is [`notebooks/NB02_Organizing_sampled_data_and_preprocessing.ipynb`](../notebooks/NB02_Organizing_sampled_data_and_preprocessing.ipynb).
Existing HDF5 files produced by earlier FlowCode runs can also be migrated:

```bash
flowlot-build legacy --input BLAST110.h5 --output data/stage1/stage1_raw_data.h5 \
  --dataset BLAST110 --sample-level sample_1000
```

Reorganize Stage 1 into the tube-centric Stage 2 analytical file (`data/stage2/stage2_analytics.h5`),
configure marker alignment policies (`strict` vs. `intersection`), and apply standardized preprocessing:

```bash
flowlot-build stage2 --stage1 data/stage1/stage1_raw_data.h5 --output data/stage2/stage2_analytics.h5 \
  --dataset MyCohort --cells 1000 --marker-policy intersection
flowlot-build preprocess --stage2 data/stage2/stage2_analytics.h5 --dataset MyCohort \
  --cells 1000 --id common12 --markers CD45,CD34,CD117,HLA-DR,CD33,CD13,CD14,CD15,CD11b,CD64,CD7,CD19 --arcsinh-cofactor 500
```

`Stage2Organizer.add_preprocess` stabilizes heteroskedastic cytometric variance via an arcsinh transformation ($\theta=500$) and performs training-cohort z-score standardization to prevent out-of-sample distributional leakage.

## 4. Compute LOT representations

The optimal transport and embedding workflow is implemented in [`notebooks/NB03_lot_embeddings.ipynb`](../notebooks/NB03_lot_embeddings.ipynb).
FlowLOT supports multiple reference templates:
- **Patient baseline (`patient0`):** Maximally interpretable, preserves physical cell topology.
- **Pooled subsample (`pooled`):** Subsamples across training patient clouds to form an empirical cohort distribution.
- **Wasserstein barycenter (`barycenter`):** Frechet mean under the $\mathcal{W}_2$ metric minimizing average transport discrepancy.
- **Synthetic distributions (`uniform`, `gaussian`):** Moment-matched controls.

Compute Sinkhorn LOT displacement embeddings using patient-0:

```bash
flowlot-run --stage2 data/stage2/stage2_analytics.h5 --dataset MyCohort --cells 1000 \
  --preprocess common12 --reference patient0 --solver sinkhorn --reg 0.01
```

Compare with a training-cohort Wasserstein barycenter and exact EMD:

```bash
flowlot-run --stage2 data/stage2/stage2_analytics.h5 --dataset MyCohort --cells 1000 \
  --preprocess common12 --reference barycenter --reference-size 256 --solver emd \
  --reference-patients P001,P002,P003,P004,P005
```

Key runtime features:
- **Persistent reference caching:** Frozen references are stored under `references/{ref_store_key}` (e.g. `barycenter_size256_p5`) to guarantee reproducible template reuse across runs.
- **Dual representation (`--representation both`):** Computes Monge/Kantorovich transport plans once and creates both `{base_id}_map` and `{base_id}_disp` embeddings, linking `sorted_cell_matrices` and `transport_matrices` to cut solver time by 50% without duplicating cell arrays.
- **Parallel patient solves (`--n-jobs N`):** Distributes independent patient OT problems across CPU cores via `ProcessPoolExecutor`.
- **Lightweight storage:** Utilizes fast LZF compression for sorted cell matrices and transport couplings, with optional `--no-store-transport` for large cohorts.

## 5. Classification, repeated benchmarks, and multi-tube fusion

The classification, repeated benchmark, and aggregation workflows are structured in:
- **[`notebooks/NB04_p1_Classification_split_n_NSC.ipynb`](../notebooks/NB04_p1_Classification_split_n_NSC.ipynb):** Generates balanced, nested, disjoint patient training/testing splits across sample sizes ($k \in [2, 4, 6, 8]$ for `BLAST110`, $k \in [4, 8, 12, 16]$ for `FLOWCAPII`, with `--classes` support for binary extraction). Evaluates Nearest Subspace Classifiers:
  - Standard NSC (`nsc`): Mean-centered PCA subspaces.
  - Energy-thresholded NSC (`nsc_energy`): Subspaces capturing 95% variance.
  - Uncentered full-rank NSC (`nsc_full`): Uncentered affine subspaces retaining full signal span.
- **[`notebooks/NB04_p2_aggregate_all_classification_result.ipynb`](../notebooks/NB04_p2_aggregate_all_classification_result.ipynb):** Aggregates repeated benchmark shards, evaluates scaling frontiers across cell subsampling tiers (500 vs. 1,000 vs. 5,000 cells), and constructs comparative leaderboards.

Evaluate FlowLOT classifiers with early feature fusion or late probability fusion:

```bash
flowlot-eval --stage2 data/stage2/stage2_analytics.h5 --dataset MyCohort --cells 1000 \
  --preprocess common12 --embedding patient0_sinkhorn_disp --task classification \
  --fusion early --folds 5 --output results/classification

flowlot-eval --stage2 data/stage2/stage2_analytics.h5 --dataset MyCohort --cells 1000 \
  --preprocess common12 --embedding barycenter_emd_disp --task regression \
  --fusion late --folds 5 --output results/regression
```

### 11 Evaluated Classifier Families
FlowLOT benchmarks 11 diverse classifiers across multi-scale cell tiers:
1. **FlowLOT Tangent Space Classifiers:** Nearest Subspace Classifiers (`nsc`, `nsc_energy`, `nsc_full`), Logistic Regression (`logistic`), Linear SVM (`linear_svm`), Random Forest (`random_forest`), Extra Trees (`extra_trees`), and XGBoost (`xgboost`).
2. **Deep Point-Cloud and Set Baselines:** DGCNN (`dgcnn`), PointNet++ (`pointnet2`), CytoSet (`cytoset`), Attention-MIL (`attention_mil`), and CellCNN (`cellcnn`).
3. **Clustering / Gating Baselines:** FlowSOM (`flowsom`).

> [!NOTE]
> The earlier deep baselines remain available through `flowlot.models.baselines` and the top-level `evaluate.py`. Their input is a padded `[batch, cells, markers]` tensor plus boolean mask.

### Multi-Tube Fusion Strategies
- **Single tube (`single`):** Trains and evaluates estimators per individual tube.
- **Early concatenation (`early_mean`, `early_zero`):** Concatenates tangent displacement vectors across available tubes with dimension-normalized weights $w_t = 1/D_t$ and imputes missing tubes using training-cohort means or zeros plus availability indicators.
- **Late consensus (`late_soft`, `late_hard`):** Trains independent classifiers per tube and computes soft class posterior averages or hard majority votes.
- **Meta-model stacking (`stacking`):** Available via `LateTubeFusion(method="stacking")`, training a meta-classifier on out-of-fold tube predictions.

## 6. Visualization, latent manifolds, and cMRD estimation

- **[`notebooks/NB05_Visualization.ipynb`](../notebooks/NB05_Visualization.ipynb):**
  Uses `flowlot.App.Visualization` (`visualization_utils`) to analyze tangent space manifolds via PCA and UMAP. Derives discriminant trajectory vectors between control and AML populations:
  $$\hat{v} = \frac{\mu_{\text{AML}} - \mu_{\text{NBM}}}{\|\mu_{\text{AML}} - \mu_{\text{NBM}}\|}$$
  and renders publication-grade directional marginal density evolution plots along disease trajectories.
- **[`notebooks/NB06_Estimation_cMRD.ipynb`](../notebooks/NB06_Estimation_cMRD.ipynb):**
  Performs continuous blast and LAIP/WBC percentage quantification using LOT embeddings paired with Partial Least Squares (PLS) regression. Evaluates particle count sensitivity across subsampled tiers (500, 1,000, 5,000 cells), computes regression diagnostics ($R^2$, RMSE, MAE), and assesses within-cohort vs. cross-cohort calibration transfer.
- **[`analyze_results.ipynb`](../analyze_results.ipynb):**
  Interactive notebook for inspecting repeated benchmark result directories, shard summaries, and plotting performance curves.

Clinical reporting suite utilities:

```python
from flowlot.evaluation.reporting import (
    plot_clinical_threshold_suite,
    plot_transport_geometry,
)

plot_clinical_threshold_suite(
    observed_laip_percent,
    predicted_laip_percent,
    "results/laip_thresholds",
    thresholds=(0.1, 1.0, 5.0),
    groups=["Dx", "FU"],
)
plot_transport_geometry(reference, sample, transported, "results/transport")
```

## 7. Inspect data programmatically

```python
from flowlot.io import Stage2Loader

with Stage2Loader("data/stage2/stage2_analytics.h5") as data:
    print(data.tubes("MyCohort", 1000))
    ids, z = data.embeddings("MyCohort", 1000, "P1", "common12", "patient0_sinkhorn_disp")
```

## 8. Build PDF documentation

Install Pandoc plus a LaTeX engine, then run:

```bash
python docs/build_pdf.py --output docs/FlowLOT_manual.pdf
```
