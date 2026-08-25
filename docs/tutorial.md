# End-to-end tutorial

## 1. Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[fcs,notebook]'
```

## 2. Build Stage 1 from custom tables

Prepare one manifest row per patient/tube. CSV files may contain a header; NPY
files specify markers in a semicolon-separated `markers` column. FCS channel
names are inferred when markers are omitted.

```bash
flowlot-build stage1 --manifest manifest.csv --output stage1_raw_data.h5 \
  --dataset MyCohort --cells 2048 --seed 42
```

Python API equivalent:

```python
from flowlot.io import build_stage1_from_manifest
build_stage1_from_manifest("manifest.csv", "stage1_raw_data.h5", "MyCohort", 2048, seed=42)
```

## 3. Build and preprocess Stage 2

Existing HDF5 files produced by the archived FlowCode notebooks can first be
migrated without re-reading FCS files:

```bash
flowlot-build legacy --input BLAST110.h5 --output stage1_raw_data.h5 \
  --dataset BLAST110 --sample-level sample_1000
```

Legacy LOT placeholders/embeddings are intentionally recomputed because the new
file records solver, reference, representation, cost, and convergence provenance.

```bash
flowlot-build stage2 --stage1 stage1_raw_data.h5 --output stage2_analytics.h5 \
  --dataset MyCohort --cells 2048 --marker-policy intersection
flowlot-build preprocess --stage2 stage2_analytics.h5 --dataset MyCohort \
  --cells 2048 --id immune --markers CD45,CD34,CD117,HLA-DR --arcsinh-cofactor 5
```

`Stage2Organizer.add_preprocess` also accepts a mapping of tube IDs to distinct
marker subsets.

## 4. Compute LOT representations

Patient-0 plus Sinkhorn is fast and reproduces the old reference philosophy:

```bash
flowlot-run --stage2 stage2_analytics.h5 --dataset MyCohort --cells 2048 \
  --preprocess immune --reference patient0 --solver sinkhorn --reg 0.01
```

Compare it with a training-cohort barycenter and exact EMD:

```bash
flowlot-run --stage2 stage2_analytics.h5 --dataset MyCohort --cells 2048 \
  --preprocess immune --reference barycenter --reference-size 256 --solver emd \
  --reference-patients P001,P002,P003,P004,P005
```

For equal-sized clouds, `--solver hungarian` provides the legacy one-to-one
matching. `linprog` is pedagogically useful for small clouds but becomes
impractical for thousands of cells. Use `--no-store-transport` when coupling
matrices would make the HDF5 file too large.

For confirmatory cross-validation, generate each fold's reference using only its
training patient IDs (`--reference-patients`) and store each configuration under
a distinct Stage 2 file or preprocess/reference identifier.

## 5. Classification and regression

```bash
flowlot-eval --stage2 stage2_analytics.h5 --dataset MyCohort --cells 2048 \
  --preprocess immune --embedding patient0_sinkhorn --task classification \
  --fusion early --folds 5 --output results/classification

flowlot-eval --stage2 stage2_analytics.h5 --dataset MyCohort --cells 2048 \
  --preprocess immune --embedding barycenter_emd --task regression \
  --fusion late --folds 5 --output results/regression
```

Early fusion supports training-mean or zero imputation plus tube indicators. Late
classification performs soft voting over available tubes; late regression uses
the available-tube mean. `LateTubeFusion(method="stacking")` is available through
the Python API and trains the meta-model from out-of-fold predictions.

Deep baselines are importable from `flowlot.models.baselines`. Their input is a
padded `[batch,cells,markers]` tensor plus boolean mask. The original benchmark
driver remains `python evaluate.py --manifest ...`.

## 6. Reports and quantification figures

The CLI writes per-fold CSV, aggregate CSV, a booktabs LaTeX table, and diagnostic
figures in PDF/SVG/PNG. Additional figures:

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
    groups=["Dx", "FU", ...],
)
plot_transport_geometry(reference, sample, transported, "results/transport")
```

## 7. Inspect data programmatically

```python
from flowlot.io import Stage2Loader

with Stage2Loader("stage2_analytics.h5") as data:
    print(data.tubes("MyCohort", 2048))
    ids, z = data.embeddings("MyCohort", 2048, "tube1", "immune", "patient0_sinkhorn")
```

## 8. Build PDF documentation

Install Pandoc plus a LaTeX engine, then run:

```bash
python docs/build_pdf.py --output docs/FlowLOT_manual.pdf
```
