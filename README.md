# FlowLOT comparable cytometry classifiers

This repository provides a shared, leakage-safe benchmark for variable-size
single-cell cytometry samples. It includes FlowSOM, CellCNN, gated attention MIL,
CytoSet, DGCNN, and PointNet++ baselines, plus a logistic classifier for existing
precomputed LOT representations.

## Data format

Create a CSV manifest whose paths are relative to the manifest (or absolute):

```csv
sample_id,path,label,split,lot_path
subject_001,cells/001.npy,0,train,lot/001.npy
subject_002,cells/002.npz,1,validation,lot/002.npy
subject_003,cells/003.npy,1,test,lot/003.npy
```

`path` must hold a two-dimensional `[cells, markers]` matrix. NumPy `.npy` and
`.npz`, Torch `.pt`/`.pth`, and numeric CSV/TXT files are accepted. An NPZ uses
the `cells` key (or its only array). Labels must be contiguous integers beginning
at zero. The split names are `train`, `validation`, and `test`. The optional
`lot_path` is any precomputed LOT tensor; it is flattened for classification.

Cell values should already have the biological transform appropriate to the
panel (for example, an arcsinh transform). Marker standardization is estimated
strictly from training cells and saved in each neural-network checkpoint.

## Usage

Install the Python 3.12 environment and train one model:

```bash
python -m pip install -r requirements.txt
python train.py --manifest cohort.csv --model attention_mil --output runs/mil.pt
```

Run the full baseline benchmark:

```bash
python evaluate.py --manifest cohort.csv \
  --models lot,flowsom,cellcnn,attention_mil,cytoset,dgcnn,pointnet2 \
  --output-dir benchmark_results
```

The evaluator writes `metrics.json`, `metrics.csv`, and deep-model checkpoints.
Every method reports subject-level accuracy, balanced accuracy, macro F1,
one-vs-rest macro ROC-AUC, and macro PR-AUC. For large cohorts, tune
`--max-cells`; training draws a fresh uniform subset while validation/test use a
deterministic subset. DGCNN computes k-NN distances in chunks, but graph methods
remain quadratic in the number of sampled cells.

## Python API

All deep models implement the same call:

```python
logits = model(cells, mask)  # [batch, max_cells, markers], [batch, max_cells]
```

Attention weights for cell-level interpretation are available through
`AttentionMIL(...)(cells, mask, return_attention=True)`.
