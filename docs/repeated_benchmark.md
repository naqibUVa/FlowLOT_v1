# Shared-split repeated classification benchmarks

This workflow reproduces the legacy repeated-run experiment while making its
sampling auditable. For every repeat, one stratified test cohort is fixed. The
balanced training cohorts are nested, so the two-patient cohort is a subset of
the four-patient cohort, and so on through 6 and 8 patients per class. Every LOT
classifier, fusion method, FlowSOM run, and deep model reads these exact IDs.

The default `intersection` patient policy is required for strict comparisons
across all tubes and models. It removes patients missing any selected tube before
splitting. Missing-tube experiments can use `union`, but single-tube and cell
models should then be placed in separate compatible job tables.

## Files and provenance

`flowlot-repeat splits` produces two complementary split files:

- `shared_splits.json` is the authoritative immutable registry. Its SHA-256
  checksum covers patient IDs, labels, test cohorts, nested training cohorts,
  seed, tubes, and dataset coordinates.
- `legacy_splits.h5` preserves the historical
  `run_N/subsamples/num_sub_per_cls_K` layout for older FlowCode notebooks. It
  includes the same registry checksum as an HDF5 root attribute.

`flowlot-repeat jobs` writes `jobs.tsv` plus `jobs_split_audit.csv`. Each result
shard records the registry checksum and train/test ID hashes. Aggregation refuses
mixed, missing, duplicate, or unexpected jobs unless incomplete aggregation is
explicitly requested.

## Slurm workflow

```bash
cp scripts/repeated_benchmark.env.example scripts/repeated_benchmark.env
# Edit dataset paths, preprocessing/embedding IDs, modules, and scheduler flags.

bash scripts/hpc_repeated_benchmark.sh init
bash scripts/hpc_repeated_benchmark.sh submit
bash scripts/hpc_repeated_benchmark.sh aggregate
```

The job array is resumable: a completed atomic JSON shard is reused. Failed array
indices can be resubmitted without overwriting successful results. Use
`FLOWLOT_ALLOW_INCOMPLETE=1` only for an interim summary.

For heterogeneous clusters, create two config files and result directories. A
CPU config can run `logistic,...,flowsom`; a GPU config can run
`cellcnn,attention_mil,cytoset,dgcnn,pointnet2` with GPU `SBATCH` flags. Both
experiments must use the same `shared_splits.json` to retain paired cohorts; a
single combined GPU array is simpler when resource waste is acceptable.

The legacy XGBoost entry is available after `pip install -e '.[xgboost]'` and is
enabled in the example configuration. The two nearest-subspace variants (`nsc`
and `nsc_energy`) preserve the earlier reconstruction-distance classification
scheme while using clone-compatible modern scikit-learn estimators.

For a local smoke run:

```bash
FLOWLOT_LOCAL_JOBS=2 bash scripts/hpc_repeated_benchmark.sh local
```

## Outputs

The aggregate directory contains:

- `per_run.csv`: one row per classifier/fusion/repeat/k;
- `summary.csv`: mean, standard deviation, and count;
- `paired_comparisons.csv`: paired deltas and win fractions across every method;
- `comparison_k*.tex`: booktabs tables;
- `aggregation_comparison.{pdf,svg,png}`: vector/publication figures;
- `integrity.json`: expected, completed, missing, and unexpected job IDs.

The companion notebook `notebooks/legacy_repeated_benchmark.ipynb` audits cohort
nesting, previews the job matrix, and explores accumulated comparisons.
