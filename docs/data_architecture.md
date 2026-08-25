# FlowLOT data architecture

## Design principles

The patient is the unit of prediction and splitting. Cells from one patient must
never be divided across train/test folds. Stage 1 records source data faithfully;
Stage 2 reorganizes it by tube so marker alignment, references, and models can be
fit independently per panel. HDF5 names are slash-safe but original IDs and labels
are retained as attributes or datasets.

## Stage 1: raw patient hierarchy

```text
/{dataset_name}/{cell_count}/{patient_id}_label_{label}/{tube_id}/
    raw_cell_matrix        float32 [N_i, D_t]
    marker_descriptions    UTF-8   [D_t]
    @patient_id            string
    @label                 string (lossless source value)
    @tube_id               string
    @counts                int64 (events before optional subsampling)
```

Root attributes are `schema=flowlot-stage1` and `schema_version=1.0`. Supported
inputs are FCS (optional `flowio` extra), CSV/TSV/TXT, NPY, and NPZ. Sampling is
without replacement and seeded.

## Stage 2: analytical tube hierarchy

```text
/{dataset_name}/{cell_count}/{tube_id}/
├── metadata/
│   ├── patient_ids             UTF-8 [P_t]
│   ├── labels                  numeric or UTF-8 [P_t]
│   ├── cell_counts             int64 [P_t]
│   └── marker_descriptions     UTF-8 [D_t]
├── raw/{patient_id}/raw_cell_matrix                float32 [N_i, D_t]
├── preprocess_{id}/
│   ├── marker_subset                                  UTF-8 [d_t]
│   ├── @arcsinh_cofactor                              float
│   └── {patient_id}/processed_matrix                float32 [N_i, d_t]
└── lot_embeddings/{preprocess_id}/{reference}_{solver}/
    ├── @reference_type, @solver, @representation, @flatten_order
    ├── patient_ids                                  UTF-8 [P_t]
    ├── reference_patient_ids                        UTF-8 [R_t]
    ├── reference_matrix                           float32 [M, d_t]
    ├── embeddings                                float32 [P_t, M*d_t]
    ├── transport_costs                           float64 [P_t]
    ├── converged                                    bool [P_t]
    ├── sorted_cell_matrices/{patient_id}          float32 [M, d_t]
    └── transport_matrices/{patient_id}            float32 [N_i, M]
```

`marker_policy=intersection` aligns every sample in a tube to the ordered
intersection of the first sample's markers; `strict` rejects any mismatch.
Different tubes may have different marker sets and dimensions.

### LOT tensor convention

The coupling `Γ` has shape `[N_target, M_reference]`, row marginal `a`, and
column marginal `b`. The barycentric map is

`T(X0)_j = sum_i Γ[i,j] X_i / b_j`.

The default LOT matrix is the mass-weighted displacement
`sqrt(b_j) * (T(X0)_j - X0_j)`, flattened in Fortran order to preserve contiguous
marker blocks. `representation=map` stores the earlier FlowCode behavior instead.

## Legacy archive mapping

The old path `Dataset/<dataset>/sample_<N>/patient_<id>/tube_<id>/data` maps to
Stage 1 `raw_cell_matrix`. Old `lot_hungarian/lot` vectors were mapped point
clouds of size `M*12`; `ordered` was `[M,12]`. Count arrays encoded BLAST110
`[WBC_sample, blast_sample, WBC_original, blast_original]` and LAIP29 additionally
stored LAIP quantities. These cohort-specific targets are intentionally not
hard-coded into the general schema; import them as labels/covariates in a project
manifest or derive them in an analysis module.

## Validation invariants

- `len(patient_ids) == len(labels) == len(cell_counts)` per tube.
- All raw matrices in a tube use `len(marker_descriptions)` columns.
- All matrices in a preprocess group use `len(marker_subset)` columns.
- One embedding row corresponds to one `patient_ids` entry.
- `embedding_width == reference_rows * selected_marker_count`.
- Patient-level train/validation/test partitions are established before fitting
  preprocessing, references, imputers, or models in confirmatory experiments.
