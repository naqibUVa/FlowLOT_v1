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
    sample_event_ids       UTF-8   [N_i]       (optional annotated cohorts)
    sample_source_indices  int64   [N_i]       (optional annotated cohorts)
    population_annotations float32 [N_i, Q]    (optional WBC/Blast/LAIP indicators)
    population_counts      float64 [Q, 4]      (optional counts and % of WBC)
    @patient_id            string
    @label                 string (lossless source value)
    @tube_id               string
    @counts                int64 (events before optional subsampling)
    @annotated_event_count int64 (rows in the event-label CSV, when supplied)
```

Root attributes are `schema=flowlot-stage1` and `schema_version=1.0`. Supported
inputs are FCS (optional `flowio` extra), CSV/TSV/TXT, NPY, and NPZ. Sampling is
without replacement and seeded.

For BLAST110/LAIP29-style data, the manifest can include
`event_labels_path,event_id_column,population_columns`. The event-label CSV is
joined to the raw matrix through the named event-ID channel before sampling.
`population_counts` columns are `sampled_count`, `original_count`,
`sampled_pct_wbc`, and `original_pct_wbc`; row names are stored in its
`population_names` attribute. Thus the original targets are computed as
`100 * sum(Blast) / sum(WBC)` and `100 * sum(LAIP) / sum(WBC)`, never inferred
from marker intensity.

## Stage 2: analytical tube hierarchy

```text
/{dataset_name}/{cell_count}/{tube_id}/
├── metadata/
│   ├── patient_ids             UTF-8 [P_t]
│   ├── labels                  numeric or UTF-8 [P_t]
│   ├── cell_counts             int64 [P_t]
│   └── marker_descriptions     UTF-8 [D_t]
├── raw/{patient_id}/
│   ├── raw_cell_matrix                             float32 [N_i, D_t]
│   ├── population_annotations (optional)           float32 [N_i, Q]
│   └── population_counts (optional)                float64 [Q, 4]
├── preprocess_{id}/
│   ├── marker_subset                                  UTF-8 [d_t]
│   ├── @arcsinh_cofactor                              float
│   └── {patient_id}/processed_matrix                float32 [N_i, d_t]
├── references/ (persistent cached templates)
│   └── {reference_type}_size{size}[_p{max_patients}]/
│       ├── @reference_type, @reference_size, @random_state
│       ├── reference_matrix                         float32 [M, d_t]
│       └── reference_patient_ids                    UTF-8 [R_t]
└── lot_embeddings/{preprocess_id}/{reference}_{solver}[_repr]/
    ├── @reference_type, @solver, @representation, @flatten_order
    ├── @random_state, @reference_size, @store_transport
    ├── @reference_kwargs_json, @solver_kwargs_json
    ├── patient_ids                                  UTF-8 [P_t]
    ├── reference_patient_ids                        UTF-8 [R_t]
    ├── reference_matrix                           float32 [M, d_t]
    ├── embeddings                                float32 [P_t, M*d_t]
    ├── transport_costs                           float64 [P_t]
    ├── converged                                    bool [P_t]
    ├── sorted_cell_matrices/{patient_id}          float32 [M, d_t]
    └── transport_matrices/{patient_id}            float32 [N_i, M]
```

The final embedding key may be customized (for example,
`patient0_sinkhorn_disp`, `barycenter_emd_map`, or split-specific runs)
so representations or split-specific references do not overwrite one another.
[`notebooks/NB03_lot_embeddings.ipynb`](../notebooks/NB03_lot_embeddings.ipynb) constructs and
audits these stored representations across solvers and reference types.

### Persistent reference caching
To ensure template reproducibility and avoid stochastic re-sampling between runs, references are cached under `tube["references"][ref_store_key]`. If `freeze_reference=True` (the default), subsequent pipeline steps reuse the frozen template rather than re-fitting or drifting.

### Dual representations and compression
When `representation="both"` is selected, FlowLOT solves the Monge/Kantorovich transport problem once and generates both `{base_id}_map` and `{base_id}_disp` embedding groups in parallel. The heavy point-cloud and coupling nodes (`sorted_cell_matrices` and `transport_matrices`) in the displacement group are stored as internal HDF5 links to the map group, cutting OT solver runtime by 50% without duplicating data on disk.
Point clouds and transport couplings utilize fast `lzf` chunked compression, while dense patient embedding matrices use `gzip` compression.

`marker_policy=intersection` aligns every sample in a tube to the ordered
intersection of the first sample's markers; `strict` rejects any mismatch.
Different tubes may have different marker sets and dimensions.

### LOT tensor convention

The coupling $\Gamma$ has shape $[N_{\text{target}}, M_{\text{reference}}]$, row marginal $a$, and
column marginal $b$. The barycentric map is

$$T(X_0)_j = \sum_i \Gamma_{ij} X_i / b_j.$$

The default LOT matrix is the mass-weighted displacement
$\sqrt{b_j} (T(X_0)_j - X_{0,j})$, flattened in Fortran order (`order="F"`) to preserve contiguous
marker blocks across reference anchors. `representation="map"` stores the transported template points $T(X_0)_j$ instead.

## Legacy archive mapping

The old path `Dataset/<dataset>/sample_<N>/patient_<id>/tube_<id>/data` maps to
Stage 1 `raw_cell_matrix`. Old `lot_hungarian/lot` vectors were mapped point
clouds of size $M \times 12$; `ordered` was $[M, 12]$. Count arrays encoded BLAST110
`[WBC_sample, blast_sample, WBC_original, blast_original]` and LAIP29 additionally
stored LAIP quantities. When rebuilding from raw data via [`notebooks/NB01_Sampling_from_Raw_measurements.ipynb`](../notebooks/NB01_Sampling_from_Raw_measurements.ipynb), use the per-event label
CSVs so these quantities and their cell-level provenance are represented by the
optional general population datasets above.

## Validation invariants

- `len(patient_ids) == len(labels) == len(cell_counts)` per tube.
- All raw matrices in a tube use `len(marker_descriptions)` columns.
- All matrices in a preprocess group use `len(marker_subset)` columns.
- One embedding row corresponds to one `patient_ids` entry.
- `embedding_width == reference_rows * selected_marker_count`.
- Patient-level train/validation/test partitions are established before fitting
  preprocessing, references, imputers, or models in confirmatory experiments.
- The earlier deep baselines remain available through `flowlot.models.baselines` and the top-level `evaluate.py`.

