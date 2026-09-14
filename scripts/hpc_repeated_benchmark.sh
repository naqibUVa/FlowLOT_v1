#!/usr/bin/env bash
# Slurm-compatible shared-split FlowLOT benchmark launcher.
# Resource flags are supplied through FLOWLOT_SBATCH_ARGS in the config file so
# CPU and GPU clusters can use the same script.

set -euo pipefail

export PATH="/opt/slurm/current/bin:$PATH"

ACTION=${1:-help}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG=${FLOWLOT_CONFIG:-"${SCRIPT_DIR}/repeated_benchmark.env"}

if [[ -f "${CONFIG}" ]]; then
  # This is an explicit user-owned shell configuration.
  # shellcheck disable=SC1090
  source "${CONFIG}"
elif [[ "${ACTION}" != "help" ]]; then
  echo "Configuration file not found: ${CONFIG}" >&2
  echo "Copy scripts/repeated_benchmark.env.example and set FLOWLOT_CONFIG." >&2
  exit 2
fi

require_var() {
  local name=$1
  if [[ -z "${!name:-}" ]]; then
    echo "Required configuration variable is empty: ${name}" >&2
    exit 2
  fi
}

activate_environment() {
  if [[ -n "${FLOWLOT_ENV_ACTIVATE:-}" ]]; then
    # shellcheck disable=SC1090
    source "${FLOWLOT_ENV_ACTIVATE}"
  fi
  export MPLBACKEND=Agg
  export HDF5_USE_FILE_LOCKING=FALSE
  export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-${FLOWLOT_CPUS:-4}}
  export MKL_NUM_THREADS=${OMP_NUM_THREADS}
}

run_module() {
  "${FLOWLOT_PYTHON:-python3}" -m flowlot.evaluation.repeated_benchmark "$@"
}

initialize() {
  for name in FLOWLOT_STAGE2 FLOWLOT_DATASET FLOWLOT_CELL_COUNT FLOWLOT_PREPROCESS FLOWLOT_EMBEDDING FLOWLOT_RESULTS; do
    require_var "${name}"
  done
  activate_environment
  mkdir -p "${FLOWLOT_RESULTS}/logs" "${FLOWLOT_RESULTS}/shards"
  local split_args=(
    splits
    --stage2 "${FLOWLOT_STAGE2}"
    --dataset "${FLOWLOT_DATASET}"
    --cells "${FLOWLOT_CELL_COUNT}"
    --output "${FLOWLOT_RESULTS}/shared_splits.json"
    --train-per-class "${FLOWLOT_TRAIN_PER_CLASS:-2,4,6,8}"
    --repeats "${FLOWLOT_REPEATS:-10}"
    --test-size "${FLOWLOT_TEST_SIZE:-0.5}"
    --seed "${FLOWLOT_SEED:-42}"
    --patient-policy "${FLOWLOT_PATIENT_POLICY:-intersection}"
    --legacy-h5 "${FLOWLOT_RESULTS}/legacy_splits.h5"
  )
  if [[ -n "${FLOWLOT_TUBES:-}" ]]; then
    split_args+=(--tubes "${FLOWLOT_TUBES}")
  fi
  if [[ -n "${FLOWLOT_CLASSES:-}" ]]; then
    split_args+=(--classes "${FLOWLOT_CLASSES}")
  fi
  run_module "${split_args[@]}"
  run_module jobs \
    --splits "${FLOWLOT_RESULTS}/shared_splits.json" \
    --output "${FLOWLOT_RESULTS}/jobs.tsv" \
    --models "${FLOWLOT_MODELS:-logistic,linear_svm,random_forest,extra_trees,nsc,nsc_energy,flowsom,cellcnn,attention_mil,cytoset,dgcnn,pointnet2}" \
    --aggregations "${FLOWLOT_AGGREGATIONS:-single,early_mean,late_soft}"
  local count
  count=$(($(wc -l < "${FLOWLOT_RESULTS}/jobs.tsv") - 1))
  echo "Initialized ${count} jobs in ${FLOWLOT_RESULTS}"
}

worker() {
  for name in FLOWLOT_STAGE2 FLOWLOT_PREPROCESS FLOWLOT_EMBEDDING FLOWLOT_RESULTS; do
    require_var "${name}"
  done
  activate_environment
  local index=${FLOWLOT_JOB_INDEX:-${SLURM_ARRAY_TASK_ID:-}}
  if [[ -z "${index}" ]]; then
    echo "Set FLOWLOT_JOB_INDEX or run inside a Slurm array." >&2
    exit 2
  fi
  local run_args=(
    run
    --stage2 "${FLOWLOT_STAGE2}"
    --splits "${FLOWLOT_RESULTS}/shared_splits.json"
    --jobs "${FLOWLOT_RESULTS}/jobs.tsv"
    --output-dir "${FLOWLOT_RESULTS}"
    --preprocess "${FLOWLOT_PREPROCESS}"
    --embedding "${FLOWLOT_EMBEDDING}"
    --index "${index}"
    --epochs "${FLOWLOT_EPOCHS:-50}"
    --batch-size "${FLOWLOT_BATCH_SIZE:-8}"
    --max-cells "${FLOWLOT_MAX_CELLS:-2048}"
  )
  if [[ -n "${FLOWLOT_DEVICE:-}" ]]; then
    run_args+=(--device "${FLOWLOT_DEVICE}")
  fi
  run_module "${run_args[@]}"
}

submit_array() {
  require_var FLOWLOT_RESULTS
  if [[ ! -f "${FLOWLOT_RESULTS}/jobs.tsv" ]]; then
    echo "Run '$0 init' before submitting." >&2
    exit 2
  fi
  local count upper concurrency
  count=$(($(wc -l < "${FLOWLOT_RESULTS}/jobs.tsv") - 1))
  upper=$((count - 1))
  concurrency=${FLOWLOT_MAX_CONCURRENT:-20}
  mkdir -p "${FLOWLOT_RESULTS}/logs"
  # FLOWLOT_SBATCH_ARGS is deliberately shell-split to permit multiple scheduler flags.
  # shellcheck disable=SC2086
  sbatch ${FLOWLOT_SBATCH_ARGS:-} \
    --job-name=flowlot-repeat \
    --array="${FLOWLOT_ARRAY_RANGE:-0-${upper}}%${concurrency}" \
    --output="${FLOWLOT_RESULTS}/logs/%A_%a.out" \
    --error="${FLOWLOT_RESULTS}/logs/%A_%a.err" \
    --export="ALL,FLOWLOT_CONFIG=${CONFIG}" \
    "${BASH_SOURCE[0]}" worker
}

run_local() {
  require_var FLOWLOT_RESULTS
  local count parallelism
  count=$(($(wc -l < "${FLOWLOT_RESULTS}/jobs.tsv") - 1))
  parallelism=${FLOWLOT_LOCAL_JOBS:-1}
  export FLOWLOT_CONFIG="${CONFIG}"
  export -f require_var activate_environment run_module worker
  seq 0 $((count - 1)) | xargs -P "${parallelism}" -I '{}' \
    bash -c 'FLOWLOT_JOB_INDEX="$1" "$2" worker' _ '{}' "${BASH_SOURCE[0]}"
}

aggregate() {
  require_var FLOWLOT_RESULTS
  activate_environment
  local aggregate_args=(
    aggregate
    --splits "${FLOWLOT_RESULTS}/shared_splits.json"
    --jobs "${FLOWLOT_RESULTS}/jobs.tsv"
    --shards "${FLOWLOT_RESULTS}/shards"
    --output "${FLOWLOT_RESULTS}/aggregate"
    --bootstrap-iterations "${FLOWLOT_BOOTSTRAP_ITERATIONS:-1000}"
    --confidence-level "${FLOWLOT_CONFIDENCE_LEVEL:-0.95}"
    --bootstrap-seed "${FLOWLOT_BOOTSTRAP_SEED:-42}"
  )
  if [[ -n "${FLOWLOT_ALLOW_INCOMPLETE:-}" ]]; then
    aggregate_args+=(--allow-incomplete)
  fi
  run_module "${aggregate_args[@]}"
}

status() {
  require_var FLOWLOT_RESULTS
  if [[ ! -f "${FLOWLOT_RESULTS}/jobs.tsv" ]]; then
    echo "Benchmark not initialized yet in ${FLOWLOT_RESULTS}"
    return 0
  fi
  local total completed pct
  total=$(($(wc -l < "${FLOWLOT_RESULTS}/jobs.tsv") - 1))
  completed=$(find "${FLOWLOT_RESULTS}/shards" -name "*.json" ! -name "._*" 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${total}" -gt 0 ]]; then
    pct=$(awk "BEGIN { printf \"%.1f\", (${completed} / ${total}) * 100 }")
  else
    pct="0.0"
  fi
  echo "========================================================"
  echo "Benchmark:    $(basename "${FLOWLOT_RESULTS}")"
  echo "Directory:    ${FLOWLOT_RESULTS}"
  echo "Progress:     ${completed} / ${total} shards completed (${pct}%)"
  if [[ -f "${FLOWLOT_RESULTS}/aggregate/summary.csv" ]]; then
    echo "Aggregated:   Yes (${FLOWLOT_RESULTS}/aggregate/summary.csv exists)"
  else
    echo "Aggregated:   No (run '$0 aggregate' when completed)"
  fi
  if command -v squeue >/dev/null 2>&1; then
    echo "---------------- Active Slurm Tasks --------------------"
    squeue -u "${USER:-$(whoami)}" --name=flowlot-repeat || true
  fi
  echo "========================================================"
}

case "${ACTION}" in
  init) initialize ;;
  submit) submit_array ;;
  worker) worker ;;
  local) run_local ;;
  status) status ;;
  aggregate) aggregate ;;
  *)
    cat <<'HELP'
Usage: hpc_repeated_benchmark.sh {init|submit|worker|local|status|aggregate}

  init       Create immutable shared splits and jobs.tsv.
  submit     Submit jobs.tsv as a resumable Slurm array.
  worker     Execute FLOWLOT_JOB_INDEX or SLURM_ARRAY_TASK_ID.
  local      Run all jobs locally with FLOWLOT_LOCAL_JOBS workers.
  status     Check completed shards vs total planned jobs and active Slurm tasks.
  aggregate  Validate completeness and create CSV/LaTeX/vector comparisons.

Set FLOWLOT_CONFIG to a copy of repeated_benchmark.env.example.
HELP
    ;;
esac
