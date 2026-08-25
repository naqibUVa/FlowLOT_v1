#!/usr/bin/env bash
# Execute the global job table created by 05_option_sensitivity_comparison.ipynb.

set -euo pipefail

ACTION=${1:-help}
TABLE=${FLOWLOT_SENSITIVITY_TABLE:-}
SCRIPT_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")

activate_environment() {
  if [[ -n "${FLOWLOT_ENV_ACTIVATE:-}" ]]; then
    # shellcheck disable=SC1090
    source "${FLOWLOT_ENV_ACTIVATE}"
  fi
  export MPLBACKEND=Agg
  export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-${FLOWLOT_CPUS:-4}}
  export MKL_NUM_THREADS=${OMP_NUM_THREADS}
}

require_table() {
  if [[ -z "${TABLE}" || ! -f "${TABLE}" ]]; then
    echo "Set FLOWLOT_SENSITIVITY_TABLE to sensitivity_jobs.tsv" >&2
    exit 2
  fi
}

worker() {
  require_table
  activate_environment
  local index=${FLOWLOT_JOB_INDEX:-${SLURM_ARRAY_TASK_ID:-}}
  if [[ -z "${index}" ]]; then
    echo "Set FLOWLOT_JOB_INDEX or run inside a Slurm array." >&2
    exit 2
  fi
  local python=${FLOWLOT_PYTHON:-python3}
  mapfile -t row < <("${python}" - "${TABLE}" "${index}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
index = int(sys.argv[2])
if not 0 <= index < len(rows):
    raise SystemExit(f"Global job index {index} is outside [0, {len(rows) - 1}]")
record = rows[index]
for field in ("stage2", "splits", "jobs", "output_dir", "preprocess", "embedding", "job_index"):
    print(record[field])
PY
  )
  local args=(
    -m flowlot.evaluation.repeated_benchmark run
    --stage2 "${row[0]}"
    --splits "${row[1]}"
    --jobs "${row[2]}"
    --output-dir "${row[3]}"
    --preprocess "${row[4]}"
    --embedding "${row[5]}"
    --index "${row[6]}"
    --epochs "${FLOWLOT_EPOCHS:-50}"
    --batch-size "${FLOWLOT_BATCH_SIZE:-8}"
    --max-cells "${FLOWLOT_MAX_CELLS:-2048}"
  )
  if [[ -n "${FLOWLOT_DEVICE:-}" ]]; then
    args+=(--device "${FLOWLOT_DEVICE}")
  fi
  "${python}" "${args[@]}"
}

submit() {
  require_table
  local count upper concurrency log_dir
  count=$(($(wc -l < "${TABLE}") - 1))
  if ((count < 1)); then
    echo "Sensitivity table contains no jobs." >&2
    exit 2
  fi
  upper=$((count - 1))
  concurrency=${FLOWLOT_MAX_CONCURRENT:-20}
  log_dir=$(dirname -- "${TABLE}")/logs
  mkdir -p "${log_dir}"
  # Scheduler flags are intentionally shell-split.
  # shellcheck disable=SC2086
  sbatch ${FLOWLOT_SENSITIVITY_SBATCH_ARGS:-${FLOWLOT_SBATCH_ARGS:-}} \
    --job-name=flowlot-sensitivity \
    --array="0-${upper}%${concurrency}" \
    --output="${log_dir}/%A_%a.out" \
    --error="${log_dir}/%A_%a.err" \
    --export="ALL,FLOWLOT_SENSITIVITY_TABLE=${TABLE}" \
    "${SCRIPT_PATH}" worker
}

local_run() {
  require_table
  local count parallelism
  count=$(($(wc -l < "${TABLE}") - 1))
  parallelism=${FLOWLOT_LOCAL_JOBS:-1}
  export FLOWLOT_SENSITIVITY_TABLE="${TABLE}"
  seq 0 $((count - 1)) | xargs -P "${parallelism}" -I '{}' \
    env FLOWLOT_JOB_INDEX='{}' "${SCRIPT_PATH}" worker
}

case "${ACTION}" in
  submit) submit ;;
  worker) worker ;;
  local) local_run ;;
  *)
    cat <<'HELP'
Usage: hpc_sensitivity_benchmark.sh {submit|worker|local}

Set FLOWLOT_SENSITIVITY_TABLE to the absolute sensitivity_jobs.tsv path created
by notebooks/05_option_sensitivity_comparison.ipynb.
HELP
    ;;
esac
