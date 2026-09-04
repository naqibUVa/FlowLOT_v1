#!/usr/bin/env python
"""Run FlowLOT repeated-classification jobs locally with process parallelism.

This script is the local equivalent of the Slurm arrays in
`scripts/hpc_repeated_benchmark.sh` and `scripts/hpc_sensitivity_benchmark.sh`.
Each worker writes one atomic JSON shard into `<output_dir>/shards`, and the
notebooks' `aggregate_results(...)` step later consumes exactly those shards.

Input: a TSV job table in one of two shapes
  * plain table (written by `03_split_verification.ipynb` / `create_job_table`):
    columns include `index`; global settings come from command-line flags.
  * "global" sensitivity table (written by `05_option_sensitivity_comparison.ipynb`):
    one row per job with columns
    `stage2, splits, jobs, output_dir, preprocess, embedding, job_index`.
    This shape is detected automatically.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
import torch


def _worker_init() -> None:
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _run_row(args: tuple) -> int:
    from flowlot.evaluation.repeated_benchmark import run_job

    stage2, splits, jobs_path, output_dir, preprocess, embedding, index, epochs, batch_size, max_cells = args
    run_job(
        stage2, splits, jobs_path, output_dir, preprocess, embedding, index,
        epochs=epochs, batch_size=batch_size, max_cells=max_cells,
    )
    return int(index)


def build_tasks(args: argparse.Namespace) -> list[tuple]:
    table = pd.read_csv(args.job_table, sep="\t")
    is_global = "job_index" in table.columns and "stage2" in table.columns
    if is_global:
        tasks = [
            (
                str(row.stage2), str(row.splits), str(row.jobs), str(row.output_dir),
                str(row.preprocess), str(row.embedding), int(row.job_index),
                args.epochs, args.batch_size, args.max_cells,
            )
            for row in table.itertuples(index=False)
        ]
    else:
        assert args.stage2 and args.registry and args.output_dir, (
            "plain job table requires --stage2/--registry/--output-dir/--preprocess/--embedding"
        )
        tasks = [
            (
                str(args.stage2), str(args.registry), str(args.job_table), str(args.output_dir),
                str(args.preprocess), str(args.embedding), int(row.index),
                args.epochs, args.batch_size, args.max_cells,
            )
            for row in table.itertuples(index=False)
        ]
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job-table", required=True, type=Path, help="TSV job table")
    parser.add_argument("--stage2", type=Path, help="stage2 HDF5 (plain table mode)")
    parser.add_argument("--registry", type=Path, help="shared_splits.json (plain table mode)")
    parser.add_argument("--output-dir", type=Path, help="directory that receives shards/ (plain table mode)")
    parser.add_argument("--preprocess", default="common12")
    parser.add_argument("--embedding", default="patient0_sinkhorn_disp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-cells", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=min(40, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=0, help="only the first N jobs (smoke test)")
    parser.add_argument("--resume", action="store_true", default=True, help="skip already-completed shards")
    args = parser.parse_args()

    tasks = build_tasks(args)
    if args.limit:
        tasks = tasks[: args.limit]
    # run_job(..., resume=True) skips shards that already exist for this registry.
    print(f"jobs to run: {len(tasks)} (workers={args.workers})")
    if not tasks:
        print("no jobs")
        return
    with mp.Pool(args.workers, initializer=_worker_init) as pool:
        for _ in pool.imap_unordered(_run_row, tasks, chunksize=1):
            pass
    print(f"done: {len(tasks)} shards ready")


if __name__ == "__main__":
    main()