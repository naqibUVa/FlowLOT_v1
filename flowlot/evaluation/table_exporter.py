"""CSV and booktabs LaTeX exports for cross-validation summaries."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def export_csv(rows: Sequence[Mapping[str, object]], output: str | Path) -> Path:
    if not rows:
        raise ValueError("Cannot export an empty result table")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarize_runs(
    rows: Sequence[Mapping[str, object]],
    model_key: str = "model",
    identifier_keys: Sequence[str] = ("fold", "run", "seed"),
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[model_key])].append(row)
    summary = []
    for model, values in grouped.items():
        result: dict[str, object] = {model_key: model, "n_runs": len(values)}
        numeric = [
            key
            for key in values[0]
            if key != model_key
            and key not in identifier_keys
            and all(isinstance(row.get(key), (int, float, np.number)) for row in values)
        ]
        for metric in numeric:
            array = np.asarray([row[metric] for row in values], dtype=float)
            result[f"{metric}_mean"] = float(np.nanmean(array))
            result[f"{metric}_std"] = float(np.nanstd(array, ddof=1)) if len(array) > 1 else 0.0
        summary.append(result)
    return summary


def export_latex(
    summary: Sequence[Mapping[str, object]],
    output: str | Path,
    model_key: str = "model",
    lower_is_better: Sequence[str] = ("mae", "rmse"),
    caption: str = "Cross-validated FlowLOT benchmark performance.",
    label: str = "tab:flowlot-benchmark",
) -> Path:
    if not summary:
        raise ValueError("Cannot export an empty summary")
    means = [key[:-5] for key in summary[0] if key.endswith("_mean")]
    best: dict[str, float] = {}
    for metric in means:
        values = np.asarray([row[f"{metric}_mean"] for row in summary], dtype=float)
        best[metric] = float(np.nanmin(values) if metric in lower_is_better else np.nanmax(values))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(means) + "}",
        "\\toprule",
        "Model & " + " & ".join(metric.replace("_", " ").title() for metric in means) + " \\\\",
        "\\midrule",
    ]
    for row in summary:
        cells = [str(row[model_key]).replace("_", "\\_")]
        for metric in means:
            mean, std = float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])
            value = f"{mean:.3f} $\\pm$ {std:.3f}"
            if np.isclose(mean, best[metric], equal_nan=False):
                value = f"\\textbf{{{value}}}"
            cells.append(value)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
