"""Compile the Markdown knowledge bundle into a single PDF with Pandoc."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/FlowLOT_manual.pdf"))
    args = parser.parse_args()
    if shutil.which("pandoc") is None:
        raise SystemExit("Pandoc is required: https://pandoc.org/installing.html")
    root = Path(__file__).resolve().parent.parent
    sources = [root / "README.md", root / "docs/data_architecture.md", root / "docs/knowledge_map.md", root / "docs/tutorial.md"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pandoc",
            *map(str, sources),
            "--from=gfm",
            "--toc",
            "--number-sections",
            "--metadata=title:FlowLOT Manual",
            "--output",
            str(args.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
