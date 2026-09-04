"""Gut Aging Index data preparation and calculation package."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from .gai_cal import main as run_gai

__all__ = ["main"]

PROJECT_ROOT = Path.cwd().parent


def main() -> None:

    if len(sys.argv) != 2 or sys.argv[1].lower() not in ("agp", "ggmp"):
        print("Invalid arguments!")
        print("Usage: uv run gai-cal ggmp | uv run gai-cal agp")
        return

    type DATASET = Literal["AGP", "GGMP"]
    dataset: DATASET = "AGP" if sys.argv[1].lower() == "agp" else "GGMP"

    meta_path = PROJECT_ROOT / "datasets/processed" / dataset / "meta.tsv"
    otu_path = PROJECT_ROOT / "datasets/processed" / dataset / "otu.tsv"
    output_path = PROJECT_ROOT / "result" / dataset

    run_gai(meta_path, otu_path, output_path)
