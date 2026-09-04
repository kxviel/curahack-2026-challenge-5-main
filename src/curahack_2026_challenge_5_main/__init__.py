"""Gut Aging Index data preparation and calculation package."""

from __future__ import annotations

import sys

from .gai_cal import main as run_gai

__all__ = ["main"]


def main() -> None:
    meta_path = "./datasets/processed/"
    otu_path = "./datasets/processed/"
    output_dir = "./result/"

    if len(sys.argv) != 2:
        print("Invalid arguments!")
        print("Usage: uv run gai-cal ggmp | uv run gai-cal agp")
        return

    if sys.argv[1] == "agp":
        meta_path += "AGP/meta.tsv"
        otu_path += "AGP/otu.tsv"
        output_dir += "agp/"

    elif sys.argv[1] == "ggmp":
        meta_path += "GGMP/meta.tsv"
        otu_path += "GGMP/otu.tsv"
        output_dir += "ggmp/"

    else:
        print("Invalid arguments!")
        print("Usage: uv run gai-cal ggmp | uv run gai-cal agp")
        return

    run_gai(meta_path, otu_path, output_dir)
