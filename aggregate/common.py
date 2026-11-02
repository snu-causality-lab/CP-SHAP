"""Shared paths and helpers for the aggregate/ scripts.

Each script reads the raw outputs that the matching reproduce/ script wrote under outputs/ and writes the
numbers shown in one paper result to results/<name>.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
RESULTS = REPO / "results"


def parser(doc: str, default_in: Path) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.set_defaults(inp=default_in, out=RESULTS)
    return p


def write(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df.to_csv(path, index=False)
    print(f"[aggregate] wrote {path} ({len(df)} rows)")
    return path
