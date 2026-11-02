"""Thin import wrapper: the estimator itself lives in experiments/sampling_validation."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(REPO_ROOT))

from experiments.sampling_validation.sampler import (  # noqa: E402,F401
    antithetic_perm_global,
    antithetic_perm_local,
)

__all__ = ["antithetic_perm_global", "antithetic_perm_local"]
