"""Aggregate paired baseline-sensitivity cells.

The aggregator is strict: every configured ``(seed, rho)`` cell and every
baseline must exist.  It compares attribution arrays from the same fitted model
and writes numerical, sign-flip, and component-conflict tables (Table 7, Appendix E.2).

Output labels top3_overlap, top5_overlap and active1pct_* describe the shipped paper
config (primary_top_k=3, top_k=5, active_sign_fraction=0.01). If these settings are
changed, the calculations follow the config; interpret the unchanged column labels
with those configured values.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

try:
    from .run_cell import (
        BASELINE_NAMES,
        JOINT_MARGINAL_BASELINE_NAME,
        MARGINAL_BASELINE_NAME,
        rho_label,
    )
except ImportError:  # Direct script execution.
    from run_cell import (
        BASELINE_NAMES,
        JOINT_MARGINAL_BASELINE_NAME,
        MARGINAL_BASELINE_NAME,
        rho_label,
    )


REFERENCE = "empirical_mean"
COMPONENTS = {
    "phiG": "phi_global",
    "phiL": "phi_local",
    "phiT": "phi_total",
}
DEFAULT_ACTIVE_SIGN_FRACTION = 0.01
DEFAULT_PRIMARY_TOP_K = 3


def load_cell(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        cell = {key: archive[key] for key in archive.files}
    cell["metadata"] = json.loads(str(cell.pop("metadata_json")))
    cell["baseline_names"] = [str(value) for value in cell["baseline_names"]]
    if cell["baseline_names"] != list(BASELINE_NAMES):
        raise ValueError(
            f"{path}: baseline order {cell['baseline_names']} != {list(BASELINE_NAMES)}"
        )
    if "marginal_baseline_name" in cell:
        cell["marginal_baseline_name"] = str(
            np.asarray(cell["marginal_baseline_name"]).item()
        )
        if cell["marginal_baseline_name"] != MARGINAL_BASELINE_NAME:
            raise ValueError(
                f"{path}: unexpected marginal baseline "
                f"{cell['marginal_baseline_name']}"
            )
    if "joint_marginal_baseline_name" in cell:
        cell["joint_marginal_baseline_name"] = str(
            np.asarray(cell["joint_marginal_baseline_name"]).item()
        )
        if (
            cell["joint_marginal_baseline_name"]
            != JOINT_MARGINAL_BASELINE_NAME
        ):
            raise ValueError(
                f"{path}: unexpected joint marginal baseline "
                f"{cell['joint_marginal_baseline_name']}"
            )
    return cell


def safe_corr(a: np.ndarray, b: np.ndarray, kind: str) -> tuple[float, int]:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or np.ptp(x) <= 1e-15 or np.ptp(y) <= 1e-15:
        return float("nan"), int(len(x))
    statistic = pearsonr if kind == "pearson" else spearmanr
    return float(statistic(x, y).statistic), int(len(x))


def sign_tolerance(a: np.ndarray, b: np.ndarray, cfg: dict) -> float:
    scale = max(
        float(np.max(np.abs(a))) if np.size(a) else 0.0,
        float(np.max(np.abs(b))) if np.size(b) else 0.0,
    )
    return max(
        float(cfg.get("absolute_zero_floor", 1e-9)),
        float(cfg.get("relative_zero_threshold", 1e-6)) * scale,
    )


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    left = np.asarray(a, dtype=float).ravel()
    right = np.asarray(b, dtype=float).ravel()
    if left.shape != right.shape or not len(left):
        return float("nan")
    k = min(int(k), len(left))
    left_top = set(np.argsort(np.abs(left), kind="stable")[-k:])
    right_top = set(np.argsort(np.abs(right), kind="stable")[-k:])
    return float(len(left_top & right_top) / k)


def topk_overlap_default_sort(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Top-k overlap using NumPy default argsort; tied entries may differ from the stable-sort helper."""

    left = np.asarray(a, dtype=float).ravel()
    right = np.asarray(b, dtype=float).ravel()
    k = min(int(k), len(left))
    if left.shape != right.shape or k <= 0:
        return float("nan")
    left_top = set(np.argsort(np.abs(left))[-k:])
    right_top = set(np.argsort(np.abs(right))[-k:])
    return float(len(left_top & right_top) / k)


def active_mask(values: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return configured relative-activity masks and thresholds (1% in the paper config)."""

    array = np.asarray(values, dtype=float)
    fraction = float(
        cfg.get("active_sign_fraction", DEFAULT_ACTIVE_SIGN_FRACTION)
    )
    floor = float(cfg.get("absolute_zero_floor", 1e-9))
    if not 0.0 <= fraction < 1.0:
        raise ValueError("active_sign_fraction must lie in [0, 1)")
    if array.ndim == 1:
        scale = float(np.max(np.abs(array))) if array.size else 0.0
        thresholds = np.asarray(max(floor, fraction * scale))
    elif array.ndim == 2:
        scale = np.max(np.abs(array), axis=1, keepdims=True)
        thresholds = np.maximum(floor, fraction * scale)
    else:
        raise ValueError("attribution arrays must be one- or two-dimensional")
    return np.isfinite(array) & (np.abs(array) >= thresholds), thresholds


def symmetric_normalized_l1(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric normalized L1, scaled to equal 2 for opposite vectors."""

    left = np.asarray(a, dtype=float).ravel()
    right = np.asarray(b, dtype=float).ravel()
    valid = np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        return float("nan")
    numerator = 2.0 * float(np.sum(np.abs(left[valid] - right[valid])))
    denominator = float(
        np.sum(np.abs(left[valid])) + np.sum(np.abs(right[valid]))
    )
    return numerator / denominator if denominator > 0.0 else 0.0


def _finite_summary(values: list[float]) -> tuple[float, float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        float(np.mean(finite)),
        float(np.median(finite)),
        float(np.min(finite)),
        int(finite.size),
    )


def _vector_primary_stats(a: np.ndarray, b: np.ndarray, cfg: dict) -> dict:
    signed_spearman, _ = safe_corr(a, b, "spearman")
    abs_rank_spearman, _ = safe_corr(np.abs(a), np.abs(b), "spearman")
    active_a, _ = active_mask(np.asarray(a, dtype=float), cfg)
    active_b, _ = active_mask(np.asarray(b, dtype=float), cfg)
    eligible = active_a & active_b
    sign_agreement = (
        float(np.mean(np.sign(a[eligible]) == np.sign(b[eligible])))
        if eligible.any()
        else float("nan")
    )
    return {
        "signed_spearman": signed_spearman,
        "abs_rank_spearman": abs_rank_spearman,
        "top3_overlap": topk_overlap(
            a, b, int(cfg.get("primary_top_k", DEFAULT_PRIMARY_TOP_K))
        ),
        "top5_overlap": topk_overlap(a, b, int(cfg.get("top_k", 5))),
        "symmetric_normalized_l1": symmetric_normalized_l1(a, b),
        "active1pct_sign_agreement": sign_agreement,
        "active1pct_eligible_fraction": float(np.mean(eligible)),
        "active1pct_eligible_n": int(eligible.sum()),
    }


def _row_pearson(x, y):
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    num = np.sum(xc * yc, axis=1)
    den = np.sqrt(np.sum(xc * xc, axis=1) * np.sum(yc * yc, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return np.clip(r, -1.0, 1.0)

def _row_corr(x, y, kind):
    """Row-wise safe_corr for finite 2-D inputs."""
    bad = (x.shape[1] < 2) | (np.ptp(x, axis=1) <= 1e-15) | (np.ptp(y, axis=1) <= 1e-15)
    if kind == "spearman":
        x, y = rankdata(x, axis=1), rankdata(y, axis=1)
    r = _row_pearson(x, y)
    r[bad] = np.nan
    return r

def _row_topk(a, b, k, stable):
    d = a.shape[1]
    k = min(int(k), d)
    kind = "stable" if stable else None
    ia = np.argsort(np.abs(a), axis=1, kind=kind)[:, d - k:]
    ib = np.argsort(np.abs(b), axis=1, kind=kind)[:, d - k:]
    ma = np.zeros(a.shape, dtype=bool)
    mb = np.zeros(b.shape, dtype=bool)
    rows = np.arange(a.shape[0])[:, None]
    ma[rows, ia] = True
    mb[rows, ib] = True
    return np.sum(ma & mb, axis=1) / k

def _row_pair_stats(a, b, cfg):
    """Compute per-query metrics in batches; keep the scalar definitions and tie rules."""
    row = {}
    active = active_mask(a, cfg)[0] & active_mask(b, cfg)[0]
    n_el = active.sum(axis=1)
    same = (np.sign(a) == np.sign(b)) & active
    with np.errstate(invalid="ignore", divide="ignore"):
        sign_agree = np.where(n_el > 0, same.sum(axis=1) / np.maximum(n_el, 1), np.nan)
    num = 2.0 * np.sum(np.abs(a - b), axis=1)
    den = np.sum(np.abs(a), axis=1) + np.sum(np.abs(b), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        l1 = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0)
    per_row = {
        "signed_spearman": _row_corr(a, b, "spearman"),
        "abs_rank_spearman": _row_corr(np.abs(a), np.abs(b), "spearman"),
        "top3_overlap": _row_topk(a, b, int(cfg.get("primary_top_k", DEFAULT_PRIMARY_TOP_K)), stable=True),
        "top5_overlap": _row_topk(a, b, int(cfg.get("top_k", 5)), stable=True),
        "symmetric_normalized_l1": l1,
        "active1pct_sign_agreement": sign_agree,
        "active1pct_eligible_fraction": active.mean(axis=1),
    }
    for metric, values in per_row.items():
        mean, median, minimum, valid_n = _finite_summary(values)
        row[metric] = mean
        row[f"query_{metric}_median"] = median
        row[f"query_{metric}_min"] = minimum
        row[f"query_{metric}_valid_n"] = valid_n
    row["active1pct_eligible_n"] = int(n_el.sum())
    row["topk_overlap"] = float(np.mean(_row_topk(a, b, int(cfg.get("top_k", 5)), stable=False)))
    corr = _row_corr(a, b, "pearson")
    corr = corr[np.isfinite(corr)]
    row["query_pearson_median"] = float(np.median(corr)) if corr.size else float("nan")
    row["query_pearson_min"] = float(np.min(corr)) if corr.size else float("nan")
    row["query_pearson_valid_n"] = int(corr.size)
    row["query_primary_valid_n"] = row["query_signed_spearman_valid_n"]
    return row


def pair_stats(
    a: np.ndarray,
    b: np.ndarray,
    cfg: dict,
) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired attribution shape mismatch: {a.shape} vs {b.shape}")
    diff = b - a
    pearson, pearson_n = safe_corr(a, b, "pearson")
    spearman, spearman_n = safe_corr(a, b, "spearman")
    tolerance = sign_tolerance(a, b, cfg)
    sign_valid = (np.abs(a) > tolerance) & (np.abs(b) > tolerance)
    sign_flip = sign_valid & (np.sign(a) != np.sign(b))
    row = {
        "pearson": pearson,
        "pearson_n": pearson_n,
        "spearman": spearman,
        "spearman_n": spearman_n,
        "mean_absolute_difference": float(np.mean(np.abs(diff))),
        "sign_tolerance": tolerance,
        "sign_valid_n": int(sign_valid.sum()),
        "cross_baseline_sign_flip_n": int(sign_flip.sum()),
        "cross_baseline_sign_flip_rate": (
            float(sign_flip.sum() / sign_valid.sum())
            if sign_valid.any()
            else float("nan")
        ),
        "cross_baseline_sign_agreement": (
            float(1.0 - sign_flip.sum() / sign_valid.sum())
            if sign_valid.any()
            else float("nan")
        ),
    }

    if a.ndim == 1:
        primary = _vector_primary_stats(a, b, cfg)
        row.update(primary)
        row["topk_overlap"] = topk_overlap_default_sort(
            a, b, int(cfg.get("top_k", 5))
        )
        row["query_pearson_median"] = float("nan")
        row["query_pearson_min"] = float("nan")
        row["query_pearson_valid_n"] = 0
        row["query_signed_spearman_median"] = float("nan")
        row["query_signed_spearman_min"] = float("nan")
        row["query_primary_valid_n"] = 0
    else:
        # Preserve the scalar path for empty, non-finite or unusual inputs.
        if (a.ndim == 2 and a.shape[0] > 0 and a.shape[1] > 0
                and np.isfinite(a).all() and np.isfinite(b).all()
                and int(cfg.get("primary_top_k", DEFAULT_PRIMARY_TOP_K)) > 0
                and int(cfg.get("top_k", 5)) > 0):
            row.update(_row_pair_stats(a, b, cfg))
            return row
        primary_rows = []
        correlations = []
        for left, right in zip(a, b):
            primary_rows.append(_vector_primary_stats(left, right, cfg))
            corr, _ = safe_corr(left, right, "pearson")
            if np.isfinite(corr):
                correlations.append(corr)
        primary_metrics = [
            "signed_spearman",
            "abs_rank_spearman",
            "top3_overlap",
            "top5_overlap",
            "symmetric_normalized_l1",
            "active1pct_sign_agreement",
            "active1pct_eligible_fraction",
        ]
        for metric in primary_metrics:
            mean, median, minimum, valid_n = _finite_summary(
                [item[metric] for item in primary_rows]
            )
            row[metric] = mean
            row[f"query_{metric}_median"] = median
            row[f"query_{metric}_min"] = minimum
            row[f"query_{metric}_valid_n"] = valid_n
        row["active1pct_eligible_n"] = int(
            sum(item["active1pct_eligible_n"] for item in primary_rows)
        )
        row["topk_overlap"] = float(
            np.mean(
                [
                    topk_overlap_default_sort(
                        left, right, int(cfg.get("top_k", 5))
                    )
                    for left, right in zip(a, b)
                ]
            )
        )
        row["query_pearson_median"] = (
            float(np.median(correlations)) if correlations else float("nan")
        )
        row["query_pearson_min"] = (
            float(np.min(correlations)) if correlations else float("nan")
        )
        row["query_pearson_valid_n"] = len(correlations)
        row["query_primary_valid_n"] = row[
            "query_signed_spearman_valid_n"
        ]
    return row


def feature_stats(
    a: np.ndarray,
    b: np.ndarray,
    cfg: dict,
) -> list[dict]:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.ndim == 1:
        left, right = left[None, :], right[None, :]
    rows = []
    for feature in range(left.shape[1]):
        stats = pair_stats(left[:, feature], right[:, feature], cfg)
        rows.append({"feature": feature + 1, **stats})
    return rows


def component_event_rows(
    phi_global: np.ndarray,
    phi_local: np.ndarray,
    phi_total: np.ndarray,
    cfg: dict,
) -> tuple[list[dict], list[dict]]:
    global_full = np.broadcast_to(phi_global[None, :], phi_local.shape)
    tolerance = max(
        sign_tolerance(phi_local, global_full, cfg),
        sign_tolerance(phi_local, phi_total, cfg),
    )
    valid_lg = (np.abs(phi_local) > tolerance) & (
        np.abs(global_full) > tolerance
    )
    valid_lt = (np.abs(phi_local) > tolerance) & (
        np.abs(phi_total) > tolerance
    )
    conflict = valid_lg & (np.sign(phi_local) != np.sign(global_full))
    total_local_flip = valid_lt & (np.sign(phi_local) != np.sign(phi_total))
    ratio = np.abs(phi_total) / (
        np.abs(phi_local) + np.abs(global_full)
        + float(cfg.get("absolute_zero_floor", 1e-9))
    )
    cancellation = conflict & (
        ratio <= float(cfg.get("cancellation_threshold", 0.1))
    )
    active_local, _ = active_mask(phi_local, cfg)
    active_global_vector, _ = active_mask(phi_global, cfg)
    active_global = np.broadcast_to(
        active_global_vector[None, :], phi_local.shape
    )
    active_total, _ = active_mask(phi_total, cfg)
    active_lg = active_local & active_global
    active_lt = active_local & active_total
    active_conflict = active_lg & (
        np.sign(phi_local) != np.sign(global_full)
    )
    active_total_local_flip = active_lt & (
        np.sign(phi_local) != np.sign(phi_total)
    )
    active_cancellation = active_conflict & (
        ratio <= float(cfg.get("cancellation_threshold", 0.1))
    )
    events = {
        "local_global_sign_conflict": (conflict, valid_lg),
        "total_local_sign_flip": (total_local_flip, valid_lt),
        "cancellation": (cancellation, valid_lg),
        "active1pct_local_global_sign_conflict": (
            active_conflict,
            active_lg,
        ),
        "active1pct_total_local_sign_flip": (
            active_total_local_flip,
            active_lt,
        ),
        "active1pct_cancellation": (
            active_cancellation,
            active_lg,
        ),
    }
    summary_rows, feature_rows = [], []
    for event, (mask, eligible) in events.items():
        eligible_count = int(eligible.sum())
        summary_rows.append(
            {
                "event": event,
                "cell_rate": float(mask.mean()),
                "row_any_rate": float(mask.any(axis=1).mean()),
                "event_count": int(mask.sum()),
                "n_cells": int(mask.size),
                "eligible_count": eligible_count,
                "eligible_fraction": float(eligible.mean()),
                "eligible_event_rate": (
                    float(mask.sum() / eligible_count)
                    if eligible_count
                    else float("nan")
                ),
                "sign_tolerance": tolerance,
            }
        )
        for feature in range(mask.shape[1]):
            feature_eligible_count = int(eligible[:, feature].sum())
            feature_rows.append(
                {
                    "event": event,
                    "feature": feature + 1,
                    "cell_rate": float(mask[:, feature].mean()),
                    "event_count": int(mask[:, feature].sum()),
                    "n_cells": int(mask.shape[0]),
                    "eligible_count": feature_eligible_count,
                    "eligible_fraction": float(
                        eligible[:, feature].mean()
                    ),
                    "eligible_event_rate": (
                        float(
                            mask[:, feature].sum()
                            / feature_eligible_count
                        )
                        if feature_eligible_count
                        else float("nan")
                    ),
                }
            )
    return summary_rows, feature_rows


def result_rows(cell: dict) -> list[dict]:
    metadata = cell["metadata"]
    rows = []
    for idx, baseline in enumerate(cell["baseline_names"]):
        rows.append(
            {
                "rho_x": metadata["rho_x"],
                "data_seed": metadata["data_seed"],
                "baseline": baseline,
                "global_runtime_sec": float(cell["global_runtime_sec"][idx]),
                "local_runtime_sec": float(cell["local_runtime_sec"][idx]),
                "local_runtime_sec_per_query": float(
                    cell["local_runtime_sec"][idx] / metadata["n_explain"]
                ),
                "efficiency_global_abs": float(
                    abs(cell["efficiency_global"][idx])
                ),
                "efficiency_local_max_abs": float(
                    np.max(np.abs(cell["efficiency_local"][idx]))
                ),
                **metadata,
            }
        )
    if "phi_global_marginal" in cell:
        rows.append(
            {
                "rho_x": metadata["rho_x"],
                "data_seed": metadata["data_seed"],
                "baseline": MARGINAL_BASELINE_NAME,
                "global_runtime_sec": float(
                    cell["global_runtime_sec_marginal"]
                ),
                "local_runtime_sec": float(
                    cell["local_runtime_sec_marginal"]
                ),
                "local_runtime_sec_per_query": float(
                    cell["local_runtime_sec_marginal"]
                    / metadata["n_explain"]
                ),
                "efficiency_global_abs": float(
                    abs(cell["efficiency_global_marginal"])
                ),
                "efficiency_local_max_abs": float(
                    np.max(np.abs(cell["efficiency_local_marginal"]))
                ),
                **metadata,
            }
        )
    if "phi_global_joint_marginal" in cell:
        rows.append(
            {
                "rho_x": metadata["rho_x"],
                "data_seed": metadata["data_seed"],
                "baseline": JOINT_MARGINAL_BASELINE_NAME,
                "global_runtime_sec": float(
                    cell["global_runtime_sec_joint_marginal"]
                ),
                "local_runtime_sec": float(
                    cell["local_runtime_sec_joint_marginal"]
                ),
                "local_runtime_sec_per_query": float(
                    cell["local_runtime_sec_joint_marginal"]
                    / metadata["n_explain"]
                ),
                "efficiency_global_abs": float(
                    abs(cell["efficiency_global_joint_marginal"])
                ),
                "efficiency_local_max_abs": float(
                    np.max(
                        np.abs(cell["efficiency_local_joint_marginal"])
                    )
                ),
                **metadata,
            }
        )
    return rows


def validate_raw_artifact(cell: dict, path: Path) -> None:
    required = {
        "X_full",
        "y_full",
        "train_indices",
        "calib_indices",
        "test_pool_indices",
        "explain_indices",
        "X_explain",
        "y_explain",
        "phi_global",
        "phi_local",
        "phi_total",
    }
    missing = sorted(required - set(cell))
    if missing:
        raise ValueError(f"{path}: missing raw artifact keys {missing}")

    X_full = np.asarray(cell["X_full"])
    y_full = np.asarray(cell["y_full"]).reshape(-1)
    explain = np.asarray(cell["explain_indices"], dtype=int)
    if X_full.shape != (len(y_full), int(cell["metadata"]["dimension"])):
        raise ValueError(f"{path}: X_full/y_full shape mismatch")
    if not np.array_equal(X_full[explain], cell["X_explain"]):
        raise ValueError(f"{path}: X_explain is not indexed from X_full")
    if not np.array_equal(y_full[explain], cell["y_explain"]):
        raise ValueError(f"{path}: y_explain is not indexed from y_full")

    split_parts = [
        np.asarray(cell["train_indices"], dtype=int),
        np.asarray(cell["calib_indices"], dtype=int),
        np.asarray(cell["test_pool_indices"], dtype=int),
    ]
    joined = np.concatenate(split_parts)
    if len(joined) != len(y_full) or len(np.unique(joined)) != len(y_full):
        raise ValueError(f"{path}: train/calibration/test split is not a partition")
    if not np.all(np.isin(explain, split_parts[-1])):
        raise ValueError(f"{path}: explanation rows are not held-out test rows")

    n_baselines = len(BASELINE_NAMES)
    n_explain = int(cell["metadata"]["n_explain"])
    dimension = int(cell["metadata"]["dimension"])
    expected = {
        "phi_global": (n_baselines, dimension),
        "phi_local": (n_baselines, n_explain, dimension),
        "phi_total": (n_baselines, n_explain, dimension),
    }
    for key, shape in expected.items():
        values = np.asarray(cell[key], dtype=float)
        if values.shape != shape:
            raise ValueError(f"{path}: {key} shape {values.shape} != {shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: {key} contains non-finite values")

    if int(cell["metadata"]["artifact_schema_version"]) >= 3:
        marginal_required = {
            "marginal_baseline_name",
            "marginal_mask_matrix",
            "marginal_sample_indices",
            "marginal_mask_digest",
            "phi_global_marginal",
            "phi_local_marginal",
            "phi_total_marginal",
            "global_runtime_sec_marginal",
            "local_runtime_sec_marginal",
            "efficiency_global_marginal",
            "efficiency_local_marginal",
            "endpoint_global_marginal",
            "endpoint_local_marginal",
        }
        marginal_missing = sorted(marginal_required - set(cell))
        if marginal_missing:
            raise ValueError(
                f"{path}: missing marginal artifact keys {marginal_missing}"
            )
        X_calib = X_full[split_parts[1]]
        sample_indices = np.asarray(
            cell["marginal_sample_indices"], dtype=int
        )
        marginal_matrix = np.asarray(
            cell["marginal_mask_matrix"], dtype=float
        )
        if sample_indices.shape != X_calib.shape:
            raise ValueError(f"{path}: marginal sample-index shape mismatch")
        if np.any(sample_indices < 0) or np.any(
            sample_indices >= len(X_calib)
        ):
            raise ValueError(f"{path}: marginal sample index out of bounds")
        expected_matrix = X_calib[
            sample_indices, np.arange(dimension)[None, :]
        ]
        if not np.array_equal(marginal_matrix, expected_matrix):
            raise ValueError(
                f"{path}: marginal mask matrix does not match sampled columns"
            )
        phi_marginal = np.asarray(
            cell["phi_global_marginal"], dtype=float
        )
        if phi_marginal.shape != (dimension,) or not np.isfinite(
            phi_marginal
        ).all():
            raise ValueError(f"{path}: invalid marginal global attribution")
        expected_marginal_shapes = {
            "phi_local_marginal": (n_explain, dimension),
            "phi_total_marginal": (n_explain, dimension),
            "efficiency_local_marginal": (n_explain,),
            "endpoint_local_marginal": (n_explain, 2),
        }
        for key, shape in expected_marginal_shapes.items():
            values = np.asarray(cell[key], dtype=float)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(
                    f"{path}: invalid {key}; expected finite shape {shape}"
                )
        if not np.allclose(
            cell["phi_total_marginal"],
            cell["phi_local_marginal"]
            + cell["phi_global_marginal"][None, :],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{path}: marginal decomposition mismatch")
    if int(cell["metadata"]["artifact_schema_version"]) >= 5:
        joint_required = {
            "joint_marginal_baseline_name",
            "joint_marginal_draw_count",
            "joint_marginal_donor_rows",
            "joint_marginal_donor_indices",
            "joint_marginal_donor_digest",
            "phi_global_joint_marginal",
            "phi_local_joint_marginal",
            "phi_total_joint_marginal",
            "global_runtime_sec_joint_marginal",
            "local_runtime_sec_joint_marginal",
            "efficiency_global_joint_marginal",
            "efficiency_local_joint_marginal",
            "endpoint_global_joint_marginal",
            "endpoint_local_joint_marginal",
        }
        joint_missing = sorted(joint_required - set(cell))
        if joint_missing:
            raise ValueError(
                f"{path}: missing joint-marginal keys {joint_missing}"
            )
        X_calib = X_full[split_parts[1]]
        donor_indices = np.asarray(
            cell["joint_marginal_donor_indices"], dtype=int
        )
        donor_rows = np.asarray(
            cell["joint_marginal_donor_rows"], dtype=float
        )
        if donor_indices.ndim != 1 or donor_rows.shape != (
            len(donor_indices),
            dimension,
        ):
            raise ValueError(f"{path}: joint donor shape mismatch")
        if np.any(donor_indices < 0) or np.any(
            donor_indices >= len(X_calib)
        ):
            raise ValueError(f"{path}: joint donor index out of bounds")
        if not np.array_equal(donor_rows, X_calib[donor_indices]):
            raise ValueError(f"{path}: joint donor rows are not intact rows")
        primary_draws = int(np.asarray(
            cell["joint_marginal_draw_count"]
        ).item())
        if primary_draws <= 0 or primary_draws > len(donor_rows):
            raise ValueError(f"{path}: invalid joint draw count")
        expected_joint_shapes = {
            "phi_global_joint_marginal": (dimension,),
            "phi_local_joint_marginal": (n_explain, dimension),
            "phi_total_joint_marginal": (n_explain, dimension),
            "efficiency_local_joint_marginal": (n_explain,),
            "endpoint_global_joint_marginal": (2,),
            "endpoint_local_joint_marginal": (n_explain, 2),
        }
        for key, shape in expected_joint_shapes.items():
            values = np.asarray(cell[key], dtype=float)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(
                    f"{path}: invalid {key}; expected finite shape {shape}"
                )
        if not np.allclose(
            cell["phi_total_joint_marginal"],
            cell["phi_local_joint_marginal"]
            + cell["phi_global_joint_marginal"][None, :],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{path}: joint-marginal decomposition mismatch")
        convergence_draws = cell["metadata"].get(
            "joint_marginal_convergence_draws"
        )
        if convergence_draws is not None:
            convergence_required = {
                "joint_convergence_draw_count",
                "phi_global_joint_marginal_convergence",
                "global_runtime_sec_joint_marginal_convergence",
                "efficiency_global_joint_marginal_convergence",
                "endpoint_global_joint_marginal_convergence",
            }
            missing_convergence = sorted(
                convergence_required - set(cell)
            )
            if missing_convergence:
                raise ValueError(
                    f"{path}: missing convergence keys "
                    f"{missing_convergence}"
                )
            if int(cell["joint_convergence_draw_count"]) != int(
                convergence_draws
            ):
                raise ValueError(f"{path}: convergence draw-count mismatch")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_artifact_manifest(cfg: dict, cells_dir: Path) -> pd.DataFrame:
    rows = []
    for seed in cfg["data_seeds"]:
        for rho in cfg["correlations"]:
            path = cells_dir / f"cell_seed{int(seed)}_{rho_label(float(rho))}.npz"
            cell = load_cell(path)
            validate_raw_artifact(cell, path)
            rows.append(
                {
                    "data_seed": int(seed),
                    "rho_x": float(rho),
                    "file": path.name,
                    "file_size_bytes": int(path.stat().st_size),
                    "file_sha256": file_sha256(path),
                    "artifact_schema_version": int(
                        cell["metadata"]["artifact_schema_version"]
                    ),
                    "X_full_shape": str(tuple(cell["X_full"].shape)),
                    "y_full_shape": str(tuple(cell["y_full"].shape)),
                    "X_explain_shape": str(tuple(cell["X_explain"].shape)),
                    "phi_global_shape": str(tuple(cell["phi_global"].shape)),
                    "phi_local_shape": str(tuple(cell["phi_local"].shape)),
                    "phi_total_shape": str(tuple(cell["phi_total"].shape)),
                    "phi_global_marginal_shape": str(
                        tuple(cell["phi_global_marginal"].shape)
                    ),
                    "phi_local_marginal_shape": str(
                        tuple(cell["phi_local_marginal"].shape)
                    ),
                    "phi_total_marginal_shape": str(
                        tuple(cell["phi_total_marginal"].shape)
                    ),
                    "marginal_mask_matrix_shape": str(
                        tuple(cell["marginal_mask_matrix"].shape)
                    ),
                    "marginal_mask_digest": str(
                        np.asarray(cell["marginal_mask_digest"]).item()
                    ),
                    "joint_marginal_draw_count": int(
                        cell["joint_marginal_draw_count"]
                    ),
                    "joint_marginal_donor_rows_shape": str(
                        tuple(cell["joint_marginal_donor_rows"].shape)
                    ),
                    "phi_global_joint_marginal_shape": str(
                        tuple(cell["phi_global_joint_marginal"].shape)
                    ),
                    "phi_local_joint_marginal_shape": str(
                        tuple(cell["phi_local_joint_marginal"].shape)
                    ),
                    "phi_total_joint_marginal_shape": str(
                        tuple(cell["phi_total_joint_marginal"].shape)
                    ),
                    "joint_marginal_donor_digest": str(
                        np.asarray(
                            cell["joint_marginal_donor_digest"]
                        ).item()
                    ),
                    "joint_convergence_draw_count": (
                        int(cell["joint_convergence_draw_count"])
                        if "joint_convergence_draw_count" in cell
                        else ""
                    ),
                    "data_digest": str(cell["data_digest"]),
                    "split_digest": cell["metadata"]["split_digest"],
                    "prediction_digest": cell["metadata"][
                        "prediction_digest"
                    ],
                    "raw_draw_digest": cell["metadata"]["raw_draw_digest"],
                }
            )
    return pd.DataFrame(rows).sort_values(["data_seed", "rho_x"])


def aggregate_cells(cfg: dict, cells_dir: Path) -> tuple[pd.DataFrame, ...]:
    agreement_rows = []
    feature_agreement_rows = []
    event_rows = []
    feature_event_rows = []
    metadata_rows = []

    for seed in cfg["data_seeds"]:
        raw_digest = None
        for rho in cfg["correlations"]:
            path = cells_dir / f"cell_seed{int(seed)}_{rho_label(float(rho))}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"missing required cell: {path}")
            cell = load_cell(path)
            validate_raw_artifact(cell, path)
            metadata = cell["metadata"]
            if not np.isclose(float(metadata["rho_x"]), float(rho)):
                raise ValueError(f"{path}: rho metadata mismatch")
            if int(metadata["data_seed"]) != int(seed):
                raise ValueError(f"{path}: seed metadata mismatch")
            if raw_digest is None:
                raw_digest = metadata["raw_draw_digest"]
            elif raw_digest != metadata["raw_draw_digest"]:
                raise ValueError(
                    f"seed={seed}: correlation modes do not share raw RNG draws"
                )
            metadata_rows.extend(result_rows(cell))

            names = cell["baseline_names"]
            for left_name, right_name in combinations(names, 2):
                left_idx, right_idx = names.index(left_name), names.index(right_name)
                for component, key in COMPONENTS.items():
                    left, right = cell[key][left_idx], cell[key][right_idx]
                    agreement_rows.append(
                        {
                            "rho_x": float(rho),
                            "data_seed": int(seed),
                            "baseline_a": left_name,
                            "baseline_b": right_name,
                            "reference_comparison": left_name == REFERENCE,
                            "component": component,
                            **pair_stats(left, right, cfg),
                        }
                    )
                    for row in feature_stats(left, right, cfg):
                        feature_agreement_rows.append(
                            {
                                "rho_x": float(rho),
                                "data_seed": int(seed),
                                "baseline_a": left_name,
                                "baseline_b": right_name,
                                "reference_comparison": left_name == REFERENCE,
                                "component": component,
                                **row,
                            }
                        )

            reference_idx = names.index(REFERENCE)
            distribution_components = {
                MARGINAL_BASELINE_NAME: {
                    "phiG": np.asarray(
                        cell["phi_global_marginal"], dtype=float
                    ),
                    "phiL": np.asarray(
                        cell["phi_local_marginal"], dtype=float
                    ),
                    "phiT": np.asarray(
                        cell["phi_total_marginal"], dtype=float
                    ),
                },
                JOINT_MARGINAL_BASELINE_NAME: {
                    "phiG": np.asarray(
                        cell["phi_global_joint_marginal"], dtype=float
                    ),
                    "phiL": np.asarray(
                        cell["phi_local_joint_marginal"], dtype=float
                    ),
                    "phiT": np.asarray(
                        cell["phi_total_joint_marginal"], dtype=float
                    ),
                },
            }
            for distribution_name, components in (
                distribution_components.items()
            ):
                for component, distribution_values in components.items():
                    reference_values = np.asarray(
                        cell[COMPONENTS[component]][reference_idx],
                        dtype=float,
                    )
                    agreement_rows.append(
                        {
                            "rho_x": float(rho),
                            "data_seed": int(seed),
                            "baseline_a": REFERENCE,
                            "baseline_b": distribution_name,
                            "reference_comparison": True,
                            "component": component,
                            **pair_stats(
                                reference_values, distribution_values, cfg
                            ),
                        }
                    )
                    for row in feature_stats(
                        reference_values, distribution_values, cfg
                    ):
                        feature_agreement_rows.append(
                            {
                                "rho_x": float(rho),
                                "data_seed": int(seed),
                                "baseline_a": REFERENCE,
                                "baseline_b": distribution_name,
                                "reference_comparison": True,
                                "component": component,
                                **row,
                            }
                        )

            for baseline_idx, baseline in enumerate(names):
                summaries, features = component_event_rows(
                    cell["phi_global"][baseline_idx],
                    cell["phi_local"][baseline_idx],
                    cell["phi_total"][baseline_idx],
                    cfg,
                )
                for row in summaries:
                    event_rows.append(
                        {
                            "rho_x": float(rho),
                            "data_seed": int(seed),
                            "baseline": baseline,
                            **row,
                        }
                    )
                for row in features:
                    feature_event_rows.append(
                        {
                            "rho_x": float(rho),
                            "data_seed": int(seed),
                            "baseline": baseline,
                            **row,
                        }
                    )
            for distribution_name, components in (
                distribution_components.items()
            ):
                summaries, features = component_event_rows(
                    components["phiG"],
                    components["phiL"],
                    components["phiT"],
                    cfg,
                )
                for row in summaries:
                    event_rows.append(
                        {
                            "rho_x": float(rho),
                            "data_seed": int(seed),
                            "baseline": distribution_name,
                            **row,
                        }
                    )
                for row in features:
                    feature_event_rows.append(
                        {
                            "rho_x": float(rho),
                            "data_seed": int(seed),
                            "baseline": distribution_name,
                            **row,
                        }
                    )

    return (
        pd.DataFrame(metadata_rows),
        pd.DataFrame(agreement_rows),
        pd.DataFrame(feature_agreement_rows),
        pd.DataFrame(event_rows),
        pd.DataFrame(feature_event_rows),
    )


def make_summary(
    agreement: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    reference = agreement[agreement["reference_comparison"]].copy()
    for keys, frame in reference.groupby(["rho_x", "baseline_b", "component"]):
        rho, baseline, component = keys
        row = {
            "rho_x": rho,
            "baseline": baseline,
            "component": component,
        }
        for metric in [
            "pearson",
            "spearman",
            "mean_absolute_difference",
            "topk_overlap",
            "signed_spearman",
            "abs_rank_spearman",
            "top3_overlap",
            "top5_overlap",
            "symmetric_normalized_l1",
            "active1pct_sign_agreement",
            "active1pct_eligible_fraction",
            "query_pearson_median",
            "query_pearson_min",
            "query_signed_spearman_median",
            "query_signed_spearman_min",
        ]:
            row[metric] = float(frame[metric].mean())
            row[f"{metric}_sd"] = float(frame[metric].std())
        for event_name in [
            "local_global_sign_conflict",
            "total_local_sign_flip",
            "cancellation",
            "active1pct_local_global_sign_conflict",
            "active1pct_total_local_sign_flip",
            "active1pct_cancellation",
        ]:
            for prefix, event_baseline in [
                ("", baseline),
                ("reference_", REFERENCE),
            ]:
                event_frame = events[
                    np.isclose(events["rho_x"], rho)
                    & (events["baseline"] == event_baseline)
                    & (events["event"] == event_name)
                ]
                for metric in [
                    "cell_rate",
                    "eligible_fraction",
                    "eligible_event_rate",
                ]:
                    column = f"{prefix}{event_name}_{metric}"
                    row[column] = float(event_frame[metric].mean())
                    row[f"{column}_sd"] = float(event_frame[metric].std())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["component", "baseline", "rho_x"])


def make_joint_convergence(cfg: dict, cells_dir: Path) -> pd.DataFrame:
    convergence_cfg = cfg.get("joint_marginal_convergence")
    if not convergence_cfg:
        return pd.DataFrame()
    seed = int(convergence_cfg["seed"])
    rho = float(convergence_cfg["rho"])
    path = cells_dir / f"cell_seed{seed}_{rho_label(rho)}.npz"
    cell = load_cell(path)
    validate_raw_artifact(cell, path)
    primary = np.asarray(cell["phi_global_joint_marginal"], dtype=float)
    extended = np.asarray(
        cell["phi_global_joint_marginal_convergence"], dtype=float
    )
    return pd.DataFrame(
        [
            {
                "data_seed": seed,
                "rho_x": rho,
                "primary_draws": int(cell["joint_marginal_draw_count"]),
                "extended_draws": int(cell["joint_convergence_draw_count"]),
                "max_absolute_difference": float(
                    np.max(np.abs(primary - extended))
                ),
                **pair_stats(primary, extended, cfg),
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config_full.json",
    )
    parser.add_argument(
        "--cells-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "cells",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results, agreement, feature_agreement, events, feature_events = (
        aggregate_cells(cfg, args.cells_dir)
    )
    summary = make_summary(agreement, events)
    manifest = make_artifact_manifest(cfg, args.cells_dir)
    joint_convergence = make_joint_convergence(cfg, args.cells_dir)

    results.to_csv(args.out_dir / "results.csv", index=False)
    agreement.to_csv(args.out_dir / "baseline_agreement.csv", index=False)
    feature_agreement.to_csv(
        args.out_dir / "feature_agreement.csv", index=False
    )
    events.to_csv(args.out_dir / "component_events.csv", index=False)
    feature_events.to_csv(
        args.out_dir / "feature_component_events.csv", index=False
    )
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    manifest.to_csv(args.out_dir / "artifacts_manifest.csv", index=False)
    joint_convergence.to_csv(
        args.out_dir / "joint_marginal_convergence.csv", index=False
    )
    print(
        f"[aggregate] wrote baseline-sensitivity outputs under {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
