"""Strict aggregation for exact-vs-sampled multicollinearity cells (Table 5, Appendix B.6)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

try:
    from .run_cell import rho_label
except ImportError:  # Direct script execution.
    from run_cell import rho_label


DEFAULT_ACTIVE_SIGN_FRACTION = 0.01
DEFAULT_TOP_K = 3


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {key: z[key] for key in z.files}
    out["metadata"] = json.loads(str(out.pop("metadata_json")))
    return out


def safe_corr(a, b, method: str) -> tuple[float, int]:
    x = np.asarray(a).ravel()
    y = np.asarray(b).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or np.ptp(x) <= 1e-15 or np.ptp(y) <= 1e-15:
        return float("nan"), int(len(x))
    fn = pearsonr if method == "pearson" else spearmanr
    return float(fn(x, y).statistic), int(len(x))


def zero_tol(exact, cfg) -> float:
    scale = float(np.max(np.abs(exact))) if np.size(exact) else 0.0
    return max(
        float(cfg.get("absolute_zero_floor", 1e-9)),
        float(cfg.get("relative_zero_threshold", 1e-6)) * scale,
    )


def topk_overlap(exact, approx, k: int) -> float:
    a = np.asarray(exact, dtype=float).ravel()
    b = np.asarray(approx, dtype=float).ravel()
    k = min(int(k), len(a))
    if k <= 0 or len(a) != len(b):
        return float("nan")
    exact_top = set(np.argsort(np.abs(a), kind="stable")[-k:])
    approx_top = set(np.argsort(np.abs(b), kind="stable")[-k:])
    return float(len(exact_top & approx_top) / k)


def active_sign_mask(exact, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Return an exact-reference 1%-active mask and its scalar/row thresholds."""
    a = np.asarray(exact, dtype=float)
    fraction = float(
        cfg.get("active_sign_fraction", DEFAULT_ACTIVE_SIGN_FRACTION)
    )
    floor = float(cfg.get("absolute_zero_floor", 1e-9))
    if not 0.0 <= fraction < 1.0:
        raise ValueError("active_sign_fraction must lie in [0, 1)")
    if a.ndim == 1:
        scale = float(np.max(np.abs(a))) if a.size else 0.0
        thresholds = np.asarray(max(floor, fraction * scale))
    elif a.ndim == 2:
        scale = np.max(np.abs(a), axis=1, keepdims=True)
        thresholds = np.maximum(floor, fraction * scale)
    else:
        raise ValueError("Attribution arrays must be one- or two-dimensional")
    return np.isfinite(a) & (np.abs(a) >= thresholds), thresholds


def normalized_mae(exact, approx) -> tuple[float, float]:
    a = np.asarray(exact, dtype=float)
    b = np.asarray(approx, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        return float("nan"), float("nan")
    exact_mean_abs = float(np.mean(np.abs(a[valid])))
    mae = float(np.mean(np.abs(b[valid] - a[valid])))
    value = mae / exact_mean_abs if exact_mean_abs > 0.0 else float("nan")
    return value, exact_mean_abs


def numerical_stats(exact, approx, prefix: str, cfg) -> dict:
    a = np.asarray(exact, dtype=float)
    b = np.asarray(approx, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for {prefix}: {a.shape} vs {b.shape}")
    diff = b - a
    tol = zero_tol(a, cfg)
    pearson, pearson_n = safe_corr(a, b, "pearson")
    spearman, spearman_n = safe_corr(a, b, "spearman")
    abs_spearman, abs_spearman_n = safe_corr(np.abs(a), np.abs(b), "spearman")
    norm_mae, exact_mean_abs = normalized_mae(a, b)
    valid_sign = (np.abs(a) > tol) & (np.abs(b) > tol)
    sign_agreement = (
        float(np.mean(np.sign(a[valid_sign]) == np.sign(b[valid_sign])))
        if valid_sign.any() else float("nan")
    )
    active, active_thresholds = active_sign_mask(a, cfg)
    active &= np.isfinite(b)
    active_sign_agreement = (
        float(np.mean(np.sign(a[active]) == np.sign(b[active])))
        if active.any() else float("nan")
    )
    rel = np.abs(diff) / np.maximum(np.abs(a), tol)
    row = {
        f"{prefix}_pearson": pearson,
        f"{prefix}_pearson_n": pearson_n,
        f"{prefix}_spearman": spearman,
        f"{prefix}_spearman_n": spearman_n,
        f"{prefix}_abs_spearman": abs_spearman,
        f"{prefix}_abs_spearman_n": abs_spearman_n,
        f"{prefix}_mae": float(np.mean(np.abs(diff))),
        f"{prefix}_normalized_mae": norm_mae,
        f"{prefix}_exact_mean_abs": exact_mean_abs,
        f"{prefix}_rmse": float(np.sqrt(np.mean(diff ** 2))),
        f"{prefix}_max_abs_diff": float(np.max(np.abs(diff))),
        f"{prefix}_max_relative_diff": float(np.max(rel)),
        f"{prefix}_sign_agreement": sign_agreement,
        f"{prefix}_sign_valid_n": int(valid_sign.sum()),
        f"{prefix}_zero_tol": tol,
        f"{prefix}_active1pct_sign_agreement": active_sign_agreement,
        f"{prefix}_active1pct_fraction": float(active.mean()),
        f"{prefix}_active1pct_n": int(active.sum()),
        f"{prefix}_active1pct_threshold_min": float(np.min(active_thresholds)),
        f"{prefix}_active1pct_threshold_max": float(np.max(active_thresholds)),
        f"{prefix}_top3_overlap": topk_overlap(a, b, DEFAULT_TOP_K)
        if a.ndim == 1 else float("nan"),
    }
    if a.ndim == 1:
        row[f"{prefix}_top5_overlap"] = topk_overlap(a, b, 5)
    else:
        top3, top5 = [], []
        query_r, query_s, query_abs_s, query_norm_mae = [], [], [], []
        query_active_sign, query_active_fraction = [], []
        for ae, be in zip(a, b):
            top3.append(topk_overlap(ae, be, DEFAULT_TOP_K))
            top5.append(topk_overlap(ae, be, 5))
            query_r.append(safe_corr(ae, be, "pearson")[0])
            query_s.append(safe_corr(ae, be, "spearman")[0])
            query_abs_s.append(
                safe_corr(np.abs(ae), np.abs(be), "spearman")[0]
            )
            query_norm_mae.append(normalized_mae(ae, be)[0])
            query_active, _ = active_sign_mask(ae, cfg)
            query_active &= np.isfinite(be)
            query_active_fraction.append(float(query_active.mean()))
            query_active_sign.append(
                float(
                    np.mean(
                        np.sign(ae[query_active]) == np.sign(be[query_active])
                    )
                )
                if query_active.any() else float("nan")
            )
        finite_r = np.asarray(query_r)[np.isfinite(query_r)]
        finite_s = np.asarray(query_s)[np.isfinite(query_s)]
        finite_abs_s = np.asarray(query_abs_s)[np.isfinite(query_abs_s)]
        finite_norm = np.asarray(query_norm_mae)[np.isfinite(query_norm_mae)]
        finite_active_sign = np.asarray(query_active_sign)[
            np.isfinite(query_active_sign)
        ]
        row[f"{prefix}_top3_overlap"] = float(np.mean(top3))
        row[f"{prefix}_top3_overlap_median_query"] = float(np.median(top3))
        row[f"{prefix}_top3_overlap_min_query"] = float(np.min(top3))
        row[f"{prefix}_top5_overlap"] = float(np.mean(top5))
        row[f"{prefix}_query_pearson_median"] = (
            float(np.median(finite_r)) if finite_r.size else float("nan")
        )
        row[f"{prefix}_query_pearson_mean"] = (
            float(np.mean(finite_r)) if finite_r.size else float("nan")
        )
        row[f"{prefix}_query_pearson_min"] = (
            float(np.min(finite_r)) if finite_r.size else float("nan")
        )
        row[f"{prefix}_query_pearson_valid_n"] = int(finite_r.size)
        row[f"{prefix}_query_spearman_median"] = (
            float(np.median(finite_s)) if finite_s.size else float("nan")
        )
        row[f"{prefix}_query_spearman_mean"] = (
            float(np.mean(finite_s)) if finite_s.size else float("nan")
        )
        row[f"{prefix}_query_spearman_min"] = (
            float(np.min(finite_s)) if finite_s.size else float("nan")
        )
        row[f"{prefix}_query_abs_spearman_median"] = (
            float(np.median(finite_abs_s))
            if finite_abs_s.size else float("nan")
        )
        row[f"{prefix}_query_abs_spearman_mean"] = (
            float(np.mean(finite_abs_s))
            if finite_abs_s.size else float("nan")
        )
        row[f"{prefix}_query_abs_spearman_min"] = (
            float(np.min(finite_abs_s))
            if finite_abs_s.size else float("nan")
        )
        row[f"{prefix}_query_normalized_mae_median"] = (
            float(np.median(finite_norm)) if finite_norm.size else float("nan")
        )
        row[f"{prefix}_query_normalized_mae_p90"] = (
            float(np.quantile(finite_norm, 0.9))
            if finite_norm.size else float("nan")
        )
        row[f"{prefix}_query_normalized_mae_max"] = (
            float(np.max(finite_norm)) if finite_norm.size else float("nan")
        )
        row[f"{prefix}_query_active1pct_sign_agreement_median"] = (
            float(np.median(finite_active_sign))
            if finite_active_sign.size else float("nan")
        )
        row[f"{prefix}_query_active1pct_sign_agreement_min"] = (
            float(np.min(finite_active_sign))
            if finite_active_sign.size else float("nan")
        )
        row[f"{prefix}_query_active1pct_fraction_mean"] = float(
            np.mean(query_active_fraction)
        )
    return row


def component_events(phi_l, phi_g, tol_l, tol_g, cancel_threshold):
    local = np.asarray(phi_l, dtype=float)
    global_ = np.broadcast_to(np.asarray(phi_g, dtype=float), local.shape)
    eligible = (np.abs(local) > tol_l) & (np.abs(global_) > tol_g)
    conflict = eligible & (np.sign(local) != np.sign(global_))
    ratio = np.abs(local + global_) / (
        np.abs(local) + np.abs(global_) + np.finfo(float).eps
    )
    cancellation = conflict & (ratio <= cancel_threshold)
    return conflict, cancellation, ratio, eligible


def event_comparison(exact_event, approx_event, event_name, rho, repeat,
                     threshold, exact_ratio=None) -> tuple[dict, list[dict]]:
    e = np.asarray(exact_event, dtype=bool)
    a = np.asarray(approx_event, dtype=bool)
    tp = int(np.sum(e & a))
    fn = int(np.sum(e & ~a))
    fp = int(np.sum(~e & a))
    tn = int(np.sum(~e & ~a))
    e_n, a_n, union = int(e.sum()), int(a.sum()), int((e | a).sum())
    precision = tp / a_n if a_n else float("nan")
    recall = tp / e_n if e_n else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall
        else float("nan")
    )
    row = {
        "rho_x": rho, "sampler_repeat": repeat, "event": event_name,
        "cancellation_threshold": threshold,
        "n_cells": int(e.size), "exact_count": e_n, "approx_count": a_n,
        "intersection_count": tp, "union_count": union,
        "exact_only_count": fn,
        "approx_only_count": fp, "true_negative_count": tn,
        "exact_cell_rate": float(e.mean()), "approx_cell_rate": float(a.mean()),
        "exact_row_any_rate": float(e.any(axis=1).mean()),
        "approx_row_any_rate": float(a.any(axis=1).mean()),
        "agreement": (tp + tn) / e.size,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
        "false_negative_rate": fn / (fn + tp) if fn + tp else float("nan"),
        "jaccard": tp / union if union else float("nan"),
    }
    if exact_ratio is not None:
        values = np.asarray(exact_ratio)[np.isfinite(exact_ratio)]
        row["exact_ratio_median"] = (
            float(np.median(values)) if values.size else float("nan")
        )
        row["exact_ratio_p10"] = (
            float(np.quantile(values, 0.1)) if values.size else float("nan")
        )
    features = []
    for j in range(e.shape[1]):
        ej, aj = e[:, j], a[:, j]
        inter = int(np.sum(ej & aj))
        e_count, a_count = int(ej.sum()), int(aj.sum())
        u_count = int((ej | aj).sum())
        features.append({
            "rho_x": rho, "sampler_repeat": repeat, "event": event_name,
            "cancellation_threshold": threshold, "feature": j + 1,
            "n_cells": int(ej.size),
            "exact_count": e_count, "approx_count": a_count,
            "exact_rate": float(ej.mean()), "approx_rate": float(aj.mean()),
            "intersection_count": inter, "union_count": u_count,
            "exact_only_count": e_count - inter,
            "approx_only_count": a_count - inter,
            "recall": inter / e_count if e_count else float("nan"),
            "precision": inter / a_count if a_count else float("nan"),
            "jaccard": inter / u_count if u_count else float("nan"),
        })
    return row, features


def distribution_stats(values, higher_is_better: bool | None = None) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        keys = ["mean", "sd", "median", "p10", "p90", "min", "max", "worst"]
        return {key: float("nan") for key in keys} | {"n": 0}
    stats = {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.1)),
        "p90": float(np.quantile(x, 0.9)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
    if higher_is_better is None:
        stats["worst"] = float("nan")
    else:
        stats["worst"] = stats["min"] if higher_is_better else stats["max"]
    return stats


def add_distribution(
    row: dict, frame: pd.DataFrame, column: str, output_name: str | None = None,
    higher_is_better: bool | None = None,
) -> None:
    name = output_name or column
    for stat, value in distribution_stats(
        frame[column], higher_is_better=higher_is_better
    ).items():
        if stat == "mean":
            row[name] = value
        else:
            row[f"{name}_{stat}"] = value


def event_count_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["rho_x", "event", "cancellation_threshold"]
    for keys, frame in events.groupby(group_cols, sort=True, dropna=False):
        rho, event, threshold = keys
        row = {
            "rho_x": float(rho),
            "event": event,
            "cancellation_threshold": float(threshold),
            "n_cells": int(frame["n_cells"].iloc[0]),
            "sampler_repeats": int(len(frame)),
            "exact_count": int(frame["exact_count"].iloc[0]),
        }
        for column in [
            "approx_count",
            "intersection_count",
            "union_count",
            "exact_only_count",
            "approx_only_count",
        ]:
            add_distribution(
                row, frame, column, higher_is_better=None
            )
        for column in ["precision", "recall", "jaccard"]:
            add_distribution(
                row, frame, column, higher_is_better=True
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config_full.json"))
    parser.add_argument(
        "--cells-dir", default=str(Path(__file__).parent / "results" / "cells")
    )
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).parent / "results")
    )
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as f:
        cfg = json.load(f)
    cells_dir, out_dir = Path(args.cells_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agreement_rows, event_rows, feature_rows, result_rows, exact_rate_rows = [], [], [], [], []
    for rho in cfg["correlations"]:
        label = rho_label(float(rho))
        exact_path = cells_dir / f"exact_{label}.npz"
        sample_paths = [
            cells_dir / f"sampled_{label}_repeat{r}.npz"
            for r in range(int(cfg["sampler_repeats"]))
        ]
        missing = [p for p in [exact_path, *sample_paths] if not p.is_file()]
        if missing:
            raise SystemExit("Missing required cells:\n" + "\n".join(map(str, missing)))
        exact = load_npz(exact_path)
        tol_l, tol_g = zero_tol(exact["phi_local"], cfg), zero_tol(exact["phi_global"], cfg)
        exact_sign, _, exact_ratio, _ = component_events(
            exact["phi_local"], exact["phi_global"], tol_l, tol_g, 0.1
        )
        exact_rate_rows.append({
            "rho_x": rho, "event": "component_sign_conflict",
            "cancellation_threshold": 0.0, "cell_rate": float(exact_sign.mean()),
            "row_any_rate": float(exact_sign.any(axis=1).mean()),
            "event_count": int(exact_sign.sum()),
        })
        for threshold in cfg["cancellation_thresholds"]:
            _, exact_cancel, _, _ = component_events(
                exact["phi_local"], exact["phi_global"], tol_l, tol_g, threshold
            )
            exact_rate_rows.append({
                "rho_x": rho, "event": "cancellation",
                "cancellation_threshold": threshold,
                "cell_rate": float(exact_cancel.mean()),
                "row_any_rate": float(exact_cancel.any(axis=1).mean()),
                "event_count": int(exact_cancel.sum()),
            })
        result_rows.append({
            "rho_x": rho, "method": "exact", "sampler_repeat": np.nan,
            "offline_sec": float(exact["offline_sec"]),
            "online_total_sec": float(exact["online_total_sec"]),
            "online_per_query_sec": float(exact["online_total_sec"]) / int(cfg["n_test"]),
            "eff_err_global": float(exact["eff_err_global"]),
            "eff_err_local_max": float(np.max(np.abs(exact["eff_err_local"]))),
            **exact["metadata"],
        })

        for repeat, sample_path in enumerate(sample_paths):
            sampled = load_npz(sample_path)
            if not np.array_equal(exact["test_indices"], sampled["test_indices"]):
                raise RuntimeError(f"Pair mismatch: {exact_path} vs {sample_path}")
            row = {"rho_x": rho, "sampler_repeat": repeat}
            for name, key in [
                ("phiG", "phi_global"), ("phiL", "phi_local"), ("phiT", "phi_total")
            ]:
                row.update(numerical_stats(exact[key], sampled[key], name, cfg))
            agreement_rows.append(row)
            result_rows.append({
                "rho_x": rho, "method": "sampled", "sampler_repeat": repeat,
                "sampler_seed": int(sampled["sampler_seed"]),
                "offline_sec": float(sampled["offline_sec"]),
                "online_total_sec": float(sampled["online_total_sec"]),
                "online_per_query_sec": float(sampled["online_total_sec"]) / int(cfg["n_test"]),
                "eff_err_global": float(sampled["eff_err_global"]),
                "eff_err_local_max": float(np.max(np.abs(sampled["eff_err_local"]))),
                **sampled["metadata"],
            })
            approx_sign, _, _, _ = component_events(
                sampled["phi_local"], sampled["phi_global"], tol_l, tol_g, 0.1
            )
            event_row, features = event_comparison(
                exact_sign, approx_sign, "component_sign_conflict", rho, repeat, 0.0
            )
            event_rows.append(event_row)
            feature_rows.extend(features)
            for threshold in cfg["cancellation_thresholds"]:
                _, exact_cancel, ratio, _ = component_events(
                    exact["phi_local"], exact["phi_global"], tol_l, tol_g, threshold
                )
                _, approx_cancel, _, _ = component_events(
                    sampled["phi_local"], sampled["phi_global"], tol_l, tol_g, threshold
                )
                event_row, features = event_comparison(
                    exact_cancel, approx_cancel, "cancellation", rho, repeat,
                    threshold, exact_ratio=ratio,
                )
                event_rows.append(event_row)
                feature_rows.extend(features)

    agreement = pd.DataFrame(agreement_rows).sort_values(["rho_x", "sampler_repeat"])
    events = pd.DataFrame(event_rows).sort_values(
        ["event", "cancellation_threshold", "rho_x", "sampler_repeat"]
    )
    exact_rates = pd.DataFrame(exact_rate_rows).sort_values(
        ["event", "cancellation_threshold", "rho_x"]
    )
    pd.DataFrame(result_rows).to_csv(out_dir / "results.csv", index=False)
    agreement.to_csv(out_dir / "agreement.csv", index=False)
    events.to_csv(out_dir / "event_agreement.csv", index=False)
    exact_rates.to_csv(out_dir / "exact_event_rates.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(out_dir / "feature_event_agreement.csv", index=False)
    event_counts = event_count_summary(events)
    event_counts.to_csv(out_dir / "event_overlap_counts.csv", index=False)

    summary_rows = []
    for rho in cfg["correlations"]:
        agr = agreement[np.isclose(agreement["rho_x"], rho)]
        sign = events[
            np.isclose(events["rho_x"], rho) &
            (events["event"] == "component_sign_conflict")
        ]
        cancel = events[
            np.isclose(events["rho_x"], rho) &
            (events["event"] == "cancellation") &
            np.isclose(events["cancellation_threshold"], 0.1)
        ]
        row = {"rho_x": rho}
        metric_specs = []
        for component in ["phiG", "phiL", "phiT"]:
            metric_specs.extend([
                (f"{component}_pearson", True),
                (f"{component}_spearman", True),
                (f"{component}_abs_spearman", True),
                (f"{component}_mae", False),
                (f"{component}_normalized_mae", False),
                (f"{component}_exact_mean_abs", None),
                (f"{component}_rmse", False),
                (f"{component}_max_abs_diff", False),
                (f"{component}_sign_agreement", True),
                (f"{component}_active1pct_sign_agreement", True),
                (f"{component}_active1pct_fraction", None),
                (f"{component}_top3_overlap", True),
                (f"{component}_top5_overlap", True),
            ])
        for component in ["phiL", "phiT"]:
            metric_specs.extend([
                (f"{component}_query_pearson_mean", True),
                (f"{component}_query_pearson_median", True),
                (f"{component}_query_pearson_min", True),
                (f"{component}_query_spearman_mean", True),
                (f"{component}_query_spearman_median", True),
                (f"{component}_query_spearman_min", True),
                (f"{component}_query_abs_spearman_mean", True),
                (f"{component}_query_abs_spearman_median", True),
                (f"{component}_query_abs_spearman_min", True),
                (f"{component}_query_normalized_mae_median", False),
                (f"{component}_query_normalized_mae_p90", False),
                (f"{component}_query_normalized_mae_max", False),
                (f"{component}_top3_overlap_median_query", True),
                (f"{component}_top3_overlap_min_query", True),
                (
                    f"{component}_query_active1pct_sign_agreement_median",
                    True,
                ),
                (
                    f"{component}_query_active1pct_sign_agreement_min",
                    True,
                ),
            ])
        for column, direction in metric_specs:
            add_distribution(
                row, agr, column, higher_is_better=direction
            )
        for prefix, frame in [("sign", sign), ("cancel", cancel)]:
            event_metric_specs = [
                ("exact_cell_rate", None),
                ("approx_cell_rate", None),
                ("exact_row_any_rate", None),
                ("approx_row_any_rate", None),
                ("exact_count", None),
                ("approx_count", None),
                ("intersection_count", None),
                ("union_count", None),
                ("exact_only_count", None),
                ("approx_only_count", None),
                ("recall", True),
                ("precision", True),
                ("f1", True),
                ("jaccard", True),
            ]
            for column, direction in event_metric_specs:
                add_distribution(
                    row, frame, column, output_name=f"{prefix}_{column}",
                    higher_is_better=direction,
                )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"[aggregate] wrote outputs under {out_dir} from {len(agreement)} pairs")


if __name__ == "__main__":
    main()
