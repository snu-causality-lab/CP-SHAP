"""
Plug-in comparison: LIME and occlusion versus Shapley (Table 6, Appendix D).

This script keeps the same CP uncertainty target used in the paper,

    width(x) = q_hi(x) - q_lo(x),

and swaps the local attribution operator:

    1. exact local Shapley                    (reference used in the paper)
    2. LIME-style local weighted surrogate    (non-Shapley)
    3. occlusion sensitivity                  (non-Shapley)

The experiment reports runtime, local-surrogate fidelity, and sanity agreement
with exact Shapley rankings/signs. Agreement is not treated as a theorem or a
replacement guarantee; it simply verifies that the plug-in produces compatible
feature-level signed scores for the same downstream interface.

Outputs:
    results.csv       one row per setting × seed × method
    per_instance.csv  row-level agreement diagnostics for non-Shapley methods
    component_results.csv  local/global/total agreement diagnostics
    component_summary.csv  mean/std of component diagnostics across seeds
    summary.csv       mean/std over seeds
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.conformal.cq import calibration_width_adjustment_from_models  # noqa: E402
from src.conformal.quantile import QuantileModelConfig, fit_quantile_models  # noqa: E402
from data.synthetic.synthetic_heteroscedastic import generate_heteroscedastic_data  # noqa: E402
from data.synthetic.synthetic_homoscedastic import generate_homoscedastic_data  # noqa: E402
from data.synthetic.synthetic_source_separation import generate_source_separation_data  # noqa: E402
from src.shapley.shapley_global import compute_global_shapley  # noqa: E402
from src.shapley.shapley_local import shapley_local_single  # noqa: E402


METHOD_ORDER = ["shapley_exact", "lime_surrogate", "occlusion_sensitivity"]
METHOD_LABELS = {
    "shapley_exact": "Exact Shapley",
    "lime_surrogate": "LIME",
    "occlusion_sensitivity": "Occlusion",
}
COMPONENT_ORDER = ["local", "global", "total"]


def generate_data(setting: str, n_samples: int, seed: int):
    setting = setting.lower()
    if setting == "heteroscedastic":
        return generate_heteroscedastic_data(n_samples=n_samples, seed=seed)
    if setting == "homoscedastic":
        return generate_homoscedastic_data(n_samples=n_samples, seed=seed)
    if setting == "source_separation":
        return generate_source_separation_data(n=n_samples, seed=seed)
    raise ValueError(f"Unknown setting: {setting}")


def split_arrays(X: np.ndarray, y: np.ndarray, seed: int, n_calib: int, n_test: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = len(X) - n_calib - n_test
    if n_train <= 0:
        raise ValueError(
            f"Not enough samples: n={len(X)} but n_calib+n_test={n_calib + n_test}"
        )
    tr = idx[:n_train]
    ca = idx[n_train : n_train + n_calib]
    te = idx[n_train + n_calib : n_train + n_calib + n_test]
    return (X[tr], y[tr]), (X[ca], y[ca]), (X[te], y[te])


def interval_width(q_lo, q_hi, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return np.asarray(q_hi.predict(X), dtype=float) - np.asarray(q_lo.predict(X), dtype=float)


def impute_to_baseline(x0: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    return np.where(np.isnan(x0), baseline, x0)


def weighted_ridge_with_intercept(
    Z: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Small closed-form weighted ridge used to avoid an extra dependency."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    weight = np.asarray(weight, dtype=float).reshape(-1)

    X_design = np.column_stack([np.ones(len(Z)), Z])
    sw = np.sqrt(np.maximum(weight, 1e-12))
    Xw = X_design * sw[:, None]
    yw = y * sw

    reg = np.eye(X_design.shape[1]) * float(l2)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(Xw.T @ Xw + reg, Xw.T @ yw)
    pred = X_design @ coef
    return float(coef[0]), coef[1:], pred


def weighted_r2(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    weight = np.asarray(weight, dtype=float).reshape(-1)
    w_sum = float(np.sum(weight))
    if w_sum <= 0:
        return float("nan")
    y_bar = float(np.sum(weight * y) / w_sum)
    sst = float(np.sum(weight * (y - y_bar) ** 2))
    if sst <= 1e-12:
        return float("nan")
    sse = float(np.sum(weight * (y - pred) ** 2))
    return 1.0 - sse / sst


def binary_coalition_design(
    d: int,
    n_samples: int,
    rng: np.random.Generator,
    p_include: float,
) -> np.ndarray:
    anchors = [np.zeros(d, dtype=int), np.ones(d, dtype=int)]
    anchors.extend(np.eye(d, dtype=int))
    anchors.extend(1 - np.eye(d, dtype=int))

    n_random = max(0, int(n_samples) - len(anchors))
    random_Z = rng.binomial(1, p_include, size=(n_random, d)).astype(int)
    Z = np.vstack(anchors + ([random_Z] if n_random > 0 else []))
    return np.unique(Z, axis=0)


def coalition_kernel_weight(Z: np.ndarray, kernel_width: float) -> np.ndarray:
    d = Z.shape[1]
    hamming_dist = d - Z.sum(axis=1)
    weight = np.exp(-((hamming_dist.astype(float) ** 2) / (kernel_width**2)))
    return np.maximum(weight, 1e-8)


def lime_surrogate_single(
    x0: np.ndarray,
    baseline: np.ndarray,
    q_lo,
    q_hi,
    rng: np.random.Generator,
    n_samples: int = 256,
    p_include: float = 0.5,
    kernel_width: float | None = None,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """
    LIME-style binary coalition surrogate around x0.

    z_j=1 keeps feature j at x0[j]; z_j=0 replaces it by baseline[j]. The
    weighted linear coefficient beta_j is the non-Shapley local attribution.
    """
    x0 = impute_to_baseline(x0, baseline)
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    d = len(x0)
    if kernel_width is None:
        kernel_width = max(1.0, 0.75 * np.sqrt(d))

    Z = binary_coalition_design(d, n_samples, rng, p_include)

    X_eval = np.tile(baseline, (len(Z), 1))
    X_eval[Z.astype(bool)] = np.tile(x0, (len(Z), 1))[Z.astype(bool)]
    y_eval = interval_width(q_lo, q_hi, X_eval)

    # LIME locality: rows closer to the full instance get larger weight.
    weight = coalition_kernel_weight(Z, kernel_width)

    _intercept, beta, pred = weighted_ridge_with_intercept(Z, y_eval, weight, l2=l2)
    return beta, weighted_r2(y_eval, pred, weight)


def masked_calib_from_binary(
    X_calib: np.ndarray,
    baseline: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    X_calib = np.asarray(X_calib, dtype=float)
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=int).reshape(-1)
    X_masked = np.tile(baseline, (X_calib.shape[0], 1))
    keep = np.flatnonzero(z)
    if len(keep) > 0:
        X_masked[:, keep] = X_calib[:, keep]
    return X_masked


def global_calibration_width_adjustment(
    z: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    baseline: np.ndarray,
    q_lo,
    q_hi,
    alpha: float,
) -> float:
    """Return the calibration width adjustment (2*q_hat) for a coalition."""
    X_masked = masked_calib_from_binary(X_calib, baseline, z)
    return calibration_width_adjustment_from_models(X_masked, y_calib, q_lo, q_hi, alpha=alpha)


def lime_surrogate_global(
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    baseline: np.ndarray,
    q_lo,
    q_hi,
    alpha: float,
    rng: np.random.Generator,
    n_samples: int = 256,
    p_include: float = 0.5,
    kernel_width: float | None = None,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """
    LIME-style surrogate for the global CP calibration game.

    z_j=1 keeps feature j in the calibration matrix; z_j=0 occludes that
    feature for every calibration row by replacing it with the baseline.
    """
    X_calib = np.asarray(X_calib, dtype=float)
    d = X_calib.shape[1]
    if kernel_width is None:
        kernel_width = max(1.0, 0.75 * np.sqrt(d))

    Z = binary_coalition_design(d, n_samples, rng, p_include)
    y_eval = np.array(
        [
            global_calibration_width_adjustment(z, X_calib, y_calib, baseline, q_lo, q_hi, alpha)
            for z in Z
        ],
        dtype=float,
    )
    weight = coalition_kernel_weight(Z, kernel_width)
    _intercept, beta, pred = weighted_ridge_with_intercept(Z, y_eval, weight, l2=l2)
    return beta, weighted_r2(y_eval, pred, weight)


def occlusion_sensitivity_single(
    x0: np.ndarray,
    baseline: np.ndarray,
    q_lo,
    q_hi,
) -> np.ndarray:
    """
    Tabular occlusion sensitivity score:
        score_j = width(x0) - width(x0 with feature j replaced by baseline_j).

    This is the feature-level analogue of image occlusion sensitivity: instead
    of masking an image patch, we occlude one tabular feature by replacing it
    with the calibration baseline.

    Positive means the observed feature value increases local interval width
    relative to the baseline replacement.
    """
    x0 = impute_to_baseline(x0, baseline)
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    d = len(x0)
    full_width = float(interval_width(q_lo, q_hi, x0)[0])

    X_occ = np.tile(x0, (d, 1))
    X_occ[np.arange(d), np.arange(d)] = baseline
    occ_width = interval_width(q_lo, q_hi, X_occ)
    return full_width - occ_width


def occlusion_sensitivity_global(
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    baseline: np.ndarray,
    q_lo,
    q_hi,
    alpha: float,
) -> np.ndarray:
    """
    Global occlusion sensitivity for the conformal calibration component.

    score_j = V(full calibration features) - V(feature j occluded in all
    calibration rows), where V is the conformal correction width 2*q_hat.
    """
    X_calib = np.asarray(X_calib, dtype=float)
    d = X_calib.shape[1]
    full = np.ones(d, dtype=int)
    full_adjustment = global_calibration_width_adjustment(full, X_calib, y_calib, baseline, q_lo, q_hi, alpha)

    out = np.zeros(d, dtype=float)
    for j in range(d):
        z = full.copy()
        z[j] = 0
        out[j] = full_adjustment - global_calibration_width_adjustment(
            z, X_calib, y_calib, baseline, q_lo, q_hi, alpha
        )
    return out


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    ar = pd.Series(np.asarray(a, dtype=float)).rank(method="average").to_numpy()
    br = pd.Series(np.asarray(b, dtype=float)).rank(method="average").to_numpy()
    return corr_safe(ar, br)


def topk_idx(v: np.ndarray, k: int) -> set[int]:
    v = np.asarray(v, dtype=float).reshape(-1)
    k = min(int(k), len(v))
    if k <= 0:
        return set()
    return set(np.argsort(-np.abs(v))[:k].tolist())


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    aa = topk_idx(a, k)
    bb = topk_idx(b, k)
    if not aa:
        return float("nan")
    return len(aa & bb) / len(aa)


def sign_agreement(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = (np.abs(a) > eps) | (np.abs(b) > eps)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(a[mask]) == np.sign(b[mask])))


def retest_stats(scores: np.ndarray, width: np.ndarray, top_quantile: float) -> dict:
    """
    Decision-interface sanity check: retest if any feature has a positive local
    uncertainty contribution; high-uncertainty rows are top quantile by width.
    """
    scores = np.asarray(scores, dtype=float)
    width = np.asarray(width, dtype=float).reshape(-1)
    retest = np.max(scores, axis=1) > 0
    thr = float(np.quantile(width, top_quantile))
    high = width >= thr
    tp = int(np.sum(retest & high))
    fp = int(np.sum(retest & ~high))
    fn = int(np.sum(~retest & high))
    tn = int(np.sum(~retest & ~high))
    recall = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    precision = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    return {
        "uncertainty_threshold": thr,
        "retest_rate": float(np.mean(retest)),
        "uncertainty_recall": recall,
        "uncertainty_precision": precision,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def summarize_method(
    setting: str,
    seed: int,
    method: str,
    scores: np.ndarray,
    width: np.ndarray,
    runtime_sec: float,
    shap_scores: np.ndarray,
    lime_r2: np.ndarray | None,
    args,
) -> dict:
    n_test, d = scores.shape
    row = {
        "setting": setting,
        "seed": seed,
        "method": method,
        "d": d,
        "n_test": n_test,
        "n_samples": args.n_samples,
        "n_calib": args.n_calib,
        "backend": args.backend,
        "lime_samples": args.lime_samples if method == "lime_surrogate" else 0,
        "runtime_total_sec": float(runtime_sec),
        "runtime_per_query_sec": float(runtime_sec / n_test) if n_test else float("nan"),
        "mean_abs_score": float(np.mean(np.abs(scores))),
    }

    if method == "shapley_exact":
        row.update(
            {
                "row_pearson_vs_shapley": 1.0,
                "row_spearman_vs_shapley": 1.0,
                "top1_overlap_vs_shapley": 1.0,
                "top3_overlap_vs_shapley": 1.0,
                "sign_agreement_vs_shapley": 1.0,
                "lime_weighted_r2": float("nan"),
            }
        )
    else:
        pearson = [corr_safe(shap_scores[i], scores[i]) for i in range(n_test)]
        spearman = [spearman_safe(shap_scores[i], scores[i]) for i in range(n_test)]
        top1 = [topk_overlap(shap_scores[i], scores[i], 1) for i in range(n_test)]
        top3 = [topk_overlap(shap_scores[i], scores[i], 3) for i in range(n_test)]
        signs = [sign_agreement(shap_scores[i], scores[i]) for i in range(n_test)]
        row.update(
            {
                "row_pearson_vs_shapley": float(np.nanmean(pearson)),
                "row_spearman_vs_shapley": float(np.nanmean(spearman)),
                "top1_overlap_vs_shapley": float(np.nanmean(top1)),
                "top3_overlap_vs_shapley": float(np.nanmean(top3)),
                "sign_agreement_vs_shapley": float(np.nanmean(signs)),
                "lime_weighted_r2": float(np.nanmean(lime_r2)) if lime_r2 is not None else float("nan"),
            }
        )

    row.update(retest_stats(scores, width, top_quantile=args.high_width_quantile))
    return row


def per_instance_rows(
    setting: str,
    seed: int,
    method: str,
    scores: np.ndarray,
    shap_scores: np.ndarray,
    width: np.ndarray,
    lime_r2: np.ndarray | None,
):
    rows = []
    for i in range(len(scores)):
        rows.append(
            {
                "setting": setting,
                "seed": seed,
                "row": i,
                "method": method,
                "width": float(width[i]),
                "row_pearson_vs_shapley": corr_safe(shap_scores[i], scores[i]),
                "row_spearman_vs_shapley": spearman_safe(shap_scores[i], scores[i]),
                "top1_overlap_vs_shapley": topk_overlap(shap_scores[i], scores[i], 1),
                "top3_overlap_vs_shapley": topk_overlap(shap_scores[i], scores[i], 3),
                "sign_agreement_vs_shapley": sign_agreement(shap_scores[i], scores[i]),
                "lime_weighted_r2": float(lime_r2[i]) if lime_r2 is not None else float("nan"),
            }
        )
    return rows


def component_metric_row(
    setting: str,
    seed: int,
    method: str,
    component: str,
    values: np.ndarray,
    shap_values: np.ndarray,
) -> dict:
    values = np.asarray(values, dtype=float)
    shap_values = np.asarray(shap_values, dtype=float)

    if values.ndim == 1:
        pearson = corr_safe(shap_values, values)
        spearman = spearman_safe(shap_values, values)
        top1 = topk_overlap(shap_values, values, 1)
        top3 = topk_overlap(shap_values, values, 3)
        sign = sign_agreement(shap_values, values)
    else:
        n = values.shape[0]
        pearson = float(np.nanmean([corr_safe(shap_values[i], values[i]) for i in range(n)]))
        spearman = float(np.nanmean([spearman_safe(shap_values[i], values[i]) for i in range(n)]))
        top1 = float(np.nanmean([topk_overlap(shap_values[i], values[i], 1) for i in range(n)]))
        top3 = float(np.nanmean([topk_overlap(shap_values[i], values[i], 3) for i in range(n)]))
        sign = float(np.nanmean([sign_agreement(shap_values[i], values[i]) for i in range(n)]))

    return {
        "setting": setting,
        "seed": seed,
        "method": method,
        "component": component,
        "mean_abs_score": float(np.mean(np.abs(values))),
        "pearson_vs_shapley": pearson,
        "spearman_vs_shapley": spearman,
        "top1_overlap_vs_shapley": top1,
        "top3_overlap_vs_shapley": top3,
        "sign_agreement_vs_shapley": sign,
    }


def component_rows_for_methods(
    setting: str,
    seed: int,
    components_by_method: dict[str, dict[str, np.ndarray]],
) -> list[dict]:
    rows = []
    shap = components_by_method["shapley_exact"]
    for method in METHOD_ORDER:
        for component in COMPONENT_ORDER:
            rows.append(
                component_metric_row(
                    setting=setting,
                    seed=seed,
                    method=method,
                    component=component,
                    values=components_by_method[method][component],
                    shap_values=shap[component],
                )
            )
    return rows


def save_cell_npz(
    out_dir: Path,
    setting: str,
    seed: int,
    X_test: np.ndarray,
    width: np.ndarray,
    features: list[str],
    components_by_method: dict[str, dict[str, np.ndarray]],
):
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "setting": setting,
        "seed": int(seed),
        "X_test": np.asarray(X_test, dtype=float),
        "width": np.asarray(width, dtype=float),
        "features": np.asarray(features, dtype=object),
    }
    for method, comps in components_by_method.items():
        for component, values in comps.items():
            payload[f"{method}_{component}"] = np.asarray(values, dtype=float)
    np.savez(cells_dir / f"cell_{setting}_seed{seed}.npz", **payload)


def run_one(setting: str, seed: int, args, out_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    print(f"[plugin_comparison] setting={setting} seed={seed}", flush=True)
    X, y, cols = generate_data(setting, n_samples=args.n_samples, seed=seed)
    (X_tr, y_tr), (X_ca, y_ca), (X_te, _y_te) = split_arrays(
        X, y, seed=seed, n_calib=args.n_calib, n_test=args.n_test
    )

    cfg = QuantileModelConfig(
        alpha_lower=args.alpha / 2,
        alpha_upper=1 - args.alpha / 2,
        random_state=seed,
        backend=args.backend,
    )
    q_lo, q_hi = fit_quantile_models(X_tr, y_tr, cfg)
    baseline = np.nanmean(X_ca, axis=0)
    width = interval_width(q_lo, q_hi, X_te)

    n_test, d = X_te.shape
    print(f"[plugin_comparison]   d={d} n_test={n_test} features={cols}", flush=True)

    # Reference: exact local Shapley.
    shap_scores = np.zeros((n_test, d), dtype=float)
    t0 = time.perf_counter()
    for i in range(n_test):
        shap_scores[i] = shapley_local_single(
            X_te[i], baseline=baseline, q_lo=q_lo, q_hi=q_hi, impute_nan_x0=True
        )
    shap_sec = time.perf_counter() - t0

    # Reference global component: exact Shapley over the calibration correction game.
    global_shap, _ = compute_global_shapley(
        X_calib=X_ca,
        y_calib=y_ca,
        q_lo=q_lo,
        q_hi=q_hi,
        alpha=args.alpha,
        baseline=baseline,
        n_jobs=1,
        show_progress=False,
    )

    # Non-Shapley 1: LIME-style local surrogate.
    lime_scores = np.zeros((n_test, d), dtype=float)
    lime_r2 = np.zeros(n_test, dtype=float)
    t0 = time.perf_counter()
    for i in range(n_test):
        rng = np.random.default_rng(seed * 100_000 + i)
        lime_scores[i], lime_r2[i] = lime_surrogate_single(
            X_te[i],
            baseline=baseline,
            q_lo=q_lo,
            q_hi=q_hi,
            rng=rng,
            n_samples=args.lime_samples,
            p_include=args.lime_p_include,
            kernel_width=args.lime_kernel_width,
            l2=args.lime_l2,
        )
    lime_sec = time.perf_counter() - t0

    rng_global = np.random.default_rng(seed * 100_000 + 99_999)
    global_lime, global_lime_r2 = lime_surrogate_global(
        X_calib=X_ca,
        y_calib=y_ca,
        baseline=baseline,
        q_lo=q_lo,
        q_hi=q_hi,
        alpha=args.alpha,
        rng=rng_global,
        n_samples=args.lime_samples,
        p_include=args.lime_p_include,
        kernel_width=args.lime_kernel_width,
        l2=args.lime_l2,
    )

    # Non-Shapley 2: occlusion sensitivity / single-feature ablation.
    occlusion_scores = np.zeros((n_test, d), dtype=float)
    t0 = time.perf_counter()
    for i in range(n_test):
        occlusion_scores[i] = occlusion_sensitivity_single(X_te[i], baseline, q_lo, q_hi)
    occlusion_sec = time.perf_counter() - t0

    global_occlusion = occlusion_sensitivity_global(
        X_calib=X_ca,
        y_calib=y_ca,
        baseline=baseline,
        q_lo=q_lo,
        q_hi=q_hi,
        alpha=args.alpha,
    )

    scores_by_method = {
        "shapley_exact": (shap_scores, shap_sec, None),
        "lime_surrogate": (lime_scores, lime_sec, lime_r2),
        "occlusion_sensitivity": (occlusion_scores, occlusion_sec, None),
    }
    components_by_method = {
        "shapley_exact": {
            "local": shap_scores,
            "global": global_shap,
            "total": shap_scores + global_shap[None, :],
        },
        "lime_surrogate": {
            "local": lime_scores,
            "global": global_lime,
            "total": lime_scores + global_lime[None, :],
        },
        "occlusion_sensitivity": {
            "local": occlusion_scores,
            "global": global_occlusion,
            "total": occlusion_scores + global_occlusion[None, :],
        },
    }

    result_rows = []
    instance_rows = []
    for method in METHOD_ORDER:
        scores, runtime_sec, r2 = scores_by_method[method]
        result_rows.append(
            summarize_method(
                setting=setting,
                seed=seed,
                method=method,
                scores=scores,
                width=width,
                runtime_sec=runtime_sec,
                shap_scores=shap_scores,
                lime_r2=r2,
                args=args,
            )
        )
        if method != "shapley_exact":
            instance_rows.extend(
                per_instance_rows(setting, seed, method, scores, shap_scores, width, r2)
            )

    print(
        "[plugin_comparison]   runtime/query "
        f"shap={shap_sec / n_test:.4f}s "
        f"lime={lime_sec / n_test:.4f}s "
        f"occlusion={occlusion_sec / n_test:.4f}s "
        f"global_lime_r2={global_lime_r2:.3f}",
        flush=True,
    )
    component_rows = component_rows_for_methods(setting, seed, components_by_method)

    if args.save_cells:
        save_cell_npz(out_dir, setting, seed, X_te, width, cols, components_by_method)

    return result_rows, instance_rows, component_rows


def make_summary(results: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "runtime_per_query_sec",
        "row_pearson_vs_shapley",
        "row_spearman_vs_shapley",
        "top1_overlap_vs_shapley",
        "top3_overlap_vs_shapley",
        "sign_agreement_vs_shapley",
        "lime_weighted_r2",
        "retest_rate",
        "uncertainty_recall",
        "uncertainty_precision",
    ]
    summary = (
        results.groupby(["setting", "method"], as_index=False)[value_cols]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary.reset_index(drop=True)


def make_component_summary(component_results: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "mean_abs_score",
        "pearson_vs_shapley",
        "spearman_vs_shapley",
        "top1_overlap_vs_shapley",
        "top3_overlap_vs_shapley",
        "sign_agreement_vs_shapley",
    ]
    summary = (
        component_results.groupby(["setting", "method", "component"], as_index=False)[value_cols]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary.reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=str(Path(__file__).parent))
    p.add_argument(
        "--settings",
        nargs="+",
        default=["heteroscedastic", "homoscedastic", "source_separation"],
        choices=["heteroscedastic", "homoscedastic", "source_separation"],
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 72)))
    p.add_argument("--n_samples", type=int, default=3000)
    p.add_argument("--n_calib", type=int, default=900)
    p.add_argument("--n_test", type=int, default=600)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument(
        "--backend",
        default="sklearn_gbrt",
        choices=["sklearn_gbrt", "lightgbm"],
        help="Use sklearn_gbrt by default for CPU-only reproducibility.",
    )
    p.add_argument("--lime_samples", type=int, default=256)
    p.add_argument("--lime_p_include", type=float, default=0.5)
    p.add_argument("--lime_kernel_width", type=float, default=None)
    p.add_argument("--lime_l2", type=float, default=1e-3)
    p.add_argument("--save_cells", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--high_width_quantile",
        type=float,
        default=0.75,
        help="Rows above this width quantile define high-uncertainty recall.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Fast sanity run: one setting, one seed, small sample/test size.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.settings = ["heteroscedastic"]
        args.seeds = [0]
        args.n_samples = 500
        args.n_calib = 100
        args.n_test = 5
        args.lime_samples = 64

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    all_instances: list[dict] = []
    all_components: list[dict] = []
    for setting in args.settings:
        for seed in args.seeds:
            rows, inst, comp = run_one(setting, seed, args, out_dir)
            all_results.extend(rows)
            all_instances.extend(inst)
            all_components.extend(comp)

    results = pd.DataFrame(all_results)
    per_instance = pd.DataFrame(all_instances)
    component_results = pd.DataFrame(all_components)
    summary = make_summary(results)
    component_summary = make_component_summary(component_results)

    suffix = "_smoke" if args.smoke else ""
    results_path = out_dir / f"results{suffix}.csv"
    per_instance_path = out_dir / f"per_instance{suffix}.csv"
    component_results_path = out_dir / f"component_results{suffix}.csv"
    component_summary_path = out_dir / f"component_summary{suffix}.csv"
    summary_path = out_dir / f"summary{suffix}.csv"

    results.to_csv(results_path, index=False)
    per_instance.to_csv(per_instance_path, index=False)
    component_results.to_csv(component_results_path, index=False)
    component_summary.to_csv(component_summary_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"[plugin_comparison] wrote {results_path}", flush=True)
    print(f"[plugin_comparison] wrote {per_instance_path}", flush=True)
    print(f"[plugin_comparison] wrote {component_results_path}", flush=True)
    print(f"[plugin_comparison] wrote {component_summary_path}", flush=True)
    print(f"[plugin_comparison] wrote {summary_path}", flush=True)
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        print(summary.to_string(index=False), flush=True)
        print(component_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
