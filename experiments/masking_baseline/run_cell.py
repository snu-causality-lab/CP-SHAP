"""Run one correlated source-separation baseline-sensitivity cell (Table 7, Appendix E.2).

One invocation fits a single pair of quantile models for one ``(seed, rho)``
dataset and enumerates all coalitions at d=10 for the specified completion schemes:
four mean-completion rules plus product-marginal and joint-marginal resampling.
The paper compares four alternatives against the calibration-mean reference;
the final aggregation retains those five rules for Table 7 and its accompanying
text. Joint-marginal results and the configured higher-draw global check are
additional diagnostics, not selected for that paper output.
The paired design isolates the completion/baseline choice:
the data, split, fitted models, calibration labels, and test queries are shared.

Stored baseline IDs: empirical_mean is the calibration-set fixed mean (reference);
empirical_marginal is the pooled fixed mean over training + calibration rows,
not distributional resampling. The config key marginal_background identifies this
fixed-mean pool; product-marginal resampling has a separate distributional baseline.

This module only runs when invoked explicitly.  Importing it does not fit a
model or write results.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import hashlib
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.conformal.cq import calibration_width_adjustment  # noqa: E402
from src.conformal.quantile import QuantileModelConfig, fit_quantile_models  # noqa: E402
from data.synthetic.synthetic_source_separation import (  # noqa: E402
    poly2_feature_map,
    sample_gamma,
)
from src.shapley.shapley_global import all_subsets, shapley_from_v_cache  # noqa: E402


# Stored IDs: empirical_mean = calibration-set fixed mean (reference);
# empirical_marginal = training + calibration pooled fixed mean, not resampling.
BASELINE_NAMES = (
    "empirical_mean",
    "empirical_marginal",
    "oracle_conditional",
    "gaussian_conditional",
)
MARGINAL_BASELINE_NAME = "empirical_product_marginal"
JOINT_MARGINAL_BASELINE_NAME = "empirical_joint_marginal"


@dataclass(frozen=True)
class CorrelatedSourceSeparation:
    X: np.ndarray
    y: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    covariance: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray
    raw_X: np.ndarray
    z: np.ndarray
    e: np.ndarray


def rho_label(rho: float) -> str:
    """Stable filename label such as ``corr00`` or ``corr95``."""

    return f"corr{int(round(100.0 * float(rho))):02d}"


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def generate_correlated_source_separation(
    n_samples: int,
    rho_x: float,
    seed: int,
    *,
    p_mean: int = 5,
    p_unc: int = 5,
    alpha: float = 2.0,
    sigma_eps2: float = 0.02,
    sigma_delta2: float = 0.05,
) -> CorrelatedSourceSeparation:
    """Source-separation DGP with correlation added only to ``X``.

    The RNG calls match ``generate_source_separation_data``.  At ``rho_x=0``
    this function therefore reproduces its ``X`` and ``y`` bitwise.  Across
    correlation modes, the raw independent features, DGP coefficients, and
    noise draws are common random numbers.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if p_mean + p_unc != 10:
        raise ValueError("this experiment is fixed at d=10 (5 mean + 5 uncertainty)")
    if not 0.0 <= rho_x < 1.0:
        raise ValueError("rho_x must lie in [0, 1)")

    rng = np.random.default_rng(seed)
    raw_v = rng.normal(0.0, 1.0, size=(n_samples, p_mean))
    beta = rng.uniform(-1.0, 1.0, size=p_mean)
    raw_u = rng.normal(0.0, 1.0, size=(n_samples, p_unc))

    # The number and order of RNG draws below match the repository DGP.
    raw_phi = poly2_feature_map(raw_u, include_bias=True)
    gamma = sample_gamma(raw_phi.shape[1], rng)
    delta = rng.normal(0.0, np.sqrt(sigma_delta2), size=n_samples)
    z = rng.normal(0.0, 1.0, size=n_samples)
    e = rng.normal(0.0, 1.0, size=n_samples)

    raw_X = np.hstack([raw_v, raw_u])
    covariance = (
        (1.0 - rho_x) * np.eye(raw_X.shape[1])
        + rho_x * np.ones((raw_X.shape[1], raw_X.shape[1]))
    )
    if rho_x == 0.0:
        X = raw_X.copy()
    else:
        chol = np.linalg.cholesky(covariance)
        X = raw_X @ chol.T

    v = X[:, :p_mean]
    u = X[:, p_mean:]
    mu = v @ beta
    sigma = np.abs(poly2_feature_map(u, include_bias=True) @ gamma + delta)
    y = mu + np.sqrt(alpha) * sigma * z + np.sqrt(sigma_eps2) * e
    return CorrelatedSourceSeparation(
        X=X,
        y=y,
        mu=mu,
        sigma=sigma,
        covariance=covariance,
        beta=beta,
        gamma=gamma,
        delta=delta,
        raw_X=raw_X,
        z=z,
        e=e,
    )


def split_indices(
    n_samples: int,
    test_size: float,
    calib_size: float,
    n_explain: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce ``experiments/run_synthetic.py`` and select query rows.

    The full test pool stays held out.  Only its first ``n_explain`` rows are
    used for the expensive exact local-Shapley calculation.
    """

    if n_samples <= 0 or not 0.0 < test_size < 1.0:
        raise ValueError("n_samples must be positive and test_size in (0, 1)")
    if not 0.0 < calib_size < 1.0 or n_explain <= 0:
        raise ValueError("calib_size must be in (0, 1) and n_explain positive")
    indices = np.arange(n_samples)
    temp_idx, test_pool_idx = train_test_split(
        indices, test_size=test_size, random_state=seed
    )
    train_idx, calib_idx = train_test_split(
        temp_idx, test_size=calib_size, random_state=seed
    )
    if n_explain > len(test_pool_idx):
        raise ValueError(
            f"n_explain={n_explain} exceeds test pool size={len(test_pool_idx)}"
        )
    explain_idx = test_pool_idx[:n_explain]
    return (
        np.asarray(train_idx),
        np.asarray(calib_idx),
        np.asarray(test_pool_idx),
        np.asarray(explain_idx),
    )


def complete_fixed(
    X: np.ndarray,
    observed: tuple[int, ...],
    baseline: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    out = np.tile(np.asarray(baseline, dtype=float), (len(X), 1))
    if observed:
        out[:, observed] = X[:, observed]
    return out


def pooled_empirical_mean(
    X_train: np.ndarray,
    X_calib: np.ndarray,
) -> np.ndarray:
    """Return one fixed masking vector from the train+calibration pool."""

    X_train = np.asarray(X_train, dtype=float)
    X_calib = np.asarray(X_calib, dtype=float)
    if X_train.ndim != 2 or X_calib.ndim != 2:
        raise ValueError("X_train and X_calib must be two-dimensional")
    if X_train.shape[1] != X_calib.shape[1]:
        raise ValueError("X_train and X_calib must have the same dimension")
    if len(X_train) + len(X_calib) == 0:
        raise ValueError("the train+calibration background must be non-empty")
    return np.vstack([X_train, X_calib]).mean(axis=0)


def sample_empirical_product_marginal(
    X_calib: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample each calibration feature independently with replacement.

    The returned matrix has the same shape as ``X_calib``.  Column ``j`` is an
    independent bootstrap sample from the empirical marginal distribution of
    calibration feature ``j``.
    """

    values = np.asarray(X_calib, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("X_calib must be a non-empty two-dimensional array")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0, len(values), size=values.shape, endpoint=False
    )
    columns = np.arange(values.shape[1])[None, :]
    return values[indices, columns], indices


def sample_empirical_joint_marginal(
    X_calib: np.ndarray,
    n_draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample intact calibration donor rows with replacement."""

    values = np.asarray(X_calib, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("X_calib must be a non-empty two-dimensional array")
    if int(n_draws) <= 0:
        raise ValueError("n_draws must be positive")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=int(n_draws), endpoint=False)
    return values[indices], indices


def complete_empirical_product_marginal(
    X: np.ndarray,
    observed: tuple[int, ...],
    marginal_mask_matrix: np.ndarray,
) -> np.ndarray:
    """Keep observed columns and fill omitted columns from a fixed draw matrix."""

    values = np.asarray(X, dtype=float)
    mask = np.asarray(marginal_mask_matrix, dtype=float)
    if values.shape != mask.shape:
        raise ValueError(
            "X and marginal_mask_matrix must have the same shape"
        )
    out = mask.copy()
    if observed:
        out[:, observed] = values[:, observed]
    return out


def complete_conditional_mean(
    X: np.ndarray,
    observed: tuple[int, ...],
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    ridge: float = 0.0,
) -> np.ndarray:
    """Complete missing coordinates with a Gaussian conditional mean."""

    X = np.asarray(X, dtype=float)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    if X.ndim != 2 or X.shape[1] != mean.size:
        raise ValueError("X and mean have incompatible shapes")
    if covariance.shape != (mean.size, mean.size):
        raise ValueError("covariance has incompatible shape")

    d = X.shape[1]
    obs = np.asarray(observed, dtype=int)
    missing = np.setdiff1d(np.arange(d), obs, assume_unique=True)
    out = X.copy()
    if missing.size == 0:
        return out
    if obs.size == 0:
        out[:, missing] = mean[missing]
        return out

    sigma_oo = covariance[np.ix_(obs, obs)].copy()
    if ridge:
        sigma_oo.flat[:: sigma_oo.shape[0] + 1] += float(ridge)
    sigma_mo = covariance[np.ix_(missing, obs)]
    coef = np.linalg.solve(sigma_oo, sigma_mo.T).T
    out[:, missing] = (
        mean[missing]
        + (X[:, obs] - mean[obs]) @ coef.T
    )
    return out


def _predict_width(X: np.ndarray, q_lo, q_hi) -> np.ndarray:
    return np.asarray(q_hi.predict(X) - q_lo.predict(X), dtype=float)


def _global_value(
    completed: np.ndarray,
    y_calib: np.ndarray,
    q_lo,
    q_hi,
    alpha: float,
) -> float:
    return float(
        calibration_width_adjustment(
            y_calib,
            q_lo.predict(completed),
            q_hi.predict(completed),
            alpha=alpha,
        )
    )


def global_cache_deterministic(
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    q_lo,
    q_hi,
    alpha: float,
    subsets: Iterable[tuple[int, ...]],
    completer,
) -> dict[tuple[int, ...], float]:
    cache = {}
    for subset in subsets:
        cache[subset] = _global_value(
            completer(X_calib, subset), y_calib, q_lo, q_hi, alpha
        )
    return cache


def local_cache_deterministic(
    x: np.ndarray,
    q_lo,
    q_hi,
    subsets: tuple[tuple[int, ...], ...],
    completer,
) -> dict[tuple[int, ...], float]:
    # Batch all 1024 completed coalitions into one model prediction call.
    rows = np.vstack([completer(x[None, :], subset)[0] for subset in subsets])
    values = _predict_width(rows, q_lo, q_hi)
    return {subset: float(values[i]) for i, subset in enumerate(subsets)}


def compute_empirical_product_marginal(
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_explain: np.ndarray,
    marginal_mask_matrix: np.ndarray,
    q_lo,
    q_hi,
    alpha: float,
    subsets: tuple[tuple[int, ...], ...],
) -> dict:
    """Compute global, local, and total SHAP for one realized marginal draw.

    The full masking matrix is used row-wise for the calibration game.  Its
    first ``n_explain`` rows are paired with the held-out explanation rows for
    the local game.  Each paired mask row is itself one draw from the empirical
    product of the calibration feature marginals.
    """

    start = time.perf_counter()
    cache = global_cache_deterministic(
        X_calib,
        y_calib,
        q_lo,
        q_hi,
        alpha,
        subsets,
        lambda rows, subset: complete_empirical_product_marginal(
            rows, subset, marginal_mask_matrix
        ),
    )
    dimension = X_calib.shape[1]
    phi_global = shapley_from_v_cache(cache, dimension)
    empty, full = (), tuple(range(dimension))
    efficiency = phi_global.sum() - (cache[full] - cache[empty])
    global_runtime = time.perf_counter() - start

    X_explain = np.asarray(X_explain, dtype=float)
    if X_explain.ndim != 2 or X_explain.shape[1] != dimension:
        raise ValueError("X_explain and X_calib have incompatible shapes")
    if len(X_explain) > len(marginal_mask_matrix):
        raise ValueError("X_explain cannot exceed the marginal mask row count")
    local_masks = np.asarray(
        marginal_mask_matrix[: len(X_explain)], dtype=float
    )
    phi_local = np.zeros_like(X_explain, dtype=float)
    efficiency_local = np.zeros(len(X_explain), dtype=float)
    endpoint_local = np.zeros((len(X_explain), 2), dtype=float)
    start = time.perf_counter()
    for query_idx, (x, mask_row) in enumerate(
        zip(X_explain, local_masks, strict=True)
    ):
        local_cache = local_cache_deterministic(
            x,
            q_lo,
            q_hi,
            subsets,
            lambda rows, subset, baseline=mask_row: complete_fixed(
                rows, subset, baseline
            ),
        )
        phi_local[query_idx] = shapley_from_v_cache(
            local_cache, dimension
        )
        endpoint_local[query_idx] = [
            local_cache[empty],
            local_cache[full],
        ]
        efficiency_local[query_idx] = (
            phi_local[query_idx].sum()
            - (local_cache[full] - local_cache[empty])
        )
    local_runtime = time.perf_counter() - start
    return {
        "phi_global_marginal": phi_global,
        "phi_local_marginal": phi_local,
        "phi_total_marginal": phi_local + phi_global[None, :],
        "global_runtime_sec_marginal": float(global_runtime),
        "local_runtime_sec_marginal": float(local_runtime),
        "efficiency_global_marginal": float(efficiency),
        "efficiency_local_marginal": efficiency_local,
        "endpoint_global_marginal": np.asarray(
            [cache[empty], cache[full]], dtype=float
        ),
        "endpoint_local_marginal": endpoint_local,
    }


def compute_empirical_joint_marginal_global(
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    donor_rows: np.ndarray,
    q_lo,
    q_hi,
    alpha: float,
    subsets: tuple[tuple[int, ...], ...],
) -> dict:
    """Average draw-specific conformal payoffs over intact donor rows."""

    X_calib = np.asarray(X_calib, dtype=float)
    donors = np.asarray(donor_rows, dtype=float)
    if donors.ndim != 2 or donors.shape[1] != X_calib.shape[1]:
        raise ValueError("donor_rows and X_calib have incompatible shapes")
    n_draws, dimension = donors.shape
    n_calib = len(X_calib)
    cache: dict[tuple[int, ...], float] = {}
    start = time.perf_counter()
    for subset in subsets:
        completed = np.repeat(donors, n_calib, axis=0)
        if subset:
            completed[:, subset] = np.tile(
                X_calib[:, subset], (n_draws, 1)
            )
        pred_lo = q_lo.predict(completed).reshape(n_draws, n_calib)
        pred_hi = q_hi.predict(completed).reshape(n_draws, n_calib)
        draw_values = [
            calibration_width_adjustment(
                y_calib,
                pred_lo[draw_idx],
                pred_hi[draw_idx],
                alpha=alpha,
            )
            for draw_idx in range(n_draws)
        ]
        cache[subset] = float(np.mean(draw_values))
    phi_global = shapley_from_v_cache(cache, dimension)
    empty, full = (), tuple(range(dimension))
    return {
        "phi_global": phi_global,
        "runtime_sec": float(time.perf_counter() - start),
        "efficiency": float(
            phi_global.sum() - (cache[full] - cache[empty])
        ),
        "endpoint": np.asarray([cache[empty], cache[full]], dtype=float),
    }


def compute_empirical_joint_marginal(
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_explain: np.ndarray,
    donor_rows: np.ndarray,
    q_lo,
    q_hi,
    alpha: float,
    subsets: tuple[tuple[int, ...], ...],
) -> dict:
    """Compute exact SHAP for a Monte Carlo joint-marginal payoff game."""

    global_result = compute_empirical_joint_marginal_global(
        X_calib,
        y_calib,
        donor_rows,
        q_lo,
        q_hi,
        alpha,
        subsets,
    )
    donors = np.asarray(donor_rows, dtype=float)
    X_explain = np.asarray(X_explain, dtype=float)
    n_draws, dimension = donors.shape
    n_subsets = len(subsets)
    phi_local = np.zeros_like(X_explain, dtype=float)
    efficiency_local = np.zeros(len(X_explain), dtype=float)
    endpoint_local = np.zeros((len(X_explain), 2), dtype=float)
    empty, full = (), tuple(range(dimension))
    start = time.perf_counter()
    for query_idx, x in enumerate(X_explain):
        completed = np.tile(donors, (n_subsets, 1))
        for subset_idx, subset in enumerate(subsets):
            if subset:
                row_slice = slice(
                    subset_idx * n_draws, (subset_idx + 1) * n_draws
                )
                completed[row_slice, subset] = x[np.asarray(subset)]
        widths = _predict_width(completed, q_lo, q_hi).reshape(
            n_subsets, n_draws
        )
        cache = {
            subset: float(widths[subset_idx].mean())
            for subset_idx, subset in enumerate(subsets)
        }
        phi_local[query_idx] = shapley_from_v_cache(cache, dimension)
        endpoint_local[query_idx] = [cache[empty], cache[full]]
        efficiency_local[query_idx] = (
            phi_local[query_idx].sum() - (cache[full] - cache[empty])
        )
    return {
        "phi_global_joint_marginal": global_result["phi_global"],
        "phi_local_joint_marginal": phi_local,
        "phi_total_joint_marginal": (
            phi_local + global_result["phi_global"][None, :]
        ),
        "global_runtime_sec_joint_marginal": global_result["runtime_sec"],
        "local_runtime_sec_joint_marginal": float(
            time.perf_counter() - start
        ),
        "efficiency_global_joint_marginal": global_result["efficiency"],
        "efficiency_local_joint_marginal": efficiency_local,
        "endpoint_global_joint_marginal": global_result["endpoint"],
        "endpoint_local_joint_marginal": endpoint_local,
    }


def compute_all_baselines(
    X_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: np.ndarray,
    covariance: np.ndarray,
    q_lo,
    q_hi,
    cfg: dict,
    marginal_seed: int,
    joint_marginal_seed: int,
    joint_convergence_draws: int | None,
) -> dict:
    d = X_train.shape[1]
    subsets = tuple(all_subsets(d))
    b_emp = np.mean(X_calib, axis=0)
    background = np.vstack([X_train, X_calib])
    b_pool = pooled_empirical_mean(X_train, X_calib)

    lw = LedoitWolf().fit(X_train)
    gaussian_mean = np.asarray(lw.location_, dtype=float)
    gaussian_cov = np.asarray(lw.covariance_, dtype=float)
    ridge = float(cfg.get("conditional_ridge", 1e-10))
    marginal_mask_matrix, marginal_sample_indices = (
        sample_empirical_product_marginal(X_calib, marginal_seed)
    )
    primary_joint_draws = int(cfg["joint_marginal_draws"])
    stored_joint_draws = max(
        primary_joint_draws,
        int(joint_convergence_draws or primary_joint_draws),
    )
    joint_donor_rows, joint_donor_indices = sample_empirical_joint_marginal(
        X_calib, stored_joint_draws, joint_marginal_seed
    )

    # The two empirical IDs select fixed vectors; neither draws replacement values.
    # b_emp is the calibration mean; b_pool is the training + calibration pooled mean.
    deterministic = {
        "empirical_mean": lambda rows, subset: complete_fixed(rows, subset, b_emp),
        "empirical_marginal": lambda rows, subset: complete_fixed(
            rows, subset, b_pool
        ),
        "oracle_conditional": lambda rows, subset: complete_conditional_mean(
            rows, subset, np.zeros(d), covariance
        ),
        "gaussian_conditional": lambda rows, subset: complete_conditional_mean(
            rows, subset, gaussian_mean, gaussian_cov, ridge=ridge
        ),
    }

    phi_global = np.zeros((len(BASELINE_NAMES), d), dtype=float)
    phi_local = np.zeros((len(BASELINE_NAMES), len(X_test), d), dtype=float)
    global_runtime = np.zeros(len(BASELINE_NAMES), dtype=float)
    local_runtime = np.zeros(len(BASELINE_NAMES), dtype=float)
    efficiency_global = np.zeros(len(BASELINE_NAMES), dtype=float)
    efficiency_local = np.zeros((len(BASELINE_NAMES), len(X_test)), dtype=float)
    endpoint_global = np.zeros((len(BASELINE_NAMES), 2), dtype=float)
    endpoint_local = np.zeros((len(BASELINE_NAMES), len(X_test), 2), dtype=float)

    for baseline_idx, baseline_name in enumerate(BASELINE_NAMES):
        start = time.perf_counter()
        global_cache = global_cache_deterministic(
            X_calib,
            y_calib,
            q_lo,
            q_hi,
            float(cfg["alpha"]),
            subsets,
            deterministic[baseline_name],
        )
        phi_global[baseline_idx] = shapley_from_v_cache(global_cache, d)
        global_runtime[baseline_idx] = time.perf_counter() - start
        empty, full = (), tuple(range(d))
        endpoint_global[baseline_idx] = [global_cache[empty], global_cache[full]]
        efficiency_global[baseline_idx] = (
            phi_global[baseline_idx].sum()
            - (global_cache[full] - global_cache[empty])
        )

        start = time.perf_counter()
        for query_idx, x in enumerate(X_test):
            local_cache = local_cache_deterministic(
                x, q_lo, q_hi, subsets, deterministic[baseline_name]
            )
            phi_local[baseline_idx, query_idx] = shapley_from_v_cache(
                local_cache, d
            )
            endpoint_local[baseline_idx, query_idx] = [
                local_cache[empty],
                local_cache[full],
            ]
            efficiency_local[baseline_idx, query_idx] = (
                phi_local[baseline_idx, query_idx].sum()
                - (local_cache[full] - local_cache[empty])
            )
        local_runtime[baseline_idx] = time.perf_counter() - start

    marginal_results = compute_empirical_product_marginal(
        X_calib,
        y_calib,
        X_test,
        marginal_mask_matrix,
        q_lo,
        q_hi,
        float(cfg["alpha"]),
        subsets,
    )
    joint_results = compute_empirical_joint_marginal(
        X_calib,
        y_calib,
        X_test,
        joint_donor_rows[:primary_joint_draws],
        q_lo,
        q_hi,
        float(cfg["alpha"]),
        subsets,
    )
    convergence_results = {}
    if joint_convergence_draws is not None:
        convergence = compute_empirical_joint_marginal_global(
            X_calib,
            y_calib,
            joint_donor_rows[:joint_convergence_draws],
            q_lo,
            q_hi,
            float(cfg["alpha"]),
            subsets,
        )
        convergence_results = {
            "joint_convergence_draw_count": np.asarray(
                int(joint_convergence_draws)
            ),
            "phi_global_joint_marginal_convergence": convergence["phi_global"],
            "global_runtime_sec_joint_marginal_convergence": np.asarray(
                convergence["runtime_sec"]
            ),
            "efficiency_global_joint_marginal_convergence": np.asarray(
                convergence["efficiency"]
            ),
            "endpoint_global_joint_marginal_convergence": convergence[
                "endpoint"
            ],
        }

    return {
        "baseline_names": np.asarray(BASELINE_NAMES),
        "phi_global": phi_global,
        "phi_local": phi_local,
        "phi_total": phi_local + phi_global[:, None, :],
        "global_runtime_sec": global_runtime,
        "local_runtime_sec": local_runtime,
        "efficiency_global": efficiency_global,
        "efficiency_local": efficiency_local,
        "endpoint_global": endpoint_global,
        "endpoint_local": endpoint_local,
        "empirical_mean_baseline": b_emp,
        "pooled_empirical_mean_baseline": b_pool,
        "gaussian_mean": gaussian_mean,
        "gaussian_covariance": gaussian_cov,
        "background_pool_digest": array_digest(background),
        "marginal_baseline_name": np.asarray(MARGINAL_BASELINE_NAME),
        "marginal_mask_matrix": marginal_mask_matrix,
        "marginal_sample_indices": marginal_sample_indices,
        "marginal_mask_digest": np.asarray(
            array_digest(marginal_mask_matrix, marginal_sample_indices)
        ),
        "joint_marginal_baseline_name": np.asarray(
            JOINT_MARGINAL_BASELINE_NAME
        ),
        "joint_marginal_draw_count": np.asarray(primary_joint_draws),
        "joint_marginal_donor_rows": joint_donor_rows,
        "joint_marginal_donor_indices": joint_donor_indices,
        "joint_marginal_donor_digest": np.asarray(
            array_digest(joint_donor_rows, joint_donor_indices)
        ),
        **marginal_results,
        **joint_results,
        **convergence_results,
    }


def validate_config(cfg: dict) -> None:
    required = {
        "name",
        "n_samples",
        "test_size",
        "calib_size",
        "n_explain",
        "data_seeds",
        "correlations",
        "alpha",
        "backend",
        "marginal_background",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    if int(cfg.get("dimension", 10)) != 10:
        raise ValueError("masking-baseline experiment is fixed at d=10")
    if list(cfg.get("baseline_methods", BASELINE_NAMES)) != list(BASELINE_NAMES):
        raise ValueError(f"baseline_methods must be {list(BASELINE_NAMES)}")
    if cfg["marginal_background"] != "train_plus_calibration":
        raise ValueError(
            "marginal_background must be 'train_plus_calibration'"
        )
    if list(cfg.get("distributional_baseline_methods", [])) != [
        MARGINAL_BASELINE_NAME,
        JOINT_MARGINAL_BASELINE_NAME,
    ]:
        raise ValueError(
            "distributional_baseline_methods must be "
            f"['{MARGINAL_BASELINE_NAME}', "
            f"'{JOINT_MARGINAL_BASELINE_NAME}']"
        )
    if cfg.get("marginal_distribution_background") != (
        "calibration_columnwise_bootstrap"
    ):
        raise ValueError(
            "marginal_distribution_background must be "
            "'calibration_columnwise_bootstrap'"
        )
    if cfg.get("joint_marginal_background") != (
        "calibration_row_bootstrap"
    ):
        raise ValueError(
            "joint_marginal_background must be "
            "'calibration_row_bootstrap'"
        )
    if int(cfg.get("joint_marginal_draws", 0)) <= 0:
        raise ValueError("joint_marginal_draws must be positive")
    for rho in cfg["correlations"]:
        if not 0.0 <= float(rho) < 1.0:
            raise ValueError(f"invalid correlation: {rho}")


def atomic_savez(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_cell(cfg: dict, rho_x: float, seed: int) -> dict:
    validate_config(cfg)
    data = generate_correlated_source_separation(
        int(cfg["n_samples"]), float(rho_x), int(seed)
    )
    train_idx, calib_idx, test_pool_idx, explain_idx = split_indices(
        len(data.X),
        float(cfg["test_size"]),
        float(cfg["calib_size"]),
        int(cfg["n_explain"]),
        int(seed),
    )
    X_train, y_train = data.X[train_idx], data.y[train_idx]
    X_calib, y_calib = data.X[calib_idx], data.y[calib_idx]
    X_test = data.X[explain_idx]

    model_cfg = QuantileModelConfig(
        alpha_lower=float(cfg["alpha"]) / 2.0,
        alpha_upper=1.0 - float(cfg["alpha"]) / 2.0,
        random_state=int(seed),
        backend=str(cfg["backend"]),
        params=cfg.get("model_params"),
    )
    fit_start = time.perf_counter()
    q_lo, q_hi = fit_quantile_models(X_train, y_train, model_cfg)
    fit_runtime = time.perf_counter() - fit_start
    predictions = (
        q_lo.predict(X_test),
        q_hi.predict(X_test),
        q_lo.predict(X_calib),
        q_hi.predict(X_calib),
    )
    marginal_seed = int(seed) + int(
        cfg.get("marginal_resample_seed_offset", 271828)
    )
    joint_marginal_seed = int(seed) + int(
        cfg.get("joint_marginal_seed_offset", 314159)
    )
    convergence_cfg = cfg.get("joint_marginal_convergence")
    joint_convergence_draws = None
    if convergence_cfg and (
        int(convergence_cfg["seed"]) == int(seed)
        and np.isclose(float(convergence_cfg["rho"]), float(rho_x))
    ):
        joint_convergence_draws = int(convergence_cfg["draws"])

    baseline_results = compute_all_baselines(
        X_train,
        X_calib,
        y_calib,
        X_test,
        data.covariance,
        q_lo,
        q_hi,
        cfg,
        marginal_seed,
        joint_marginal_seed,
        joint_convergence_draws,
    )
    empirical_matrix = np.corrcoef(data.X, rowvar=False)
    upper = np.triu_indices(data.X.shape[1], 1)
    eigenvalues = np.linalg.eigvalsh(data.covariance)
    metadata = {
        "artifact_schema_version": 5,
        "config_name": cfg["name"],
        "rho_x": float(rho_x),
        "data_seed": int(seed),
        "dimension": 10,
        "n_samples": int(len(data.X)),
        "n_train": int(len(train_idx)),
        "n_calib": int(len(calib_idx)),
        "n_test_pool": int(len(test_pool_idx)),
        "n_explain": int(len(explain_idx)),
        "alpha": float(cfg["alpha"]),
        "backend": str(cfg["backend"]),
        "fit_runtime_sec": float(fit_runtime),
        "empirical_pearson_mean": float(empirical_matrix[upper].mean()),
        "empirical_pearson_min": float(empirical_matrix[upper].min()),
        "empirical_pearson_max": float(empirical_matrix[upper].max()),
        "covariance_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "prediction_digest": array_digest(*predictions),
        "split_digest": array_digest(
            train_idx, calib_idx, test_pool_idx, explain_idx
        ),
        "raw_draw_digest": array_digest(
            data.raw_X, data.beta, data.gamma, data.delta, data.z, data.e
        ),
        "marginal_background": str(cfg["marginal_background"]),
        "marginal_background_size": int(len(X_train) + len(X_calib)),
        "conditional_ridge": float(cfg.get("conditional_ridge", 1e-10)),
        "marginal_resample_seed": marginal_seed,
        "marginal_distribution_background": str(
            cfg["marginal_distribution_background"]
        ),
        "marginal_distribution_background_size": int(len(X_calib)),
        "joint_marginal_background": str(cfg["joint_marginal_background"]),
        "joint_marginal_background_size": int(len(X_calib)),
        "joint_marginal_draw_count": int(cfg["joint_marginal_draws"]),
        "joint_marginal_seed": joint_marginal_seed,
        "joint_marginal_convergence_draws": joint_convergence_draws,
    }
    return {
        **baseline_results,
        "metadata_json": json.dumps(metadata, sort_keys=True),
        "X_full": data.X,
        "y_full": data.y,
        "train_indices": train_idx,
        "calib_indices": calib_idx,
        "test_pool_indices": test_pool_idx,
        "test_indices": explain_idx,
        "explain_indices": explain_idx,
        "X_test": X_test,
        "X_explain": X_test,
        "y_explain": data.y[explain_idx],
        "covariance": data.covariance,
        "beta": data.beta,
        "gamma": data.gamma,
        "data_digest": array_digest(data.X, data.y),
        "config_json": json.dumps(cfg, sort_keys=True),
    }


def augment_empirical_product_marginal(
    cfg: dict,
    path: Path,
    rho_x: float,
    seed: int,
) -> None:
    """Add or refresh the full-component marginal arm in a raw cell."""

    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(payload["metadata_json"]))
    if not np.isclose(float(metadata["rho_x"]), float(rho_x)):
        raise ValueError(f"{path}: rho metadata mismatch")
    if int(metadata["data_seed"]) != int(seed):
        raise ValueError(f"{path}: seed metadata mismatch")

    X_full = np.asarray(payload["X_full"], dtype=float)
    y_full = np.asarray(payload["y_full"], dtype=float)
    train_idx = np.asarray(payload["train_indices"], dtype=int)
    calib_idx = np.asarray(payload["calib_indices"], dtype=int)
    explain_idx = np.asarray(payload["explain_indices"], dtype=int)
    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_calib, y_calib = X_full[calib_idx], y_full[calib_idx]
    X_explain = X_full[explain_idx]

    model_cfg = QuantileModelConfig(
        alpha_lower=float(cfg["alpha"]) / 2.0,
        alpha_upper=1.0 - float(cfg["alpha"]) / 2.0,
        random_state=int(seed),
        backend=str(cfg["backend"]),
        params=cfg.get("model_params"),
    )
    q_lo, q_hi = fit_quantile_models(X_train, y_train, model_cfg)
    prediction_digest = array_digest(
        q_lo.predict(X_explain),
        q_hi.predict(X_explain),
        q_lo.predict(X_calib),
        q_hi.predict(X_calib),
    )
    if prediction_digest != metadata["prediction_digest"]:
        raise ValueError(
            f"{path}: refitted prediction digest does not match raw artifact"
        )

    marginal_seed = int(seed) + int(
        cfg.get("marginal_resample_seed_offset", 271828)
    )
    marginal_mask_matrix, marginal_sample_indices = (
        sample_empirical_product_marginal(X_calib, marginal_seed)
    )
    marginal_results = compute_empirical_product_marginal(
        X_calib,
        y_calib,
        X_explain,
        marginal_mask_matrix,
        q_lo,
        q_hi,
        float(cfg["alpha"]),
        tuple(all_subsets(X_full.shape[1])),
    )
    payload.update(marginal_results)
    payload.update(
        {
            "marginal_baseline_name": np.asarray(MARGINAL_BASELINE_NAME),
            "marginal_mask_matrix": marginal_mask_matrix,
            "marginal_sample_indices": marginal_sample_indices,
            "marginal_mask_digest": np.asarray(
                array_digest(
                    marginal_mask_matrix, marginal_sample_indices
                )
            ),
            "config_json": np.asarray(json.dumps(cfg, sort_keys=True)),
        }
    )
    metadata.update(
        {
            "artifact_schema_version": 4,
            "marginal_resample_seed": marginal_seed,
            "marginal_distribution_background": str(
                cfg["marginal_distribution_background"]
            ),
            "marginal_distribution_background_size": int(len(X_calib)),
        }
    )
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True)
    )
    atomic_savez(path, **payload)


def augment_empirical_joint_marginal(
    cfg: dict,
    path: Path,
    rho_x: float,
    seed: int,
) -> None:
    """Add the donor-row joint-marginal expected-payoff arm."""

    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(payload["metadata_json"]))
    if not np.isclose(float(metadata["rho_x"]), float(rho_x)):
        raise ValueError(f"{path}: rho metadata mismatch")
    if int(metadata["data_seed"]) != int(seed):
        raise ValueError(f"{path}: seed metadata mismatch")

    X_full = np.asarray(payload["X_full"], dtype=float)
    y_full = np.asarray(payload["y_full"], dtype=float)
    train_idx = np.asarray(payload["train_indices"], dtype=int)
    calib_idx = np.asarray(payload["calib_indices"], dtype=int)
    explain_idx = np.asarray(payload["explain_indices"], dtype=int)
    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_calib, y_calib = X_full[calib_idx], y_full[calib_idx]
    X_explain = X_full[explain_idx]

    model_cfg = QuantileModelConfig(
        alpha_lower=float(cfg["alpha"]) / 2.0,
        alpha_upper=1.0 - float(cfg["alpha"]) / 2.0,
        random_state=int(seed),
        backend=str(cfg["backend"]),
        params=cfg.get("model_params"),
    )
    q_lo, q_hi = fit_quantile_models(X_train, y_train, model_cfg)
    prediction_digest = array_digest(
        q_lo.predict(X_explain),
        q_hi.predict(X_explain),
        q_lo.predict(X_calib),
        q_hi.predict(X_calib),
    )
    if prediction_digest != metadata["prediction_digest"]:
        raise ValueError(
            f"{path}: refitted prediction digest does not match raw artifact"
        )

    primary_draws = int(cfg["joint_marginal_draws"])
    convergence_cfg = cfg.get("joint_marginal_convergence")
    convergence_draws = None
    if convergence_cfg and (
        int(convergence_cfg["seed"]) == int(seed)
        and np.isclose(float(convergence_cfg["rho"]), float(rho_x))
    ):
        convergence_draws = int(convergence_cfg["draws"])
    stored_draws = max(primary_draws, int(convergence_draws or primary_draws))
    joint_seed = int(seed) + int(
        cfg.get("joint_marginal_seed_offset", 314159)
    )
    donor_rows, donor_indices = sample_empirical_joint_marginal(
        X_calib, stored_draws, joint_seed
    )
    subsets = tuple(all_subsets(X_full.shape[1]))
    joint_results = compute_empirical_joint_marginal(
        X_calib,
        y_calib,
        X_explain,
        donor_rows[:primary_draws],
        q_lo,
        q_hi,
        float(cfg["alpha"]),
        subsets,
    )
    convergence_results = {}
    if convergence_draws is not None:
        convergence = compute_empirical_joint_marginal_global(
            X_calib,
            y_calib,
            donor_rows[:convergence_draws],
            q_lo,
            q_hi,
            float(cfg["alpha"]),
            subsets,
        )
        convergence_results = {
            "joint_convergence_draw_count": np.asarray(convergence_draws),
            "phi_global_joint_marginal_convergence": convergence["phi_global"],
            "global_runtime_sec_joint_marginal_convergence": np.asarray(
                convergence["runtime_sec"]
            ),
            "efficiency_global_joint_marginal_convergence": np.asarray(
                convergence["efficiency"]
            ),
            "endpoint_global_joint_marginal_convergence": convergence[
                "endpoint"
            ],
        }
    payload.update(joint_results)
    payload.update(convergence_results)
    payload.update(
        {
            "joint_marginal_baseline_name": np.asarray(
                JOINT_MARGINAL_BASELINE_NAME
            ),
            "joint_marginal_draw_count": np.asarray(primary_draws),
            "joint_marginal_donor_rows": donor_rows,
            "joint_marginal_donor_indices": donor_indices,
            "joint_marginal_donor_digest": np.asarray(
                array_digest(donor_rows, donor_indices)
            ),
            "config_json": np.asarray(json.dumps(cfg, sort_keys=True)),
        }
    )
    metadata.update(
        {
            "artifact_schema_version": 5,
            "joint_marginal_background": str(
                cfg["joint_marginal_background"]
            ),
            "joint_marginal_background_size": int(len(X_calib)),
            "joint_marginal_draw_count": primary_draws,
            "joint_marginal_seed": joint_seed,
            "joint_marginal_convergence_draws": convergence_draws,
        }
    )
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True)
    )
    atomic_savez(path, **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config_full.json",
    )
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "cells",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--augment-marginal",
        action="store_true",
        help="augment an existing cell with the full-component marginal arm",
    )
    parser.add_argument(
        "--augment-joint-marginal",
        action="store_true",
        help="augment an existing cell with donor-row joint-marginal payoffs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    validate_config(cfg)
    if not any(np.isclose(args.rho, value) for value in cfg["correlations"]):
        raise SystemExit(f"rho={args.rho} is not in config correlations")
    if int(args.seed) not in [int(value) for value in cfg["data_seeds"]]:
        raise SystemExit(f"seed={args.seed} is not in config data_seeds")

    out_path = args.out_dir / (
        f"cell_seed{int(args.seed)}_{rho_label(float(args.rho))}.npz"
    )
    if args.augment_marginal and args.augment_joint_marginal:
        raise SystemExit("choose only one augmentation mode")
    if args.augment_marginal:
        if not out_path.is_file():
            raise SystemExit(f"cannot augment missing cell: {out_path}")
        print(
            f"[masking-baseline] augmenting marginal arm seed={args.seed} "
            f"rho={args.rho:.2f}",
            flush=True,
        )
        augment_empirical_product_marginal(
            cfg, out_path, float(args.rho), int(args.seed)
        )
        print(f"[masking-baseline] augmented {out_path}", flush=True)
        return
    if args.augment_joint_marginal:
        if not out_path.is_file():
            raise SystemExit(f"cannot augment missing cell: {out_path}")
        print(
            f"[masking-baseline] augmenting joint-marginal arm "
            f"seed={args.seed} rho={args.rho:.2f}",
            flush=True,
        )
        augment_empirical_joint_marginal(
            cfg, out_path, float(args.rho), int(args.seed)
        )
        print(f"[masking-baseline] augmented {out_path}", flush=True)
        return
    if out_path.exists() and not args.force:
        print(f"[masking-baseline] exists; skipping {out_path}", flush=True)
        return

    print(
        f"[masking-baseline] seed={args.seed} rho={args.rho:.2f} "
        f"config={cfg['name']}",
        flush=True,
    )
    result = run_cell(cfg, float(args.rho), int(args.seed))
    atomic_savez(out_path, **result)
    print(f"[masking-baseline] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
