# src/shapley/shapley_global.py

import math
from itertools import combinations
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from src.conformal.cq import calibration_width_adjustment_from_models


def masked_calib_matrix(
    X_calib: np.ndarray,
    baseline: np.ndarray,
    S: tuple[int, ...],
) -> np.ndarray:
    """
    Construct masked calibration matrix X^(S):
      - features in S: take from X_calib
      - others: set to baseline
    """
    X_calib = np.asarray(X_calib, dtype=float)
    baseline = np.asarray(baseline, dtype=float).reshape(-1)

    X_masked = np.tile(baseline, (X_calib.shape[0], 1))
    if len(S) > 0:
        X_masked[:, list(S)] = X_calib[:, list(S)]
    return X_masked


def all_subsets(d: int):
    """Return list of all subsets S of [0..d-1] as sorted tuples."""
    sets = []
    for k in range(d + 1):
        for S in combinations(range(d), k):
            sets.append(tuple(S))
    return sets


def _worker_vS(args):
    """
    Worker for parallel computation of v(S).
    args = (S, X_calib, y_calib, baseline, q_lo, q_hi, alpha)
    """
    S, X_calib, y_calib, baseline, q_lo, q_hi, alpha = args
    X_masked = masked_calib_matrix(X_calib, baseline, S)
    vS = calibration_width_adjustment_from_models(X_masked, y_calib, q_lo, q_hi, alpha=alpha)
    return S, vS


def compute_v_cache(
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    baseline: np.ndarray,
    q_lo,
    q_hi,
    alpha: float = 0.1,
    n_jobs: int | None = None,
    use_ratio: float = 0.8,
    show_progress: bool = True,
) -> dict:
    """
    Compute v(S) for all subsets S using calibration data.

    v(S) = calibration width adjustment (2*q_hat) under masked calibration inputs.

    Returns:
      v_cache: dict mapping S(tuple) -> v(S) float
    """
    X_calib = np.asarray(X_calib, dtype=float)
    y_calib = np.asarray(y_calib, dtype=float).reshape(-1)
    baseline = np.asarray(baseline, dtype=float).reshape(-1)

    n, d = X_calib.shape
    if baseline.shape[0] != d:
        raise ValueError("baseline must have shape (d,) matching X_calib.shape[1].")

    subsets = all_subsets(d)

    # Decide parallelism
    if n_jobs is None:
        n_jobs = max(1, int(cpu_count() * use_ratio))

    # Serial
    if n_jobs <= 1:
        v_cache = {}
        it = subsets
        if show_progress:
            it = tqdm(it, desc="[Global] v(S)")
        for S in it:
            X_masked = masked_calib_matrix(X_calib, baseline, S)
            v_cache[S] = calibration_width_adjustment_from_models(X_masked, y_calib, q_lo, q_hi, alpha=alpha)
        return v_cache

    # Parallel
    args_list = [(S, X_calib, y_calib, baseline, q_lo, q_hi, alpha) for S in subsets]
    v_cache = {}
    with Pool(n_jobs) as pool:
        it = pool.imap(_worker_vS, args_list)
        if show_progress:
            it = tqdm(it, total=len(args_list), desc=f"[Global] v(S) (mp, {n_jobs} jobs)")
        for S, vS in it:
            v_cache[S] = float(vS)
    return v_cache


def shapley_from_v_cache(v_cache: dict, d: int) -> np.ndarray:
    """
    Compute Shapley values phi from cached v(S) for all subsets.

    phi_j = sum_{S subset of N \\ {j}} w(S) [v(S ∪ {j}) - v(S)]
    """
    phi = np.zeros(d, dtype=float)
    fact_d = math.factorial(d)

    for j in range(d):
        others = [k for k in range(d) if k != j]
        for k in range(len(others) + 1):
            weight = math.factorial(k) * math.factorial(d - k - 1) / fact_d
            for S_subset in combinations(others, k):
                S = tuple(sorted(S_subset))
                Sj = tuple(sorted(S_subset + (j,)))
                phi[j] += weight * (v_cache[Sj] - v_cache[S])

    return phi


def compute_global_shapley(
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    q_lo,
    q_hi,
    alpha: float = 0.1,
    baseline: np.ndarray | None = None,
    n_jobs: int | None = None,
    use_ratio: float = 0.8,
    show_progress: bool = True,
):
    """
    End-to-end global Shapley.

    baseline default: mean of X_calib (feature-wise)
    Returns:
      phi_global: (d,) ndarray
      v_cache: dict for reuse (optional)
    """
    X_calib = np.asarray(X_calib, dtype=float)
    y_calib = np.asarray(y_calib, dtype=float).reshape(-1)
    n, d = X_calib.shape

    if baseline is None:
        baseline = np.nanmean(X_calib, axis=0)

    v_cache = compute_v_cache(
        X_calib=X_calib,
        y_calib=y_calib,
        baseline=baseline,
        q_lo=q_lo,
        q_hi=q_hi,
        alpha=alpha,
        n_jobs=n_jobs,
        use_ratio=use_ratio,
        show_progress=show_progress,
    )
    phi_global = shapley_from_v_cache(v_cache, d)
    return phi_global, v_cache