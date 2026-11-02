"""
Antithetic-permutation Shapley estimators for the local and global
games. Each estimator uses paired permutations
(forward + reverse) for variance reduction (Mitchell et al. 2022).

Local game value:   v_L(S, x) = q_hi(x_S) - q_lo(x_S)
                    where x_S has features in S taken from x, others from baseline.
Global game value:  v_G(S) = 2 * q_hat( q_lo(X_c^{(S)}), q_hi(X_c^{(S)}); y_c )
                    where X_c^{(S)} is the block-masked calibration matrix.

Both estimators are unbiased and reuse the value functions in src/conformal/ and src/shapley/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.conformal.cq import calibration_width_adjustment_from_models  # noqa: E402
from src.shapley.shapley_global import masked_calib_matrix  # noqa: E402


def _local_width(x_row: np.ndarray, q_lo, q_hi) -> float:
    x = x_row.reshape(1, -1)
    return float(q_hi.predict(x)[0] - q_lo.predict(x)[0])


def antithetic_perm_local(
    x: np.ndarray,
    baseline: np.ndarray,
    q_lo,
    q_hi,
    budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Antithetic-permutation local Shapley for a single instance x.

    `budget` is the total number of permutations (forward + reverse pairs).
    Returns phi of shape (d,).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    d = x.shape[0]

    n_pairs = max(1, budget // 2)
    phi = np.zeros(d, dtype=float)

    v_empty = _local_width(baseline, q_lo, q_hi)

    for _ in range(n_pairs):
        perm = rng.permutation(d)
        for direction in (perm, perm[::-1]):
            x_masked = baseline.copy()
            v_prev = v_empty
            for j in direction:
                x_masked[j] = x[j]
                v_new = _local_width(x_masked, q_lo, q_hi)
                phi[j] += v_new - v_prev
                v_prev = v_new

    phi /= 2 * n_pairs
    return phi


def antithetic_perm_global(
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    q_lo,
    q_hi,
    alpha: float,
    baseline: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Antithetic-permutation global Shapley over calibration features.

    `budget` is the total number of permutations (paired forward + reverse).
    Returns phi of shape (d,). Coalition values are cached so repeated
    coalitions across permutations are evaluated only once.
    """
    X_calib = np.asarray(X_calib, dtype=float)
    y_calib = np.asarray(y_calib, dtype=float).reshape(-1)
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    d = X_calib.shape[1]

    n_pairs = max(1, budget // 2)
    phi = np.zeros(d, dtype=float)

    v_cache: dict[tuple[int, ...], float] = {}

    def v(S_tuple: tuple[int, ...]) -> float:
        if S_tuple not in v_cache:
            X_masked = masked_calib_matrix(X_calib, baseline, S_tuple)
            v_cache[S_tuple] = float(
                calibration_width_adjustment_from_models(X_masked, y_calib, q_lo, q_hi, alpha=alpha)
            )
        return v_cache[S_tuple]

    v_empty = v(())

    for _ in range(n_pairs):
        perm = rng.permutation(d)
        for direction in (perm, perm[::-1]):
            S: tuple[int, ...] = ()
            v_prev = v_empty
            for j in direction:
                S_new = tuple(sorted(S + (int(j),)))
                v_new = v(S_new)
                phi[j] += v_new - v_prev
                S = S_new
                v_prev = v_new

    phi /= 2 * n_pairs
    return phi
