# src/shapley/shapley_local.py

import math
from itertools import combinations
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm


def _eval_width(x: np.ndarray, q_lo, q_hi) -> float:
    """
    Width for a single input:
        f(x) = q_hi(x) - q_lo(x)
    """
    x = np.asarray(x, dtype=float).reshape(1, -1)
    lo = float(q_lo.predict(x)[0])
    hi = float(q_hi.predict(x)[0])
    return hi - lo


def shapley_local_single(
    x0: np.ndarray,
    baseline: np.ndarray,
    q_lo,
    q_hi,
    impute_nan_x0: bool = True,
) -> np.ndarray:
    """
    Exact local Shapley for interval width f(x)=hi(x)-lo(x) at a single instance x0.

    Coalition evaluation uses baseline for missing features:
        x_S = baseline, then replace j in S with x0[j].

    Caches up to 2**d coalition values for this query only; each is evaluated once.
    If impute_nan_x0=True, any NaN in x0 is replaced by baseline before Shapley computation.
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    baseline = np.asarray(baseline, dtype=float).reshape(-1)

    d = x0.shape[0]
    if baseline.shape[0] != d:
        raise ValueError("baseline must have shape (d,) matching x0.")

    if impute_nan_x0:
        x0 = np.where(np.isnan(x0), baseline, x0)

    # One cache per query; values are discarded when this call returns.
    v_cache = {}

    def coalition_value(S):
        key = frozenset(S)
        if key not in v_cache:
            x_S = baseline.copy()
            for j in key:
                x_S[j] = x0[j]
            v_cache[key] = _eval_width(x_S, q_lo, q_hi)
        return v_cache[key]

    phi = np.zeros(d, dtype=float)
    idx_all = list(range(d))
    fact_d = math.factorial(d)

    for k in idx_all:
        others = [i for i in idx_all if i != k]
        for s in range(len(others) + 1):
            weight = math.factorial(s) * math.factorial(d - s - 1) / fact_d
            for S in combinations(others, s):
                S = set(S)

                f_S = coalition_value(S)
                f_Sk = coalition_value(S | {k})

                phi[k] += weight * (f_Sk - f_S)

    return phi


def _worker_local(args):
    x0, baseline, q_lo, q_hi, impute_nan_x0 = args
    return shapley_local_single(x0, baseline, q_lo, q_hi, impute_nan_x0=impute_nan_x0)


def compute_local_shapley(
    X: np.ndarray,
    baseline: np.ndarray,
    q_lo,
    q_hi,
    n_jobs: int | None = None,
    use_ratio: float = 0.8,
    chunksize: int = 10,
    show_progress: bool = True,
    impute_nan_x0: bool = True,
) -> np.ndarray:
    """
    Compute local Shapley for each instance in X.

    Returns:
      phi_local: (n, d) ndarray
    """
    X = np.asarray(X, dtype=float)
    baseline = np.asarray(baseline, dtype=float).reshape(-1)

    n, d = X.shape
    if baseline.shape[0] != d:
        raise ValueError("baseline must have shape (d,) matching X.shape[1].")

    if n_jobs is None:
        n_jobs = max(1, int(cpu_count() * use_ratio))

    # Serial
    if n_jobs <= 1:
        out = []
        it = X
        if show_progress:
            it = tqdm(it, total=n, desc="[Local] phi(x)")
        for x0 in it:
            out.append(shapley_local_single(x0, baseline, q_lo, q_hi, impute_nan_x0=impute_nan_x0))
        return np.asarray(out)

    # Parallel
    args_list = [(x0, baseline, q_lo, q_hi, impute_nan_x0) for x0 in X]
    with Pool(n_jobs) as pool:
        it = pool.imap(_worker_local, args_list, chunksize=chunksize)
        if show_progress:
            it = tqdm(it, total=len(args_list), desc=f"[Local] phi(x) (mp, {n_jobs} jobs)")
        out = list(it)
    return np.asarray(out)