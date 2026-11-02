"""
Runtime of global / local Shapley in the exact regime (Table 4, Appendix B.5).

The worst-case cost is O(2^d · n_c · C_eval); this script measures wall-clock time.

For d in {3, 5, 7, 10}:
  1) Generate synthetic heteroscedastic-style data with d features.
  2) Fit two quantile GBRT models (same config as run_synthetic.py).
  3) Time the offline φ^G computation (one-time, amortized across all queries).
  4) Time the per-query φ^L computation (mean over a small batch).

All times are wall-clock on the current host. Single thread by default so
timings are hardware-comparable (set OMP/MKL env vars to 1).

Outputs:
  results.csv     columns: d, n_calib, n_test, offline_phiG_sec,
                   online_phiL_sec_per_query, online_phiL_sec_total
"""

from __future__ import annotations

import os

# Force single-thread BLAS BEFORE numpy/sklearn import so wall-clock timings
# are hardware-comparable when the script is invoked directly (without a
# wrapper). setdefault preserves user/CI overrides via the environment.
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

# Repo root importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.conformal.quantile import QuantileModelConfig, fit_quantile_models  # noqa: E402
from src.shapley.shapley_global import compute_global_shapley  # noqa: E402
from src.shapley.shapley_local import shapley_local_single  # noqa: E402


def generate_synthetic(d: int, n: int, seed: int):
    """Heteroscedastic-style DGP extended to arbitrary d.

    Mirrors synthetic_heteroscedastic.py (uniform X, sigma tied to two features).
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    # pick two feature indices for sigma to depend on; spread them if d is tiny
    i1 = 0
    i2 = d - 1
    sigma = 0.05 + 0.95 * (X[:, i1] + X[:, i2]) / 2.0
    y = rng.normal(0.0, sigma, n)
    cols = [f"X{i + 1}" for i in range(d)]
    return X, y, cols


def split(X: np.ndarray, y: np.ndarray, seed: int, n_calib: int, n_test: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = len(X) - n_calib - n_test
    assert n_train > 0, "Not enough samples; increase --n_samples"
    tr = idx[:n_train]
    ca = idx[n_train : n_train + n_calib]
    te = idx[n_train + n_calib : n_train + n_calib + n_test]
    return (X[tr], y[tr]), (X[ca], y[ca]), (X[te], y[te])


def run_one_d(d: int, args) -> dict:
    print(f"[runtime] d={d} ...", flush=True)
    X, y, _cols = generate_synthetic(d=d, n=args.n_samples, seed=args.seed)
    (X_tr, y_tr), (X_ca, y_ca), (X_te, y_te) = split(
        X, y, seed=args.seed, n_calib=args.n_calib, n_test=args.n_test
    )

    cfg = QuantileModelConfig(
        alpha_lower=0.05,
        alpha_upper=0.95,
        random_state=args.seed,
        backend="sklearn_gbrt",
    )
    q_lo, q_hi = fit_quantile_models(X_tr, y_tr, cfg)

    baseline = np.nanmean(X_ca, axis=0)

    # ----- Offline: phi_global (one-time) -----
    t0 = time.perf_counter()
    phi_global, _ = compute_global_shapley(
        X_calib=X_ca,
        y_calib=y_ca,
        q_lo=q_lo,
        q_hi=q_hi,
        alpha=0.1,
        baseline=baseline,
        n_jobs=1,                  # single process for hardware-comparable timings
        show_progress=False,
    )
    offline_phiG_sec = time.perf_counter() - t0

    # ----- Online: phi_local per query -----
    n_q = min(args.n_test, len(X_te))
    t0 = time.perf_counter()
    for i in range(n_q):
        _ = shapley_local_single(
            X_te[i], baseline=baseline, q_lo=q_lo, q_hi=q_hi, impute_nan_x0=True
        )
    online_total = time.perf_counter() - t0
    online_per_query = online_total / n_q if n_q > 0 else float("nan")

    row = {
        "d": d,
        "n_calib": len(X_ca),
        "n_test": n_q,
        "offline_phiG_sec": offline_phiG_sec,
        "online_phiL_sec_per_query": online_per_query,
        "online_phiL_sec_total": online_total,
        "phi_global_sum": float(np.sum(phi_global)),  # sanity
    }
    print(
        f"[runtime]   offline phi^G = {offline_phiG_sec:.3f}s  "
        f"online phi^L = {online_per_query*1000:.2f} ms/query  ({n_q} queries)"
    )
    return row


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=str, default=str(Path(__file__).parent))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[3, 5, 7, 10],
        help="Feature dimensions to benchmark.",
    )
    p.add_argument("--n_samples", type=int, default=4000)
    p.add_argument("--n_calib", type=int, default=500)
    p.add_argument("--n_test", type=int, default=20,
                   help="Number of online queries to time (mean reported).")
    p.add_argument("--smoke", action="store_true",
                   help="Use smaller dims + fewer samples/queries.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.dims = [3, 5]
        args.n_samples = 300
        args.n_calib = 60
        args.n_test = 5

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [run_one_d(d, args) for d in args.dims]
    df = pd.DataFrame(rows)

    out_csv = out_dir / ("results_smoke.csv" if args.smoke else "results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[runtime] saved to {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
