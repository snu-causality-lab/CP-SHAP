"""
Runtime of global / local Shapley in the sampled regime (Table 4, Appendix B.5).

runtime_scaling.py covers d ∈ {3, 5, 7, 10} with the exact (2^d) Shapley enumeration.
This script extends to d ∈ {15, 20, 30} using the antithetic-permutation
estimator of experiments/sampling_validation (paired forward/reverse permutations
for variance reduction).

For each d:
  1) Synthetic heteroscedastic data with d features.
  2) Fit two quantile GBRT models (paper default backend).
  3) Time offline phi^G via antithetic sampling, fixed budget.
  4) Time per-query phi^L via antithetic sampling.
  5) Sanity: efficiency error
       phi-sum vs (v(N) - v(empty)),
     for both games. With antithetic sampling these should be 0 up to
     numerical jitter — efficiency holds *deterministically* per
     forward+reverse pair, so any drift exposes a sampler bug.

Single-threaded BLAS for hardware-comparable wall-clocks.

Outputs
-------
results.csv     d, budget, n_calib, n_test, offline_phiG_sec,
                online_phiL_sec_per_query, eff_err_global,
                eff_err_local_mean, eff_err_local_max
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

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
from src.shapley.shapley_global import masked_calib_matrix  # noqa: E402

# Reuse the antithetic sampler of experiments/sampling_validation.
SAMPLER_DIR = REPO_ROOT / "experiments" / "sampling_validation"
sys.path.insert(0, str(SAMPLER_DIR))
from sampler import antithetic_perm_global, antithetic_perm_local  # noqa: E402


def generate_synthetic(d: int, n: int, seed: int):
    """Heteroscedastic DGP extended to arbitrary d (same as runtime_scaling.py and sampling_validation)."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    sigma = 0.05 + 0.95 * (X[:, 0] + X[:, d - 1]) / 2.0
    y = rng.normal(0.0, sigma, n)
    return X, y


def split(X, y, seed, n_calib, n_test):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = len(X) - n_calib - n_test
    if n_train <= 0:
        raise ValueError(
            f"n={len(X)} too small for n_calib+n_test={n_calib + n_test}"
        )
    tr = idx[:n_train]
    ca = idx[n_train : n_train + n_calib]
    te = idx[n_train + n_calib : n_train + n_calib + n_test]
    return (X[tr], y[tr]), (X[ca], y[ca]), (X[te], y[te])


def _local_v(x_row: np.ndarray, q_lo, q_hi) -> float:
    x = x_row.reshape(1, -1)
    return float(q_hi.predict(x)[0] - q_lo.predict(x)[0])


def run_one_d(d: int, args) -> dict:
    print(f"[runtime] d={d} budget={args.budget} ...", flush=True)
    X, y = generate_synthetic(d=d, n=args.n_samples, seed=args.seed)
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

    sampler_rng = np.random.default_rng(args.seed + 100_000)

    # ---- Offline phi^G ----
    t0 = time.perf_counter()
    phi_global = antithetic_perm_global(
        X_calib=X_ca,
        y_calib=y_ca,
        q_lo=q_lo,
        q_hi=q_hi,
        alpha=args.alpha,
        baseline=baseline,
        budget=args.budget,
        rng=sampler_rng,
    )
    offline_sec = time.perf_counter() - t0

    # Efficiency check (global game)
    v_full_g = calibration_width_adjustment_from_models(
        masked_calib_matrix(X_ca, baseline, tuple(range(d))),
        y_ca,
        q_lo,
        q_hi,
        alpha=args.alpha,
    )
    v_empty_g = calibration_width_adjustment_from_models(
        masked_calib_matrix(X_ca, baseline, ()),
        y_ca,
        q_lo,
        q_hi,
        alpha=args.alpha,
    )
    eff_err_g = float(np.sum(phi_global) - (v_full_g - v_empty_g))

    # ---- Online phi^L per query ----
    n_q = min(args.n_test, len(X_te))
    phi_locals = np.zeros((n_q, d))
    t0 = time.perf_counter()
    for i in range(n_q):
        phi_locals[i] = antithetic_perm_local(
            X_te[i],
            baseline=baseline,
            q_lo=q_lo,
            q_hi=q_hi,
            budget=args.budget,
            rng=sampler_rng,
        )
    online_total = time.perf_counter() - t0
    online_per_query = online_total / n_q if n_q > 0 else float("nan")

    # Efficiency check (local game)
    v_empty_l = _local_v(baseline, q_lo, q_hi)
    eff_errs_l = np.array(
        [
            float(np.sum(phi_locals[i])) - (_local_v(X_te[i], q_lo, q_hi) - v_empty_l)
            for i in range(n_q)
        ]
    )

    row = {
        "d": int(d),
        "budget": int(args.budget),
        "n_calib": int(len(X_ca)),
        "n_test": int(n_q),
        "offline_phiG_sec": float(offline_sec),
        "online_phiL_sec_per_query": float(online_per_query),
        "online_phiL_sec_total": float(online_total),
        "eff_err_global": eff_err_g,
        "eff_err_local_mean": float(eff_errs_l.mean()),
        "eff_err_local_max": float(np.max(np.abs(eff_errs_l))),
    }
    print(
        f"[runtime]   offline {offline_sec:.2f}s  "
        f"online {online_per_query * 1000:.1f} ms/query ({n_q} q)  "
        f"eff_err_g={eff_err_g:+.2e}  eff_err_l_max={float(np.max(np.abs(eff_errs_l))):+.2e}",
        flush=True,
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
        default=[15, 20, 30],
        help="High-d settings to time (default extends the exact regime 3..10).",
    )
    p.add_argument("--n_samples", type=int, default=4000)
    p.add_argument("--n_calib", type=int, default=500)
    p.add_argument("--n_test", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument(
        "--budget",
        type=int,
        default=256,
        help="Permutation budget: 256 gives 128 forward/reverse pairs.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="d ∈ {12, 15} with smaller config (quick sanity).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.dims = [12, 15]
        args.n_samples = 1500
        args.n_calib = 200
        args.n_test = 4
        args.budget = 64

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [run_one_d(d, args) for d in args.dims]
    df = pd.DataFrame(rows)

    suffix = "_smoke" if args.smoke else ""
    out_csv = out_dir / f"results{suffix}.csv"

    df.to_csv(out_csv, index=False)
    print(f"\n[runtime] saved to {out_csv}")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
