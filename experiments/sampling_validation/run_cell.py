"""
Single-cell runner for sampling-vs-exact Shapley validation (Appendix A.3 and B.1).

Each invocation runs ONE cell: (method, d, budget, seed) and saves
phi_global (d,), phi_local (n_test, d), and wall-clock numbers to a per-cell
.npz file. summarize.py builds the validation table from the cells.

Cell config:
  method ∈ {"exact", "sampled"}
  d      ∈ {10, 15}      (agreement check reported in Appendix B.1)
  budget : int           (permutation budget; 256 = 128 forward/reverse pairs; ignored for exact)
  seed   ∈ {0, 1, 2, 3, 4}

Single-thread enforced before numpy import.

Usage (from the repository root):
    python experiments/sampling_validation/run_cell.py --method sampled --d 15 --n_test 5 --budget 256 --seed 0 --out outputs/sampling_validation/cells/cell_sampled_d15_b256_seed0.npz
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.conformal.quantile import QuantileModelConfig, fit_quantile_models  # noqa: E402
from src.shapley.shapley_global import compute_global_shapley  # noqa: E402
from src.shapley.shapley_local import shapley_local_single  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
from sampler import antithetic_perm_global, antithetic_perm_local  # noqa: E402


def generate_synthetic(d: int, n: int, seed: int):
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
            f"Not enough samples: n={len(X)} but n_calib+n_test={n_calib + n_test}"
        )
    tr = idx[:n_train]
    ca = idx[n_train : n_train + n_calib]
    te = idx[n_train + n_calib : n_train + n_calib + n_test]
    return (X[tr], y[tr]), (X[ca], y[ca]), (X[te], y[te])


def run_cell(method, d, budget, seed, n_samples, n_calib, n_test, alpha, backend):
    print(
        f"[sampling_validation] cell method={method} d={d} budget={budget} seed={seed} backend={backend}",
        flush=True,
    )

    if n_samples < n_calib + n_test + 200:
        n_samples = n_calib + n_test + 1000
        print(f"[sampling_validation] bumped n_samples to {n_samples}", flush=True)

    X, y = generate_synthetic(d=d, n=n_samples, seed=seed)
    (X_tr, y_tr), (X_ca, y_ca), (X_te, y_te) = split(X, y, seed, n_calib, n_test)

    cfg = QuantileModelConfig(
        alpha_lower=alpha / 2,
        alpha_upper=1 - alpha / 2,
        random_state=seed,
        backend=backend,
    )
    q_lo, q_hi = fit_quantile_models(X_tr, y_tr, cfg)
    baseline = np.nanmean(X_ca, axis=0)

    # Sampler RNG (independent of the data RNG).
    sampler_rng = np.random.default_rng(seed + 100_000)

    # ---- Offline: phi_global ----
    t0 = time.perf_counter()
    if method == "exact":
        phi_global, _ = compute_global_shapley(
            X_calib=X_ca,
            y_calib=y_ca,
            q_lo=q_lo,
            q_hi=q_hi,
            alpha=alpha,
            baseline=baseline,
            n_jobs=1,
            show_progress=False,
        )
    elif method == "sampled":
        phi_global = antithetic_perm_global(
            X_calib=X_ca,
            y_calib=y_ca,
            q_lo=q_lo,
            q_hi=q_hi,
            alpha=alpha,
            baseline=baseline,
            budget=budget,
            rng=sampler_rng,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    offline_sec = time.perf_counter() - t0

    # ---- Online: phi_local per query ----
    n_q = min(n_test, len(X_te))
    phi_local = np.zeros((n_q, d), dtype=float)
    t0 = time.perf_counter()
    for i in range(n_q):
        if method == "exact":
            phi_local[i] = shapley_local_single(
                X_te[i], baseline=baseline, q_lo=q_lo, q_hi=q_hi, impute_nan_x0=True
            )
        else:
            phi_local[i] = antithetic_perm_local(
                X_te[i], baseline=baseline, q_lo=q_lo, q_hi=q_hi,
                budget=budget, rng=sampler_rng,
            )
    online_total_sec = time.perf_counter() - t0
    online_per_query_sec = online_total_sec / n_q if n_q > 0 else float("nan")

    print(
        f"[sampling_validation]   offline {offline_sec:.3f}s  online {online_per_query_sec*1000:.2f} ms/query"
        f"  ({n_q} queries)",
        flush=True,
    )

    return {
        "method": method,
        "d": int(d),
        "budget": int(budget),
        "seed": int(seed),
        "n_calib": int(n_calib),
        "n_test": int(n_q),
        "n_samples": int(n_samples),
        "backend": backend,
        "alpha": float(alpha),
        "offline_sec": float(offline_sec),
        "online_per_query_sec": float(online_per_query_sec),
        "online_total_sec": float(online_total_sec),
        "phi_global": phi_global,
        "phi_local": phi_local,
        "X_test": X_te,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["exact", "sampled"], required=True)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--budget", type=int, default=0,
                   help="Permutation budget: 256 gives 128 forward/reverse pairs; ignored for exact.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n_samples", type=int, default=4000)
    p.add_argument("--n_calib", type=int, default=500)
    p.add_argument("--n_test", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument(
        "--backend", default="sklearn_gbrt", choices=["sklearn_gbrt", "lightgbm"]
    )
    p.add_argument("--out", required=True, help="per-cell .npz output path")
    return p.parse_args()


def main():
    args = parse_args()
    out = run_cell(
        args.method,
        args.d,
        args.budget,
        args.seed,
        args.n_samples,
        args.n_calib,
        args.n_test,
        args.alpha,
        args.backend,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # np.savez requires arrays; scalars get wrapped automatically.
    np.savez(
        out_path,
        method=out["method"],
        d=out["d"],
        budget=out["budget"],
        seed=out["seed"],
        n_calib=out["n_calib"],
        n_test=out["n_test"],
        n_samples=out["n_samples"],
        backend=out["backend"],
        alpha=out["alpha"],
        offline_sec=out["offline_sec"],
        online_per_query_sec=out["online_per_query_sec"],
        online_total_sec=out["online_total_sec"],
        phi_global=out["phi_global"],
        phi_local=out["phi_local"],
        X_test=out["X_test"],
    )
    print(f"[sampling_validation] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
