"""
Calibration-set bootstrap stability on UCI Concrete (LACP): the phi^G interval in Figure 2.

Holds (data, fitted f_hat, fitted sigma_hat, X_test) fixed; only the
calibration-set realisation varies across N_BOOTSTRAPS bootstrap replicas.
Quantifies how stable phi^G is under cal-set choice for a fitted CP family.

Reports per-feature mean / std / CV / 95% CI of phi^G plus pairwise Spearman
of |phi^G| across replicas.
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

EXP_DIR = Path(__file__).resolve().parent
CP_FAMILIES_DIR = EXP_DIR.parent
RESULTS_DIR = EXP_DIR.parents[2] / "outputs" / "cp_families" / EXP_DIR.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_bootstraps', type=int, default=int(os.environ.get('N_BOOTSTRAPS', 30)))
    ap.add_argument('--seed', type=int, default=int(os.environ.get('SEED', 42)))
    ap.add_argument('--alpha', type=float, default=float(os.environ.get('ALPHA', 0.1)))
    ap.add_argument('--n_calib', type=int, default=int(os.environ.get('N_CALIB', 0)),
                    help='0 = use the natural calib split size')
    args = ap.parse_args()

    sys.path.insert(0, str(CP_FAMILIES_DIR / "lacp_concrete"))
    import importlib
    mod = importlib.import_module('main')
    importlib.reload(mod)

    X, y, feats = mod.load_dataset()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=args.seed)
    X_calib_full, X_test, y_calib_full, y_test = train_test_split(
        X_temp, y_temp, test_size=0.4, random_state=args.seed)
    n_calib = args.n_calib if args.n_calib > 0 else len(X_calib_full)

    base_model = mod.make_regressor(args.seed)
    base_model.fit(X_train, y_train)
    res_train = np.abs(y_train - base_model.predict(X_train))
    scale_model = mod.make_regressor(args.seed)
    # Predict absolute residual scale, not residual variance.
    scale_model.fit(X_train, res_train)

    d = X.shape[1]
    rng = np.random.default_rng(args.seed)
    n_total_pool = len(X_calib_full)

    print(f"[lacp_calibration_bootstrap] d={d} n_pool={n_total_pool} n_calib={n_calib} n_bootstraps={args.n_bootstraps} alpha={args.alpha}")

    phi_G_replicas = np.zeros((args.n_bootstraps, d), dtype=float)
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    per_bootstrap_rows = []

    # Manuscript §3 convention: interventional baseline = mean over D_c (calibration).
    # Stability test holds the BASELINE fixed across replicas (uses X_calib_full mean,
    # not per-replica X_cb mean) — this isolates "phi_G stable under cal-set realisation"
    # without confounding from baseline drift.
    means = X_calib_full.mean(axis=0)

    t0 = time.perf_counter()
    for r in range(args.n_bootstraps):
        idx = rng.choice(n_total_pool, size=n_calib, replace=True)
        X_cb = X_calib_full[idx]
        y_cb = y_calib_full[idx]

        v_G = mod.compute_v_G(X_cb, y_cb, base_model, scale_model, means, alpha=args.alpha)
        phi_G = mod.exact_shapley(v_G, d)
        phi_G_replicas[r] = phi_G

        for j, name in enumerate(feats):
            per_bootstrap_rows.append({
                'bootstrap_id': r, 'feature': name, 'phi_G': float(phi_G[j]),
                # Resampled feature mean; the masking baseline stays fixed at means[j].
                'bootstrap_sample_mean': float(X_cb[:, j].mean()),
            })
        if (r + 1) % 5 == 0 or r + 1 == args.n_bootstraps:
            elapsed = time.perf_counter() - t0
            print(f"[lacp_calibration_bootstrap] {r+1}/{args.n_bootstraps} replicas done ({elapsed:.1f}s)")

    pd.DataFrame(per_bootstrap_rows).to_csv(out_dir / 'per_bootstrap.csv', index=False)

    rows = []
    for j, name in enumerate(feats):
        col = phi_G_replicas[:, j]
        rows.append({
            'feature': name,
            'phi_G_mean': float(col.mean()),
            'phi_G_std': float(col.std(ddof=1)),
            'phi_G_cv': float(col.std(ddof=1) / (np.abs(col.mean()) + 1e-12)),
            'phi_G_ci95_lo': float(np.quantile(col, 0.025)),
            'phi_G_ci95_hi': float(np.quantile(col, 0.975)),
        })
    pd.DataFrame(rows).to_csv(out_dir / 'per_feature.csv', index=False)

    pairwise = []
    n = phi_G_replicas.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(np.abs(phi_G_replicas[i]), np.abs(phi_G_replicas[j]))
            pairwise.append(r)
    top1_match = []
    top3_overlap = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = phi_G_replicas[i], phi_G_replicas[j]
            top1_match.append(int(np.argmax(np.abs(a))) == int(np.argmax(np.abs(b))))
            ai = set(np.argsort(-np.abs(a))[:3])
            bi = set(np.argsort(-np.abs(b))[:3])
            top3_overlap.append(len(ai & bi) / 3.0)

    summary = {
        'family': 'locally_adaptive_concrete',
        'd': d, 'n_calib': n_calib, 'n_bootstraps': args.n_bootstraps,
        'mean_pairwise_spearman_abs': float(np.nanmean(pairwise)),
        'top1_stability': float(np.mean(top1_match)),
        'top3_overlap_mean': float(np.mean(top3_overlap)),
    }
    pd.DataFrame([summary]).to_csv(out_dir / 'summary.csv', index=False)
    print(f"[lacp_calibration_bootstrap] saved {out_dir}/{{per_bootstrap,per_feature,summary}}.csv")


if __name__ == '__main__':
    main()
