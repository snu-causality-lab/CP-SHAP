"""
CQR (Conformalized Quantile Regression) on UCI Concrete (d=8): Figure 8 (Appendix B.4).

Mirrors experiments/cp_families/lacp_concrete but swaps the locally adaptive (f_hat, sigma_hat)
recipe for two quantile regressors q_lo (alpha/2) and q_hi (1 - alpha/2).

Two-game decomposition (Prop 3, identity transform — no log):
  W^T(x) = (q_hi(x) - q_lo(x)) + 2 Q
         = v_L(S=full)            + v_G(S=full)

  v_L(S) = q_hi(x_S) - q_lo(x_S)
  v_G(S) = 2 Q(S) where Q(S) is the (1-alpha)-quantile of CQR nonconformity
           scores max(q_lo(X_calib_S) - y_calib, y_calib - q_hi(X_calib_S))
"""
import os
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from joblib import Parallel, delayed

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root for src/
from src.conformal.cq import calibration_width_adjustment_from_models

np.random.seed(42)

BASE_MODEL = os.environ.get('BASE_MODEL', 'gbr').lower()
N_EST = int(os.environ.get('N_EST', 100))
MAX_DEPTH = int(os.environ.get('MAX_DEPTH', 3))
OUT_SUFFIX = os.environ.get('OUT_SUFFIX', '')

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR.parents[2] / "data" / "downloads"
RESULTS_DIR = EXP_DIR.parents[2] / "outputs" / "cp_families" / EXP_DIR.name


def make_quantile_regressor(seed, q):
    """Quantile regressor for level q (e.g. alpha/2 for q_lo, 1-alpha/2 for q_hi).
    GBR is the only sklearn family with stable quantile loss out-of-the-box.
    """
    if BASE_MODEL != 'gbr':
        raise ValueError("This experiment currently supports only BASE_MODEL=gbr (sklearn quantile loss).")
    return GradientBoostingRegressor(
        loss='quantile', alpha=q,
        n_estimators=N_EST, max_depth=MAX_DEPTH, random_state=seed,
    )


def load_dataset():
    df = pd.read_csv(DATA_DIR / "concrete.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1].tolist()
    return X, y, feature_names


def _eval_v_L(S, d, x_test, q_lo_model, q_hi_model, means):
    mask = np.zeros(d, dtype=bool)
    mask[list(S)] = True

    x_masked = np.copy(x_test)
    x_masked[~mask] = means[~mask]

    lo = q_lo_model.predict(x_masked.reshape(1, -1))[0]
    hi = q_hi_model.predict(x_masked.reshape(1, -1))[0]
    return S, hi - lo


def compute_v_L(x_test, q_lo_model, q_hi_model, means):
    d = len(x_test)
    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

    all_S = [S for r in range(d + 1) for S in itertools.combinations(range(d), r)]

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_eval_v_L)(S, d, x_test, q_lo_model, q_hi_model, means)
        for S in all_S
    )
    return {S: v for S, v in results}


def _eval_v_G(S, d, X_calib, y_calib, q_lo_model, q_hi_model, means, alpha):
    mask = np.zeros(d, dtype=bool)
    mask[list(S)] = True

    X_masked = np.copy(X_calib)
    X_masked[:, ~mask] = means[~mask]

    return S, calibration_width_adjustment_from_models(X_masked, y_calib, q_lo_model, q_hi_model, alpha=alpha)


def compute_v_G(X_calib, y_calib, q_lo_model, q_hi_model, means, alpha=0.1):
    d = X_calib.shape[1]
    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

    all_S = [S for r in range(d + 1) for S in itertools.combinations(range(d), r)]

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_eval_v_G)(S, d, X_calib, y_calib, q_lo_model, q_hi_model, means, alpha)
        for S in all_S
    )
    return {S: v for S, v in results}


def exact_shapley(v, d):
    phi = np.zeros(d)
    import math
    for i in range(d):
        for r in range(d):
            for S in itertools.combinations([j for j in range(d) if j != i], r):
                S_with_i = tuple(sorted(list(S) + [i]))
                weight = math.factorial(r) * math.factorial(d - r - 1) / math.factorial(d)
                marginal = v[S_with_i] - v[S]
                phi[i] += weight * marginal
    return phi


def run_one_config(seed, alpha, outdir, X, y, feature_names):
    done_marker = os.path.join(outdir, 'summary.csv')
    if os.path.exists(done_marker):
        print(f"[skip] {outdir} already done.")
        return
    os.makedirs(outdir, exist_ok=True)

    np.random.seed(seed)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=seed)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=seed)

    print(f"[seed={seed}, alpha={alpha:.2f}] training CQR quantile regressors (n_est={N_EST}, max_depth={MAX_DEPTH})...")
    q_lo_model = make_quantile_regressor(seed, alpha / 2.0)
    q_hi_model = make_quantile_regressor(seed, 1.0 - alpha / 2.0)
    q_lo_model.fit(X_train, y_train)
    q_hi_model.fit(X_train, y_train)

    means = np.mean(X_calib, axis=0)
    d = X.shape[1]

    print(f"[seed={seed}, alpha={alpha:.2f}] computing v_G...")
    v_G = compute_v_G(X_calib, y_calib, q_lo_model, q_hi_model, means, alpha=alpha)
    phi_G = exact_shapley(v_G, d)

    num_test = int(os.environ.get('NUM_TEST', 100))
    print(f"[seed={seed}, alpha={alpha:.2f}] computing v_L for {num_test} test points...")
    results_list = []

    for idx in range(min(num_test, len(X_test))):
        x_star = X_test[idx]
        v_L = compute_v_L(x_star, q_lo_model, q_hi_model, means)
        phi_L = exact_shapley(v_L, d)

        phi_T = phi_L + phi_G
        v_T = {S: v_L[S] + v_G[S] for S in v_L.keys()}
        phi_T_direct = exact_shapley(v_T, d)
        np.testing.assert_allclose(phi_T, phi_T_direct, atol=1e-12)

        for i, name in enumerate(feature_names):
            results_list.append({
                'seed': seed, 'alpha': alpha,
                'instance_idx': idx, 'feature': name,
                'phi_L': phi_L[i], 'phi_G': phi_G[i],
                'phi_T': phi_T[i],
                'phi_T_direct': phi_T_direct[i],
            })

    df_results = pd.DataFrame(results_list)
    df_results.to_csv(os.path.join(outdir, 'attributions.csv'), index=False)

    summary = df_results.groupby('feature').agg(
        mean_abs_phi_L=('phi_L', lambda x: np.mean(np.abs(x))),
        mean_phi_G=('phi_G', 'mean'),
        mean_abs_phi_T=('phi_T', lambda x: np.mean(np.abs(x)))
    ).reset_index()
    summary['seed'] = seed
    summary['alpha'] = alpha

    sign_cancellation = df_results.copy()
    sign_cancellation['opp_sign'] = np.sign(sign_cancellation['phi_L']) != np.sign(sign_cancellation['phi_G'])
    sign_cancellation['opp_sign'] = sign_cancellation['opp_sign'] & (np.abs(sign_cancellation['phi_L']) > 1e-4) & (np.abs(sign_cancellation['phi_G']) > 1e-4)
    opp_rate = sign_cancellation.groupby('feature')['opp_sign'].mean().reset_index()

    summary = summary.merge(opp_rate, on='feature')
    summary.rename(columns={'opp_sign': 'cancellation_rate'}, inplace=True)
    summary.to_csv(os.path.join(outdir, 'summary.csv'), index=False)
    print(f"  saved {outdir}/{{attributions,summary}}.csv")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, feature_names = load_dataset()

    seeds = [int(s) for s in os.environ.get('SEEDS', '42').split(',') if s.strip()]
    alphas = [float(a) for a in os.environ.get('ALPHAS', '0.1').split(',') if a.strip()]
    print(f"Sweep: base_model={BASE_MODEL}  n_est={N_EST}  max_depth={MAX_DEPTH}")
    print(f"  seeds={seeds}  alphas={alphas}  total={len(seeds)*len(alphas)} configs  out_suffix='{OUT_SUFFIX}'")

    for seed in seeds:
        for alpha in alphas:
            outdir = str(RESULTS_DIR / f"seed{seed}_alpha{alpha:.2f}{OUT_SUFFIX}")
            run_one_config(seed, alpha, outdir, X, y, feature_names)


if __name__ == "__main__":
    main()
