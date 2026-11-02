"""Mondrian CQR on Folktables ACS Income (sex partition): Figure 4 (Appendix A.5) and Figure 8 (Appendix B.4)."""
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

N_EST = int(os.environ.get('N_EST', 100))
MAX_DEPTH = int(os.environ.get('MAX_DEPTH', 3))
OUT_SUFFIX = os.environ.get('OUT_SUFFIX', '')
PARTITION = os.environ.get('PARTITION', 'sex').lower()  # sex / race

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR.parents[2] / "data" / "downloads"
RESULTS_DIR = EXP_DIR.parents[2] / "outputs" / "cp_families" / EXP_DIR.name


def load_folktables():
    print(f"Loading actual folktables dataset (partition={PARTITION})...")
    df = pd.read_csv(DATA_DIR / "folktables_acs_income_CA.csv")

    if len(df) > 2000:
        df = df.sample(2000, random_state=42)

    y = df['label'].astype(float).values
    df_features = df.drop(columns=['label', 'group_race'])

    if PARTITION == 'sex':
        if 'SEX' not in df_features.columns:
            raise KeyError("'SEX' column missing from Folktables features")
        groups = df_features['SEX'].values
        df_players = df_features.drop(columns=['SEX'])
    elif PARTITION == 'race':
        # Prefer ACSIncome's standard 'group_race' (added in download_datasets.py);
        # fall back to RAC1P if present in features.
        if 'group_race' in df.columns:
            groups = df['group_race'].values
        elif 'RAC1P' in df_features.columns:
            groups = df_features['RAC1P'].values
        else:
            raise KeyError("no race column ('group_race' or 'RAC1P') found")
        # Drop the race column from features to avoid using it as a player too.
        drop_cols = [c for c in ('RAC1P',) if c in df_features.columns]
        df_players = df_features.drop(columns=drop_cols) if drop_cols else df_features
    else:
        raise ValueError(f"unknown PARTITION={PARTITION}")

    X = df_players.values
    feature_names = df_players.columns.tolist()
    return X, y, groups, feature_names

def get_coalition_means(X):
    return np.mean(X, axis=0)

def _eval_v_L(S, d, x_test, base_model_lo, base_model_hi, means):
    mask = np.zeros(d, dtype=bool)
    mask[list(S)] = True
    
    x_masked = np.copy(x_test)
    x_masked[~mask] = means[~mask]
    
    x_input = x_masked.reshape(1, -1)
    q_lo = base_model_lo.predict(x_input)[0]
    q_hi = base_model_hi.predict(x_input)[0]
    
    return S, q_hi - q_lo

def compute_v_L(x_test, base_model_lo, base_model_hi, means):
    d = len(x_test)
    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    
    all_S = [S for r in range(d + 1) for S in itertools.combinations(range(d), r)]
    
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_eval_v_L)(S, d, x_test, base_model_lo, base_model_hi, means)
        for S in all_S
    )
    return {S: v for S, v in results}

def _eval_v_G_mondrian(S, d, X_calib_g, y_calib_g, base_model_lo, base_model_hi, means, alpha):
    mask = np.zeros(d, dtype=bool)
    mask[list(S)] = True

    X_masked = np.copy(X_calib_g)
    X_masked[:, ~mask] = means[~mask]

    return S, calibration_width_adjustment_from_models(X_masked, y_calib_g, base_model_lo, base_model_hi, alpha=alpha)

def compute_v_G_mondrian(X_calib, y_calib, groups_calib, group_val, base_model_lo, base_model_hi, means, alpha=0.1):
    """
    Our Mondrian construction uses the CQR signed-distance score.
    Boström & Johansson (2020) develop Mondrian regressors with absolute-residual 
    or normalized-residual scores. Both are valid CP nonconformity scores under 
    our two-game framework. This is a design choice (Mondrian-CQR hybrid).
    """
    group_mask = groups_calib == group_val
    X_calib_g = X_calib[group_mask]
    y_calib_g = y_calib[group_mask]
    
    d = X_calib.shape[1]
    n_c_g = len(X_calib_g)
    
    if n_c_g == 0:
        v_G = {}
        for r in range(d + 1):
            for S in itertools.combinations(range(d), r):
                v_G[S] = 0.0
        return v_G
        
    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    all_S = [S for r in range(d + 1) for S in itertools.combinations(range(d), r)]
    
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_eval_v_G_mondrian)(S, d, X_calib_g, y_calib_g, base_model_lo, base_model_hi, means, alpha)
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

def run_one_config(seed, alpha, outdir, X, y, groups, feature_names):
    done_marker = os.path.join(outdir, 'summary.csv')
    if os.path.exists(done_marker):
        print(f"[skip] {outdir} already done.")
        return
    os.makedirs(outdir, exist_ok=True)

    np.random.seed(seed)
    X_train, X_temp, y_train, y_temp, g_train, g_temp = train_test_split(
        X, y, groups, test_size=0.5, random_state=seed)
    X_calib, X_test, y_calib, y_test, g_calib, g_test = train_test_split(
        X_temp, y_temp, g_temp, test_size=0.4, random_state=seed)

    print(f"[seed={seed}, alpha={alpha:.2f}] training CQR quantile regressors (n_est={N_EST}, max_depth={MAX_DEPTH})...")
    base_model_lo = GradientBoostingRegressor(
        loss='quantile', alpha=alpha / 2.0,
        n_estimators=N_EST, max_depth=MAX_DEPTH, random_state=seed)
    base_model_hi = GradientBoostingRegressor(
        loss='quantile', alpha=1 - alpha / 2.0,
        n_estimators=N_EST, max_depth=MAX_DEPTH, random_state=seed)
    base_model_lo.fit(X_train, y_train)
    base_model_hi.fit(X_train, y_train)

    # Manuscript §3 convention: D_c (calibration) mean, not training mean.
    # Note: for Mondrian, we use the *full* calibration mean as the global baseline,
    # not per-group baseline. Per-group baseline would be a stricter "Mondrian-pure"
    # variant; both are valid SHAP design choices (we follow the §3 convention).
    means = get_coalition_means(X_calib)
    d = X.shape[1]

    unique_groups = np.unique(g_train)
    print(f"[seed={seed}, alpha={alpha:.2f}] computing v_G per group {list(unique_groups)} ...")
    v_G_dict, phi_G_dict = {}, {}
    for g_val in unique_groups:
        v_G = compute_v_G_mondrian(X_calib, y_calib, g_calib, g_val,
                                    base_model_lo, base_model_hi, means, alpha=alpha)
        v_G_dict[g_val] = v_G
        phi_G_dict[g_val] = exact_shapley(v_G, d)

    num_test = int(os.environ.get('NUM_TEST', 100))
    print(f"[seed={seed}, alpha={alpha:.2f}] computing v_L for {num_test} test points...")
    results_list = []

    for idx in range(min(num_test, len(X_test))):
        x_star = X_test[idx]
        g_star = g_test[idx]
        v_L = compute_v_L(x_star, base_model_lo, base_model_hi, means)
        phi_L = exact_shapley(v_L, d)

        v_G = v_G_dict[g_star]
        phi_G = phi_G_dict[g_star]
        phi_T = phi_L + phi_G

        v_T = {S: v_L[S] + v_G[S] for S in v_L.keys()}
        phi_T_direct = exact_shapley(v_T, d)
        np.testing.assert_allclose(phi_T, phi_T_direct, atol=1e-12)

        for i, name in enumerate(feature_names):
            results_list.append({
                'seed': seed, 'alpha': alpha,
                'instance_idx': idx, 'group': g_star, 'feature': name,
                'phi_L': phi_L[i], 'phi_G': phi_G[i],
                'phi_T': phi_T[i],
                'phi_T_direct': phi_T_direct[i],
            })

    df_results = pd.DataFrame(results_list)
    df_results.to_csv(os.path.join(outdir, 'attributions.csv'), index=False)

    summary = df_results.groupby(['group', 'feature']).agg(
        mean_abs_phi_L=('phi_L', lambda x: np.mean(np.abs(x))),
        mean_phi_G=('phi_G', 'mean'),
        mean_abs_phi_T=('phi_T', lambda x: np.mean(np.abs(x)))
    ).reset_index()
    summary['seed'] = seed
    summary['alpha'] = alpha
    summary.to_csv(os.path.join(outdir, 'summary.csv'), index=False)
    print(f"  saved {outdir}/{{attributions,summary}}.csv")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, groups, feature_names = load_folktables()

    seeds = [int(s) for s in os.environ.get('SEEDS', '42').split(',') if s.strip()]
    alphas = [float(a) for a in os.environ.get('ALPHAS', '0.1').split(',') if a.strip()]
    print(f"Sweep: partition={PARTITION}  n_est={N_EST}  max_depth={MAX_DEPTH}")
    print(f"  seeds={seeds}  alphas={alphas}  total={len(seeds)*len(alphas)} configs  out_suffix='{OUT_SUFFIX}'")

    for seed in seeds:
        for alpha in alphas:
            outdir = str(RESULTS_DIR / f"seed{seed}_alpha{alpha:.2f}{OUT_SUFFIX}")
            run_one_config(seed, alpha, outdir, X, y, groups, feature_names)


if __name__ == "__main__":
    main()
