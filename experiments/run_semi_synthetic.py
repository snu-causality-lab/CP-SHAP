"""Semi-synthetic clinical decision experiment with masked patient features.

Use reproduce/clinical.py for the paper settings.

Split parameters: test_size is a fraction of all rows; calib_size is a fraction
of the remaining non-test rows. Values 0.2 and 0.375 give a 50/30/20 split.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.semi_synthetic.semi_synthetic import generate_semi_synthetic_data
from src.conformal.quantile import QuantileModelConfig, fit_quantile_models
from src.shapley.shapley_global import compute_global_shapley
from src.shapley.shapley_local import compute_local_shapley


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)


def mask_random_k_features(
    X: np.ndarray,
    cols: list[str],
    k: int,
    seed: int = 42,
    mask_value: float = np.nan,
):
    """
    Randomly mask k features (set to NaN) per row.

    Returns:
      X_masked: (n,d)
      masked_tuples: list[tuple[str,...]] same length n
    """
    rng = np.random.default_rng(seed)

    X = np.asarray(X, dtype=float)
    n, d = X.shape

    if d < k:
        raise ValueError(f"Need at least k={k} features to mask, but got d={d}.")

    X_masked = X.copy()
    masked_tuples = []

    for i in range(n):
        idx = rng.choice(d, size=k, replace=False)
        X_masked[i, idx] = mask_value
        masked_tuples.append(tuple(cols[j] for j in idx))

    return X_masked, masked_tuples


def run_semi_synthetic(
    raw_csv_path: str,
    output_dir: str,
    seed: int = 42,
    alpha: float = 0.1,
    test_size: float = 0.2,
    calib_size: float = 0.2,  
    mask_k: int = 3,
    model_backend: str = "lightgbm",
    model_params: dict | None = None,
    n_jobs_global: int | None = None,
    n_jobs_local: int = 1,
):
    """
    Semi-synthetic experiment runner.

    Saves:
      - data.csv (original X + y, before split)
      - train.csv, calib.csv, test.csv (optionally masked), test_raw.csv (unmasked + masked tuple info)
      - phi_global.csv
      - phi_local.csv
      - phi_total.csv
    """

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------
    # 1) Data (semi-synthetic DGP)
    # --------------------------
    X, y, cols = generate_semi_synthetic_data(csv_path=raw_csv_path, seed=seed)

    df_all = pd.DataFrame(X, columns=cols)
    df_all["y"] = y
    df_all.to_csv(os.path.join(output_dir, "data.csv"), index=False)

    # --------------------------
    # 2) Split: train / calib / test
    # --------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=calib_size, random_state=seed
    )

    pd.DataFrame(X_train, columns=cols).assign(y=y_train).to_csv(
        os.path.join(output_dir, "train.csv"), index=False
    )
    pd.DataFrame(X_calib, columns=cols).assign(y=y_calib).to_csv(
        os.path.join(output_dir, "calib.csv"), index=False
    )

    # --------------------------
    # 3) Masking on test (decision-making scenario)
    # --------------------------
    X_test_masked, masked_tuples = mask_random_k_features(
        X_test, cols=cols, k=mask_k, seed=seed, mask_value=np.nan
    )

    df_test_raw = pd.DataFrame(X_test, columns=cols).assign(y=y_test)
    df_test_raw["masked"] = masked_tuples
    df_test_raw.to_csv(os.path.join(output_dir, "test_raw.csv"), index=False)

    df_test_masked = pd.DataFrame(X_test_masked, columns=cols).assign(y=y_test)
    df_test_masked.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # --------------------------
    # 4) Fit quantile models
    # --------------------------
    cfg = QuantileModelConfig(
        alpha_lower=0.05,
        alpha_upper=0.95,
        random_state=seed,
        backend=model_backend,
        params=model_params,
    )
    q_lo, q_hi = fit_quantile_models(X_train, y_train, cfg)

    # --------------------------
    # 5) Global Shapley (dataset-level uncertainty)
    # --------------------------
    baseline = np.nanmean(X_calib, axis=0)

    phi_global, _ = compute_global_shapley(
        X_calib=X_calib,
        y_calib=y_calib,
        q_lo=q_lo,
        q_hi=q_hi,
        alpha=alpha,
        baseline=baseline,
        n_jobs=n_jobs_global,
        show_progress=True,
    )
    pd.DataFrame([phi_global], columns=cols).to_csv(
        os.path.join(output_dir, "phi_global.csv"), index=False
    )

    # --------------------------
    # 6) Local Shapley on MASKED test (instance-level uncertainty)
    # --------------------------
    # NaN in x0 are imputed by baseline inside compute_local_shapley (impute_nan_x0=True)
    phi_local = compute_local_shapley(
        X=X_test_masked,
        baseline=baseline,
        q_lo=q_lo,
        q_hi=q_hi,
        n_jobs=n_jobs_local,
        show_progress=True,
        impute_nan_x0=True,
    )
    pd.DataFrame(phi_local, columns=cols).to_csv(
        os.path.join(output_dir, "phi_local.csv"), index=False
    )

    # --------------------------
    # 7) Total = local + global
    # --------------------------
    phi_total = phi_local + phi_global[None, :]
    pd.DataFrame(phi_total, columns=cols).to_csv(
        os.path.join(output_dir, "phi_total.csv"), index=False
    )

    return {
        "cols": cols,
        "phi_global": phi_global,
        "phi_local_shape": phi_local.shape,
        "output_dir": output_dir,
    }