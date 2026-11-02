"""Synthetic conformal-uncertainty experiments on three data-generating processes.

Use reproduce/synthetic.py for the paper settings.

Split parameters: test_size is a fraction of all rows; calib_size is a fraction
of the remaining non-test rows. Values 0.2 and 0.375 give a 50/30/20 split.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.conformal.quantile import QuantileModelConfig, fit_quantile_models
from src.shapley.shapley_global import compute_global_shapley
from src.shapley.shapley_local import compute_local_shapley

from data.synthetic.synthetic_heteroscedastic import generate_heteroscedastic_data
from data.synthetic.synthetic_homoscedastic import generate_homoscedastic_data
from data.synthetic.synthetic_source_separation import generate_source_separation_data


def run_synthetic(
    setting: str,
    output_dir: str,
    seed: int = 42,
    n_samples: int = 20000,
    test_size: float = 0.2,
    calib_size: float = 0.2,
    alpha: float = 0.1,
    model_backend: str = "sklearn_gbrt",
    model_params: dict | None = None,
    n_jobs_global: int | None = None,
    n_jobs_local: int = 1,
):
    """
    Synthetic experiments runner.

    setting:
      - "heteroscedastic"
      - "homoscedastic"
      - "source_separation"

    Saves:
      data.csv, train.csv, calib.csv, test.csv
      phi_global.csv, phi_local.csv, phi_total.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------
    # 1) Data
    # --------------------------
    setting = setting.lower()
    if setting == "heteroscedastic":
        X, y, cols = generate_heteroscedastic_data(n_samples=n_samples, seed=seed)
    elif setting == "homoscedastic":
        X, y, cols = generate_homoscedastic_data(n_samples=n_samples, seed=seed)
    elif setting == "source_separation":
        X, y, cols = generate_source_separation_data(n=n_samples, seed=seed)
    else:
        raise ValueError(f"Unknown synthetic setting: {setting}")

    pd.DataFrame(X, columns=cols).assign(y=y).to_csv(os.path.join(output_dir, "data.csv"), index=False)

    # --------------------------
    # 2) Split
    # --------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, test_size=calib_size, random_state=seed)

    pd.DataFrame(X_train, columns=cols).assign(y=y_train).to_csv(os.path.join(output_dir, "train.csv"), index=False)
    pd.DataFrame(X_calib, columns=cols).assign(y=y_calib).to_csv(os.path.join(output_dir, "calib.csv"), index=False)
    pd.DataFrame(X_test,  columns=cols).assign(y=y_test ).to_csv(os.path.join(output_dir, "test.csv"),  index=False)
    
    # --------------------------
    # 3) Fit quantile models
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
    # 4) Global Shapley
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
    pd.DataFrame([phi_global], columns=cols).to_csv(os.path.join(output_dir, "phi_global.csv"), index=False)

    # --------------------------
    # 5) Local Shapley (on test)
    # --------------------------
    phi_local = compute_local_shapley(
        X=X_test,
        baseline=baseline,
        q_lo=q_lo,
        q_hi=q_hi,
        n_jobs=n_jobs_local,
        show_progress=True,
        impute_nan_x0=True,
    )
    pd.DataFrame(phi_local, columns=cols).to_csv(os.path.join(output_dir, "phi_local.csv"), index=False)

    # --------------------------
    # 6) Total = local + global
    # --------------------------
    phi_total = phi_local + phi_global[None, :]
    pd.DataFrame(phi_total, columns=cols).to_csv(os.path.join(output_dir, "phi_total.csv"), index=False)

    return {
        "setting": setting,
        "cols": cols,
        "phi_global": phi_global,
        "phi_local_shape": phi_local.shape,
        "output_dir": output_dir,
    }