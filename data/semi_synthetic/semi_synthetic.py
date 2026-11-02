# data/semi_synthetic/semi_synthetic.py

import numpy as np
import pandas as pd


def generate_semi_synthetic_data(
    csv_path: str = "semi-syn-raw.csv",
    seed: int = 42,
):
    """
    Semi-synthetic risk regression DGP.

    Steps:
      1) Load raw clinical covariates from csv_path
      2) Robust-scale features: (x - median) / IQR, then clip to [-6, 6]
      3) Define a clinically-plausible latent risk via main effects + interactions
      4) Generate heteroskedastic outcome:
            risk = sigmoid(lp)
            y = clip(24 * risk + eps, 0, 24)
            eps ~ N(0, sigma(risk)^2), sigma(risk) = 0.7 + 2.5 * risk^2

    Returns:
      X: (n, 6) float ndarray
      y: (n,) float ndarray
      cols: list of feature names in order
    """
    rng = np.random.default_rng(seed)

    df0 = pd.read_csv(csv_path, index_col=0)

    cols = ["Lactate", "MAP", "O2Sat", "Resp", "Age", "pH"]
    X_df = df0[cols].astype(float).copy()

    # robust scaling (median / IQR)
    med = X_df.median()
    iqr = (X_df.quantile(0.75) - X_df.quantile(0.25)).replace(0, np.nan)
    Xz = ((X_df - med) / iqr).fillna(0).clip(-6, 6)

    lactate = Xz["Lactate"].to_numpy()
    map_    = Xz["MAP"].to_numpy()
    o2      = Xz["O2Sat"].to_numpy()
    resp    = Xz["Resp"].to_numpy()
    age     = Xz["Age"].to_numpy()
    ph      = Xz["pH"].to_numpy()

    # direction alignment: low MAP, low O2Sat, low pH => higher risk
    low_map = -map_
    low_o2  = -o2
    low_ph  = -ph

    # latent risk score with interactions
    lp = (
        1.3 * lactate
        + 1.1 * low_map
        + 0.9 * low_o2
        + 0.8 * resp
        + 0.6 * age
        + 1.0 * low_ph
        + 0.9 * (lactate * low_map)
        + 0.7 * (resp * low_o2)
        + 0.8 * (lactate * low_ph)
    )

    risk = 1.0 / (1.0 + np.exp(-lp))

    # heteroskedastic noise (high risk -> larger noise)
    sigma = 0.7 + 2.5 * (risk ** 2)
    eps = rng.normal(0.0, sigma, size=risk.shape[0])

    y = np.clip(24.0 * risk + eps, 0.0, 24.0)

    X = X_df.to_numpy(dtype=float)

    return X, y.astype(float), cols