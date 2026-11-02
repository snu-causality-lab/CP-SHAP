# src/conformal/quantile.py

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np


@dataclass
class QuantileModelConfig:
    """
    Configuration for quantile regression models.
    """
    alpha_lower: float = 0.05
    alpha_upper: float = 0.95
    random_state: int = 42

    # which backend to use: "sklearn_gbrt" or "lightgbm"
    backend: str = "lightgbm"

    # optional backend-specific hyperparameters
    params: Optional[Dict[str, Any]] = None


def fit_quantile_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: QuantileModelConfig,
):
    """
    Fit lower/upper quantile models and return (q_lo, q_hi).

    Returns:
        q_lo, q_hi: fitted models with .predict(X) available
    """
    backend = config.backend.lower()
    params = config.params or {}

    if backend == "sklearn_gbrt":
        from sklearn.ensemble import GradientBoostingRegressor

        q_lo = GradientBoostingRegressor(
            loss="quantile",
            alpha=config.alpha_lower,
            random_state=config.random_state,
            subsample=0.7,
            **params,
        )
        q_hi = GradientBoostingRegressor(
            loss="quantile",
            alpha=config.alpha_upper,
            random_state=config.random_state,
            subsample=0.7,
            **params,
        )

    elif backend == "lightgbm":
        from lightgbm import LGBMRegressor

        # sensible defaults; can be overridden by config.params
        base_params = dict(
            objective="quantile",
            n_estimators=100,
            max_depth=3,
            num_leaves=7,
            min_child_samples=50,
            min_split_gain=0.1,
            subsample=0.6,
            colsample_bytree=0.6,
            learning_rate=0.1,
            random_state=config.random_state,
            # Keep LightGBM n_jobs=1, including in config.params: multiple threads can hang in forked Shapley workers.
            # Use --workers / --jobs-per-run in reproduce/ for process-level parallelism.
            n_jobs=1,
        )
        base_params.update(params)

        q_lo = LGBMRegressor(alpha=config.alpha_lower, **base_params)
        q_hi = LGBMRegressor(alpha=config.alpha_upper, **base_params)

    else:
        raise ValueError(f"Unknown backend: {config.backend}")

    q_lo.fit(X_train, y_train)
    q_hi.fit(X_train, y_train)

    return q_lo, q_hi


def predict_interval(
    q_lo,
    q_hi,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict (lo, hi) quantiles.
    """
    lo = np.asarray(q_lo.predict(X), dtype=float)
    hi = np.asarray(q_hi.predict(X), dtype=float)
    return lo, hi