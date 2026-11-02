"""Calibration quantiles and calibration width adjustments for conformal prediction.

conformal_qhat_from_scores is shared with locally adaptive conformal prediction.
The CQR wrappers use two-sided quantile-regression scores and return the calibration
adjustment 2*q_hat, not the total interval width. This adjustment can be negative;
the total CQR endpoint difference is q_hi(x) - q_lo(x) + 2*q_hat.
"""

from __future__ import annotations
import numpy as np
import math


def conformal_qhat_from_scores(scores: np.ndarray, alpha: float = 0.1) -> float:
    """
    Split-CP threshold from nonconformity scores:
        k = ceil((n+1) * (1-alpha)), clamped to [1, n]
        q_hat = V_(k)  (k-th order statistic, 1-indexed)
    """
    V = np.asarray(scores, dtype=float).reshape(-1)
    n = V.shape[0]
    if n == 0:
        raise ValueError("Empty calibration set.")

    k = math.ceil((n + 1) * (1 - alpha))
    k = min(max(k, 1), n)  # clamp to [1, n]

    q_hat = np.sort(V)[k - 1]
    return float(q_hat)


def conformal_qhat(
    y_calib: np.ndarray,
    y_pred_lo: np.ndarray,
    y_pred_hi: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Compute conformal quantile (q_hat) using calibration residuals for two-sided interval.

    Residual definition (CQR style):
        V_i = max( q_lo(x_i) - y_i,  y_i - q_hi(x_i) )

    Then:
        k = ceil((n+1) * (1-alpha))
        q_hat = V_(k)  (k-th order statistic, 1-indexed)

    Returns:
        q_hat (float)
    """
    y_calib = np.asarray(y_calib, dtype=float).reshape(-1)
    y_pred_lo = np.asarray(y_pred_lo, dtype=float).reshape(-1)
    y_pred_hi = np.asarray(y_pred_hi, dtype=float).reshape(-1)

    if not (len(y_calib) == len(y_pred_lo) == len(y_pred_hi)):
        raise ValueError("y_calib, y_pred_lo, y_pred_hi must have the same length.")

    V = np.maximum(y_pred_lo - y_calib, y_calib - y_pred_hi)
    return conformal_qhat_from_scores(V, alpha=alpha)


def calibration_width_adjustment(
    y_calib: np.ndarray,
    y_pred_lo: np.ndarray,
    y_pred_hi: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Return the calibration width adjustment for two-sided CQR: 2 * q_hat.

    This can be negative. The total endpoint difference at x is
    q_hi(x) - q_lo(x) + this adjustment.
    """
    q_hat = conformal_qhat(y_calib, y_pred_lo, y_pred_hi, alpha=alpha)
    return 2.0 * q_hat


def calibration_width_adjustment_from_models(
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    q_lo_model,
    q_hi_model,
    alpha: float = 0.1,
) -> float:
    """
    Compute the calibration width adjustment (2*q_hat) from model predictions.

    Requires:
        q_lo_model.predict(X), q_hi_model.predict(X)
    """
    y_pred_lo = q_lo_model.predict(X_calib)
    y_pred_hi = q_hi_model.predict(X_calib)
    return calibration_width_adjustment(y_calib, y_pred_lo, y_pred_hi, alpha=alpha)