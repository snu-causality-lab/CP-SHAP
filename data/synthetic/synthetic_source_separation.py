# data/synthetic/synthetic_source_separation.py

import numpy as np


def poly2_feature_map(U: np.ndarray, include_bias: bool = True) -> np.ndarray:
    """
    2nd-degree polynomial feature map:
      phi(u1,...,u_p') -> (1, u1,...,u_p', u1^2, u1 u2, ..., u_p'^2)
    Order: [bias], [linear], [quadratic i<=j].
    """
    U = np.asarray(U)
    n, p_prime = U.shape

    feats = []
    if include_bias:
        feats.append(np.ones((n, 1)))
    feats.append(U)

    quad_cols = []
    for i in range(p_prime):
        for j in range(i, p_prime):
            quad_cols.append((U[:, i] * U[:, j]).reshape(n, 1))
    feats.append(np.hstack(quad_cols))

    return np.hstack(feats)


def sample_gamma(d_phi: int, rng: np.random.Generator) -> np.ndarray:
    """gamma_k ~ Uniform([-1,-0.5] U [0.5,1])."""
    signs = rng.choice([-1.0, 1.0], size=d_phi)
    mags = rng.uniform(0.5, 1.0, size=d_phi)
    return signs * mags


def generate_source_separation_data(
    n: int = 20000,
    p_mean: int = 5,          # X1-X5
    p_unc: int = 5,           # X6-X10
    alpha: float = 2.0,
    sigma_eps2: float = 0.02,
    sigma_delta2: float = 0.05,
    seed: int = 42,
):
    """
    Mean sources: X1-X5
      V_ij ~ N(0,1), beta_i ~ Uniform([-1,1]), mu = V beta

    Uncertainty sources: X6-X10
      U_ij ~ N(0,1)
      sigma = | phi(U) gamma + delta |
        gamma_k ~ Uniform([-1,-0.5] U [0.5,1])
        delta ~ N(0, sigma_delta2)

    Observation:
      y = mu + sqrt(alpha)*sigma*z + sqrt(sigma_eps2)*e
      z,e iid N(0,1)

    Returns:
      X, y, cols
    """
    rng = np.random.default_rng(seed)

    # Mean sources: X1-X5
    V = rng.normal(0.0, 1.0, size=(n, p_mean))
    beta = rng.uniform(-1.0, 1.0, size=(p_mean,))
    mu = V @ beta

    # Uncertainty sources: X6-X10
    U = rng.normal(0.0, 1.0, size=(n, p_unc))
    Phi = poly2_feature_map(U, include_bias=True)
    gamma = sample_gamma(Phi.shape[1], rng)

    delta = rng.normal(0.0, np.sqrt(sigma_delta2), size=(n,))
    sigma = np.abs(Phi @ gamma + delta)

    # Sample y
    z = rng.normal(0.0, 1.0, size=(n,))
    e = rng.normal(0.0, 1.0, size=(n,))
    y = mu + np.sqrt(alpha) * sigma * z + np.sqrt(sigma_eps2) * e

    # Full X: X1-X10
    X = np.hstack([V, U])
    cols = [f"X{i+1}" for i in range(X.shape[1])]

    return X, y, cols