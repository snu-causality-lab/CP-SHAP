# data/synthetic/synthetic_heteroscedastic.py

import numpy as np

def generate_heteroscedastic_data(n_samples: int, seed: int = 42):
    rng = np.random.default_rng(seed)

    X = rng.random((n_samples, 5))
    sigma = 0.05 + 0.95 * (X[:, 0] + X[:, 3])
    y = rng.normal(0.0, sigma, n_samples)

    cols = [f"X{i+1}" for i in range(X.shape[1])]
    return X, y, cols