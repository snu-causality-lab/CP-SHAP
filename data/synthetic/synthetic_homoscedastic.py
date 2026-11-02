# data/synthetic/synthetic_homoscedastic.py

import numpy as np

def generate_homoscedastic_data(n_samples, seed=42):
    rng = np.random.default_rng(seed)

    X = rng.random((n_samples, 5))
    sigma = 0.5
    y = rng.normal(0.0, sigma, n_samples)

    cols = [f"X{i+1}" for i in range(X.shape[1])]
    return X, y, cols