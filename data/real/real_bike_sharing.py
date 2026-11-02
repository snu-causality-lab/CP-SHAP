# data/real/real_bike_sharing.py

from ucimlrepo import fetch_ucirepo

def load_bike_sharing_data(seed: int = 42):
    """
    Load UCI Bike Sharing dataset (id=275).

    - Drops 'dteday'
    - Returns X (float ndarray), y (1D ndarray), and column names
    """
    bike_sharing = fetch_ucirepo(id=275)

    X_df = bike_sharing.data.features.drop("dteday", axis=1)
    y_df = bike_sharing.data.targets

    cols = X_df.columns.tolist()

    X = X_df.to_numpy(dtype=float)
    y = y_df.to_numpy(dtype=float).ravel()

    return X, y, cols