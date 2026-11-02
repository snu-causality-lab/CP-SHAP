"""Figure 7 (Appendix B.3): mixing coefficient of the decision rule, clinical seed 19.

Score = lambda_l * phi^L + (1 - lambda_l) * phi^G for 21 values of lambda_l in [0, 1] with
threshold tau = 0: a patient is re-tested when any score is positive; high risk = y >= 18.
Writes retest rate, recall, precision, FP and FN for each lambda_l (the plotted values) and the
other counts.
"""
import numpy as np
import pandas as pd

from common import OUTPUTS, parser, write

N_LAMBDA = 21
K = 3


def load_phi(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if str(c).lower().startswith("unnamed")])


def evaluate_scores(scores: pd.DataFrame, high: np.ndarray, K: int) -> dict:
    retest = scores.gt(0).any(axis=1).to_numpy()
    n_tests = scores.where(scores > 0).notna().sum(axis=1).clip(upper=K).to_numpy(dtype=int).copy()
    n_tests[~retest] = 0
    tp = int(np.sum(retest & high))
    fp = int(np.sum(retest & ~high))
    fn = int(np.sum(~retest & high))
    tn = int(np.sum(~retest & ~high))
    return {
        "retest_rate": float(np.mean(retest)),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "avg_tests_if_retest": float(n_tests[retest].mean()) if np.any(retest) else 0.0,
    }


def main() -> None:
    a = parser(__doc__, OUTPUTS / "semi_synthetic" / "semi_synthetic" / "seed_19").parse_args()
    test = pd.read_csv(a.inp / "test_raw.csv")
    high = test["y"].to_numpy() >= 18.0
    phi_L = load_phi(a.inp / "phi_local.csv")
    phi_G = load_phi(a.inp / "phi_global.csv")
    common = sorted(set(phi_L.columns) & set(phi_G.columns))
    phi_L, g = phi_L[common], phi_G[common].iloc[0]
    rows = []
    for lam in np.linspace(0.0, 1.0, N_LAMBDA):
        scores = float(lam) * phi_L + float(1.0 - lam) * g
        rows.append({"lambda_l": float(lam), "lambda_g": float(1.0 - lam), **evaluate_scores(scores, high, K)})
    write(pd.DataFrame(rows), a.out, "figure7_lambda_path.csv")


if __name__ == "__main__":
    main()
