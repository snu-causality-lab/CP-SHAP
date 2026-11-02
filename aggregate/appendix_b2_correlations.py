"""Appendix B.2: Pearson correlation between each feature and its phi^L (heteroscedastic DGP).

For seeds 1-100, the correlation is computed over the 4,000 test instances of each run and then
averaged over seeds.
"""
import numpy as np
import pandas as pd

from common import OUTPUTS, parser, write

SEEDS = range(1, 101)


def main() -> None:
    a = parser(__doc__, OUTPUTS / "synthetic" / "heteroscedastic").parse_args()
    rows = []
    for s in SEEDS:
        run = a.inp / f"seed_{s}"
        X = pd.read_csv(run / "test.csv").drop(columns="y")
        L = pd.read_csv(run / "phi_local.csv")
        rows.append({"seed": s, "n_test": len(X), **{f: np.corrcoef(X[f], L[f])[0, 1] for f in L.columns}})
    per_seed = pd.DataFrame(rows)
    features = [c for c in per_seed.columns if c not in ("seed", "n_test")]
    summary = pd.DataFrame({"feature": features, "mean": per_seed[features].mean().to_numpy(),
                            "sd": per_seed[features].std().to_numpy(), "min": per_seed[features].min().to_numpy(),
                            "max": per_seed[features].max().to_numpy(), "n_seeds": len(per_seed)})
    write(per_seed, a.out, "appendix_b2_correlations_per_seed.csv")
    write(summary, a.out, "appendix_b2_correlations.csv")


if __name__ == "__main__":
    main()
