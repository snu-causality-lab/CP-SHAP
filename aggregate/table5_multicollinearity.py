"""Table 5 (Appendix B.6): antithetic sampling against exact Shapley values under correlated covariates.

For data/model seeds 42-71, each metric of agg/agreement.csv is averaged over the five
permutation streams, then over seeds, per target correlation rho. Writes the table (normalized
MAE) with the Pearson and top-3 means, and the ranges over rho quoted in the text.
"""
import pandas as pd

from common import OUTPUTS, parser, write

SEEDS = range(42, 72)
METRICS = ["phiG_normalized_mae", "phiL_normalized_mae", "phiT_normalized_mae",
           "phiG_pearson", "phiL_pearson", "phiT_pearson", "phiG_top3_overlap"]


def main() -> None:
    a = parser(__doc__, OUTPUTS / "multicollinearity").parse_args()
    per_seed = []
    for s in SEEDS:
        g = pd.read_csv(a.inp / f"seed_{s}" / "agg" / "agreement.csv").groupby("rho_x")[METRICS].mean().reset_index()
        per_seed.append(g.assign(seed=s))
    per_seed = pd.concat(per_seed, ignore_index=True)
    table = per_seed.groupby("rho_x")[METRICS].mean().reset_index().assign(n_seeds=per_seed.groupby("rho_x").size().to_numpy())
    ranges = pd.DataFrame([{"metric": m, "min_over_rho": table[m].min(), "max_over_rho": table[m].max()} for m in METRICS])
    write(per_seed, a.out, "table5_multicollinearity_per_seed.csv")
    write(table, a.out, "table5_multicollinearity.csv")
    write(ranges, a.out, "table5_multicollinearity_ranges.csv")


if __name__ == "__main__":
    main()
