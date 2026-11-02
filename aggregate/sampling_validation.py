"""Section 6.2, Appendix A.3 and B.1: antithetic sampling (128 pairs) versus exact Shapley values.

Reads agreement.csv of reproduce/sampling_validation.py (one row per sampled cell paired with the
exact cell of the same d and seed) and reports, for d = 10 and d = 15 at budget 256 (128 pairs),
the mean, minimum and maximum over seeds of the Pearson correlation and the mean absolute
error of each game. The Pearson correlation pools all (query, feature) entries of a game.
"""
import pandas as pd

from common import OUTPUTS, parser, write

METRICS = ["phiG_pearson_r", "phiL_pearson_r", "phiG_mae", "phiL_mae"]


def main() -> None:
    a = parser(__doc__, OUTPUTS / "sampling_validation").parse_args()
    agr = pd.read_csv(a.inp / "agreement.csv")
    agr = agr[(agr["budget"] == 256) & agr["d"].isin([10, 15])]
    rows = []
    for d, sub in agr.groupby("d"):
        for m in METRICS:
            rows.append({"d": int(d), "budget": 256, "metric": m, "mean_over_seeds": sub[m].mean(),
                         "min_over_seeds": sub[m].min(), "max_over_seeds": sub[m].max(), "n_seeds": len(sub)})
    write(pd.DataFrame(rows), a.out, "sampling_validation.csv")


if __name__ == "__main__":
    main()
