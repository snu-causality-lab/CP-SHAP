"""Figures 5 and 6: phi^L scatter and phi^G labels on the synthetic DGPs (seed 42).

The scatter points are phi^L of every (test instance, feature) coloured by the feature value,
i.e. phi_local.csv and test.csv of each run. The grey phi^G labels (two decimals) are written to
results/figure5_6_phiG.csv.
"""
import pandas as pd

from common import OUTPUTS, parser, write

SETTINGS = {"heteroscedastic": "Figure 5(a)", "homoscedastic": "Figure 5(b)", "source_separation": "Figure 6"}


def main() -> None:
    a = parser(__doc__, OUTPUTS / "synthetic").parse_args()
    labels = []
    for setting, figure in SETTINGS.items():
        run = a.inp / setting / "seed_42"
        G = pd.read_csv(run / "phi_global.csv").iloc[0]
        L = pd.read_csv(run / "phi_local.csv")
        for f in L.columns:
            labels.append({"figure": figure, "setting": setting, "feature": f, "phi_G": G[f], "label": f"{G[f]:.2f}"})
    write(pd.DataFrame(labels), a.out, "figure5_6_phiG.csv")


if __name__ == "__main__":
    main()
