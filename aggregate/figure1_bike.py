"""Figure 1 and Section 6.1: Bike Sharing test instance #124.

The figure shows the six features with the largest |phi^L| of instance 124, ordered by
increasing |phi^L| from the bottom. Writes all 12 features with phi^T, phi^L and phi^G, whether
the feature is shown, and its plot position (0 = bottom, -1 = not shown).
"""
import pandas as pd

from common import OUTPUTS, parser, write

INSTANCE = 124
TOP_K = 6


def main() -> None:
    a = parser(__doc__, OUTPUTS / "real" / "real_bike" / "seed_42").parse_args()
    phi_T = pd.read_csv(a.inp / "phi_total.csv")
    phi_L = pd.read_csv(a.inp / "phi_local.csv")
    phi_G = pd.read_csv(a.inp / "phi_global.csv")
    features = list(phi_T.columns)
    T, L, G = phi_T.iloc[INSTANCE], phi_L.iloc[INSTANCE], phi_G.iloc[0]
    top = sorted(features, key=lambda f: abs(L[f]), reverse=True)[:TOP_K]
    order = sorted(top, key=lambda f: abs(L[f]))
    rows = [{"instance": INSTANCE, "feature": f, "phi_T": T[f], "phi_L": L[f], "phi_G": G[f],
             "shown": f in top, "plot_position": order.index(f) if f in order else -1} for f in features]
    write(pd.DataFrame(rows), a.out, "figure1_bike.csv")


if __name__ == "__main__":
    main()
