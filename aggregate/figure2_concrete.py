"""Figure 2: calibration-side phi^G and mean |phi^L| per feature, LACP on UCI Concrete.

phi^G bar and interval: mean and 95% percentile interval over 30 calibration bootstraps
(experiments/cp_families/lacp_calibration_bootstrap, seed 42, alpha 0.1).
|phi^L| bar and interval: for seeds 42-51, the mean |phi^L| over the first 100 test instances is
averaged over the three alphas (phi^L does not depend on alpha), then mean +- 1.96 * SD / sqrt(10)
over seeds. Features are listed in plot order (increasing |phi^G| from the bottom).
"""
import glob

import numpy as np
import pandas as pd

from common import OUTPUTS, parser, write


def main() -> None:
    a = parser(__doc__, OUTPUTS / "cp_families").parse_args()
    G = pd.read_csv(a.inp / "lacp_calibration_bootstrap" / "per_feature.csv").set_index("feature")
    files = sorted(glob.glob(str(a.inp / "lacp_concrete" / "seed*_alpha*" / "summary.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    per_seed = df.groupby(["feature", "seed"])["mean_abs_phi_L"].mean().reset_index()
    L = per_seed.groupby("feature")["mean_abs_phi_L"].agg(["mean", "std", "count"])
    sem = L["std"] / np.sqrt(L["count"])
    L["ci95_lo"], L["ci95_hi"] = L["mean"] - 1.96 * sem, L["mean"] + 1.96 * sem
    order = sorted(G.index, key=lambda f: abs(G.loc[f, "phi_G_mean"]))
    rows = [{"plot_position": i, "feature": f,
             "phi_G_mean": G.loc[f, "phi_G_mean"], "phi_G_ci95_lo": G.loc[f, "phi_G_ci95_lo"],
             "phi_G_ci95_hi": G.loc[f, "phi_G_ci95_hi"],
             "mean_abs_phi_L": L.loc[f, "mean"], "mean_abs_phi_L_ci95_lo": L.loc[f, "ci95_lo"],
             "mean_abs_phi_L_ci95_hi": L.loc[f, "ci95_hi"], "n_seeds": int(L.loc[f, "count"])}
            for i, f in enumerate(order)]
    write(pd.DataFrame(rows), a.out, "figure2_concrete.csv")


if __name__ == "__main__":
    main()
