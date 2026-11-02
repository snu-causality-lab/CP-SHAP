"""Figure 4 (Appendix A.5): numerical check of Theorem 1 on the LACP and Mondrian runs.

For every (seed, alpha, instance, feature) cell of the LACP and Mondrian runs
(experiments/cp_families), phi^T_pred = phi^L + phi^G is compared with phi^T_direct, the
Shapley value computed directly on v^T. Writes per-family counts, residual statistics and the
Pearson correlation.
"""
import numpy as np
import pandas as pd

from common import OUTPUTS, parser, write

FAMILIES = {"locally_adaptive": "lacp_concrete", "mondrian": "mondrian_folktables"}
COLS = ["seed", "alpha", "instance_idx", "feature", "phi_L", "phi_G", "phi_T", "phi_T_direct"]


def main() -> None:
    a = parser(__doc__, OUTPUTS / "cp_families").parse_args()
    frames = []
    for family, folder in FAMILIES.items():
        for sub in sorted((a.inp / folder).glob("seed*_alpha*")):
            f = sub / "attributions.csv"
            if f.exists():
                frames.append(pd.read_csv(f)[COLS].assign(family=family, source=sub.name))
    df = pd.concat(frames, ignore_index=True)
    df["residual"] = df["phi_T_direct"] - df["phi_T"]  # phi_T == phi_L + phi_G by construction
    df["abs_residual"] = df["residual"].abs()
    summary = [{"family": fam, "n_cells": len(sub), "mean_abs_residual": sub["abs_residual"].mean(),
                "max_abs_residual": sub["abs_residual"].max(), "median_abs_residual": sub["abs_residual"].median(),
                "pearson": float(np.corrcoef(sub["phi_T"], sub["phi_T_direct"])[0, 1])}
               for fam, sub in df.groupby("family")]
    write(pd.DataFrame(summary), a.out, "figure4_theorem1.csv")


if __name__ == "__main__":
    main()
