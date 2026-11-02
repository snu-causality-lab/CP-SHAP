"""Section 6.1: sign flips between phi^L and phi^T on the heteroscedastic synthetic DGP (seed 42).

A cell (test instance, feature) flips when phi^L * phi^T < 0; an instance flips when at least
one of its cells does.
"""
import pandas as pd

from common import OUTPUTS, parser, write


def main() -> None:
    a = parser(__doc__, OUTPUTS / "synthetic" / "heteroscedastic" / "seed_42").parse_args()
    L = pd.read_csv(a.inp / "phi_local.csv").to_numpy()
    T = pd.read_csv(a.inp / "phi_total.csv").to_numpy()
    flip = (L * T) < 0
    n, d = flip.shape
    row = {"n_instances": n, "n_features": d, "n_cells": n * d,
           "flipped_cells": int(flip.sum()), "cell_percent": 100.0 * flip.sum() / (n * d),
           "flipped_instances": int(flip.any(axis=1).sum()), "instance_percent": 100.0 * flip.any(axis=1).mean()}
    write(pd.DataFrame([row]), a.out, "section6_1_signflip.csv")


if __name__ == "__main__":
    main()
