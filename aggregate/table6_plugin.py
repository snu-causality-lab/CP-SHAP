"""Table 6 (Appendix D): LIME and occlusion versus Shapley, mean and SD over seeds 42-71.

Per (setting, method, component), the per-seed agreement statistics of plugin_comparison.py are
averaged over seeds (SD with ddof = 1). The sign-agreement breakdown quoted in Appendix D is
computed from the saved attribution cells: sign agreement on all entries where either value is
non-zero (as in the table), on entries where the Shapley value is non-zero, and the share of
entries whose Shapley value is zero (|value| <= 1e-10). Local and total agreement is computed per
instance and averaged; global over the feature vector; then averaged over seeds.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUTPUTS, parser, write

SEEDS = list(range(42, 72))
COLS = ["mean_abs_score", "pearson_vs_shapley", "spearman_vs_shapley", "sign_agreement_vs_shapley"]
EPS = 1e-10


def agree(x, y, mask) -> float:
    return float(np.mean(np.sign(x[mask]) == np.sign(y[mask]))) if mask.any() else float("nan")


def main() -> None:
    a = parser(__doc__, OUTPUTS / "plugin_comparison").parse_args()
    r = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(a.inp / "seeds*" / "component_results.csv")))],
                  ignore_index=True)
    if sorted(r["seed"].unique()) != SEEDS:
        raise SystemExit(f"expected seeds 42-71, found {sorted(r['seed'].unique())}")
    s = r.groupby(["setting", "method", "component"])[COLS].agg(["mean", "std"])
    s.columns = ["_".join(c) for c in s.columns]
    write(s.reset_index(), a.out, "table6_plugin.csv")

    files = {}
    for f in sorted(Path(a.inp).glob("**/cells/cell_*_seed*.npz")):
        setting, seed = f.stem[len("cell_"):].rsplit("_seed", 1)
        files.setdefault((setting, int(seed)), f)
    rows = []
    for setting in ("heteroscedastic", "homoscedastic", "source_separation"):
        for method in ("lime_surrogate", "occlusion_sensitivity"):
            for comp in ("global", "local", "total"):
                both, nonzero, zero = [], [], []
                for seed in SEEDS:
                    z = np.load(files[(setting, seed)], allow_pickle=True)
                    x_all, y_all = np.atleast_2d(z[f"shapley_exact_{comp}"]), np.atleast_2d(z[f"{method}_{comp}"])
                    both.append(np.nanmean([agree(x, y, (np.abs(x) > EPS) | (np.abs(y) > EPS)) for x, y in zip(x_all, y_all)]))
                    nonzero.append(np.nanmean([agree(x, y, np.abs(x) > EPS) for x, y in zip(x_all, y_all)]))
                    zero.append(np.mean(np.abs(x_all) <= EPS))
                rows.append({"setting": setting, "method": method, "component": comp,
                             "sign_agreement": np.mean(both), "sign_agreement_shapley_nonzero": np.mean(nonzero),
                             "shapley_zero_percent": 100.0 * np.mean(zero)})
    write(pd.DataFrame(rows), a.out, "table6_sign_zero.csv")


if __name__ == "__main__":
    main()
