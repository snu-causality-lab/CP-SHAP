"""Antithetic sampling versus exact Shapley values: Section 6.2, Appendix A.3 and B.1.

Independent-covariate heteroscedastic DGP, n_c = 500, alpha = 0.1, seeds 0-4:
  d = 10: exact and 128-pair antithetic sampling on 20 test queries
  d = 15: exact and 128-pair antithetic sampling on 5 test queries
then summarize.py pairs each sampled cell with the exact cell of the same (d, seed).
Outputs: outputs/sampling_validation/cells/*.npz and outputs/sampling_validation/{agreement,results}.csv
"""
import sys
from pathlib import Path

from common import REPO, Run, execute, parser, python

HERE = REPO / "experiments" / "sampling_validation"
CELLS = [("exact", 10, 0, 20), ("sampled", 10, 256, 20), ("exact", 15, 0, 5), ("sampled", 15, 256, 5)]
SEEDS = range(5)


def main() -> int:
    a = parser(__doc__, "sampling_validation").parse_args()
    out = Path(a.out_root)
    cells = out / "cells"
    runs = []
    for method, d, budget, n_test in CELLS:
        for seed in SEEDS:
            path = cells / f"cell_{method}_d{d}_b{budget}_seed{seed}.npz"
            cmd = [python(), "run_cell.py", "--method", method, "--d", str(d), "--budget", str(budget),
                   "--seed", str(seed), "--n_test", str(n_test), "--out", str(path)]
            runs.append(Run(path.stem, cmd, HERE, path, cells / "logs" / f"{path.stem}.log"))
    rc = execute(runs, a.workers, a.dry_run)
    if rc:
        return rc
    agg = Run("summarize", [python(), "summarize.py", "--cells_dir", str(cells),
                            "--out_results", str(out / "results.csv"), "--out_agreement", str(out / "agreement.csv"),
                            "--out_convergence", str(out / "convergence.csv")],
              HERE, out / "agreement.csv", out / "summarize.log")
    return execute([agg], 1, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
