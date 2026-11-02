"""Sensitivity to the masking baseline: Table 7 (Appendix E.2).

For every (seed, rho) of experiments/masking_baseline/config_full.json (seeds 42-71,
rho in {0, 0.25, 0.5, 0.75, 0.95}; equicorrelated source-separation DGP, n = 20,000 split
50/30/20, alpha = 0.1, all 4,000 test rows), run_cell.py enumerates all coalitions at
d = 10 for four mean-completion rules plus product-marginal and joint-marginal resampling.
summarize.py computes per-query statistics in vectorized batches.
aggregate/table7_masking_baseline.py selects four alternatives and the calibration-mean
reference for Table 7 and its accompanying text. Joint-marginal results and the configured
higher-draw global check are additional diagnostics, not selected for that paper output.
Outputs: outputs/masking_baseline/cells/*.npz and outputs/masking_baseline/agg/*.csv
"""
import json
import sys
from pathlib import Path

from common import REPO, Run, execute, parser, python

HERE = REPO / "experiments" / "masking_baseline"
CONFIG = "config_full.json"


def rho_label(rho: float) -> str:
    return f"corr{int(round(100.0 * float(rho))):02d}"  # same label as run_cell.py


def main() -> int:
    a = parser(__doc__, "masking_baseline").parse_args()
    out = Path(a.out_root)
    cfg = json.loads((HERE / CONFIG).read_text())
    conv = cfg.get("joint_marginal_convergence") or {}
    first = (int(conv.get("seed", -1)), float(conv.get("rho", -1.0)))
    cells = sorted(((int(s), float(r)) for s in cfg["data_seeds"] for r in cfg["correlations"]),
                   key=lambda c: c != first)  # the slow convergence cell starts first
    runs = []
    for seed, rho in cells:
        name = f"cell_seed{seed}_{rho_label(rho)}"
        runs.append(Run(name, [python(), "run_cell.py", "--config", CONFIG, "--seed", str(seed), "--rho", str(rho),
                               "--out-dir", str(out / "cells")],
                        HERE, out / "cells" / f"{name}.npz", out / "cells" / "logs" / f"{name}.log"))
    rc = execute(runs, a.workers, a.dry_run)
    if rc:
        return rc
    agg = Run("summarize", [python(), "summarize.py", "--config", CONFIG, "--cells-dir", str(out / "cells"),
                            "--out-dir", str(out / "agg")],
              HERE, out / "agg" / "summary.csv", out / "summarize.log")
    return execute([agg], 1, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
