"""Sampling under multicollinearity: Table 5 (Appendix B.6).

For data/model seeds 42-71, writes a copy of experiments/multicollinearity/config_full.json with
only data_seed changed, runs run_cell.py once per (seed, rho) for
rho in {0, 0.2, 0.4, 0.6, 0.8, 0.95} (exact d = 15 Shapley values and five 128-pair antithetic
streams), then summarize.py per seed. The across-seed numbers are made by aggregate/table5_multicollinearity.py.
Outputs: outputs/multicollinearity/seed_<k>/{cells,agg}/ and outputs/multicollinearity/configs/
"""
import json
import sys
from pathlib import Path

from common import REPO, Run, execute, parser, python

HERE = REPO / "experiments" / "multicollinearity"
SEEDS = list(range(42, 72))


def rho_label(rho: float) -> str:
    return f"cor_{int(round(100.0 * float(rho))):02d}"  # same label as run_cell.py


def main() -> int:
    p = parser(__doc__, "multicollinearity")
    p.add_argument("--seeds", type=int, nargs="*", default=SEEDS, help="default: 42-71")
    a = p.parse_args()
    out = Path(a.out_root)
    base = json.loads((HERE / "config_full.json").read_text())
    cells, aggs = [], []
    for seed in a.seeds:
        cfg = dict(base, data_seed=seed)
        cfg_path = out / "configs" / f"config_seed{seed}.json"
        if not a.dry_run:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        seed_dir = out / f"seed_{seed}"
        last = int(cfg["sampler_repeats"]) - 1
        for rho in cfg["correlations"]:
            label = rho_label(rho)
            cells.append(Run(f"seed{seed}:{label}",
                             [python(), "run_cell.py", "--config", str(cfg_path), "--rho", str(rho), "--out-dir", str(seed_dir)],
                             HERE, seed_dir / "cells" / f"sampled_{label}_repeat{last}.npz", seed_dir / "logs" / f"{label}.log"))
        aggs.append(Run(f"summarize:seed{seed}",
                        [python(), "summarize.py", "--config", str(cfg_path), "--cells-dir", str(seed_dir / "cells"),
                         "--out-dir", str(seed_dir / "agg")],
                        HERE, seed_dir / "agg" / "agreement.csv", seed_dir / "summarize.log"))
    rc = execute(cells, a.workers, a.dry_run)
    if rc:
        return rc
    return execute(aggs, a.workers, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
