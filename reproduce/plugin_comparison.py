"""LIME and occlusion versus Shapley: Table 6 (Appendix D).

For seeds 42-71, plugin_comparison.py on the three synthetic DGPs with n = 3,000 split
50/30/20 (1,500/900/600) and alpha = 0.1, one process per seed. The across-seed table is made
by aggregate/table6_plugin.py.
Outputs: outputs/plugin_comparison/seeds<k>-<k>/{component_results,results,...}.csv
"""
import sys
from pathlib import Path

from common import REPO, Run, execute, parser, python

HERE = REPO / "experiments" / "plugin_comparison"
SEEDS = list(range(42, 72))


def main() -> int:
    p = parser(__doc__, "plugin_comparison")
    p.add_argument("--seeds", type=int, nargs="*", default=SEEDS, help="default: 42-71")
    a = p.parse_args()
    runs = []
    for seed in a.seeds:
        out = Path(a.out_root) / f"seeds{seed}-{seed}"
        cmd = [python(), "plugin_comparison.py", "--out_dir", str(out), "--seeds", str(seed),
               "--n_samples", "3000", "--n_calib", "900", "--n_test", "600", "--alpha", "0.1"]
        runs.append(Run(f"seed:{seed}", cmd, HERE, out / "component_results.csv", out / "run.log"))
    return execute(runs, a.workers, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
