"""Semi-synthetic clinical decision task: Table 2, Section 6.3, Table 3 and Figure 7.

experiments/run.py with the LightGBM quantile backend, 50/30/20 split, alpha = 0.1 and three masked
features per test patient, for seeds 0-29 (Table 2). Table 3, Figure 7 and the Section 6.3
numbers use seed 19.
Outputs: outputs/semi_synthetic/semi_synthetic/seed_<k>/{phi_local,phi_global,phi_total,test_raw,...}.csv
"""
import sys
from pathlib import Path

from common import REPO, Run, execute, parser, python

SEEDS = list(range(30))


def main() -> int:
    p = parser(__doc__, "semi_synthetic/semi_synthetic")
    p.add_argument("--seeds", type=int, nargs="*", default=SEEDS, help="default: 0-29")
    p.add_argument("--jobs-per-run", type=int, default=1, help="n_jobs_global and n_jobs_local of each run.py call")
    a = p.parse_args()
    runs = []
    for seed in a.seeds:
        out = Path(a.out_root) / f"seed_{seed}"
        cmd = [python(), "experiments/run.py", "--exp", "semi_synthetic", "--backend", "lightgbm", "--seed", str(seed),
               "--calib_size", "0.375", "--alpha", "0.1", "--mask_k", "3",
               "--n_jobs_global", str(a.jobs_per_run), "--n_jobs_local", str(a.jobs_per_run), "--out_dir", str(out)]
        runs.append(Run(f"seed:{seed}", cmd, REPO, out / "phi_total.csv", out / "run.log"))
    return execute(runs, a.workers, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
