"""Synthetic CQR experiments: Section 6.1, Figure 5, Figure 6 and Appendix B.2.

experiments/run.py on the three synthetic DGPs with the paper's 50/30/20 split (n = 20,000, alpha = 0.1):
  heteroscedastic seeds 1-100   Appendix B.2 correlations; seed 42 also gives Section 6.1 and Figure 5(a)
  homoscedastic seed 42         Figure 5(b)
  source_separation seed 42     Figure 6
Outputs: outputs/synthetic/<setting>/seed_<k>/{phi_local,phi_global,phi_total,test,...}.csv
"""
import sys
from pathlib import Path

from common import REPO, Run, execute, parser, python

RUNS = [("heteroscedastic", s) for s in range(1, 101)] + [("homoscedastic", 42), ("source_separation", 42)]


def main() -> int:
    p = parser(__doc__, "synthetic")
    p.add_argument("--only", nargs="*", metavar="SETTING:SEED", help="run only these, e.g. heteroscedastic:42")
    p.add_argument("--jobs-per-run", type=int, default=1, help="n_jobs_global and n_jobs_local of each run.py call")
    a = p.parse_args()
    runs = []
    for setting, seed in RUNS:
        if a.only and f"{setting}:{seed}" not in a.only:
            continue
        out = Path(a.out_root) / setting / f"seed_{seed}"
        cmd = [python(), "experiments/run.py", "--exp", "synthetic", "--setting", setting, "--seed", str(seed),
               "--n_samples", "20000", "--calib_size", "0.375", "--alpha", "0.1",
               "--n_jobs_global", str(a.jobs_per_run), "--n_jobs_local", str(a.jobs_per_run), "--out_dir", str(out)]
        runs.append(Run(f"{setting}:{seed}", cmd, REPO, out / "phi_total.csv", out / "run.log"))
    return execute(runs, a.workers, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
