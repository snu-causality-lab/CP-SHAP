"""UCI Bike Sharing case study: Figure 1 and Section 6.1.

experiments/run.py on Bike Sharing (downloaded from the UCI repository, id 275) with seed 42,
50/30/20 split, alpha = 0.1 and exact 2^12 Shapley values for all 3,476 test rows. This is a single
run, so --workers sets the number of processes inside it (n_jobs_global and n_jobs_local).
Outputs: outputs/real/real_bike/seed_42/{phi_local,phi_global,phi_total,test,...}.csv
"""
import sys
from pathlib import Path

from common import REPO, Run, execute, parser, python


def main() -> int:
    p = parser(__doc__, "real/real_bike")
    a = p.parse_args()
    out = Path(a.out_root) / "seed_42"
    cmd = [python(), "experiments/run.py", "--exp", "real", "--dataset", "real_bike", "--seed", "42",
           "--calib_size", "0.375", "--alpha", "0.1",
           "--n_jobs_global", str(a.workers), "--n_jobs_local", str(a.workers), "--out_dir", str(out)]
    return execute([Run("bike:42", cmd, REPO, out / "phi_total.csv", out / "run.log")], 1, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
