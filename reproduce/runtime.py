"""Runtime measurements: Table 4 (Appendix B.5) and the timings quoted in Section 6.2.

Single-threaded wall-clock time of the offline phi^G and per-query phi^L computations on the
heteroscedastic DGP with n_c = 500: exact enumeration for d in {3, 5, 7, 10} (20 test queries)
and antithetic sampling with 128 pairs for d in {15, 20, 30} (10 test queries). Timings
depend on the machine.
Outputs: outputs/runtime/{exact,sampled}/results.csv
"""
import sys
from pathlib import Path

from common import REPO, Run, execute, parser, python

HERE = REPO / "experiments" / "runtime"


def main() -> int:
    a = parser(__doc__, "runtime").parse_args()
    out = Path(a.out_root)
    runs = [
        Run("exact", [python(), "runtime_scaling.py", "--dims", "3", "5", "7", "10", "--n_samples", "4000",
                      "--n_calib", "500", "--n_test", "20", "--seed", "42", "--out_dir", str(out / "exact")],
            HERE, out / "exact" / "results.csv", out / "exact" / "run.log"),
        Run("sampled", [python(), "high_d_scaling.py", "--dims", "15", "20", "30", "--n_samples", "4000",
                        "--n_calib", "500", "--n_test", "10", "--budget", "256", "--seed", "42",
                        "--out_dir", str(out / "sampled")],
            HERE, out / "sampled" / "results.csv", out / "sampled" / "run.log"),
    ]
    return execute(runs, 1, a.dry_run)  # one at a time so the timings do not compete


if __name__ == "__main__":
    sys.exit(main())
