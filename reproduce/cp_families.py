"""LACP, CQR and Mondrian experiments on UCI Concrete and Folktables: Figures 2, 4 and 8.

CP means conformal prediction; CQR means conformalized quantile regression;
LACP means locally adaptive conformal prediction.
Downloads the two datasets into data/downloads/ when missing, then runs:
  lacp_concrete              seeds 42-51 x alpha {0.05, 0.10, 0.20}, first 100 test rows   Figures 2, 4, 8
  mondrian_folktables        same configurations, sex partition                           Figures 4, 8
  cqr_concrete               seed 42, alpha 0.10, first 100 test rows                     Figure 8
  lacp_calibration_bootstrap seed 42, alpha 0.10, 30 calibration bootstraps               Figure 2
The experiment code reads its settings from environment variables and writes to
outputs/cp_families/<experiment>/. Joblib workers per run come from SLURM_CPUS_PER_TASK
(set from --jobs-per-run here).
"""
import sys

from common import REPO, Run, execute, parser, python

FAM = REPO / "experiments" / "cp_families"
OUT = REPO / "outputs" / "cp_families"
SWEEP = {"SEEDS": ",".join(str(s) for s in range(42, 52)), "ALPHAS": "0.05,0.1,0.2", "NUM_TEST": "100"}


def main() -> int:
    p = parser(__doc__, None)
    p.add_argument("--jobs-per-run", type=int, default=1)
    a = p.parse_args()
    jobs = {"SLURM_CPUS_PER_TASK": str(a.jobs_per_run)}
    data = REPO / "data" / "downloads"
    download_target = (data / "concrete.csv" if not (data / "concrete.csv").is_file()
                       else data / "folktables_acs_income_CA.csv")
    download = Run("download", [python(), "download_datasets.py"], FAM,
                   download_target, OUT / "download.log")
    runs = [
        Run("lacp_concrete", [python(), "main.py"], FAM / "lacp_concrete",
            OUT / "lacp_concrete" / "seed51_alpha0.20" / "summary.csv",
            OUT / "lacp_concrete" / "run.log", {**SWEEP, **jobs}),
        Run("mondrian_folktables", [python(), "main.py"], FAM / "mondrian_folktables",
            OUT / "mondrian_folktables" / "seed51_alpha0.20" / "summary.csv",
            OUT / "mondrian_folktables" / "run.log", {**SWEEP, "PARTITION": "sex", **jobs}),
        Run("cqr_concrete", [python(), "main.py"], FAM / "cqr_concrete",
            OUT / "cqr_concrete" / "seed42_alpha0.10" / "summary.csv",
            OUT / "cqr_concrete" / "run.log", {"SEEDS": "42", "ALPHAS": "0.1", "NUM_TEST": "100", **jobs}),
        Run("lacp_calibration_bootstrap", [python(), "main.py", "--seed", "42", "--alpha", "0.1", "--n_bootstraps", "30"],
            FAM / "lacp_calibration_bootstrap",
            OUT / "lacp_calibration_bootstrap" / "per_feature.csv",
            OUT / "lacp_calibration_bootstrap" / "run.log", jobs),
    ]
    rc = execute([download], 1, a.dry_run)
    if rc:
        return rc
    return execute(runs, a.workers, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
