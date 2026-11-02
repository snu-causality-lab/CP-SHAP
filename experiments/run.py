# experiments/run.py

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # repository root for experiments/, src/ and data/

from experiments.run_synthetic import run_synthetic
from experiments.run_real import run_real
from experiments.run_semi_synthetic import run_semi_synthetic



def _default_outdir(base: str, exp: str, name: str, seed: int) -> str:
    return os.path.join(base, exp, name, f"seed_{seed}")


def parse_args():
    p = argparse.ArgumentParser("Run experiments (synthetic / semi_synthetic / real)")

    p.add_argument("--exp", type=str, required=True,
                   choices=["synthetic", "semi_synthetic", "real"],
                   help="Which experiment to run.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.1, help="Conformal alpha (e.g., 0.1).")

    p.add_argument("--out_base", type=str, default=str(REPO_ROOT / "outputs"),
                   help="Base output directory.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="If set, overrides auto output directory.")

    # common model args
    p.add_argument("--backend", type=str, default="sklearn_gbrt",
                   choices=["lightgbm", "sklearn_gbrt"],
                   help="Quantile model backend.")
    p.add_argument("--n_jobs_global", type=int, default=None,
                   help="MP workers for global Shapley.")
    p.add_argument("--n_jobs_local", type=int, default=None,
                   help="MP workers for local Shapley.")

    # split args
    p.add_argument("--test_size", type=float, default=0.2,
                   help="Fraction of all rows reserved for testing (paper: 0.2).")
    p.add_argument("--calib_size", type=float, default=0.375,
                   help="Calibration fraction of the remaining non-test rows; test_size=0.2 and calib_size=0.375 give train/calibration/test=50/30/20.")

    # synthetic args
    p.add_argument("--setting", type=str, default="heteroscedastic",
                   choices=["heteroscedastic", "homoscedastic", "source_separation"],
                   help="Synthetic setting.")
    p.add_argument("--n_samples", type=int, default=20000)

    # real args
    p.add_argument("--dataset", type=str, default="real_bike",
                   help="Real dataset name (currently: real_bike).")

    # semi-synthetic args
    p.add_argument("--raw_csv", type=str, default=str(REPO_ROOT / "data" / "semi_synthetic" / "semi-syn-raw.csv"),
                   help="Path to semi-synthetic raw csv.")
    p.add_argument("--mask_k", type=int, default=3,
                   help="How many features to mask (NaN) per test instance in semi-synthetic.")

    return p.parse_args()


def main():
    args = parse_args()

    # decide output directory name component
    if args.exp == "synthetic":
        name = args.setting
    elif args.exp == "real":
        name = args.dataset
    else:
        name = "semi_synthetic"

    out_dir = args.out_dir or _default_outdir(args.out_base, args.exp, name, args.seed)
    os.makedirs(out_dir, exist_ok=True)

    if args.exp == "synthetic":
        run_synthetic(
            setting=args.setting,
            output_dir=out_dir,
            seed=args.seed,
            n_samples=args.n_samples,
            test_size=args.test_size,
            calib_size=args.calib_size,
            alpha=args.alpha,
            model_backend=args.backend,
            model_params=None,
            n_jobs_global=args.n_jobs_global,
            n_jobs_local=args.n_jobs_local,
        )

    elif args.exp == "real":
        run_real(
            dataset=args.dataset,
            output_dir=out_dir,
            seed=args.seed,
            test_size=args.test_size,
            calib_size=args.calib_size,
            alpha=args.alpha,
            model_backend=args.backend,
            model_params=None,
            n_jobs_global=args.n_jobs_global,
            n_jobs_local=args.n_jobs_local,
        )

    elif args.exp == "semi_synthetic":
        run_semi_synthetic(
            raw_csv_path=args.raw_csv,
            output_dir=out_dir,
            seed=args.seed,
            test_size=args.test_size,
            calib_size=args.calib_size,
            alpha=args.alpha,
            mask_k=args.mask_k,
            model_backend=args.backend,
            model_params=None,
            n_jobs_global=args.n_jobs_global,
            n_jobs_local=args.n_jobs_local,
        )

    print(f"\n[Done] outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()