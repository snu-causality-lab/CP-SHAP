# Decomposing Conformal Uncertainty: Calibration- and Instance-Driven Feature Attribution

Official implementation of our NeurIPS 2026 paper.

[Paper (OpenReview)](https://openreview.net/forum?id=lZavBaxJ68)

CP-SHAP decomposes feature contributions to conformal prediction uncertainty into
calibration-driven and instance-driven components using Shapley values.

This repository provides the method implementations, experiment scripts, and aggregation
scripts for reproducing the paper's numerical results. The experiment scripts save raw
attributions, and the aggregation scripts summarize them as CSV files. Plotting and LaTeX
formatting are left to the reader.

Throughout this README, CP stands for conformal prediction, CQR for conformalized quantile
regression, and LACP for locally adaptive conformal prediction.

## Installation

Use Python 3.11 and install the dependencies:

```bash
pip install -r requirements.txt
```

The requirements file pins the package versions. We observed different results for the
scikit-learn experiments (synthetic data and Bike Sharing) when using newer versions of NumPy,
pandas, scikit-learn, and LightGBM together. We have not determined which package caused the
differences.

## Quick Start

Run the following commands from the repository root to preview the synthetic experiments,
run them with the paper's settings, and summarize the sign-flip rates in Section 6.1:

```bash
python reproduce/synthetic.py --dry-run
python reproduce/synthetic.py --workers 4
python aggregate/section6_1_signflip.py
```

This runs the full synthetic experiment group, including the 100-seed study in Appendix B.2.
Adjust `--workers` to the available CPU resources. Raw outputs are saved to
`outputs/synthetic/`, and the sign-flip summary is written to
`results/section6_1_signflip.csv`. See [Reproducing the results](#reproducing-the-results)
for the commands corresponding to each paper result.

## Repository Structure

| Folder | Content |
|---|---|
| `src/` | Method implementations: split CQR widths in `conformal/` and the calibration (phi^G) and instance (phi^L) Shapley games in `shapley/`. |
| `data/` | Data generators, loaders, and the clinical data file. |
| `experiments/` | Synthetic, clinical, and Bike pipelines (`run.py`, `run_*.py`), clinical decision rules (`decision_process.py`), and CP-family experiments (`cp_families/`: LACP and CQR on UCI Concrete, Mondrian CQR on Folktables, and calibration bootstrap). Also includes runtime, sampling, matched decision, multicollinearity, plug-in, and masking baseline experiments. |
| `reproduce/` | Scripts that run each experiment group with the paper's settings and save raw outputs to `outputs/`. |
| `aggregate/` | Scripts that summarize each paper result and write `results/*.csv`. |

## Data

- The synthetic experiments generate their own data.
- The clinical experiments use `data/semi_synthetic/semi-syn-raw.csv`, derived from the
  PhysioNet/CinC Challenge 2019 data and distributed under the Open Database License.
  See `data/semi_synthetic/README.md` for its source, modifications, and the citations requested
  by the data provider.
- The Bike Sharing experiment downloads its data from the UCI repository (id 275).
- `reproduce/cp_families.py` downloads UCI Concrete (UCI repository, id 165) and Folktables ACS
  Income (California 2018) to `data/downloads/`. You can also download them separately with
  `python experiments/cp_families/download_datasets.py`.
  Folktables data are subject to the U.S. Census Bureau terms of use.

## Reproducing the results

Run the commands below from the repository root. Start with a script in `reproduce/` to run an
experiment group, then use the corresponding script in `aggregate/` to summarize the results.
Raw outputs are saved to `outputs/`, and the summary CSVs are written to `results/`.

Use `--dry-run` to preview the planned runs and `--workers` to control parallel execution.
The runtime script always runs one experiment at a time. Runs are skipped when their outputs
match the saved successful-completion records.

For exact local Shapley values, each coalition is evaluated once per query and its value is
reused. The original Table 4 timings were measured with an implementation that repeated these
evaluations, so the exact local timings from this code are not directly comparable to those in
the original table. The sampling implementation is unchanged.

The table below lists the settings, scripts, and output CSVs for each result in the paper.

| Paper result | Settings | Experiment | Aggregation | Result CSV |
|---|---|---|---|---|
| Section 6.1, sign flips | heteroscedastic DGP, d = 5, n = 20,000 split 50/30/20 (4,000 test), alpha = 0.1, seed 42 | `reproduce/synthetic.py` | `aggregate/section6_1_signflip.py` | `section6_1_signflip.csv` |
| Figure 1 and Section 6.1, Bike #124 | Bike Sharing, d = 12, split 50/30/20 (8,689/5,214/3,476), alpha = 0.1, seed 42, exact 2^12 coalitions, test instance 124 | `reproduce/bike.py` | `aggregate/figure1_bike.py` | `figure1_bike.csv` |
| Section 6.2, runtime at d = 30 | see Table 4 | `reproduce/runtime.py` | `aggregate/table4_runtime.py` | `table4_runtime.csv` |
| Section 6.2, Appendix A.3 and B.1, sampled vs exact agreement | heteroscedastic DGP, independent covariates, n = 4,000, n_c = 500, alpha = 0.1, d = 10 (20 queries) and d = 15 (5 queries), seeds 0-4, 128 antithetic pairs | `reproduce/sampling_validation.py` | `aggregate/sampling_validation.py` | `sampling_validation.csv` |
| Table 2 | clinical task, d = 6, split 50/30/20 (4,000 test), alpha = 0.1, LightGBM, 3 masked features per patient, seeds 0-29; mean (std) over seeds | `reproduce/clinical.py` | `aggregate/table2_clinical.py` | `table2_clinical.csv` |
| Section 6.3, ceiling | clinical task, seed 19 | `reproduce/clinical.py` | `aggregate/table3_matched.py` | `section6_3_ceiling.csv` |
| Table 3 | clinical task, seed 19, matched retest rate (target 0.46 lowered to the smallest rule ceiling) | `reproduce/clinical.py` | `aggregate/table3_matched.py` | `table3_matched.csv` |
| Figure 7 | clinical task, seed 19, 21 values of the mixing coefficient, tau = 0 | `reproduce/clinical.py` | `aggregate/figure7_lambda_path.py` | `figure7_lambda_path.csv` |
| Figure 2 | UCI Concrete, LACP, d = 8, split 50/30/20 (515/309/206), first 100 test instances, seeds 42-51 x alpha in {0.05, 0.10, 0.20}; calibration bootstrap with 30 replicas at seed 42, alpha = 0.1 | `reproduce/cp_families.py` | `aggregate/figure2_concrete.py` | `figure2_concrete.csv` |
| Figure 4 and Appendix A.5 | LACP on Concrete and Mondrian CQR on Folktables (n = 2,000, d = 9, sex partition), first 100 test instances, seeds 42-51 x alpha in {0.05, 0.10, 0.20} | `reproduce/cp_families.py` | `aggregate/figure4_theorem1.py` | `figure4_theorem1.csv` |
| Figure 5 | heteroscedastic (a) and homoscedastic (b) DGPs, d = 5, n = 20,000 split 50/30/20, alpha = 0.1, seed 42 | `reproduce/synthetic.py` | `aggregate/figure5_6_synthetic.py` | `figure5_6_phiG.csv` |
| Figure 6 | source-separation DGP, d = 10, n = 20,000 split 50/30/20, alpha = 0.1, seed 42 | `reproduce/synthetic.py` | `aggregate/figure5_6_synthetic.py` | `figure5_6_phiG.csv` |
| Appendix B.2, correlations | heteroscedastic DGP, n = 20,000 split 50/30/20, alpha = 0.1, seeds 1-100 | `reproduce/synthetic.py` | `aggregate/appendix_b2_correlations.py` | `appendix_b2_correlations.csv` |
| Figure 8 and Appendix B.4 | LACP and CQR on Concrete, Mondrian CQR on Folktables, seed 42, alpha = 0.10, first 100 test instances | `reproduce/cp_families.py` | `aggregate/figure8_signflip.py` | `figure8_signflip.csv`, `figure8_selection.csv` |
| Table 4 | heteroscedastic DGP, n = 4,000, n_c = 500, seed 42; exact for d in {3, 5, 7, 10} (20 test queries), 128 antithetic pairs for d in {15, 20, 30} (10 test queries); single thread | `reproduce/runtime.py` | `aggregate/table4_runtime.py` | `table4_runtime.csv` |
| Table 5 | correlated heteroscedastic DGP, d = 15, 3,490/500/10 rows, alpha = 0.1, rho in {0, 0.2, 0.4, 0.6, 0.8, 0.95}, five 128-pair streams, seeds 42-71 | `reproduce/multicollinearity.py` | `aggregate/table5_multicollinearity.py` | `table5_multicollinearity.csv`, `table5_multicollinearity_ranges.csv` |
| Table 6 | three synthetic DGPs, n = 3,000 split 50/30/20 (1,500/900/600), alpha = 0.1, seeds 42-71 | `reproduce/plugin_comparison.py` | `aggregate/table6_plugin.py` | `table6_plugin.csv`, `table6_sign_zero.csv` |
| Table 7 | equicorrelated source-separation DGP, d = 10, n = 20,000 split 50/30/20 (4,000 test), alpha = 0.1, rho in {0, 0.25, 0.5, 0.75, 0.95}, seeds 42-71 | `reproduce/masking_baseline.py` | `aggregate/table7_masking_baseline.py` | `table7_masking_baseline.csv`, `table7_masking_baseline_ranges.csv` |

For Figures 4–6, the CSVs listed above contain summaries or labels. To find the individual
scatter points, use the raw experiment outputs:

- **Figure 4:** `outputs/cp_families/<family>/seed<seed>_alpha<alpha>/attributions.csv`,
  where `<family>` is `lacp_concrete` or `mondrian_folktables`, `<seed>` is 42–51,
  and `<alpha>` is `0.05`, `0.10`, or `0.20`. Each row pairs `phi_T` (= `phi_L + phi_G`)
  with `phi_T_direct` for a particular seed, alpha, instance, and feature.
  `results/figure4_theorem1.csv` summarizes the residuals and correlations for each family.
- **Figures 5–6:** `outputs/synthetic/<setting>/seed_42/phi_local.csv` contains
  the local attribution points. The corresponding feature values, used to color the points,
  are in `test.csv` in the same folder. Match each point to the same row and feature column
  in that file. `<setting>` is `heteroscedastic` for Figure 5(a), `homoscedastic` for
  Figure 5(b), and `source_separation` for Figure 6. The global attribution labels
  are in `results/figure5_6_phiG.csv`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{cho2026decomposing,
  title = {Decomposing Conformal Uncertainty: Calibration- and Instance-Driven Feature Attribution},
  author = {Cho, Sangyeon and Cho, Minyoung and Kim, Jungsoo and Oh, Sujeong and Lee, Sanghack},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026}
}
```

Citation metadata are also available in [`CITATION.cff`](CITATION.cff).

## License

The code is released under the MIT license (`LICENSE`). The clinical data file is distributed
under the Open Database License; see `data/semi_synthetic/` for details.
