"""
Matched-aggressiveness decision analysis (Table 3, Appendix B.3).

The rules of Table 2 fire at different rates, so their recall is not directly
comparable. We fix a target retest rate and tune a per-rule threshold on the
decision statistic so that every rule fires at (approximately) the same rate;
a target above the smallest rule ceiling is lowered to that ceiling.
Recall is then a matched comparison.

This script is a pure wrapper: it imports helpers from decision_process.py and
uses the pre-computed phi_*.csv files under
`outputs/semi_synthetic/semi_synthetic/seed_*`. It reads existing attributions and writes decision metrics.

Usage:
    python matched_decision.py [--out_dir <dir>] [--seeds 19] [--target 0.46]
                               [--smoke]

Outputs:
    results.csv — columns: seed, rule_key, rule, requested_target, feasible_target,
                 ceiling, threshold, retest_rate, recall, precision, TP, FP, TN, FN,
                 avg_tests_if_retest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Make repo importable (matched_decision.py lives under experiments/matched_decision)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.decision_process import (  # noqa: E402  (after sys.path insert)
    load_phi_df,
    align_feature_columns,
    build_scores,
)


# --------------------------------------------------------------------------- #
# Thresholded decision rule
# --------------------------------------------------------------------------- #
def decide_retest_thresholded(scores_row: pd.Series, threshold: float, K: int):
    """
    Retest iff max of positive scores is strictly > threshold.
    Same counting logic as decision_process.decide_retest_and_count (top-K by
    |score|) once the retest is triggered.

    When threshold <= 0, this reduces to "retest iff any positive score" which
    matches the paper's original rule. Increasing threshold makes the rule
    more conservative (lower retest rate).
    """
    pos = scores_row[scores_row > 0]
    if len(pos) == 0 or pos.max() <= threshold:
        return False, 0, []
    ordered_idx = pos.abs().sort_values(ascending=False).index
    m = min(K, len(ordered_idx))
    return True, m, list(ordered_idx[:m])


def tune_threshold_for_target(
    scores: pd.DataFrame,
    target: float,
    K: int,
    n_grid: int = 401,
) -> tuple[float, float]:
    """
    Sweep threshold over a grid covering all positive score maxima; return the
    threshold whose empirical retest rate is closest to `target`. Ties broken
    by preferring the threshold that yields >= target (conservative upper bound
    on aggressiveness).

    Returns (threshold, achieved_retest_rate).
    """
    # Max positive score per row; 0.0 if none positive.
    pos_max = scores.where(scores > 0).max(axis=1).fillna(0.0).to_numpy()

    lo = float(pos_max[pos_max > 0].min()) if (pos_max > 0).any() else 0.0
    hi = float(pos_max.max()) if len(pos_max) else 0.0

    # Equally spaced threshold grid plus quantiles of positive per-row score maxima; deduplicate below.
    cand = np.unique(
        np.concatenate(
            [
                np.linspace(max(lo - 1e-6, 0.0), hi + 1e-6, n_grid),
                np.quantile(pos_max[pos_max > 0], np.linspace(0, 1, min(200, len(pos_max)))) if (pos_max > 0).any() else np.array([0.0]),
            ]
        )
    )
    cand = np.sort(cand)

    best_thr = cand[0]
    best_rate = (pos_max > cand[0]).mean()
    best_gap = abs(best_rate - target)

    for thr in cand:
        rate = float((pos_max > thr).mean())
        gap = abs(rate - target)
        # Prefer smaller gap, tie-break: rate >= target > rate < target
        better = (gap < best_gap - 1e-12) or (
            abs(gap - best_gap) <= 1e-12 and rate >= target > best_rate
        )
        if better:
            best_gap = gap
            best_rate = rate
            best_thr = float(thr)

    return best_thr, best_rate


def evaluate_rule_with_threshold(
    scores: pd.DataFrame,
    high: np.ndarray,
    threshold: float,
    K: int,
):
    n = len(scores)
    TP = FP = TN = FN = 0
    retest_flag = np.zeros(n, dtype=bool)
    m_used = np.zeros(n, dtype=int)

    for i in range(n):
        retest, m, _ = decide_retest_thresholded(scores.iloc[i], threshold, K)
        retest_flag[i] = retest
        m_used[i] = m
        if retest and high[i]:
            TP += 1
        elif retest and not high[i]:
            FP += 1
        elif (not retest) and high[i]:
            FN += 1
        else:
            TN += 1

    retest_rate = float(retest_flag.mean())
    recall = float(TP / (TP + FN)) if (TP + FN) > 0 else float("nan")
    precision = float(TP / (TP + FP)) if (TP + FP) > 0 else float("nan")
    avg_tests_if_retest = float(m_used[retest_flag].mean()) if retest_flag.any() else 0.0

    return {
        "retest_rate": retest_rate,
        "recall": recall,
        "precision": precision,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "avg_tests_if_retest": avg_tests_if_retest,
    }


# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #
RULE_ALIASES = {
    "total": "Total",
    "mix_05_05": "Combined Equal (0.5/0.5)",
    "local": "Local",
    "mix_08_02": "Combined Local-W (0.8/0.2)",
}
RULE_ORDER = ["total", "mix_05_05", "local", "mix_08_02"]


def _max_retest_rate(scores: pd.DataFrame) -> float:
    """Retest ceiling for this rule (threshold -> -inf ⇒ retest iff any score > 0)."""
    pos_max = scores.where(scores > 0).max(axis=1).fillna(0.0).to_numpy()
    return float((pos_max > 0).mean())


def run_one_seed(
    seed: int,
    base_output_dir: Path,
    target: float,
    tau: float,
    K: int,
    smoke: bool,
):
    seed_dir = base_output_dir / "semi_synthetic" / "semi_synthetic" / f"seed_{seed}"
    if not seed_dir.exists():
        raise FileNotFoundError(
            f"Expected pre-computed outputs at: {seed_dir}\n"
            "Run `python experiments/run.py --exp semi_synthetic --seed <seed> ...` first."
        )

    data_csv = seed_dir / "test_raw.csv"
    df = pd.read_csv(data_csv)
    if smoke:
        df = df.head(100)
    high = (df["y"].to_numpy() >= tau)

    phi_total = load_phi_df(str(seed_dir / "phi_total.csv"))
    phi_local = load_phi_df(str(seed_dir / "phi_local.csv"))
    phi_global = load_phi_df(str(seed_dir / "phi_global.csv"))

    if len(phi_global) != 1:
        raise ValueError("phi_global.csv should have exactly 1 row.")

    phi_total, phi_local, phi_global, _cols = align_feature_columns(
        phi_total, phi_local, phi_global
    )
    if smoke:
        phi_total = phi_total.head(100).reset_index(drop=True)
        phi_local = phi_local.head(100).reset_index(drop=True)
    g_row = phi_global.iloc[0]

    # Determine feasible target: min of the rule ceilings clipped at the user target.
    scores_by_rule = {r: build_scores(phi_total, phi_local, g_row, r) for r in RULE_ORDER}
    ceilings = {r: _max_retest_rate(s) for r, s in scores_by_rule.items()}
    min_ceiling = min(ceilings.values())
    feasible_target = min(target, min_ceiling - 1e-6) if min_ceiling < target else target

    rows = []
    for rule in RULE_ORDER:
        scores = scores_by_rule[rule]
        threshold, _grid_rate = tune_threshold_for_target(
            scores, target=feasible_target, K=K
        )
        out = evaluate_rule_with_threshold(scores, high=high, threshold=threshold, K=K)
        rows.append(
            {
                "seed": seed,
                "rule_key": rule,
                "rule": RULE_ALIASES[rule],
                "requested_target": target,
                "feasible_target": feasible_target,
                "ceiling": ceilings[rule],
                "threshold": threshold,
                **out,
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=str, default=str(Path(__file__).parent))
    p.add_argument(
        "--outputs_root",
        type=str,
        default=str(REPO_ROOT / "outputs"),
        help="Root `outputs/` directory with pre-computed phi_*.csv",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[19])
    p.add_argument(
        "--target",
        type=float,
        default=0.46,
        help="Target retest rate to match across rules.",
    )
    p.add_argument("--tau", type=float, default=18.0, help="High-risk label threshold on y.")
    p.add_argument("--K", type=int, default=3, help="Top-K retest features.")
    p.add_argument("--smoke", action="store_true", help="Use only first 100 rows.")
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(args.outputs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in args.seeds:
        print(f"[matched_decision] seed={seed} target={args.target} tau={args.tau} K={args.K}"
              f" smoke={args.smoke}")
        df = run_one_seed(
            seed=seed,
            base_output_dir=base,
            target=args.target,
            tau=args.tau,
            K=args.K,
            smoke=args.smoke,
        )
        print(df.to_string(index=False))
        all_rows.append(df)

    final = pd.concat(all_rows, ignore_index=True)
    out_csv = out_dir / ("results_smoke.csv" if args.smoke else "results.csv")
    final.to_csv(out_csv, index=False)
    print(f"\n[matched_decision] saved to {out_csv}")

    # Also print a clean summary
    summary = (
        final.groupby("rule", sort=False)[
            ["ceiling", "retest_rate", "recall", "precision"]
        ]
        .mean()
        .reset_index()
    )
    print("\n[matched_decision] averaged across seeds (ceiling = max retest rate achievable):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
