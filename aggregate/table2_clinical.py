"""Table 2: decision rules on the semi-synthetic clinical task, mean (std) over seeds 0-29.

Per seed, decision_process.evaluate_rules applies each rule (a patient is re-tested when any
score is positive; up to K = 3 features with positive scores are ordered; high risk = y >= 18).
Recall = TP / (TP + FN), precision = TP / (TP + FP). The standard deviation uses ddof = 1.
"""
import sys

import pandas as pd

from common import OUTPUTS, REPO, parser, write

sys.path.insert(0, str(REPO))
from experiments.decision_process import evaluate_rules  # noqa: E402

SEEDS = range(30)
RULES = {"total": "phi^T", "local": "phi^L", "mix_08_02": "0.8 phi^L + 0.2 phi^G"}
METRICS = ["retest_rate", "recall", "precision", "TP"]


def main() -> None:
    a = parser(__doc__, OUTPUTS / "semi_synthetic" / "semi_synthetic").parse_args()
    rows = []
    for s in SEEDS:
        run = a.inp / f"seed_{s}"
        r = evaluate_rules(str(run / "test_raw.csv"), str(run / "phi_total.csv"), str(run / "phi_local.csv"),
                           str(run / "phi_global.csv"), tau=18.0, K=3)
        r = r[r["rule"].isin(RULES)].copy()
        r["recall"] = r["TP"] / (r["TP"] + r["FN"])
        r["precision"] = r["TP"] / (r["TP"] + r["FP"])
        r.insert(0, "seed", s)
        rows.append(r[["seed", "rule"] + METRICS + ["FP", "TN", "FN"]])
    per_seed = pd.concat(rows, ignore_index=True)
    summary = []
    for rule, label in RULES.items():
        sub = per_seed[per_seed["rule"] == rule]
        for m in METRICS:
            summary.append({"rule": rule, "label": label, "metric": m, "mean": sub[m].mean(),
                            "std": sub[m].std(ddof=1), "n_seeds": len(sub)})
    write(per_seed, a.out, "table2_clinical_per_seed.csv")
    write(pd.DataFrame(summary), a.out, "table2_clinical.csv")


if __name__ == "__main__":
    main()
