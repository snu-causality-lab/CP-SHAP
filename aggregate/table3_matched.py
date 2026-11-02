"""Table 3 (Appendix B.3) and the Section 6.3 ceiling numbers, clinical seed 19.

Table 3: experiments/matched_decision/matched_decision.py tunes one threshold per rule so that all
rules re-test the same fraction of patients (target 0.46, lowered to the smallest rule ceiling).
Section 6.3: share of test patients whose phi^T is <= 0 for every feature, and the resulting
ceiling of the phi^T rule (share of patients with at least one positive phi^T).
"""
import sys

import pandas as pd

from common import OUTPUTS, REPO, parser, write

sys.path.insert(0, str(REPO / "experiments" / "matched_decision"))
from matched_decision import run_one_seed  # noqa: E402

SEED = 19
RULES = {"total": "Total (phi^T)", "local": "Local (phi^L)", "mix_08_02": "Combined local-W (0.8 phi^L + 0.2 phi^G)"}


def main() -> None:
    a = parser(__doc__, OUTPUTS).parse_args()  # the matched-rule code reads semi_synthetic/semi_synthetic/seed_19 below this root
    res = run_one_seed(SEED, a.inp, target=0.46, tau=18.0, K=3, smoke=False)
    res = res[res["rule_key"].isin(RULES)].copy()
    res["label"] = res["rule_key"].map(RULES)
    write(res, a.out, "table3_matched.csv")

    T = pd.read_csv(a.inp / "semi_synthetic" / "semi_synthetic" / f"seed_{SEED}" / "phi_total.csv")
    all_nonpositive = (T <= 0).all(axis=1)
    row = {"seed": SEED, "n_test": len(T), "share_all_phiT_nonpositive": float(all_nonpositive.mean()),
           "percent_all_phiT_nonpositive": 100.0 * float(all_nonpositive.mean()),
           "total_rule_ceiling": float((T > 0).any(axis=1).mean())}
    write(pd.DataFrame([row]), a.out, "section6_3_ceiling.csv")


if __name__ == "__main__":
    main()
