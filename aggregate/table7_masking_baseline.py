"""Table 7 (Appendix E.2): sensitivity of phi^G to the masking baseline, mean over seeds 42-71.

For each completion rule and rho: D = symmetric normalized L1 distance and T3 = top-3 overlap
between phi^G under the rule and phi^G under the calibration mean (baseline_agreement.csv,
component phiG), and the cancellation rate in percent within the rule (component_events.csv,
event active1pct_cancellation). Also writes the ranges quoted in the text.
In the code, the pooled fixed mean is `empirical_marginal` and the calibration mean is
`empirical_mean`.
"""
import pandas as pd

from common import OUTPUTS, parser, write

ROWS = {"empirical_marginal": "Pooled fixed mean", "empirical_product_marginal": "Product-marginal resampling",
        "oracle_conditional": "Oracle conditional mean", "gaussian_conditional": "Estimated conditional mean",
        "empirical_mean": "Calibration mean (reference)"}


def main() -> None:
    a = parser(__doc__, OUTPUTS / "masking_baseline" / "agg").parse_args()
    agree = pd.read_csv(a.inp / "baseline_agreement.csv")
    agree = agree[(agree.baseline_a == "empirical_mean") & (agree.component == "phiG")]
    agree = agree[["rho_x", "data_seed", "baseline_b", "symmetric_normalized_l1", "top3_overlap"]].rename(
        columns={"baseline_b": "baseline", "symmetric_normalized_l1": "D", "top3_overlap": "T3"})
    events = pd.read_csv(a.inp / "component_events.csv")
    events = events[events.event == "active1pct_cancellation"][["rho_x", "data_seed", "baseline", "cell_rate"]]
    events = events.assign(cancel_percent=100.0 * events.cell_rate).drop(columns="cell_rate")
    m = agree.merge(events, on=["rho_x", "data_seed", "baseline"], how="outer")
    m = m[m["baseline"].isin(ROWS)]
    t = m.groupby(["baseline", "rho_x"]).agg(D=("D", "mean"), T3=("T3", "mean"), cancel_percent=("cancel_percent", "mean"),
                                             n_seeds=("data_seed", "nunique")).reset_index()
    t.insert(1, "label", t["baseline"].map(ROWS))
    write(t, a.out, "table7_masking_baseline.csv")

    def rng(rows, col, rho=None):
        v = t[t["baseline"].isin(rows) & ((t["rho_x"] == rho) if rho is not None else True)][col]
        return v.min(), v.max()
    fixed_cond = ["empirical_marginal", "oracle_conditional", "gaussian_conditional"]
    cond = ["oracle_conditional", "gaussian_conditional"]
    texts = [("pooled fixed mean D", *rng(["empirical_marginal"], "D")),
             ("pooled fixed mean T3", *rng(["empirical_marginal"], "T3")),
             ("product-marginal D", *rng(["empirical_product_marginal"], "D")),
             ("product-marginal T3", *rng(["empirical_product_marginal"], "T3")),
             ("conditional-mean D at rho=0", *rng(cond, "D", 0.0)),
             ("conditional-mean D at rho=0.95", *rng(cond, "D", 0.95)),
             ("conditional-mean T3", *rng(cond, "T3")),
             ("cancellation %, fixed- and conditional-mean rules", *rng(fixed_cond, "cancel_percent")),
             ("cancellation %, product-marginal", *rng(["empirical_product_marginal"], "cancel_percent")),
             ("cancellation %, calibration mean", *rng(["empirical_mean"], "cancel_percent"))]
    write(pd.DataFrame(texts, columns=["quantity", "min", "max"]), a.out, "table7_masking_baseline_ranges.csv")


if __name__ == "__main__":
    main()
