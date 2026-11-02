"""Table 4 (Appendix B.5) and the Section 6.2 timings: wall-clock runtime by feature dimension.

Reads results.csv of the exact (d <= 10) and sampled (d >= 15, 128 antithetic pairs) runtime
runs. The per-query phi^L time is the total over the timed queries divided by their number.
Timings depend on the machine.
"""
import pandas as pd

from common import OUTPUTS, parser, write


def main() -> None:
    a = parser(__doc__, OUTPUTS / "runtime").parse_args()
    exact = pd.read_csv(a.inp / "exact" / "results.csv").assign(regime="exact", pairs=0)
    sampled = pd.read_csv(a.inp / "sampled" / "results.csv").assign(regime="antithetic")
    sampled["pairs"] = sampled["budget"] // 2
    cols = ["d", "regime", "pairs", "n_calib", "n_test", "offline_phiG_sec", "online_phiL_sec_per_query"]
    write(pd.concat([exact[cols], sampled[cols]], ignore_index=True).sort_values("d"), a.out, "table4_runtime.csv")


if __name__ == "__main__":
    main()
