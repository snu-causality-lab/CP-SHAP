"""Figure 8 (Appendix B.4): sign flips between phi^L and phi^T across CP families (seed 42, alpha 0.10).

Per family, instances are ranked by the number of clean flips (sign(phi^L) != sign(phi^T) with
|phi^L| and |phi^T| at least 10% of the instance's largest |phi^L|), then by the largest
|phi^T - phi^L|; instances with the same rounded sum |phi^L| are skipped as duplicates and the
top three are kept. Writes the per-feature bars of the selected instances (highlighted = clean
flip) and one summary row per selected instance.
Reads the fixed seed42_alpha0.10 output of reproduce/cp_families.py; suffixed runs are not used.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUTPUTS, parser, write

FAMILIES = [("Concrete (locally adaptive)", "lacp_concrete"), ("Concrete (CQR)", "cqr_concrete"),
            ("Folktables (Mondrian)", "mondrian_folktables")]


def find_attributions(d: Path) -> list[str]:
    path = d / "seed42_alpha0.10" / "attributions.csv"
    return [str(path)] if path.is_file() else []


def find_signflip_instances(df: pd.DataFrame, top_n: int = 3) -> list[int]:
    df = df.copy()
    abs_L = df.groupby("instance_idx")["phi_L"].transform(lambda x: np.abs(x).max())
    tau = 0.10 * abs_L
    df["abs_L"] = np.abs(df["phi_L"])
    df["abs_T"] = np.abs(df["phi_T"])
    df["clean_flip"] = (np.sign(df["phi_L"]) * np.sign(df["phi_T"]) < 0) & (df["abs_L"] >= tau) & (df["abs_T"] >= tau)
    df["abs_swing"] = np.abs(df["phi_T"] - df["phi_L"])
    g = df.groupby("instance_idx").agg(n_clean_flip=("clean_flip", "sum"), max_abs_swing=("abs_swing", "max"),
                                       sum_abs_L=("abs_L", "sum"))
    g = g[g["n_clean_flip"] > 0]
    if g.empty:
        return []
    g["score"] = g["n_clean_flip"] * 1000 + g["max_abs_swing"]
    g = g.sort_values("score", ascending=False)
    seen, picked = set(), []
    for ix in g.index:
        sig = round(float(g.loc[ix, "sum_abs_L"]), 3)
        if sig in seen:
            continue
        seen.add(sig)
        picked.append(ix)
        if len(picked) >= top_n:
            break
    return picked


def main() -> None:
    a = parser(__doc__, OUTPUTS / "cp_families").parse_args()
    bars, summary = [], []
    for name, folder in FAMILIES:
        attrs = find_attributions(a.inp / folder)
        if not attrs:
            raise SystemExit(f"no seed 42, alpha 0.10 attributions under {a.inp / folder}")
        df = pd.read_csv(attrs[0])
        if "group" in df.columns:  # Mondrian rows carry the group; collapse to one row per (instance, feature)
            df = df.groupby(["seed", "alpha", "instance_idx", "feature"], as_index=False).agg(
                phi_L=("phi_L", "mean"), phi_G=("phi_G", "mean"), phi_T=("phi_T", "mean"))
        for rank, ix in enumerate(find_signflip_instances(df, top_n=3)):
            sub = df[df["instance_idx"] == ix].reset_index(drop=True)
            tau = 0.10 * np.abs(sub["phi_L"]).max()
            flip = (np.sign(sub["phi_L"]) * np.sign(sub["phi_T"]) < 0) & (np.abs(sub["phi_L"]) >= tau) & (np.abs(sub["phi_T"]) >= tau)
            for j, r in sub.iterrows():
                bars.append({"family": name, "rank": rank, "instance_idx": ix, "feature": r["feature"],
                             "phi_L": r["phi_L"], "phi_G": r["phi_G"], "phi_T": r["phi_T"], "highlighted": bool(flip[j])})
            summary.append({"family": name, "rank": rank, "instance_idx": ix, "n_features": len(sub),
                            "n_clean_flipped": int(flip.sum()), "max_abs_swing": float(np.abs(sub["phi_T"] - sub["phi_L"]).max()),
                            "tau_threshold": float(tau)})
    write(pd.DataFrame(bars), a.out, "figure8_signflip.csv")
    write(pd.DataFrame(summary), a.out, "figure8_selection.csv")


if __name__ == "__main__":
    main()
