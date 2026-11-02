"""
Aggregate sampling-validation per-cell .npz files into:
  - results.csv     wall-clock numbers per cell
  - agreement.csv   sampled-vs-exact agreement, paired by dimension and seed
                    (paper settings: d=10 and d=15)
  - convergence.csv optional comparison across multiple permutation budgets,
                    using the largest budget as reference; not produced by the
                    paper's single-budget reproduction run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_cell(p: Path) -> dict:
    z = np.load(p, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    # Convert numpy scalars to Python types for CSV-friendliness.
    for k in (
        "method", "backend",
    ):
        out[k] = str(out[k])
    for k in ("d", "budget", "seed", "n_calib", "n_test", "n_samples"):
        out[k] = int(out[k])
    for k in ("alpha", "offline_sec", "online_per_query_sec", "online_total_sec"):
        out[k] = float(out[k])
    return out


def agreement_pair(exact: dict, sampled: dict) -> dict:
    """Compare phi_global and phi_local between an exact and a sampled cell."""
    pg_e, pg_s = exact["phi_global"], sampled["phi_global"]
    pl_e, pl_s = exact["phi_local"], sampled["phi_local"]

    def _stats(a, b):
        a, b = a.ravel(), b.ravel()
        scale = max(np.max(np.abs(a)), 1e-12)
        max_abs_diff = float(np.max(np.abs(a - b)))
        mae = float(np.mean(np.abs(a - b)))
        # Pearson r; guard against zero variance.
        if a.std() < 1e-12 or b.std() < 1e-12:
            r = float("nan")
        else:
            r = float(np.corrcoef(a, b)[0, 1])
        return max_abs_diff, max_abs_diff / scale, mae, r

    g_max, g_max_rel, g_mae, g_r = _stats(pg_e, pg_s)
    l_max, l_max_rel, l_mae, l_r = _stats(pl_e, pl_s)

    return {
        "d": exact["d"],
        "seed": exact["seed"],
        "budget": sampled["budget"],
        # Global
        "phiG_max_abs_diff": g_max,
        "phiG_max_rel_diff": g_max_rel,
        "phiG_mae": g_mae,
        "phiG_pearson_r": g_r,
        # Local
        "phiL_max_abs_diff": l_max,
        "phiL_max_rel_diff": l_max_rel,
        "phiL_mae": l_mae,
        "phiL_pearson_r": l_r,
        # wall-clock
        "exact_offline_sec": exact["offline_sec"],
        "sampled_offline_sec": sampled["offline_sec"],
        "exact_online_per_query_sec": exact["online_per_query_sec"],
        "sampled_online_per_query_sec": sampled["online_per_query_sec"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells_dir", default="cells")
    p.add_argument("--out_results", default="results.csv")
    p.add_argument("--out_agreement", default="agreement.csv")
    p.add_argument("--out_convergence", default="convergence.csv")
    args = p.parse_args()

    cells_dir = Path(args.cells_dir)
    files = sorted(cells_dir.glob("cell_*.npz"))
    if not files:
        raise SystemExit(f"No cell_*.npz under {cells_dir}")

    cells = [load_cell(f) for f in files]
    print(f"[aggregate] loaded {len(cells)} cells")

    # ---- results.csv: wall-clock per cell ----
    rows = []
    for c in cells:
        rows.append({
            "method": c["method"], "d": c["d"], "budget": c["budget"], "seed": c["seed"],
            "backend": c["backend"], "n_calib": c["n_calib"], "n_test": c["n_test"],
            "offline_sec": c["offline_sec"],
            "online_per_query_sec": c["online_per_query_sec"],
            "online_total_sec": c["online_total_sec"],
        })
    res_df = pd.DataFrame(rows).sort_values(
        ["method", "d", "budget", "seed"]
    ).reset_index(drop=True)
    res_df.to_csv(args.out_results, index=False)
    print(f"[aggregate] wrote {args.out_results}  ({len(res_df)} rows)")

    # ---- agreement.csv: pair exact vs sampled at the same (d, seed) ----
    pairs = []
    by_key: dict[tuple, dict] = {}
    for c in cells:
        by_key[(c["method"], c["d"], c["budget"], c["seed"])] = c

    for (method, d, budget, seed), c in by_key.items():
        if method != "sampled":
            continue
        exact_key = ("exact", d, 0, seed)
        if exact_key in by_key:
            pairs.append(agreement_pair(by_key[exact_key], c))

    if pairs:
        agr_df = pd.DataFrame(pairs).sort_values(
            ["d", "budget", "seed"]
        ).reset_index(drop=True)
        agr_df.to_csv(args.out_agreement, index=False)
        print(f"[aggregate] wrote {args.out_agreement}  ({len(agr_df)} pairs)")

        # Summary across seeds
        summ = (
            agr_df.groupby(["d", "budget"], as_index=False)
            .agg(
                phiG_pearson_r_mean=("phiG_pearson_r", "mean"),
                phiL_pearson_r_mean=("phiL_pearson_r", "mean"),
                phiG_max_rel_diff_mean=("phiG_max_rel_diff", "mean"),
                phiL_max_rel_diff_mean=("phiL_max_rel_diff", "mean"),
                phiG_mae_mean=("phiG_mae", "mean"),
                phiL_mae_mean=("phiL_mae", "mean"),
            )
        )
        with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
            print("[aggregate] agreement summary (sampled vs exact):")
            print(summ.to_string(index=False))
    else:
        print("[aggregate] no exact-vs-sampled pairs found; skipping agreement.csv")

    # ---- convergence.csv: sampler-internal across budgets per d (ref = max budget) ----
    samp_cells = [c for c in cells if c["method"] == "sampled"]
    conv_rows = []
    for d_val in sorted({c["d"] for c in samp_cells}):
        sub = [c for c in samp_cells if c["d"] == d_val]
        budgets = sorted({c["budget"] for c in sub})
        if len(budgets) < 2:
            continue
        ref_budget = max(budgets)
        for seed in sorted({c["seed"] for c in sub}):
            ref = next(
                (c for c in sub if c["budget"] == ref_budget and c["seed"] == seed),
                None,
            )
            if ref is None:
                continue
            for c in sub:
                if c["seed"] != seed or c["budget"] == ref_budget:
                    continue
                stats = {
                    "d": d_val, "seed": seed, "budget": c["budget"],
                    "ref_budget": ref_budget,
                }
                for name, a, b in (
                    ("phiG", c["phi_global"], ref["phi_global"]),
                    ("phiL", c["phi_local"], ref["phi_local"]),
                ):
                    diff = (a - b).ravel()
                    scale = max(np.max(np.abs(b)), 1e-12)
                    stats[f"{name}_max_abs_diff"] = float(np.max(np.abs(diff)))
                    stats[f"{name}_max_rel_diff"] = float(np.max(np.abs(diff)) / scale)
                    stats[f"{name}_mae"] = float(np.mean(np.abs(diff)))
                conv_rows.append(stats)

    if conv_rows:
        conv_df = pd.DataFrame(conv_rows).sort_values(
            ["d", "budget", "seed"]
        ).reset_index(drop=True)
        conv_df.to_csv(args.out_convergence, index=False)
        print(f"[aggregate] wrote {args.out_convergence}  ({len(conv_df)} rows)")


if __name__ == "__main__":
    main()
