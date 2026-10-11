"""Complete results table for the scaffold-held-out prospective test.

Every held-out compound shares no Bemis-Murcko scaffold with anything its model was trained on, so
recall here estimates how the panel behaves on a genuinely new chemical series rather than on
rearranged versions of series it already knows.

Reported per target: training and held-out counts, the number of withheld scaffolds, the decision
threshold, recall with a Wilson 95% interval, and a flag for thresholds that collapsed to the
allowed floor. A collapsed threshold means the model could not separate its actives from background
chemistry at all, so its apparently high recall is not evidence of discrimination and is excluded
from the headline average.

Output: results/tables/scaffold_holdout_results.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
M = ROOT / "models_rf" / "holdout"
TAB = ROOT / "results" / "tables"
FLOOR = 0.06   # thresholds at or below this collapsed to the allowed minimum


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    modes = json.loads((M / "binder_modes.json").read_text())
    rows = []
    for ep, v in modes.items():
        n = v["n_holdout_actives"]
        r = v["holdout_recall_at_threshold"]
        k = int(round(r * n))
        lo, hi = wilson(k, n)
        collapsed = v["threshold"] <= FLOOR
        rows.append({
            "target": ep,
            "train_actives": v["n_train_actives"],
            "holdout_actives": n,
            "holdout_scaffolds": v["n_holdout_scaffolds"],
            "threshold": v["threshold"],
            "holdout_recall": r,
            "recall_ci95_low": round(lo, 3),
            "recall_ci95_high": round(hi, 3),
            "threshold_collapsed": collapsed,
            "usable": not collapsed and r >= 0.50,
        })
    df = pd.DataFrame(rows).sort_values("holdout_recall", ascending=False)
    df.to_csv(TAB / "scaffold_holdout_results.csv", index=False)

    good = df[~df.threshold_collapsed]
    tot_n = int(good.holdout_actives.sum())
    tot_k = int((good.holdout_recall * good.holdout_actives).round().sum())
    lo, hi = wilson(tot_k, tot_n)

    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print()
    print(f"targets retrained                       : {len(df)}")
    print(f"held-out compounds (all)                : {int(df.holdout_actives.sum()):,}")
    print(f"held-out scaffolds (all)                : {int(df.holdout_scaffolds.sum()):,}")
    print(f"targets with a collapsed threshold      : {df.threshold_collapsed.sum()} "
          f"({df.loc[df.threshold_collapsed,'target'].tolist()})")
    print()
    print(f"POOLED prospective recall (excluding collapsed): {tot_k:,}/{tot_n:,} = "
          f"{tot_k/tot_n:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"mean per-target recall                  : {good.holdout_recall.mean():.3f}")
    print(f"median per-target recall                : {good.holdout_recall.median():.3f}")
    print(f"targets at or above 0.80 recall         : {(good.holdout_recall>=0.80).sum()}/{len(good)}")
    print(f"targets below 0.50 recall               : {(good.holdout_recall<0.50).sum()} "
          f"({good.loc[good.holdout_recall<0.50,'target'].tolist()})")
    print(f"\nwrote {TAB / 'scaffold_holdout_results.csv'}")


if __name__ == "__main__":
    main()
