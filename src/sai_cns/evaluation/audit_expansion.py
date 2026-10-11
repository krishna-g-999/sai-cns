"""Audit the effect of the BindingDB expansion: compare pooled models to the ChEMBL-only baseline.

Because the BindingDB affinity export contributes actives, the honest question is whether the larger
data changed cross-validated performance for the better, for the worse, or not at all. This compares
the scaffold-split headline metric (the strict, generalisation estimate) endpoint by endpoint. A
large jump would flag inflation from easy analogues; a drop in the balance-sensitive Matthews
correlation would flag class-imbalance harm. Small, stable changes mean the added measured data is
legitimate and neither inflates nor degrades the model.

Output: results/tables/expansion_audit.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TAB = ROOT / "results" / "tables"


def main() -> None:
    base = pd.read_csv(TAB / "rf_cv_summary_chembl_only_baseline.csv")
    new = pd.read_csv(TAB / "rf_cv_summary.csv")
    rows = []
    for ep in new.endpoint.unique():
        task = new[new.endpoint == ep].task.iloc[0]
        metric = "roc_auc_mean" if task == "classification" else "r2_mean"
        mcc = "mcc_mean" if task == "classification" else None
        for split in ("random", "scaffold"):
            b = base[(base.endpoint == ep) & (base.split == split)]
            n = new[(new.endpoint == ep) & (new.split == split)]
            if b.empty or n.empty:
                continue
            row = {"endpoint": ep, "task": task, "split": split,
                   "n_baseline": int(b.n.iloc[0]), "n_pooled": int(n.n.iloc[0]),
                   "n_added": int(n.n.iloc[0] - b.n.iloc[0]),
                   "metric": metric.replace("_mean", ""),
                   "baseline": round(float(b[metric].iloc[0]), 4),
                   "pooled": round(float(n[metric].iloc[0]), 4),
                   "delta": round(float(n[metric].iloc[0] - b[metric].iloc[0]), 4)}
            if mcc and mcc in b and mcc in n:
                row["mcc_baseline"] = round(float(b[mcc].iloc[0]), 4)
                row["mcc_pooled"] = round(float(n[mcc].iloc[0]), 4)
                row["mcc_delta"] = round(float(n[mcc].iloc[0] - b[mcc].iloc[0]), 4)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "expansion_audit.csv", index=False)
    print("=== expansion audit (pooled vs ChEMBL-only baseline) ===")
    print(out.to_string(index=False))
    scaf = out[out.split == "scaffold"]
    print(f"\nscaffold headline: mean delta {scaf.delta.mean():+.4f}  "
          f"(max {scaf.delta.max():+.4f}, min {scaf.delta.min():+.4f})")
    print("wrote results/tables/expansion_audit.csv")


if __name__ == "__main__":
    main()
