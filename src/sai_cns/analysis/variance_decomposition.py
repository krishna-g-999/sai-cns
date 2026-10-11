"""Why are the cross-validation error bars larger under the scaffold split?

The between-fold standard deviation reported with each endpoint mixes two distinct sources:

  1. sampling noise     - each fold metric is estimated on a finite test set, so it carries its own
                          standard error even if every fold were equally difficult;
  2. fold heterogeneity - genuine variation in how hard each held-out fold is.

For k folds with observed fold metrics m_i,

      Var_between(m)  =  Var_heterogeneity  +  mean_i Var_sampling(m_i)

so the heterogeneity component is recovered by subtraction, with Var_sampling(m_i) estimated by
bootstrapping within each fold's own test set. Under the random split folds are statistically
exchangeable, so heterogeneity should be near zero and the error bar should be almost pure sampling
noise. Under the scaffold split each fold holds out entire chemical series, so a genuine
heterogeneity term should appear. This script tests that prediction.

Output: results/tables/manuscript_T4_variance_decomposition.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score

ROOT = Path(__file__).resolve().parents[3]
OOF = ROOT / "data" / "processed" / "cv_predictions"
TAB = ROOT / "results" / "tables"
CLASSIFIERS = ["BBB", "AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG"]
REGRESSORS = ["D2", "A2A", "HT2A", "SERT", "antioxidant_DPPH"]
B = 400
rng = np.random.default_rng(7)


def _metric(y, p, task):
    if task == "classification":
        return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    return r2_score(y, p)


def _boot_var(y, p, task):
    """Sampling variance of the fold metric, by bootstrap within that fold."""
    n = len(y)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = _metric(y[idx], p[idx], task)
        if np.isfinite(v):
            vals.append(v)
    return float(np.var(vals, ddof=1)) if len(vals) > 2 else np.nan


def main():
    rows = []
    for ep in CLASSIFIERS + REGRESSORS:
        task = "classification" if ep in CLASSIFIERS else "regression"
        for split in ("random", "scaffold"):
            f = OOF / f"{ep}_{split}_oof.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            fold_m, fold_var = [], []
            for _, g in d.groupby("fold"):
                y, p = g["y_true"].to_numpy(), g["prediction"].to_numpy()
                m = _metric(y, p, task)
                if not np.isfinite(m):
                    continue
                fold_m.append(m)
                fold_var.append(_boot_var(y, p, task))
            fold_m = np.array(fold_m, float)
            fold_var = np.array(fold_var, float)
            var_between = float(np.var(fold_m, ddof=1))
            var_sampling = float(np.nanmean(fold_var))
            var_hetero = max(0.0, var_between - var_sampling)
            rows.append({
                "endpoint": ep, "task": task, "split": split, "k_folds": len(fold_m),
                "mean": round(float(fold_m.mean()), 4),
                "sd_observed": round(float(np.sqrt(var_between)), 4),
                "sd_sampling": round(float(np.sqrt(var_sampling)), 4),
                "sd_heterogeneity": round(float(np.sqrt(var_hetero)), 4),
                "pct_variance_heterogeneity": round(100 * var_hetero / var_between, 1) if var_between > 0 else 0.0,
            })
            print(f"{ep:17}{split:9} sd_obs={np.sqrt(var_between):.4f} "
                  f"sd_sampling={np.sqrt(var_sampling):.4f} sd_hetero={np.sqrt(var_hetero):.4f} "
                  f"({100*var_hetero/var_between if var_between>0 else 0:.0f}% heterogeneity)", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "manuscript_T4_variance_decomposition.csv", index=False)

    print("\n=== SUMMARY: mean share of between-fold variance that is genuine heterogeneity ===")
    print(out.groupby("split")["pct_variance_heterogeneity"].agg(["mean", "median", "min", "max"]).round(1).to_string())
    print("\nwrote", TAB / "manuscript_T4_variance_decomposition.csv")


if __name__ == "__main__":
    main()
