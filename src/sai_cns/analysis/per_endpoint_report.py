"""Complete per-endpoint validation report for the manuscript.

Recomputes, from the saved out-of-fold (OOF) cross-validation predictions, the full per-fold and
per-endpoint metrics for every trained endpoint, under both the random and the scaffold-grouped
10-fold splits. Produces:

  results/tables/manuscript_T1_endpoints.csv    one row per endpoint: data, task, both splits
  results/tables/manuscript_T2_per_fold.csv     one row per endpoint x split x fold
  results/tables/manuscript_T3_variance.csv     error-bar decomposition (why scaffold SD is larger)

Nothing is estimated; every number is computed from the saved predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             f1_score, balanced_accuracy_score, r2_score,
                             mean_absolute_error, mean_squared_error)

ROOT = Path(__file__).resolve().parents[3]
OOF = ROOT / "data" / "processed" / "cv_predictions"
TAB = ROOT / "results" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

CLASSIFIERS = ["BBB", "AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG"]
REGRESSORS = ["D2", "A2A", "HT2A", "SERT", "antioxidant_DPPH"]


def clf_metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    out = {"roc_auc": np.nan, "pr_auc": np.nan, "mcc": np.nan, "f1": np.nan, "balanced_acc": np.nan}
    if len(np.unique(y)) > 1:
        out["roc_auc"] = roc_auc_score(y, p)
        out["pr_auc"] = average_precision_score(y, p)
        out["mcc"] = matthews_corrcoef(y, yhat)
        out["f1"] = f1_score(y, yhat)
        out["balanced_acc"] = balanced_accuracy_score(y, yhat)
    return out


def reg_metrics(y, p):
    return {"r2": r2_score(y, p), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": mean_absolute_error(y, p),
            "spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman"))}


def main():
    per_fold, per_endpoint = [], []
    for ep in CLASSIFIERS + REGRESSORS:
        task = "classification" if ep in CLASSIFIERS else "regression"
        for split in ("random", "scaffold"):
            f = OOF / f"{ep}_{split}_oof.csv"
            if not f.exists():
                print(f"  missing {f.name}")
                continue
            d = pd.read_csv(f)
            y_all = d["y_true"].to_numpy()
            p_all = d["prediction"].to_numpy()
            n_scaf = int(d["scaffold_group"].nunique()) if "scaffold_group" in d else np.nan

            fold_rows = []
            for fold, g in d.groupby("fold"):
                y, p = g["y_true"].to_numpy(), g["prediction"].to_numpy()
                m = clf_metrics(y, p) if task == "classification" else reg_metrics(y, p)
                row = {"endpoint": ep, "task": task, "split": split, "fold": int(fold),
                       "n_test": len(g), "n_train": len(d) - len(g),
                       "n_scaffolds_test": int(g["scaffold_group"].nunique()) if "scaffold_group" in g else np.nan,
                       "positives_test": int(y.sum()) if task == "classification" else np.nan, **m}
                fold_rows.append(row)
            per_fold.extend(fold_rows)
            fr = pd.DataFrame(fold_rows)

            base = {"endpoint": ep, "task": task, "split": split, "n_compounds": len(d),
                    "n_scaffolds": n_scaf, "n_folds": int(d["fold"].nunique()),
                    "median_train_per_fold": int(fr["n_train"].median()),
                    "median_test_per_fold": int(fr["n_test"].median())}
            if task == "classification":
                base["positive_rate"] = round(float(y_all.mean()), 4)
                for k in ("roc_auc", "pr_auc", "mcc", "f1", "balanced_acc"):
                    base[f"{k}_mean"] = round(float(fr[k].mean()), 4)
                    base[f"{k}_sd"] = round(float(fr[k].std(ddof=1)), 4)
                base["roc_auc_min"] = round(float(fr["roc_auc"].min()), 4)
                base["roc_auc_max"] = round(float(fr["roc_auc"].max()), 4)
                base["roc_auc_pooled"] = round(float(roc_auc_score(y_all, p_all)), 4)
            else:
                for k in ("r2", "rmse", "mae", "spearman"):
                    base[f"{k}_mean"] = round(float(fr[k].mean()), 4)
                    base[f"{k}_sd"] = round(float(fr[k].std(ddof=1)), 4)
                base["r2_min"] = round(float(fr["r2"].min()), 4)
                base["r2_max"] = round(float(fr["r2"].max()), 4)
                base["r2_pooled"] = round(float(r2_score(y_all, p_all)), 4)
            per_endpoint.append(base)

    T2 = pd.DataFrame(per_fold)
    T1 = pd.DataFrame(per_endpoint)
    T2.to_csv(TAB / "manuscript_T2_per_fold.csv", index=False)
    T1.to_csv(TAB / "manuscript_T1_endpoints.csv", index=False)

    # variance decomposition: random vs scaffold spread per endpoint
    rows = []
    for ep in T1.endpoint.unique():
        a = T1[(T1.endpoint == ep) & (T1.split == "random")]
        b = T1[(T1.endpoint == ep) & (T1.split == "scaffold")]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        key = "roc_auc" if a.task == "classification" else "r2"
        rows.append({"endpoint": ep, "task": a.task, "n_compounds": a.n_compounds,
                     "n_scaffolds": a.n_scaffolds,
                     "scaffold_diversity": round(float(a.n_scaffolds / a.n_compounds), 3),
                     "random_mean": a[f"{key}_mean"], "random_sd": a[f"{key}_sd"],
                     "scaffold_mean": b[f"{key}_mean"], "scaffold_sd": b[f"{key}_sd"],
                     "sd_ratio": round(float(b[f"{key}_sd"] / a[f"{key}_sd"]), 2),
                     "generalisation_gap": round(float(a[f"{key}_mean"] - b[f"{key}_mean"]), 4),
                     "scaffold_fold_range": round(float(b[f"{key}_max"] - b[f"{key}_min"]), 4)})
    T3 = pd.DataFrame(rows).sort_values("sd_ratio", ascending=False)
    T3.to_csv(TAB / "manuscript_T3_variance.csv", index=False)

    print("=== T1 endpoints ===", len(T1), "rows,", T1.endpoint.nunique(), "endpoints")
    print("=== T2 per-fold ===", len(T2), "rows")
    print("\n=== T3 variance (why error bars differ) ===")
    print(T3.to_string(index=False))
    # correlation: does scaffold diversity predict the SD ratio?
    cc = T3[["scaffold_diversity", "sd_ratio", "generalisation_gap"]].corr(method="spearman")
    print("\nSpearman correlations:\n", cc.round(3).to_string())


if __name__ == "__main__":
    main()
