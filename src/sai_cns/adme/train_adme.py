"""Train the ADME / exposure models (ladder B) with the same protocol as the target panel.

Random forest per endpoint, random and scaffold-grouped 10-fold, identical features (ECFP-4 + 12
descriptors). One classifier (P-gp inhibition) and five regressors. Kept separate from the
target-engagement models in `models_rf/adme/`.

Outputs:
  results/tables/adme_cv_summary.csv   mean +/- sd per endpoint and split
  results/tables/adme_cv_folds.csv     every fold
  models_rf/adme/<endpoint>.joblib     deployed model
  data/processed/cv_predictions/adme/<endpoint>_<split>_oof.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.featurize import featurize, N_FEATURES  # noqa: E402
from models.train_rf import (RF_COMMON, SEED, N_SPLITS, _scaffold_groups,  # noqa: E402
                             _clf_metrics, _reg_metrics)

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
ADME = ROOT / "data" / "adme"

CLASSIFICATION = ["pgp_inhibition", "pgp_substrate"]
REGRESSION = ["solubility", "lipophilicity", "caco2_permeability",
              "plasma_protein_binding", "clearance_hepatocyte", "logbb", "kpuu"]


def _cv(task, X, y, groups, smiles):
    rows, oof = [], {}
    schemes = {
        "random": (StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED) if task == "classification"
                   else KFold(N_SPLITS, shuffle=True, random_state=SEED)),
        "scaffold": GroupKFold(N_SPLITS),
    }
    for split, splitter in schemes.items():
        it = splitter.split(X, y, groups) if split == "scaffold" else splitter.split(X, y)
        preds = np.full(len(y), np.nan); fold_id = np.full(len(y), -1, dtype=int)
        for fold, (tr, te) in enumerate(it, 1):
            if task == "classification":
                m = RandomForestClassifier(class_weight="balanced", **RF_COMMON).fit(X[tr], y[tr])
                p = m.predict_proba(X[te])[:, 1]; met = _clf_metrics(y[te], p)
            else:
                m = RandomForestRegressor(**RF_COMMON).fit(X[tr], y[tr])
                p = m.predict(X[te]); met = _reg_metrics(y[te], p)
            preds[te] = p; fold_id[te] = fold
            rows.append({"endpoint": None, "task": task, "split": split, "fold": fold,
                         "n_train": len(tr), "n_test": len(te), **met})
        oof[split] = pd.DataFrame({"smiles": smiles, "y_true": y, "fold": fold_id, "prediction": preds})
    return rows, oof


def run(endpoints=None):
    (ROOT / "models_rf" / "adme").mkdir(parents=True, exist_ok=True)
    oof_dir = ROOT / "data" / "processed" / "cv_predictions" / "adme"
    oof_dir.mkdir(parents=True, exist_ok=True)
    fold_rows, summary_rows = [], []
    for ep in (endpoints or (CLASSIFICATION + REGRESSION)):
        task = "classification" if ep in CLASSIFICATION else "regression"
        target = "label" if task == "classification" else "y"
        df = pd.read_csv(ADME / f"{ep}.csv").dropna(subset=["smiles", target]).reset_index(drop=True)
        X, mask = featurize(df["smiles"].tolist())
        df = df.loc[mask].reset_index(drop=True)
        y = df[target].to_numpy()
        y = y.astype(int) if task == "classification" else y.astype(float)
        groups = _scaffold_groups(df["smiles"].tolist())
        headline = "roc_auc" if task == "classification" else "r2"
        print(f"\n[{ep}] {task}: {len(y)} compounds, {len(set(groups))} scaffolds")

        rows, oof = _cv(task, X, y, groups, df["smiles"].tolist())
        for r in rows:
            r["endpoint"] = ep
        fold_rows.extend(rows)
        for split, frame in oof.items():
            frame.to_csv(oof_dir / f"{ep}_{split}_oof.csv", index=False)

        fr = pd.DataFrame(rows)
        metric_cols = [c for c in fr.columns if c not in
                       {"endpoint", "task", "split", "fold", "n_train", "n_test"}]
        for split, g in fr.groupby("split"):
            srow = {"endpoint": ep, "task": task, "split": split, "n": int(X.shape[0])}
            for c in metric_cols:
                srow[f"{c}_mean"] = round(float(g[c].mean()), 4)
                srow[f"{c}_sd"] = round(float(g[c].std(ddof=1)), 4)
            summary_rows.append(srow)
            print(f"[{ep}] {split:8s} {headline} = {g[headline].mean():.3f} +/- {g[headline].std(ddof=1):.3f}")

        model = (RandomForestClassifier(class_weight="balanced", **RF_COMMON) if task == "classification"
                 else RandomForestRegressor(**RF_COMMON)).fit(X, y)
        joblib.dump(model, ROOT / "models_rf" / "adme" / f"{ep}.joblib")
        (ROOT / "models_rf" / "adme" / f"{ep}_meta.json").write_text(json.dumps(
            {"endpoint": ep, "task": task, "n_compounds": int(X.shape[0]), "n_features": int(N_FEATURES),
             "hyperparameters": RF_COMMON, "cv_folds": N_SPLITS}, indent=2), encoding="utf-8")

    # merge into existing tables so running a subset does not drop the other endpoints
    fold_new = pd.DataFrame(fold_rows); summ_new = pd.DataFrame(summary_rows)
    ft = ROOT / "results" / "tables" / "adme_cv_folds.csv"
    st = ROOT / "results" / "tables" / "adme_cv_summary.csv"
    if endpoints and ft.exists():
        done = set(summ_new.endpoint)
        fold_new = pd.concat([pd.read_csv(ft).query("endpoint not in @done"), fold_new], ignore_index=True)
        summ_new = pd.concat([pd.read_csv(st).query("endpoint not in @done"), summ_new], ignore_index=True)
    fold_new.to_csv(ft, index=False)
    summ_new.to_csv(st, index=False)
    print("\nwrote results/tables/adme_cv_summary.csv and models_rf/adme/*.joblib")


if __name__ == "__main__":
    run(sys.argv[1:] or None)
