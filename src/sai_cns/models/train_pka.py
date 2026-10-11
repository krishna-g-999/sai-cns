"""Train a most-basic-pKa regressor for the CNS MPO desirability score.

The sixth CNS MPO component is the most basic pKa, which this project has until now refused to
supply rather than guess. ChEMBL records pKa without stating whether the constant is acidic or
basic, so the training set is restricted to compounds that contain an RDKit-detected basic centre
and whose assay description does not indicate acid dissociation (see fetch_pka.py). Repeat
measurements are reduced to a per-compound median.

Validation follows the protocol used elsewhere in this project: ten-fold cross-validation under both
a random split and a scaffold-grouped split that holds out entire Bemis-Murcko frameworks, with the
scaffold figure treated as the honest one. A temporal split is also reported where publication years
are available, since prospective behaviour is what matters when the model is applied to new
chemistry.

The model is deployed only if the scaffold-split performance justifies it; the decision, and the
numbers behind it, are written to results/tables/pka_model.csv so the choice is auditable.

Outputs: models_rf/pka_basic.joblib, models_rf/pka_basic_meta.json, results/tables/pka_model.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize  # noqa: E402
from models.train_rf import _scaffold_groups  # noqa: E402

DATA = ROOT / "data" / "endpoints_reg" / "pka_basic.csv"
M = ROOT / "models_rf"
TAB = ROOT / "results" / "tables"
RF = dict(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=42)
MIN_SCAFFOLD_R2 = 0.40   # deploy only above this; below it the model is not worth the false comfort


def metrics(y, p):
    return {"r2": float(r2_score(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "spearman": float(spearmanr(y, p).statistic)}


def main():
    df = pd.read_csv(DATA).dropna(subset=["smiles", "pka"])
    X, mask = featurize(df["smiles"].astype(str).tolist())
    df = df.loc[mask].reset_index(drop=True)
    y = df["pka"].to_numpy(dtype=float)
    groups = _scaffold_groups(df["smiles"].astype(str).tolist())
    print(f"{len(y):,} compounds, {len(set(groups)):,} scaffolds, "
          f"pKa median {np.median(y):.2f} sd {y.std():.2f}", flush=True)

    rows = []
    for name, cv, g in [("random 10-fold", KFold(10, shuffle=True, random_state=42), None),
                        ("scaffold 10-fold", GroupKFold(10), groups)]:
        pred = cross_val_predict(RandomForestRegressor(**RF), X, y, cv=cv, groups=g, n_jobs=-1)
        m = metrics(y, pred)
        m["split"] = name
        rows.append(m)
        print(f"  {name:18} R2 {m['r2']:.3f}  RMSE {m['rmse']:.2f}  MAE {m['mae']:.2f}  "
              f"rho {m['spearman']:.3f}", flush=True)

    # temporal split where years exist
    if "year" in df.columns and df["year"].notna().sum() > 500:
        yr = pd.to_numeric(df["year"], errors="coerce")
        ok = yr.notna().to_numpy()
        cut = np.nanpercentile(yr[ok], 75)
        tr, te = (ok & (yr <= cut)).to_numpy(), (ok & (yr > cut)).to_numpy()
        if te.sum() >= 100:
            mdl = RandomForestRegressor(**RF).fit(X[tr], y[tr])
            m = metrics(y[te], mdl.predict(X[te]))
            m["split"] = f"temporal (test after {int(cut)})"
            rows.append(m)
            print(f"  {m['split']:18} R2 {m['r2']:.3f}  RMSE {m['rmse']:.2f}  "
                  f"rho {m['spearman']:.3f}  n={int(te.sum())}", flush=True)

    res = pd.DataFrame(rows)[["split", "r2", "rmse", "mae", "spearman"]]
    TAB.mkdir(parents=True, exist_ok=True)
    res.to_csv(TAB / "pka_model.csv", index=False)

    scaffold_r2 = float(res.loc[res.split == "scaffold 10-fold", "r2"].iloc[0])
    deploy = scaffold_r2 >= MIN_SCAFFOLD_R2
    if deploy:
        model = RandomForestRegressor(**RF).fit(X, y)
        joblib.dump(model, M / "pka_basic.joblib", compress=3)
    meta = {"endpoint": "pka_basic", "n_compounds": int(len(y)),
            "n_scaffolds": int(len(set(groups))),
            "scaffold_r2": round(scaffold_r2, 3),
            "deploy_threshold_r2": MIN_SCAFFOLD_R2, "deployed": bool(deploy),
            "note": "most-basic pKa; training restricted to compounds with an RDKit basic centre "
                    "and no acid indication in the assay description"}
    (M / "pka_basic_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nscaffold R2 {scaffold_r2:.3f} against a deployment threshold of {MIN_SCAFFOLD_R2}")
    print("DEPLOYED" if deploy else "NOT deployed; CNS MPO keeps its five-component form")
    print("wrote", TAB / "pka_model.csv")


if __name__ == "__main__":
    main()
