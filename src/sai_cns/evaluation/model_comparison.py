"""Does a different estimator beat the random forest? Five families, same features, same folds.

Random and scaffold-grouped 5-fold (the scaffold column is the one that matters). This answers, with
numbers rather than assertion, whether a different estimator would materially improve the models.

Five families, because a tree ensemble beating another tree ensemble is a weak result on its own.
Two are baselines that a reader is entitled to demand: a nearest-neighbour read-across on Tanimoto
similarity, which is what a medicinal chemist does by eye and which any model must beat to justify
itself, and L2-regularised logistic regression, the simplest thing that could work on this
representation. Three are ensembles: random forest, XGBoost and histogram gradient boosting.

An earlier version of this script ran only the three ensembles while the manuscript reported all
five, so two of the quoted numbers had no artefact behind them. It also split the raw endpoint table
where train_rf.py splits the deduplicated one, which compares estimators on a task slightly easier
than the one they are deployed on. Both are corrected here.

Output: results/tables/model_comparison.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.featurize import MORGAN_BITS, featurize  # noqa: E402
from models.train_rf import (CLASSIFICATION, REGRESSION, SEED, _dedup_features,  # noqa: E402
                             _load, _scaffold_groups)

ROOT = Path(__file__).resolve().parents[3]
N_SPLITS = 5


def classifiers(pos_weight):
    return {
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                               class_weight="balanced", n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
                                 scale_pos_weight=pos_weight, n_jobs=-1, random_state=SEED),
        # class_weight="balanced" added so the comparison is like-for-like. This was the only
        # ensemble competing unweighted: the forest carries class_weight="balanced", XGBoost carries
        # scale_pos_weight, and logistic regression is both scaled and balanced. On endpoints whose
        # actives outnumber inactives four to one that is not a neutral difference, and a family
        # comparison whose conclusion is "the forest is not distinguishable from the boosters" cannot
        # rest on one booster having been handicapped.
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                                               class_weight="balanced",
                                                               random_state=SEED),
        # Jaccard distance on the binary fingerprint block is 1 - Tanimoto, so this is a
        # read-across over the five nearest measured analogues and nothing more.
        "kNN read-across": KNeighborsClassifier(n_neighbors=5, metric="jaccard", n_jobs=-1),
        "LogisticRegression": make_pipeline(StandardScaler(),
                                            LogisticRegression(max_iter=2000, C=1.0,
                                                               class_weight="balanced",
                                                               random_state=SEED)),
    }


# The read-across must see only the fingerprint, because Jaccard is undefined on continuous
# descriptors. Every other estimator sees all 1,036 columns.
FINGERPRINT_ONLY = {"kNN read-across"}


def regressors():
    return {
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                              random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                                              random_state=SEED),
        "kNN read-across": KNeighborsRegressor(n_neighbors=5, metric="jaccard", n_jobs=-1),
        "LogisticRegression": make_pipeline(StandardScaler(), Ridge(alpha=1.0,
                                                                    random_state=SEED)),
    }


def cv_score(task, model, X, y, groups, split):
    splitter = (GroupKFold(N_SPLITS) if split == "scaffold" else
                (StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED) if task == "classification"
                 else KFold(N_SPLITS, shuffle=True, random_state=SEED)))
    it = splitter.split(X, y, groups) if split == "scaffold" else splitter.split(X, y)
    scores = []
    for tr, te in it:
        # clone, not __class__(**get_params()), which cannot rebuild a Pipeline
        m = clone(model)
        m.fit(X[tr], y[tr])
        if task == "classification":
            p = m.predict_proba(X[te])[:, 1]
            scores.append(roc_auc_score(y[te], p) if len(set(y[te])) > 1 else np.nan)
        else:
            scores.append(r2_score(y[te], m.predict(X[te])))
    return float(np.nanmean(scores)), float(np.nanstd(scores, ddof=1))


def main():
    rows = []
    for ep in CLASSIFICATION + list(REGRESSION):
        task = "classification" if ep in CLASSIFICATION else "regression"
        target = "label" if task == "classification" else REGRESSION[ep]
        df = _load(ep).dropna(subset=["smiles", target]).reset_index(drop=True)
        X, mask = featurize(df["smiles"].tolist())
        df = df.loc[mask].reset_index(drop=True)
        y = df[target].to_numpy()
        y = y.astype(int) if task == "classification" else y.astype(float)
        groups = _scaffold_groups(df["smiles"].tolist())
        # Compare estimators on the matrix the deployed pipeline fits, not on the raw table.
        X, y, groups, _s, rep = _dedup_features(X, y, groups, df["smiles"].tolist(), task)
        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1) if task == "classification" else None
        models = classifiers(pos_weight) if task == "classification" else regressors()
        print(f"\n[{ep}] {task}, {len(y)} compounds after deduplication "
              f"({rep.get('duplicate_rows_removed', 0)} duplicates removed)")
        for name, model in models.items():
            Xm = X[:, :MORGAN_BITS] if name in FINGERPRINT_ONLY else X
            for split in ("random", "scaffold"):
                mean, sd = cv_score(task, model, Xm, y, groups, split)
                rows.append({"endpoint": ep, "task": task, "model": name, "split": split,
                             "metric": "roc_auc" if task == "classification" else "r2",
                             "mean": round(mean, 4), "sd": round(sd, 4)})
            sc = [r for r in rows if r["endpoint"] == ep and r["model"] == name and r["split"] == "scaffold"][0]
            print(f"    {name:22s} scaffold {sc['metric']} = {sc['mean']:.3f}")
    out = pd.DataFrame(rows)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "results" / "tables" / "model_comparison.csv", index=False)

    # headline: mean scaffold metric per model
    scaf = out[out.split == "scaffold"]
    print("\n=== mean scaffold metric by model ===")
    for task in ("classification", "regression"):
        t = scaf[scaf.task == task]
        print(f"  {task}:")
        for name in sorted(t.model.unique(), key=lambda n: -t[t.model == n]["mean"].mean()):
            print(f"    {name:22s} {t[t.model == name]['mean'].mean():.4f}")
    print("wrote results/tables/model_comparison.csv")


if __name__ == "__main__":
    main()
