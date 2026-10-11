"""Justify the retained features: which feature blocks matter, and which descriptors drive each model.

The review asked for a statistical, not hand-waving, justification of the features the model keeps.
This provides two complementary analyses per endpoint:

  1. Block ablation. Five-fold cross-validation with the fingerprint block only, the descriptor block
     only, and both combined. If both blocks together beat either alone, both are retained on
     measured evidence rather than assertion.

  2. Descriptor permutation importance. With the model fit on a held-out split, each of the twelve
     interpretable descriptors is shuffled and the drop in performance recorded (repeated ten times).
     A descriptor is judged informative if its mean importance exceeds zero by more than two standard
     deviations - a simple, defensible retention criterion - and the ranking shows which physical
     properties each endpoint actually uses (for example polar surface area and logP for BBB).

Outputs:
  results/tables/feature_block_ablation.csv       headline metric per endpoint x block
  results/tables/feature_descriptor_importance.csv  per-endpoint importance of each descriptor

Run:  python src/sai_cns/evaluation/feature_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.featurize import featurize, MORGAN_BITS  # noqa: E402
from models.train_rf import (CLASSIFICATION, REGRESSION, RF_COMMON, SEED, _load,  # noqa: E402
                             _scaffold_groups, _dedup_features)

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]

DESCRIPTOR_NAMES = ["mw", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds",
                    "aromatic_rings", "fraction_csp3", "ring_count", "heavy_atoms",
                    "formal_charge", "qed"]
BLOCKS = {
    "fingerprint_only": np.arange(0, MORGAN_BITS),
    "descriptors_only": np.arange(MORGAN_BITS, MORGAN_BITS + len(DESCRIPTOR_NAMES)),
    "combined": np.arange(0, MORGAN_BITS + len(DESCRIPTOR_NAMES)),
}


def _rf(task):
    return (RandomForestClassifier(class_weight="balanced", **RF_COMMON) if task == "classification"
            else RandomForestRegressor(**RF_COMMON))


def _score(task, model, X, y):
    if task == "classification":
        return roc_auc_score(y, model.predict_proba(X)[:, 1])
    return r2_score(y, model.predict(X))


def _cv_block(task, X, y, cols):
    splitter = (StratifiedKFold(5, shuffle=True, random_state=SEED) if task == "classification"
                else KFold(5, shuffle=True, random_state=SEED))
    scores = []
    for tr, te in splitter.split(X, y):
        model = _rf(task)
        model.fit(X[tr][:, cols], y[tr])
        scores.append(_score(task, model, X[te][:, cols], y[te]))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def analyse(endpoints):
    ablation, importance = [], []
    for endpoint in endpoints:
        task = "classification" if endpoint in CLASSIFICATION else "regression"
        target = "label" if task == "classification" else REGRESSION[endpoint]
        df = _load(endpoint).dropna(subset=["smiles", target]).reset_index(drop=True)
        X, mask = featurize(df["smiles"].tolist())
        df = df.loc[mask].reset_index(drop=True)
        y = df[target].to_numpy()
        y = y.astype(int) if task == "classification" else y.astype(float)
        headline = "roc_auc" if task == "classification" else "r2"

        # Deduplicate exactly as the training pipeline does, and for the same reason. This file
        # used to featurise and split directly, so rows identical in feature space sat on both
        # sides of every fold: the leak deduplication exists to remove. It mattered for one
        # endpoint rather than all of them, because that is where the duplicates are. Measured
        # against the deployed panel before this was fixed, seven of the eight classifiers agreed
        # to within 0.011 AUROC while BBB read 0.0596 high, BBB being the endpoint the adversarial
        # suite reports as carrying 3,773 duplicate rows before deduplication.
        n_before = len(y)
        _smi = df["smiles"].tolist()
        X, y, _, _, _rep = _dedup_features(X, y, _scaffold_groups(_smi), _smi, task)
        if len(y) != n_before:
            print(f"[{endpoint}] deduplicated {n_before} -> {len(y)} rows", flush=True)
        print(f"[{endpoint}] {task}, {len(y)} compounds")

        for block, cols in BLOCKS.items():
            mean, sd = _cv_block(task, X, y, cols)
            ablation.append({"endpoint": endpoint, "task": task, "block": block,
                             "metric": headline, "mean": round(mean, 4), "sd": round(sd, 4)})
            print(f"    {block:16s} {headline} = {mean:.3f}")

        # permutation importance of the twelve descriptors on a held-out split
        strat = y if task == "classification" else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=strat)
        model = _rf(task).fit(Xtr, ytr)
        scorer = "roc_auc" if task == "classification" else "r2"
        desc_cols = BLOCKS["descriptors_only"]
        pi = permutation_importance(model, Xte, yte, n_repeats=10, random_state=SEED,
                                    scoring=scorer)
        for j, name in zip(desc_cols, DESCRIPTOR_NAMES):
            importance.append({"endpoint": endpoint, "descriptor": name,
                               "importance_mean": round(float(pi.importances_mean[j]), 5),
                               "importance_sd": round(float(pi.importances_std[j]), 5),
                               "informative": bool(pi.importances_mean[j] >
                                                   2 * pi.importances_std[j] and
                                                   pi.importances_mean[j] > 0)})

    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ablation).to_csv(ROOT / "results" / "tables" / "feature_block_ablation.csv", index=False)
    pd.DataFrame(importance).to_csv(ROOT / "results" / "tables" / "feature_descriptor_importance.csv",
                                    index=False)
    print("\nwrote results/tables/feature_block_ablation.csv, feature_descriptor_importance.csv")


if __name__ == "__main__":
    analyse(sys.argv[1:] or (CLASSIFICATION + list(REGRESSION)))
