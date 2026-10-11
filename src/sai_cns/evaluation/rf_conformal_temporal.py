"""Conformal prediction and temporal validation for the random-forest models.

Two analyses the manuscript body needs, computed directly on the deployed random forests:

  Conformal prediction (classification). Inductive Mondrian (class-conditional) conformal at a 10%
  significance level. Data are split train / calibration / test; the nonconformity of a compound is
  1 - predicted probability of its true class; a per-class threshold is taken from the calibration
  quantile; the test prediction set contains every class whose nonconformity is below its threshold.
  Empirical coverage should be near the target 0.90, with average set size reported.

  Temporal validation. For endpoints that carry a document year, the model is trained on compounds up
  to the 75th-percentile year and tested on strictly later compounds, measuring generalisation to
  chemistry that appeared after training (AUROC for classifiers, R-squared for regressors).

Outputs: results/tables/rf_conformal.csv, results/tables/rf_temporal.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.featurize import featurize  # noqa: E402
from models.train_rf import (CLASSIFICATION, REGRESSION, RF_COMMON, SEED, _load,  # noqa: E402
                             _dedup_features, _scaffold_groups)

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
EPS = 0.10  # 90% target coverage


def conformal(endpoint, split="random"):
    """Mondrian conformal prediction at 90 per cent, on the rows the model can actually distinguish.

    Two corrections to an earlier version of this function, both found by auditing chapter 4 against
    the code.

    It did not deduplicate. train_rf.py collapses rows whose feature vectors are byte-identical
    before it cross-validates, and its docstring says why: the featuriser is stereo-blind, so
    stereoisomers and salt forms fold to the same vector and any random splitter puts copies of one
    compound on both sides of the split. This function split the raw table, so it reported BBB with
    n_test 1,561, which is a fifth of 7,807 raw rows rather than of the 3,901 the model is trained
    and scored on. Chapter 3 states 3,901 and chapter 4's table was built from 7,807, and nothing
    compared them.

    It also reported only the mean set size. On a two-class problem the mean is
    1 + P(ambiguous) - P(empty), so the excess over 1.0 cannot be read as the ambiguous fraction
    unless no set is empty, and sets are empty. Chapter 4 read it that way. Both counts are now
    recorded, so the quantity a reader wants is in the file rather than inferred from it with the
    wrong sign.

    `split` is "random" or "scaffold". The random split is the conventional estimate and the one the
    earlier figures used; the scaffold split withholds whole Bemis-Murcko classes and is the regime
    the server actually operates in, where the interesting number is how much of the panel becomes
    genuinely undecidable on unfamiliar chemistry.
    """
    df = _load(endpoint).dropna(subset=["smiles", "label"]).reset_index(drop=True)
    X, mask = featurize(df["smiles"].tolist())
    y = df.loc[mask, "label"].to_numpy().astype(int)
    smiles = [s for s, m in zip(df["smiles"].tolist(), mask) if m]
    groups = _scaffold_groups(smiles)
    X, y, groups, smiles, _rep = _dedup_features(X, y, groups, smiles, "classification")
    y = np.asarray(y).astype(int)

    n = len(y)
    rng = np.random.RandomState(SEED)
    if split == "random":
        idx = rng.permutation(n)
    else:
        # Order by scaffold group so that a group falls entirely on one side of every boundary.
        order = rng.permutation(np.unique(groups))
        rank = {g: i for i, g in enumerate(order)}
        idx = np.argsort([rank[g] for g in groups], kind="stable")
    a, b = int(0.6 * n), int(0.8 * n)
    tr, cal, te = idx[:a], idx[a:b], idx[b:]
    model = RandomForestClassifier(class_weight="balanced", **RF_COMMON).fit(X[tr], y[tr])
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    # Mondrian: per-class nonconformity threshold from calibration compounds of that class
    thr = {}
    for c in classes:
        ci = classes.index(c)
        cal_c = cal[y[cal] == c]
        if len(cal_c) == 0:
            thr[c] = 1.0; continue
        alpha = 1.0 - proba[cal_c, ci]
        k = int(np.ceil((len(cal_c) + 1) * (1 - EPS)))
        thr[c] = np.sort(alpha)[min(k, len(cal_c)) - 1]
    covered, sizes = [], []
    for i in te:
        pset = [c for c in classes if (1.0 - proba[i, classes.index(c)]) <= thr[c]]
        covered.append(y[i] in pset)
        sizes.append(len(pset))
    sizes = np.asarray(sizes)
    return {"endpoint": endpoint, "split": split, "n_total": int(n), "n_test": len(te),
            "target_coverage": 1 - EPS,
            "empirical_coverage": round(float(np.mean(covered)), 3),
            "avg_set_size": round(float(sizes.mean()), 3),
            # Recorded separately because the mean cannot separate them: it is
            # 1 + P(ambiguous) - P(empty), so an empty set and an ambiguous one move it opposite ways.
            "frac_ambiguous": round(float((sizes == 2).mean()), 4),
            "frac_empty": round(float((sizes == 0).mean()), 4),
            "frac_singleton": round(float((sizes == 1).mean()), 4)}


def temporal(endpoint, task):
    target = "label" if task == "classification" else ("y" if endpoint == "antioxidant_DPPH" else "pchembl")
    df = _load(endpoint)
    if "year" not in df.columns:
        return None
    df = df.dropna(subset=["smiles", target, "year"]).reset_index(drop=True)
    if len(df) < 200:
        return None
    cutoff = np.percentile(df["year"], 75)
    if (df["year"] > cutoff).sum() < 30:
        return None
    X, mask = featurize(df["smiles"].tolist())
    df = df.loc[mask].reset_index(drop=True)
    y = df[target].to_numpy()

    # Deduplicate on the feature vector, as the training code does and as this function did not.
    # The row positions are passed through the `smiles` slot, which _dedup_features carries and
    # filters without interpreting, so the kept positions come back and the year column can be
    # subset to match. Without this the temporal split scores the model on rows it memorised,
    # because a stereoisomer dated after the cutoff is byte-identical to one dated before it.
    groups = _scaffold_groups(df["smiles"].tolist())
    X, y, _g, kept, _rep = _dedup_features(
        X, y, groups, list(range(len(y))), task)
    df = df.iloc[list(kept)].reset_index(drop=True)
    y = np.asarray(y)

    trm = (df["year"] <= cutoff).to_numpy(); tem = (df["year"] > cutoff).to_numpy()
    if task == "classification":
        y = y.astype(int)
        if len(set(y[tem])) < 2:
            return None
        m = RandomForestClassifier(class_weight="balanced", **RF_COMMON).fit(X[trm], y[trm])
        score = roc_auc_score(y[tem], m.predict_proba(X[tem])[:, 1]); metric = "auroc"
    else:
        y = y.astype(float)
        m = RandomForestRegressor(**RF_COMMON).fit(X[trm], y[trm])
        score = r2_score(y[tem], m.predict(X[tem])); metric = "r2"
    return {"endpoint": endpoint, "task": task, "cutoff_year": int(cutoff),
            "n_train": int(trm.sum()), "n_test": int(tem.sum()),
            "metric": metric, "score": round(float(score), 3)}


def main():
    # Two files, not one table with a `split` column, and the reason is a near miss worth recording.
    # Six consumers key this table by endpoint alone: thesis_T3_uncertainty_stack.py builds
    # {endpoint: row}, so a second row per endpoint would have silently overwritten the first and
    # drawn scaffold numbers labelled as the deployed ones, while two decks and
    # check_manuscript_numbers.py take min/max across every row and would have mixed the regimes.
    # Keeping rf_conformal.csv at one row per endpoint preserves that contract; the scaffold arm is
    # opt-in, so nothing reads it by accident.
    for split, out in (("random", "rf_conformal.csv"), ("scaffold", "rf_conformal_scaffold.csv")):
        conf = []
        for ep in CLASSIFICATION:
            conf.append(conformal(ep, split))
            print(f"  [{split:8}] {ep}", flush=True)
        d = pd.DataFrame(conf)
        d.to_csv(ROOT / "results" / "tables" / out, index=False)
        print(f"=== conformal, {split} split (target coverage 0.90) -> {out} ===")
        print(d.to_string(index=False))

    temp = []
    for ep in CLASSIFICATION:
        r = temporal(ep, "classification")
        if r:
            temp.append(r)
    for ep in list(REGRESSION):
        r = temporal(ep, "regression")
        if r:
            temp.append(r)
    pd.DataFrame(temp).to_csv(ROOT / "results" / "tables" / "rf_temporal.csv", index=False)
    print("\n=== temporal (train past, test future) ===")
    print(pd.DataFrame(temp).to_string(index=False))
    print("\nwrote results/tables/rf_conformal.csv, rf_temporal.csv")


if __name__ == "__main__":
    main()
