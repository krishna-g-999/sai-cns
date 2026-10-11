"""Random-forest models for the twelve endpoints, with ten-fold cross-validation.

This is the primary predictive model requested in review: a random forest per endpoint, evaluated
by ten-fold cross-validation under two splitting schemes so the result is honest about novelty.

  random 10-fold   - StratifiedKFold / KFold; the conventional estimate.
  scaffold 10-fold - GroupKFold on the Bemis-Murcko scaffold, so no scaffold appears in both train
                     and test; this measures generalisation to genuinely new chemistry.

Eight endpoints are binary classification (active/inactive); four receptors and the antioxidant
assay are regression, because their measured sets are almost entirely active and a binary split is
ill-posed (see docs/decisions_log.md and docs/ENDPOINT_JUSTIFICATION.md).

Outputs:
  results/tables/rf_cv_folds.csv     one row per endpoint x split x fold, with all metrics
  results/tables/rf_cv_summary.csv   mean +/- sd per endpoint x split (headline metric first)
  models_rf/<endpoint>.joblib        final model refit on all data, for deployment
  models_rf/<endpoint>_meta.json     n, positives, features, hyper-parameters, CV summary

Run:  python src/sai_cns/models/train_rf.py            (all endpoints)
      python src/sai_cns/models/train_rf.py AChE hERG  (named endpoints)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.featurize import featurize, feature_names, parent_mol, N_FEATURES  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
SEED = 42
N_SPLITS = 10
RF_COMMON = dict(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)

CLASSIFICATION = ["BBB", "AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG"]
REGRESSION = {"D2": "pchembl", "A2A": "pchembl", "HT2A": "pchembl", "SERT": "pchembl",
              "antioxidant_DPPH": "y"}


def _load(endpoint: str) -> pd.DataFrame:
    if endpoint == "antioxidant_DPPH":
        return pd.read_csv(ROOT / "data" / "endpoints_reg" / "antioxidant_dpph.csv")
    return pd.read_csv(ROOT / "data" / "endpoints" / f"{endpoint}.csv")


def _scaffold_groups(smiles: list[str]) -> np.ndarray:
    """Integer group id per compound from its Bemis-Murcko scaffold.

    The scaffold retains atom and bond types; it is not the generic carbon framework, which would
    require MakeScaffoldGeneric and would group more loosely. It is computed on the same desalted
    parent the featuriser uses, so a salt and its free base cannot land in different folds while
    being identical to the model.

    Acyclic compounds have no scaffold and are all placed in one shared group, so a fold boundary
    never runs through them. Giving each its own group, as this once did, quietly turned the
    scaffold split into a random split for that part of the set.
    """
    codes, mapping = [], {}
    for smi in smiles:
        try:
            mol = parent_mol(str(smi))
            scaf = "" if mol is None else MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False)
        except Exception:
            scaf = ""
        codes.append(mapping.setdefault(scaf or "_ACYCLIC_", len(mapping)))
    return np.asarray(codes)


def _dedup_features(X, y, groups, smiles, task):
    """Collapse rows that are the same input as far as the model is concerned.

    The featuriser folds a molecule to a stereo-blind 1024-bit fingerprint over its desalted parent,
    so stereoisomers, salt forms and protonation variants of one compound produce byte-identical
    feature vectors. Left in place, any random splitter puts copies of one compound on both sides of
    a fold and the model is scored on rows it has memorised. Deduplication has to happen here, on the
    feature vector, because that is the level at which the rows are indistinguishable: neither the
    SMILES string nor the InChIKey collapses them, and both are unique for every BBB row.

    Classification groups whose members disagree on the label are dropped rather than voted on. The
    same input carrying both labels cannot be learned from, and choosing one is an arbitrary decision
    dressed as data. Regression takes the median of the group.

    Returns (X, y, groups, smiles, report).
    """
    seen: dict[bytes, list[int]] = {}
    for i, vec in enumerate(X):
        seen.setdefault(vec.tobytes(), []).append(i)

    keep, conflicts, merged = [], 0, 0
    y_out = []
    for idxs in seen.values():
        if len(idxs) > 1:
            merged += len(idxs) - 1
        if task == "classification":
            labels = {int(y[i]) for i in idxs}
            if len(labels) > 1:
                conflicts += 1
                continue
            keep.append(idxs[0])
            y_out.append(y[idxs[0]])
        else:
            keep.append(idxs[0])
            y_out.append(float(np.median([y[i] for i in idxs])))

    order = np.argsort(keep)
    keep_sorted = np.asarray(keep)[order]
    y_new = np.asarray(y_out, dtype=y.dtype)[order]
    report = {"rows_in": int(len(y)), "duplicate_rows_removed": int(merged),
              "conflicting_groups_dropped": int(conflicts), "rows_out": int(len(keep_sorted))}
    return (X[keep_sorted], y_new, groups[keep_sorted],
            [smiles[i] for i in keep_sorted], report)


def _clf_metrics(y, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba) if len(set(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, proba) if len(set(y)) > 1 else np.nan,
        "mcc": matthews_corrcoef(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
    }


def _reg_metrics(y, pred) -> dict:
    return {
        "r2": r2_score(y, pred),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": mean_absolute_error(y, pred),
        "spearman": float(spearmanr(y, pred).statistic),
    }


def _cv(endpoint, task, X, y, groups, smiles):
    """Run random and scaffold 10-fold CV.

    Returns (metric_rows, oof) where metric_rows is the per-fold metrics and oof maps each split to a
    DataFrame of every compound's out-of-fold prediction and fold id, so all 10-fold data is saved.
    """
    rows, oof = [], {}
    schemes = {
        "random": (StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED) if task == "classification"
                   else KFold(N_SPLITS, shuffle=True, random_state=SEED)),
        "scaffold": GroupKFold(N_SPLITS),
    }
    for split_name, splitter in schemes.items():
        it = (splitter.split(X, y, groups) if split_name == "scaffold"
              else splitter.split(X, y))
        preds = np.full(len(y), np.nan)
        fold_id = np.full(len(y), -1, dtype=int)
        for fold, (tr, te) in enumerate(it, 1):
            if task == "classification":
                model = RandomForestClassifier(class_weight="balanced", **RF_COMMON)
                model.fit(X[tr], y[tr])
                p = model.predict_proba(X[te])[:, 1]
                m = _clf_metrics(y[te], p)
            else:
                model = RandomForestRegressor(**RF_COMMON)
                model.fit(X[tr], y[tr])
                p = model.predict(X[te])
                m = _reg_metrics(y[te], p)
            preds[te] = p
            fold_id[te] = fold
            rows.append({"endpoint": endpoint, "task": task, "split": split_name, "fold": fold,
                         "n_train": len(tr), "n_test": len(te), **m})
        oof[split_name] = pd.DataFrame({
            "smiles": smiles, "y_true": y, "fold": fold_id,
            "prediction": preds, "scaffold_group": groups})
    return rows, oof


def _final_model(task, X, y):
    model = (RandomForestClassifier(class_weight="balanced", **RF_COMMON) if task == "classification"
             else RandomForestRegressor(**RF_COMMON))
    model.fit(X, y)
    return model


def run(endpoints: list[str]) -> None:
    (ROOT / "models_rf").mkdir(exist_ok=True)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    fold_rows, summary_rows = [], []

    for endpoint in endpoints:
        task = "classification" if endpoint in CLASSIFICATION else "regression"
        target = "label" if task == "classification" else REGRESSION[endpoint]
        df = _load(endpoint).dropna(subset=["smiles", target]).reset_index(drop=True)
        print(f"\n[{endpoint}] {task}: {len(df)} rows -> featurizing")
        X, mask = featurize(df["smiles"].tolist())
        df = df.loc[mask].reset_index(drop=True)
        y = df[target].to_numpy()
        y = y.astype(int) if task == "classification" else y.astype(float)
        groups = _scaffold_groups(df["smiles"].tolist())
        smiles = df["smiles"].tolist()

        # Before any split, and before the deployed refit: collapse rows the model cannot tell
        # apart. Doing this after the split would be pointless, and doing it only for the CV would
        # leave the shipped model fitted on a set where some compounds are silently weighted twice.
        X, y, groups, smiles, dedup = _dedup_features(X, y, groups, smiles, task)
        if dedup["duplicate_rows_removed"] or dedup["conflicting_groups_dropped"]:
            print(f"[{endpoint}] deduplicated: {dedup['rows_in']} -> {dedup['rows_out']} rows "
                  f"({dedup['duplicate_rows_removed']} duplicates of another row, "
                  f"{dedup['conflicting_groups_dropped']} groups dropped for contradictory labels)")
        print(f"[{endpoint}] {X.shape[0]} distinct compounds, {X.shape[1]} features, "
              f"{len(set(groups))} scaffolds"
              + (f", {int(y.sum())} active" if task == "classification" else ""))

        rows, oof = _cv(endpoint, task, X, y, groups, smiles)
        fold_rows.extend(rows)
        oof_dir = ROOT / "data" / "processed" / "cv_predictions"
        oof_dir.mkdir(parents=True, exist_ok=True)
        for split_name, frame in oof.items():
            frame.to_csv(oof_dir / f"{endpoint}_{split_name}_oof.csv", index=False)

        fr = pd.DataFrame(rows)
        headline = "roc_auc" if task == "classification" else "r2"
        metric_cols = [c for c in fr.columns if c not in
                       {"endpoint", "task", "split", "fold", "n_train", "n_test"}]
        for split_name, g in fr.groupby("split"):
            srow = {"endpoint": endpoint, "task": task, "split": split_name, "n": int(X.shape[0])}
            for c in metric_cols:
                srow[f"{c}_mean"] = round(float(g[c].mean()), 4)
                srow[f"{c}_sd"] = round(float(g[c].std(ddof=1)), 4)
            summary_rows.append(srow)
            print(f"[{endpoint}] {split_name:8s} {headline} = "
                  f"{g[headline].mean():.3f} +/- {g[headline].std(ddof=1):.3f}")

        # final deployment model refit on all data
        model = _final_model(task, X, y)
        joblib.dump(model, ROOT / "models_rf" / f"{endpoint}.joblib")
        meta = {
            "endpoint": endpoint, "task": task, "n_compounds": int(X.shape[0]),
            "n_features": int(N_FEATURES), "feature_layout": "ecfp4_1024 + 12 descriptors",
            "n_scaffolds": int(len(set(groups))),
            "deduplication": dedup,
            "positives": int(y.sum()) if task == "classification" else None,
            "hyperparameters": {**RF_COMMON,
                                "class_weight": "balanced" if task == "classification" else None},
            "cv_folds": N_SPLITS,
            "cv_summary": {s["split"]: {k: v for k, v in s.items()
                                        if k.endswith(("_mean", "_sd"))}
                           for s in summary_rows if s["endpoint"] == endpoint},
        }
        (ROOT / "models_rf" / f"{endpoint}_meta.json").write_text(json.dumps(meta, indent=2),
                                                                  encoding="utf-8")

    pd.DataFrame(fold_rows).to_csv(ROOT / "results" / "tables" / "rf_cv_folds.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(ROOT / "results" / "tables" / "rf_cv_summary.csv", index=False)
    (ROOT / "models_rf" / "feature_names.json").write_text(json.dumps(feature_names()), encoding="utf-8")
    print("\nWrote results/tables/rf_cv_folds.csv, rf_cv_summary.csv and models_rf/*.joblib")


if __name__ == "__main__":
    requested = sys.argv[1:] or (CLASSIFICATION + list(REGRESSION))
    run(requested)
