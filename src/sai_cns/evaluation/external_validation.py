"""External validation of the trained models on data held entirely out of training.

Two independent checks:

  1. Blood-brain barrier on approved drugs. The deployed BBB model (trained on B3DB) is applied to the
     306 FDA-curated compounds that are *not* in B3DB. Because the model never saw them, this is a
     genuine external test on approved drugs, the population a reviewer cares about most.

  2. DrugBank applicability. The eight classifiers are applied to the 11,723 DrugBank small molecules
     to show the fraction the models flag as active per endpoint - a sanity check on behaviour over a
     broad, unlabelled approved/experimental drug library (no labels exist, so this is descriptive,
     not a performance claim).

Outputs:
  results/tables/external_bbb_validation.csv     metrics on the 306 novel approved-drug BBB compounds
  results/tables/drugbank_screen_summary.csv     predicted-active fraction per endpoint on DrugBank

Run:  python src/sai_cns/evaluation/external_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.featurize import featurize  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models_rf"
CLASSIFICATION = ["BBB", "AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG"]


def _metrics(label: str, model, frame: pd.DataFrame) -> dict:
    X, mask = featurize(frame["canonical_smiles"].tolist())
    y = frame.loc[mask, "bbb_status"].to_numpy().astype(int)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "set": label,
        "n": int(len(y)),
        "n_permeable": int(y.sum()),
        "auroc": round(roc_auc_score(y, proba), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, pred), 4),
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else np.nan,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else np.nan,
    }


def bbb_external() -> pd.DataFrame:
    """Report the external result under both novelty criteria, and say which is which.

    Excluding overlap by InChIKey is not enough to call a compound unseen. The InChIKey separates
    stereoisomers, salts and protonation states; the model's features do not, so a compound can pass
    that check and still be one the model has memorised. Both numbers are reported because only the
    stricter one supports the claim, and because the difference between them is itself worth seeing.
    """
    model = joblib.load(MODELS / "BBB.joblib")
    ext = pd.read_csv(ROOT / "data" / "external" / "processed" / "external_bbb_test.csv")

    rows = [_metrics("FDA-curated, not in B3DB by InChIKey",
                     model, ext[~ext["in_b3db_training"]].reset_index(drop=True))]
    if "novel_to_model" in ext.columns:
        rows.append(_metrics("FDA-curated, also distinguishable from training in feature space",
                             model, ext[ext["novel_to_model"]].reset_index(drop=True)))
        memorised = ext[(~ext["in_b3db_training"]) & ext["feature_identical_to_training"]]
        if len(memorised):
            rows.append(_metrics("of which: feature-identical to a training compound",
                                 model, memorised.reset_index(drop=True)))
    else:
        print("  note: external_bbb_test.csv predates the feature-space check; "
              "rerun integrate_external.py to add it.")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tables" / "external_bbb_validation.csv", index=False)
    print("BBB external (approved drugs):")
    for r in rows:
        print(f"  n={r['n']:4d}  AUROC={r['auroc']}  acc={r['accuracy']}  "
              f"sens={r['sensitivity']}  spec={r['specificity']}   {r['set']}")
    if len(rows) > 1:
        print("\n  The second row is the one that supports an external claim. The first includes "
              "compounds the model cannot distinguish from its training set.")
    return out


def drugbank_screen() -> pd.DataFrame:
    drugs = pd.read_csv(ROOT / "data" / "external" / "processed" / "external_drugs.csv")
    X, mask = featurize(drugs["canonical_smiles"].tolist())
    n = int(mask.sum())
    rows = []
    for ep in CLASSIFICATION:
        path = MODELS / f"{ep}.joblib"
        if not path.exists():
            continue
        proba = joblib.load(path).predict_proba(X)[:, 1]
        rows.append({"endpoint": ep, "n_scored": n,
                     "predicted_active": int((proba >= 0.5).sum()),
                     "predicted_active_frac": round(float((proba >= 0.5).mean()), 4),
                     "median_probability": round(float(np.median(proba)), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tables" / "drugbank_screen_summary.csv", index=False)
    print(f"\nDrugBank screen ({n} molecules):")
    print(out.to_string(index=False))
    return out


def main() -> None:
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    bbb_external()
    drugbank_screen()
    print("\nwrote results/tables/external_bbb_validation.csv, drugbank_screen_summary.csv")


if __name__ == "__main__":
    main()
