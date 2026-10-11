"""Applicability domain: how far a prediction sits from the training data, and whether that matters.

A model is only trustworthy near the chemistry it was trained on. For each compound we measure the
maximum Tanimoto similarity (ECFP-4) to the training set; a compound is "in domain" if that maximum is
at least 0.30 (a standard, conservative cut). Two things are reported:

  1. The external BBB validation (306 approved drugs) is split into in- and out-of-domain, and metrics
     are computed separately - the model should be clearly better where it is in domain.
  2. The fraction of the DrugBank library that falls inside each classifier's domain, so downstream
     predictions on it can be qualified.

Outputs:
  results/tables/applicability_bbb_validation.csv   in- vs out-of-domain metrics on the 306 drugs
  results/tables/applicability_coverage.csv         in-domain fraction per endpoint on DrugBank
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.featurize import featurize  # noqa: E402
from models.train_rf import CLASSIFICATION, _load  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models_rf"
AD_THRESHOLD = 0.30
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fps(smiles):
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        fps.append(_GEN.GetFingerprint(m) if m is not None else None)
    return fps


def max_tanimoto(query_smiles, ref_smiles):
    """Maximum Tanimoto of each query to any reference compound."""
    ref = [f for f in _fps(ref_smiles) if f is not None]
    out = np.full(len(query_smiles), np.nan)
    for i, q in enumerate(_fps(query_smiles)):
        if q is None:
            continue
        out[i] = max(DataStructs.BulkTanimotoSimilarity(q, ref)) if ref else 0.0
    return out


def bbb_in_out_domain():
    train = _load("BBB").dropna(subset=["smiles", "label"])
    ext = pd.read_csv(ROOT / "data" / "external" / "processed" / "external_bbb_test.csv")
    novel = ext[~ext["in_b3db_training"]].reset_index(drop=True)
    sim = max_tanimoto(novel["canonical_smiles"].tolist(), train["smiles"].tolist())
    model = joblib.load(MODELS / "BBB.joblib")
    X, mask = featurize(novel["canonical_smiles"].tolist())
    proba = model.predict_proba(X)[:, 1]
    y = novel.loc[mask, "bbb_status"].to_numpy().astype(int)
    sim = sim[mask]
    rows = []
    for name, m in [("in_domain (T>=0.30)", sim >= AD_THRESHOLD),
                    ("out_of_domain (T<0.30)", sim < AD_THRESHOLD),
                    ("all", np.ones_like(sim, dtype=bool))]:
        if m.sum() >= 10 and len(set(y[m])) > 1:
            rows.append({"subset": name, "n": int(m.sum()),
                         "mean_max_tanimoto": round(float(np.nanmean(sim[m])), 3),
                         "auroc": round(roc_auc_score(y[m], proba[m]), 4),
                         "accuracy": round(accuracy_score(y[m], (proba[m] >= 0.5).astype(int)), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tables" / "applicability_bbb_validation.csv", index=False)
    print("=== BBB external validation by applicability domain ===")
    print(out.to_string(index=False))
    return out


def drugbank_coverage():
    drugs = pd.read_csv(ROOT / "data" / "external" / "processed" / "external_drugs.csv")
    q = drugs["canonical_smiles"].tolist()
    rows = []
    for ep in CLASSIFICATION:
        train = _load(ep).dropna(subset=["smiles"])
        sim = max_tanimoto(q, train["smiles"].tolist())
        rows.append({"endpoint": ep, "n_drugbank": int(np.isfinite(sim).sum()),
                     "in_domain_frac": round(float(np.nanmean(sim >= AD_THRESHOLD)), 4),
                     "median_max_tanimoto": round(float(np.nanmedian(sim)), 3)})
        print(f"[{ep}] DrugBank in-domain fraction {rows[-1]['in_domain_frac']}")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tables" / "applicability_coverage.csv", index=False)
    return out


def main():
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    bbb_in_out_domain()
    drugbank_coverage()
    print("\nwrote results/tables/applicability_bbb_validation.csv, applicability_coverage.csv")


if __name__ == "__main__":
    main()
