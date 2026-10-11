"""Audit the PubChem-inactive augmentation: is the change genuine or an easy-negative artefact?

Adding high-throughput inactives can inflate AUROC if the negatives are chemically trivial to tell
apart from the actives. This checks two things per endpoint:

  1. Similarity of the added inactives to the training actives (max ECFP-4 Tanimoto to any active).
     If the added inactives sit close to actives (high similarity), separating them is genuinely hard
     and any performance change is real; if they are far away (low similarity), a rise in AUROC is
     partly an easy-negative artefact and is flagged.
  2. The cross-validated metric before vs after the augmentation (scaffold split), from the saved
     baselines.

Output: results/tables/inactives_audit.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.train_rf import _load  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "_pubchem_cache"
TAB = ROOT / "results" / "tables"
TARGETS = ["AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG"]
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fps(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        out.append(_GEN.GetFingerprint(m) if m is not None else None)
    return [f for f in out if f is not None]


def main():
    pre = pd.read_csv(TAB / "rf_cv_summary_pre_inactives.csv")
    post = pd.read_csv(TAB / "rf_cv_summary.csv")
    rows = []
    for ep in TARGETS:
        added_path = CACHE / f"{ep}_added.csv"
        if not added_path.exists():
            continue
        added = pd.read_csv(added_path)
        df = _load(ep)
        actives = df[df.label == 1]["smiles"].tolist()
        act_fps = _fps(actives)
        # subsample added inactives for the similarity scan (bounded)
        samp = added["smiles"].tolist()[:2000]
        sims = []
        for f in _fps(samp):
            sims.append(max(DataStructs.BulkTanimotoSimilarity(f, act_fps)))
        sims = np.array(sims)
        b = pre[(pre.endpoint == ep) & (pre.split == "scaffold")]
        a = post[(post.endpoint == ep) & (post.split == "scaffold")]
        row = {"endpoint": ep, "added_inactives": len(added),
               "median_sim_to_active": round(float(np.median(sims)), 3),
               "frac_similar_ge_0.5": round(float((sims >= 0.5).mean()), 3),
               "auroc_before": round(float(b.roc_auc_mean.iloc[0]), 4) if len(b) else None,
               "auroc_after": round(float(a.roc_auc_mean.iloc[0]), 4) if len(a) else None,
               "mcc_before": round(float(b.mcc_mean.iloc[0]), 4) if len(b) else None,
               "mcc_after": round(float(a.mcc_mean.iloc[0]), 4) if len(a) else None}
        row["verdict"] = ("genuine (inactives overlap actives)" if row["median_sim_to_active"] >= 0.4
                          else "interpret with care (some easy negatives)")
        rows.append(row)
        print(f"[{ep}] +{len(added)} inact; median sim-to-active {row['median_sim_to_active']}; "
              f"AUROC {row['auroc_before']}->{row['auroc_after']}, MCC {row['mcc_before']}->{row['mcc_after']}")
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "inactives_audit.csv", index=False)
    print("\n=== inactives augmentation audit ===")
    print(out.to_string(index=False))
    print("wrote results/tables/inactives_audit.csv")


if __name__ == "__main__":
    main()
