"""Merge SIMILARITY-MATCHED PubChem inactives into the endpoint tables.

A previous experiment in this project added bulk PubChem high-throughput inactives and was reverted
as a documented negative result: on GSK-3-beta the added negatives had a median Tanimoto of 0.29 to
the actives, and including them inflated scaffold AUROC from 0.937 to 0.989 without improving real
discrimination. The recorded conclusion was that only similarity-matched hard negatives should be
used (docs/INACTIVES_EXPERIMENT.md, docs/decisions_log.md, 2026-07-21).

This script implements that conclusion. Screening inactives are admitted only when they are
chemically close to the target's own actives (maximum ECFP-4 Tanimoto at or above MIN_SIM), so they
constitute hard negatives that sharpen the decision boundary rather than easy negatives that flatter
it. Compounds far from the active chemistry are discarded, and the discarded count is reported.

Rules, in order of precedence:
  1. A compound already present in the endpoint table keeps its ChEMBL dose-response label. A
     single-concentration screening result never overturns a dose-response measurement.
  2. A compound recorded active for the target in PubChem is excluded upstream, so only
     never-active compounds arrive here.
  3. Surviving compounds must pass the similarity filter, and are added as label 0 with source
     "PubChem_HTS" and no pChEMBL value so they remain distinguishable in every later analysis.

The original table is copied to <target>.chembl_only.csv before the first merge so the change is
reversible and the provenance of every compound is auditable.

Output: updated data/endpoints/<target>.csv and results/tables/pubchem_merge_provenance.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "_pubchem_cache"
EP = ROOT / "data" / "endpoints"
TARGETS = ["OX1"]
MAX_ADDED_RATIO = 2.0   # do not let screening negatives swamp the measured set
MIN_SIM = 0.40          # hard-negative cut: max ECFP-4 Tanimoto to the target's actives
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m else None


def fp(s):
    m = Chem.MolFromSmiles(str(s))
    return _GEN.GetFingerprint(m) if m else None


def main():
    rows = []
    for t in TARGETS:
        src = CACHE / f"{t}_inactives.csv"
        dst = EP / f"{t}.csv"
        if not src.exists() or not dst.exists():
            print(f"[{t}] missing input, skipped", flush=True)
            continue
        df = pd.read_csv(dst)
        backup = EP / f"{t}.chembl_only.csv"
        if not backup.exists():
            df.to_csv(backup, index=False)

        base = df[df.get("source", pd.Series(dtype=str)) != "PubChem_HTS"].copy() \
            if "source" in df.columns else df.copy()
        have = {c for c in (canon(s) for s in base["smiles"].astype(str)) if c}

        pc = pd.read_csv(src)
        cand = []
        for s in pc["smiles"].astype(str):
            c = canon(s)
            if c and c not in have:
                have.add(c)
                cand.append(c)

        # similarity filter: keep only hard negatives close to this target's own actives
        act_fps = [f for f in (fp(s) for s in base.loc[base["label"] == 1, "smiles"].astype(str)) if f]
        kept, sims = [], []
        if act_fps:
            for c in cand:
                f = fp(c)
                if f is None:
                    continue
                ms = max(DataStructs.BulkTanimotoSimilarity(f, act_fps))
                if ms >= MIN_SIM:
                    kept.append(c)
                    sims.append(ms)

        n_pos = int((base["label"] == 1).sum())
        cap = int(MAX_ADDED_RATIO * max(n_pos, 1))
        added = kept[:cap]

        new = pd.DataFrame({"smiles": added, "label": 0, "pchembl": None,
                            "year": None, "source": "PubChem_HTS"})
        out = pd.concat([base, new], ignore_index=True)
        out.to_csv(dst, index=False)
        rows.append({"target": t, "chembl_actives": n_pos,
                     "chembl_inactives": int((base["label"] == 0).sum()),
                     "pubchem_candidates": len(cand),
                     "passed_similarity_filter": len(kept),
                     "median_sim_of_kept": round(float(np.median(sims)), 3) if sims else None,
                     "pubchem_added": len(added),
                     "total_inactives_after": int((out["label"] == 0).sum())})
        print(f"[{t}] candidates {len(cand)} -> hard negatives {len(kept)} "
              f"(median sim {rows[-1]['median_sim_of_kept']}) -> added {len(added)}; "
              f"inactives {rows[-1]['chembl_inactives']} -> {rows[-1]['total_inactives_after']}",
              flush=True)

    if rows:
        p = ROOT / "results" / "tables" / "pubchem_merge_provenance.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        print("\nwrote", p)
    print("DONE")


if __name__ == "__main__":
    main()
