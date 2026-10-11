"""Fetch the second target expansion: ALS, Huntington, neuroinflammation, epilepsy and sleep.

Every ChEMBL identifier below was resolved by NAME SEARCH and the returned preferred name was
checked against the intended protein (see find_targets.py); identifiers are not taken from memory,
because several memorised identifiers proved to point at unrelated proteins.

Writes data/endpoints/<TARGET>.csv in the existing schema.
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

import pandas as pd
import requests

requests.packages.urllib3.disable_warnings()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_compound_library import add_parent_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "endpoints"
BASE = "https://www.ebi.ac.uk/chembl/api/data"

# name -> (ChEMBL id, verified preferred name, rationale)
TARGETS = {
    # neuroinflammation
    "NLRP3":  ("CHEMBL1741208", "NACHT, LRR and PYD domains-containing protein 3"),
    "P2X7":   ("CHEMBL4805", "P2X purinoceptor 7"),
    "COX2":   ("CHEMBL230", "Prostaglandin G/H synthase 2"),
    "CSF1R":  ("CHEMBL1844", "Macrophage colony-stimulating factor 1 receptor"),
    # Huntington
    "PDE10A": ("CHEMBL4409", "cAMP and cAMP-inhibited cGMP 3',5'-cyclic phosphodiesterase 10A"),
    "HDAC1":  ("CHEMBL325", "Histone deacetylase 1"),
    "HDAC6":  ("CHEMBL1865", "Protein deacetylase HDAC6"),
    # excitotoxicity / epilepsy / ALS
    "GluN2B": ("CHEMBL1904", "Glutamate receptor ionotropic, NMDA 2B"),
    "mGluR5": ("CHEMBL3227", "Metabotropic glutamate receptor 5"),
    "GABA_A": ("CHEMBL2093872", "GABA-A receptor; anion channel"),
    # sleep and circadian
    "OX1":    ("CHEMBL5113", "Orexin/Hypocretin receptor type 1"),
    "OX2":    ("CHEMBL4792", "Orexin receptor type 2"),
    "MT1":    ("CHEMBL1945", "Melatonin receptor type 1A"),
    # neuroprotection / autophagy / ALS
    "mTOR":   ("CHEMBL2842", "Serine/threonine-protein kinase mTOR"),
    "SIRT1":  ("CHEMBL4506", "NAD-dependent protein deacetylase sirtuin-1"),
    "KEAP1":  ("CHEMBL2069156", "Kelch-like ECH-associated protein 1"),
    # Parkinson genetics
    "GBA1":   ("CHEMBL2179", "Lysosomal acid glucosylceramidase"),
    # cognition
    "PDE4B":  ("CHEMBL275", "3',5'-cyclic-AMP phosphodiesterase 4B"),
    # cardiac safety
    "Nav1_5": ("CHEMBL1980", "Sodium channel protein type 5 subunit alpha"),
}


def fetch(cid):
    rows, url = [], f"{BASE}/activity.json?target_chembl_id={cid}&pchembl_value__isnull=false&limit=1000"
    while url:
        for _ in range(4):
            try:
                r = requests.get(url, timeout=90, verify=False)
                r.raise_for_status()
                break
            except Exception:
                time.sleep(4)
        else:
            raise RuntimeError(f"failed {url}")
        j = r.json()
        for a in j["activities"]:
            smi, pv = a.get("canonical_smiles"), a.get("pchembl_value")
            if smi and pv:
                rows.append({"smiles": smi, "pchembl": float(pv), "year": a.get("document_year")})
        nxt = j["page_meta"]["next"]
        url = ("https://www.ebi.ac.uk" + nxt) if nxt else None
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (cid, pref) in TARGETS.items():
        dest = OUT / f"{name}.csv"
        if dest.exists():
            print(f"[{name}] exists, skipping", flush=True)
            continue
        print(f"[{name}] fetching {cid} ({pref}) ...", flush=True)
        try:
            df = fetch(cid)
        except Exception as e:
            print(f"[{name}] FAILED {e}", flush=True)
            continue
        if df.empty:
            print(f"[{name}] no data", flush=True)
            continue
        med = add_parent_key(df).groupby("inchikey").agg(
            smiles=("smiles", "first"), pchembl=("pchembl", "median"),
            year=("year", "max")).reset_index(drop=True)
        med["label"] = med["pchembl"].apply(lambda p: 1 if p >= 6 else (0 if p < 5 else None))
        med = med.dropna(subset=["label"])
        med["label"] = med["label"].astype(int)
        med["source"] = "ChEMBL"
        med[["smiles", "label", "pchembl", "year", "source"]].to_csv(dest, index=False)
        n7 = int((med["pchembl"] >= 7).sum())
        print(f"[{name}] wrote {len(med)} compounds, {n7} binders at pChEMBL>=7", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
