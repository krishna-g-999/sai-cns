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
    "TAAR1":  ("CHEMBL5857", "Trace amine-associated receptor 1"),
    "Nav1_1": ("CHEMBL1845", "Sodium channel protein type 1 subunit alpha"),
    "Nav1_7": ("CHEMBL4296", "Sodium channel protein type 9 subunit alpha"),
    "GluA2":  ("CHEMBL4016", "Glutamate receptor 2 (AMPA)"),
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
