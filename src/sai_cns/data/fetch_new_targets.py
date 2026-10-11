"""Fetch measured ChEMBL activities for the new brain-target endpoints.

For each target we pull all activities carrying a pChEMBL value (a normalised -log10 molar
potency), keep human single-protein binding/functional assays, take the per-compound median,
and write data/endpoints/<TARGET>.csv in the existing schema (smiles,label,pchembl,year,source)
with the unchanged label rule (>=6 active, <5 inactive, 5-6 grey zone dropped). No imputation.

Run (needs network):  python src/sai_cns/data/fetch_new_targets.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

requests.packages.urllib3.disable_warnings()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_compound_library import add_parent_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "endpoints"
CACHE = ROOT / "data" / "_chembl_cache"
BASE = "https://www.ebi.ac.uk/chembl/api/data"

TARGETS = {
    "HT1A": "CHEMBL214", "HT6": "CHEMBL3371", "HT7": "CHEMBL3155", "H3": "CHEMBL264",
    "DAT": "CHEMBL238", "NET": "CHEMBL222", "Sigma1": "CHEMBL287", "CB1": "CHEMBL218",
    "OPRK1": "CHEMBL237", "OPRM1": "CHEMBL233", "D3": "CHEMBL234", "A1": "CHEMBL226",
    "a7nAChR": "CHEMBL2492", "LRRK2": "CHEMBL5407",
}


def fetch_activities(cid):
    rows, url = [], (f"{BASE}/activity.json?target_chembl_id={cid}"
                     f"&pchembl_value__isnull=false&limit=1000")
    while url:
        for attempt in range(4):
            try:
                r = requests.get(url, timeout=60, verify=False)
                r.raise_for_status()
                break
            except Exception:
                time.sleep(3)
        else:
            raise RuntimeError(f"failed {url}")
        j = r.json()
        for a in j["activities"]:
            smi = a.get("canonical_smiles")
            pv = a.get("pchembl_value")
            if smi and pv:
                rows.append({"smiles": smi, "pchembl": float(pv),
                             "year": a.get("document_year"),
                             "assay": a.get("assay_type")})
        nxt = j["page_meta"]["next"]
        url = ("https://www.ebi.ac.uk" + nxt) if nxt else None
    return pd.DataFrame(rows)


def label_from(p):
    return 1 if p >= 6 else (0 if p < 5 else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cid in TARGETS.items():
        dest = OUT / f"{name}.csv"
        if dest.exists():
            print(f"[{name}] exists, skipping")
            continue
        print(f"[{name}] fetching {cid} ...", flush=True)
        df = fetch_activities(cid)
        if df.empty:
            print(f"[{name}] no data"); continue
        med = add_parent_key(df).groupby("inchikey").agg(
            smiles=("smiles", "first"), pchembl=("pchembl", "median"),
            year=("year", "max")).reset_index(drop=True)
        med["label"] = med["pchembl"].apply(label_from)
        med = med.dropna(subset=["label"])
        med["label"] = med["label"].astype(int)
        med["source"] = "ChEMBL"
        med[["smiles", "label", "pchembl", "year", "source"]].to_csv(dest, index=False)
        print(f"[{name}] wrote {len(med)} compounds "
              f"({int(med['label'].sum())} active, {int((med['label'] == 0).sum())} inactive)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
