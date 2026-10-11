"""Fetch additional targets to widen read-across beyond the modelled panel.

Read-across answers a query by pointing at measured neighbours, so its reach is set by how much of
pharmacology the index covers, not by how many models are trained. Adding measured actives for
targets we do not model therefore extends the tool's coverage without any claim that those targets
are predicted.

Every identifier below was checked by querying the target record and confirming the returned
preferred name describes the intended protein. That check was not decorative: of twenty-six
identifiers recalled from memory, four pointed at unrelated proteins, with the histamine H4 entry
resolving to the delta opioid receptor and the CDK5 entry to CDK9. Those were dropped rather than
corrected by guesswork, and only name-verified entries appear here.

These compounds enter the read-across index alone. They train nothing, and the interface labels
their evidence as similarity to measured actives rather than as prediction.

Output: data/readacross/<TARGET>.csv
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
OUT = ROOT / "data" / "readacross"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://www.ebi.ac.uk/chembl/api/data"

# name -> (ChEMBL id, preferred name confirmed by query)
TARGETS = {
    "ADRA1A": ("CHEMBL229", "Alpha-1A adrenergic receptor"),
    "ADRA2A": ("CHEMBL1867", "Alpha-2A adrenergic receptor"),
    "ADRB2": ("CHEMBL210", "Beta-2 adrenergic receptor"),
    "M1": ("CHEMBL216", "Muscarinic acetylcholine receptor M1"),
    "M2": ("CHEMBL211", "Muscarinic acetylcholine receptor M2"),
    "M3": ("CHEMBL245", "Muscarinic acetylcholine receptor M3"),
    "D1": ("CHEMBL2056", "D(1A) dopamine receptor"),
    "D4": ("CHEMBL219", "D(4) dopamine receptor"),
    "HT2C": ("CHEMBL225", "5-hydroxytryptamine receptor 2C"),
    "HT3A": ("CHEMBL1899", "5-hydroxytryptamine receptor 3A"),
    "HT4": ("CHEMBL1875", "5-hydroxytryptamine receptor 4"),
    "H1": ("CHEMBL231", "Histamine H1 receptor"),
    "CB2": ("CHEMBL253", "Cannabinoid receptor 2"),
    "OPRD1": ("CHEMBL236", "Delta-type opioid receptor"),
    "COX1": ("CHEMBL221", "Prostaglandin G/H synthase 1"),
    "PDE5": ("CHEMBL1827", "cGMP-specific 3',5'-cyclic phosphodiesterase"),
    "ROCK1": ("CHEMBL3231", "Rho-associated protein kinase 1"),
    "JNK3": ("CHEMBL2637", "Mitogen-activated protein kinase 10"),
    "p38a": ("CHEMBL260", "Mitogen-activated protein kinase 14"),
    "CTSB": ("CHEMBL4072", "Cathepsin B"),
    "CASP3": ("CHEMBL2334", "Caspase-3"),
    "PARP1": ("CHEMBL3105", "Poly [ADP-ribose] polymerase 1"),
}
ACTIVE_P = 6.0


def fetch(cid):
    rows, url = [], f"{BASE}/activity.json?target_chembl_id={cid}&pchembl_value__isnull=false&limit=1000"
    while url:
        for _ in range(4):
            try:
                r = requests.get(url, timeout=90, verify=False)
                r.raise_for_status()
                break
            except Exception:
                time.sleep(3)
        else:
            return pd.DataFrame(rows)
        j = r.json()
        for a in j["activities"]:
            smi, pv = a.get("canonical_smiles"), a.get("pchembl_value")
            if smi and pv:
                rows.append({"smiles": smi, "pchembl": float(pv)})
        nxt = j["page_meta"]["next"]
        url = ("https://www.ebi.ac.uk" + nxt) if nxt else None
    return pd.DataFrame(rows)


def main():
    for name, (cid, pref) in TARGETS.items():
        dest = OUT / f"{name}.csv"
        if dest.exists():
            print(f"[{name}] cached", flush=True)
            continue
        try:
            df = fetch(cid)
        except Exception as e:
            print(f"[{name}] FAILED {str(e)[:50]}", flush=True)
            continue
        if df.empty:
            print(f"[{name}] no data", flush=True)
            continue
        med = add_parent_key(df).groupby("inchikey").agg(
            smiles=("smiles", "first"), pchembl=("pchembl", "median")).reset_index(drop=True)
        act = med[med.pchembl >= ACTIVE_P]
        act.to_csv(dest, index=False)
        print(f"[{name}] {len(act):,} actives at pChEMBL>={ACTIVE_P:.0f}  ({pref})", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
