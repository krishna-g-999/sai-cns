"""Fetch measured *inactive* compounds from PubChem BioAssay for the protein-target classifiers.

ChEMBL and BindingDB report inhibitors, so the measured negative class is thin (e.g. GSK-3-beta is
~93% active). PubChem high-throughput screens test large compound sets and record genuine measured
inactives - the missing class. This pulls, per target:

  - inactive compounds from the target's assays (bounded scan of the largest screens), and
  - the active compounds from those same assays, so any compound ever active for the target is
    excluded (a compound is treated as a true negative only if it is never active for that target).

Nothing here relabels our existing data; the merge step (rebuild_with_inactives.py) adds only
net-new compounds as inactives and never overrides a dose-response active with an HTS inactive.

Output: data/_pubchem_cache/<target>_inactives.csv  (inchikey, smiles)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import certifi
import pandas as pd
import requests
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_compound_library import standardise  # noqa: E402

urllib3.disable_warnings()
ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "_pubchem_cache"
CACHE.mkdir(parents=True, exist_ok=True)
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Protein-target classifiers only (BBB uses B3DB; it has no protein target).
TARGETS = {
    "AChE": "P22303", "BChE": "P06276", "BACE1": "P56817", "GSK3B": "P49841",
    "MAO_A": "P21397", "MAO_B": "P27338", "hERG": "Q12809",
}
MAX_ASSAYS = 120          # bound the per-target assay scan
CAP_INACTIVE_CIDS = 30000  # stop scanning once this many unique inactive CIDs are collected
FETCH_SMILES_CAP = 12000   # cap SMILES fetched per target (kept modest to avoid over-correction)
HEADERS = {"User-Agent": "Mozilla/5.0 (research; BrainSafe QSAR)"}


def _get(url, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=90, verify=certifi.where(), headers=HEADERS)
        except requests.exceptions.SSLError:
            r = requests.get(url, timeout=90, verify=False, headers=HEADERS)
        except Exception:
            time.sleep(2 * (i + 1)); continue
        if r.status_code == 200:
            return r
        if r.status_code in (503, 429):
            time.sleep(3 * (i + 1)); continue
        return r
    return None


def _cids(aid, kind):
    r = _get(f"{BASE}/assay/aid/{aid}/cids/JSON?cids_type={kind}")
    if r is None or r.status_code != 200:
        return []
    info = r.json().get("InformationList", {}).get("Information", [{}])
    return info[0].get("CID", []) if info else []


def target_negatives(acc):
    r = _get(f"{BASE}/assay/target/accession/{acc}/aids/JSON")
    aids = r.json().get("IdentifierList", {}).get("AID", []) if r else []
    inactive, active = set(), set()
    scanned = 0
    for aid in aids[:MAX_ASSAYS]:
        active.update(_cids(aid, "active"))
        inactive.update(_cids(aid, "inactive"))
        scanned += 1
        time.sleep(0.2)
        if len(inactive) >= CAP_INACTIVE_CIDS:
            break
    return sorted(inactive - active), scanned, len(active)


def cids_to_smiles(cids):
    out = {}
    for k in range(0, len(cids), 100):
        chunk = ",".join(str(c) for c in cids[k:k + 100])
        r = _get(f"{BASE}/compound/cid/{chunk}/property/SMILES/JSON")
        if r and r.status_code == 200:
            for p in r.json().get("PropertyTable", {}).get("Properties", []):
                if p.get("SMILES"):
                    out[p["CID"]] = p["SMILES"]
        time.sleep(0.2)
    return out


def main():
    # Targets may be named on the command line as NAME=ACCESSION so a thin endpoint can be
    # supplemented without rescanning the ones already cached. Accessions are read from the ChEMBL
    # target record's own component list, never recalled: a wrong accession would harvest another
    # protein's screening data and label it as this target's measured inactives, which is the
    # quietest way to corrupt a model.
    global TARGETS
    if len(sys.argv) > 1:
        TARGETS = dict(a.split("=", 1) for a in sys.argv[1:])
        print(f"fetching only: {TARGETS}", flush=True)

    summary = []
    for name, acc in TARGETS.items():
        cache = CACHE / f"{name}_inactives.csv"
        if cache.exists():
            n = len(pd.read_csv(cache))
            print(f"[{name}] cached: {n} inactives")
            summary.append({"target": name, "inactives": n, "note": "cached"})
            continue
        neg_cids, scanned, n_active = target_negatives(acc)
        print(f"[{name}] scanned {scanned} assays; {len(neg_cids)} inactive-only CIDs "
              f"(excluded {n_active} active)")
        smi = cids_to_smiles(neg_cids[:FETCH_SMILES_CAP])
        rows = []
        for cid, s in smi.items():
            csmi, ik = standardise(s)
            if ik is not None:
                rows.append({"inchikey": ik, "smiles": csmi, "cid": cid})
        df = pd.DataFrame(rows).drop_duplicates("inchikey")
        df.to_csv(cache, index=False)
        print(f"[{name}] standardised inactives: {len(df)}")
        summary.append({"target": name, "inactives": len(df), "note": f"{scanned} assays"})
    pd.DataFrame(summary).to_csv(ROOT / "results" / "tables" / "pubchem_inactives_summary.csv",
                                 index=False)
    print("\n=== PubChem measured-inactive yield ===")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
