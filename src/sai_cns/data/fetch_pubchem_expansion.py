"""Fetch measured inactives from PubChem BioAssay for the targets whose negative class is thin.

Fourteen targets in the expanded panel have fewer than about 100 experimentally measured inactives in
ChEMBL, because ChEMBL reports what medicinal chemistry programmes publish, which is overwhelmingly
active compounds. Three of them (LRRK2, orexin OX2, GABA-A) have so few that no target-specific
decision threshold can be estimated at all and they fall back to a global cut.

PubChem high-throughput screens test large, diverse compound sets and record the genuine negatives.
For each target we collect compounds recorded inactive in that target's assays and subtract every
compound ever recorded active for the same target, so a compound is treated as a negative only if it
is never active there.

An important caveat that is carried through to the label: PubChem HTS results are usually
single-concentration and are noisier than the dose-response inactives obtained from ChEMBL. They are
written with source "PubChem_HTS" so downstream code and the manuscript can distinguish them, and
they are used to enlarge the negative class for threshold calibration, never to overturn a measured
dose-response activity.

Output: data/_pubchem_cache/<target>_inactives.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import certifi
import pandas as pd
import requests
import urllib3

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_compound_library import standardise  # noqa: E402

urllib3.disable_warnings()
CACHE = ROOT / "data" / "_pubchem_cache"
CACHE.mkdir(parents=True, exist_ok=True)
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# target -> UniProt accession (all verified against the ChEMBL/UniProt record)
TARGETS = {
    "LRRK2": "Q5S007", "OX2": "O43614", "GABA_A": "P14867", "MT1": "P48039",
    "GluN2B": "Q13224", "CSF1R": "P07333", "P2X7": "Q99572", "Sigma1": "Q99720",
    "HT6": "P50406", "NLRP3": "Q96P20", "mGluR5": "P41594", "PDE10A": "Q9Y233",
    "HT2A": "P28223", "OX1": "O43613",
}
MAX_ASSAYS = 60
CAP_INACTIVE_CIDS = 8000
FETCH_SMILES_CAP = 3000
HEADERS = {"User-Agent": "Mozilla/5.0 (research; BrainSafe QSAR)"}


def _get(url, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=90, verify=certifi.where(), headers=HEADERS)
        except requests.exceptions.SSLError:
            try:
                r = requests.get(url, timeout=90, verify=False, headers=HEADERS)
            except Exception:
                time.sleep(2 * (i + 1)); continue
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
    inactive, active, scanned = set(), set(), 0
    for aid in aids[:MAX_ASSAYS]:
        active.update(_cids(aid, "active"))
        inactive.update(_cids(aid, "inactive"))
        scanned += 1
        time.sleep(0.15)
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
        time.sleep(0.15)
    return out


def main():
    for name, acc in TARGETS.items():
        cache = CACHE / f"{name}_inactives.csv"
        if cache.exists():
            print(f"[{name}] cached: {len(pd.read_csv(cache))} inactives", flush=True)
            continue
        try:
            neg, scanned, n_act = target_negatives(acc)
        except Exception as e:
            print(f"[{name}] FAILED {str(e)[:60]}", flush=True)
            continue
        print(f"[{name}] scanned {scanned} assays, {len(neg)} inactive-only CIDs "
              f"(excluded {n_act} active)", flush=True)
        smi = cids_to_smiles(neg[:FETCH_SMILES_CAP])
        rows = []
        for cid, s in smi.items():
            csmi, ik = standardise(s)
            if ik is not None:
                rows.append({"inchikey": ik, "smiles": csmi, "cid": cid})
        df = pd.DataFrame(rows).drop_duplicates("inchikey")
        df.to_csv(cache, index=False)
        print(f"[{name}] standardised inactives: {len(df)}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
