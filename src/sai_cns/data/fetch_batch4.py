"""Fetch the fourth target expansion: the mechanisms of epilepsy and pain the panel was missing.

These six were not chosen for completeness. The clinical-indication test (inversion H6) measured that
epilepsy and chronic pain are the two conditions the tool serves worst, recovering the correct
indication for 10 and 23 percent of approved drugs, and the target-discrimination test (H7) showed
the cause is that the mechanisms those drugs act through are either absent from the panel or
represented by a thin endpoint. The data audit (results/unmodelled_target_data.csv) then found that
exactly these six clear both the volume bar and the source-diversity bar. This expansion is therefore
aimed at a measured weakness rather than at a longer list.

Identifiers come from that audit, where each was resolved by name search and its preferred name
recorded. They are re-verified here against the live ChEMBL target record before a single activity is
downloaded, and a mismatch aborts the fetch. A previous expansion had to be rebuilt because a
memorised identifier pointed at an unrelated protein, and a name search for one kainate subunit
silently returned another.

Writes data/endpoints/<TARGET>.csv in the existing schema.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_compound_library import add_parent_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "endpoints"
BASE = "https://www.ebi.ac.uk/chembl/api/data"

# endpoint name -> (ChEMBL id, exact preferred name that must match, why it is being added)
TARGETS = {
    "a4b2nAChR": ("CHEMBL1907589", "Neuronal acetylcholine receptor; alpha4/beta2",
                  "the high-affinity neuronal nicotinic subtype; varenicline and cytisine act here"),
    "a3b4nAChR": ("CHEMBL1907594", "Neuronal acetylcholine receptor; alpha3/beta4",
                  "ganglionic subtype, relevant to nicotine dependence and to off-target autonomic effects"),
    "Nav1_6":    ("CHEMBL5202", "Sodium channel protein type 8 subunit alpha",
                  "the principal axonal sodium channel; a main antiepileptic mechanism"),
    "Nav1_8":    ("CHEMBL5451", "Sodium channel protein type 10 subunit alpha",
                  "nociceptor-restricted sodium channel; a validated analgesic mechanism"),
    "Cav3_2":    ("CHEMBL1859", "Voltage-dependent T-type calcium channel subunit alpha-1H",
                  "T-type calcium current underlying absence seizures; ethosuximide acts here"),
    "GABAA_a5":  ("CHEMBL2094122", "GABA-A receptor; alpha-5/beta-3/gamma-2",
                  "a defined physiological GABA-A subtype, better evidenced than the pooled endpoint"),
}


def get(url, params=None, tries=4):
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, timeout=90, verify=False)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(4)
    return None


def verify(cid, expected):
    """Abort rather than download activities for a target that is not the one intended."""
    j = get(f"{BASE}/target/{cid}.json")
    if j is None:
        raise RuntimeError(f"could not read target {cid}; refusing to guess")
    got = j.get("pref_name")
    if got != expected:
        raise RuntimeError(f"{cid} is '{got}', expected '{expected}'. Aborting: a wrong identifier "
                           f"silently trains a model on the wrong protein.")
    if j.get("organism") != "Homo sapiens":
        raise RuntimeError(f"{cid} is {j.get('organism')}, not human")
    return j


def fetch(cid):
    rows, url = [], (f"{BASE}/activity.json?target_chembl_id={cid}"
                     f"&pchembl_value__isnull=false&limit=1000")
    while url:
        j = get(url)
        if j is None:
            raise RuntimeError(f"failed {url}")
        for a in j["activities"]:
            smi, pv = a.get("canonical_smiles"), a.get("pchembl_value")
            if smi and pv:
                rows.append({"smiles": smi, "pchembl": float(pv), "year": a.get("document_year"),
                             "assay": a.get("assay_chembl_id"), "doc": a.get("document_chembl_id")})
        nxt = j["page_meta"]["next"]
        url = ("https://www.ebi.ac.uk" + nxt) if nxt else None
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, (cid, pref, why) in TARGETS.items():
        dest = OUT / f"{name}.csv"
        if dest.exists():
            print(f"[{name}] exists, skipping", flush=True)
            continue
        print(f"[{name}] verifying {cid} ...", flush=True)
        try:
            verify(cid, pref)
        except RuntimeError as e:
            print(f"[{name}] ABORTED: {e}", flush=True)
            continue
        print(f"[{name}] {pref}: {why}", flush=True)
        try:
            df = fetch(cid)
        except Exception as e:
            print(f"[{name}] FAILED {e}", flush=True)
            continue
        if df.empty:
            print(f"[{name}] no data", flush=True)
            continue

        # Median over replicate measurements of the same structure, then the project's standard
        # label rule: active at pChEMBL >= 6, inactive below 5, the ambiguous band discarded.
        med = add_parent_key(df).groupby("inchikey").agg(
            smiles=("smiles", "first"), pchembl=("pchembl", "median"),
            year=("year", "max"), n_assays=("assay", "nunique"),
            n_docs=("doc", "nunique")).reset_index(drop=True)
        med["label"] = med["pchembl"].apply(lambda p: 1 if p >= 6 else (0 if p < 5 else None))
        med = med.dropna(subset=["label"])
        med["label"] = med["label"].astype(int)
        med["source"] = "ChEMBL"
        med[["smiles", "label", "pchembl", "year", "source"]].to_csv(dest, index=False)

        n_bind = int((med["pchembl"] >= 7).sum())
        summary.append({"target": name, "chembl_id": cid, "compounds": len(med),
                        "actives_p6": int(med.label.sum()),
                        "binders_p7": n_bind,
                        "measured_inactives": int((med.label == 0).sum()),
                        "distinct_documents": int(df["doc"].nunique()),
                        "distinct_assays": int(df["assay"].nunique())})
        print(f"[{name}] wrote {len(med)} compounds: {int(med.label.sum())} active at pChEMBL>=6, "
              f"{n_bind} binders at >=7, {int((med.label == 0).sum())} measured inactives, "
              f"from {df['doc'].nunique()} documents", flush=True)

    if summary:
        s = pd.DataFrame(summary)
        s.to_csv(ROOT / "results" / "batch4_fetch_summary.csv", index=False)
        pd.set_option("display.width", 200)
        print()
        print(s.to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
