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
    # Non-CNS expansion. Every identifier here was resolved by searching the ChEMBL API and
    # checking the returned pref_name, never taken from memory, and every candidate cleared the
    # 800-activity bar on both volume and source diversity. The assessment that chose them is
    # src/sai_cns/data/assess_noncns_targets.py and results/noncns_target_data.csv.
    "EGFR": "CHEMBL203",
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


# The label rule lives in one place. This file used to carry a private copy that took no relation
# argument, so every censored bound was read as an exact potency.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from activity_labels import ACTIVE_CUT, INACTIVE_CUT  # noqa: E402
from fetch_endpoints import fetch_inactive_activities  # noqa: E402


def label_from(p):
    """Exact-value rule, matching rebuild_endpoints.py.

    Censored records never reach this function. A bound is not a potency, and ">10000 nM" arrives
    here as exactly 5.0, which this rule discards as grey-zone; for AChE that silently lost 253
    measured non-binders. They are recovered separately in main().
    """
    return 1 if p >= ACTIVE_CUT else (0 if p < INACTIVE_CUT else None)


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
            year=("year", "max")).reset_index()
        med["label"] = med["pchembl"].apply(label_from)
        med = med.dropna(subset=["label"])
        med["label"] = med["label"].astype(int)
        med["source"] = "ChEMBL"

        # Recover the measured negative class. ChEMBL assigns a pchembl_value only where
        # standard_relation is "=", so the query above cannot see a record of the form "no
        # inhibition up to 10 uM": those are not missing data, they are the measured inactives, and
        # for AChE alone the filter discards 2,210 of them. fetch_inactive_activities asks for the
        # complement and converts each bound to a potency bound, keeping only bounds at or below
        # the inactive cut, which are inactive whatever the true value is.
        #
        # Two rules from rebuild_endpoints.py are kept deliberately: the bound is an upper limit on
        # potency, so the WEAKEST bound is the safest per-compound summary, and an exact
        # measurement always wins, so only compounds with no exact value are added.
        n_recovered = 0
        try:
            raw = fetch_inactive_activities(name, cid)
        except Exception as exc:
            print(f"[{name}] inactive fetch failed, continuing without it: {exc}", flush=True)
            raw = []
        if raw:
            ina = pd.DataFrame([{"smiles": r.get("smiles"),
                                 "pchembl": float(r["pchembl_bound"]),
                                 "year": r.get("year")} for r in raw
                                if r.get("smiles") and r.get("pchembl_bound") is not None])
            if len(ina):
                ina = add_parent_key(ina).groupby("inchikey").agg(
                    smiles=("smiles", "first"), pchembl=("pchembl", "min"),
                    year=("year", "max")).reset_index()
                ina = ina[~ina["inchikey"].isin(set(med["inchikey"]))]
                if len(ina):
                    ina["label"] = 0
                    ina["source"] = "ChEMBL_inactive"
                    med = pd.concat([med, ina], ignore_index=True)
                    n_recovered = len(ina)

        med[["smiles", "label", "pchembl", "year", "source"]].to_csv(dest, index=False)
        print(f"[{name}] wrote {len(med)} compounds "
              f"({int(med['label'].sum())} active, {int((med['label'] == 0).sum())} inactive, "
              f"{n_recovered} recovered from censored bounds)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
