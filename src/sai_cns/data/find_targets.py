"""Resolve ChEMBL target IDs by NAME SEARCH, never by memorised identifier.

Memorised ChEMBL identifiers are unreliable: an ID believed to be GABA-A can in fact be an unrelated
protein, which would silently train a mislabelled model. This script searches ChEMBL for each target
by name, keeps only human single-protein or protein-complex entries whose preferred name actually
matches the requested protein, and reports the count of activities carrying a pChEMBL value.

Output: data/_chembl_cache/target_search.json
"""
from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

import requests

requests.packages.urllib3.disable_warnings()
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "_chembl_cache" / "target_search.json"
BASE = "https://www.ebi.ac.uk/chembl/api/data"

# query -> tokens that MUST appear in the ChEMBL preferred name
WANTED = {
    "GABA-A receptor alpha-1": ["gamma-aminobutyric", "GABA"],
    "glutamate NMDA receptor GRIN2B": ["NMDA"],
    "glutamate receptor ionotropic AMPA": ["AMPA", "ionotropic"],
    "metabotropic glutamate receptor 5": ["Metabotropic glutamate receptor 5"],
    "sodium channel protein type 2 alpha": ["Sodium channel"],
    "NLR family pyrin domain containing 3": ["NACHT", "NLRP", "pyrin"],
    "serine/threonine-protein kinase mTOR": ["mTOR", "Serine/threonine-protein kinase mTOR"],
    "Kelch-like ECH-associated protein 1": ["Kelch"],
    "phosphodiesterase 10A": ["phosphodiesterase 10"],
    "histone deacetylase 6": ["deacetylase 6"],
    "histone deacetylase 1": ["deacetylase 1"],
    "P2X purinoceptor 7": ["P2X purinoceptor 7"],
    "orexin receptor type 1": ["Orexin", "Hypocretin"],
    "orexin receptor type 2": ["Orexin"],
    "melatonin receptor type 1A": ["Melatonin receptor type 1A"],
    "glucosylceramidase beta GBA": ["glucosylceramidase", "Glucosylceramidase"],
    "prostaglandin G/H synthase 2": ["Prostaglandin G/H synthase 2"],
    "macrophage colony stimulating factor 1 receptor": ["colony stimulating factor 1 receptor",
                                                        "Macrophage colony"],
    "cAMP specific phosphodiesterase 4B": ["phosphodiesterase 4B", "3',5'-cyclic-AMP"],
    "NAD-dependent protein deacetylase sirtuin-1": ["sirtuin-1", "Sirtuin 1"],
    "catechol O-methyltransferase": ["Catechol O-methyltransferase"],
    "monoamine oxidase": ["Monoamine oxidase"],
}


def search(q):
    u = f"{BASE}/target/search.json?q={urllib.parse.quote(q)}&limit=20"
    try:
        return requests.get(u, timeout=45, verify=False).json().get("targets", [])
    except Exception:
        return []


def count(cid):
    u = f"{BASE}/activity.json?target_chembl_id={cid}&pchembl_value__isnull=false&limit=1"
    try:
        return requests.get(u, timeout=45, verify=False).json()["page_meta"]["total_count"]
    except Exception:
        return None


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    found = {}
    for q, tokens in WANTED.items():
        best = None
        for t in search(q):
            if t.get("organism") != "Homo sapiens":
                continue
            if t.get("target_type") not in ("SINGLE PROTEIN", "PROTEIN COMPLEX", "PROTEIN FAMILY"):
                continue
            nm = t.get("pref_name") or ""
            if not any(tok.lower() in nm.lower() for tok in tokens):
                continue
            c = count(t["target_chembl_id"])
            if c is None:
                continue
            if best is None or c > best["n_activities"]:
                best = {"query": q, "chembl_id": t["target_chembl_id"], "pref_name": nm,
                        "target_type": t.get("target_type"), "n_activities": c}
        if best:
            found[q] = best
            print(f"{q:48} {best['chembl_id']:15}{best['n_activities']:>7}  {best['pref_name'][:44]}",
                  flush=True)
        else:
            print(f"{q:48} {'NOT FOUND':15}", flush=True)
    OUT.write_text(json.dumps(found, indent=2))
    ok = {k: v for k, v in found.items() if v["n_activities"] >= 800}
    print(f"\ntrainable (>=800 measured activities): {len(ok)}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
