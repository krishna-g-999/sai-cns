"""Fetch measured binding affinities from BindingDB and report the net-new yield per target.

BindingDB is an independent measured-affinity database (curated from papers, patents and PubChem).
It overlaps ChEMBL heavily for well-studied targets, so the only honest way to state the gain is to
standardise both to InChIKey and count how many BindingDB compounds are *not already* in our
ChEMBL-derived endpoint sets. This script fetches, standardises, and writes that comparison. It does
not modify the training data; rebuilding the endpoints is a separate, deliberate step taken only if
the net-new yield is worthwhile.

Affinities (nM) are converted to the same -log10(molar) potency scale as pChEMBL and labelled with
the identical rule (>=6 active, <5 inactive, 5-6 grey zone dropped).

Output: results/tables/bindingdb_yield.csv
"""
from __future__ import annotations

import json
import math
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
CACHE = ROOT / "data" / "_bindingdb_cache"
CACHE.mkdir(parents=True, exist_ok=True)

# Human UniProt accessions for the eleven ChEMBL targets.
UNIPROT = {
    "AChE": "P22303", "BChE": "P06276", "BACE1": "P56817", "GSK3B": "P49841",
    "MAO_B": "P27338", "MAO_A": "P21397", "D2": "P14416", "A2A": "P29274",
    "HT2A": "P28223", "SERT": "P31645", "hERG": "Q12809",
}
CUTOFF_NM = 1_000_000  # 1 mM ceiling: captures weak binders and inactives (the server rejects 10 mM)


_HEADERS = {"User-Agent": "Mozilla/5.0 (research; BrainSafe QSAR data collection)"}


def _get(url: str) -> requests.Response:
    """GET with the machine's CA bundle, falling back to unverified for hosts not in the store."""
    try:
        return requests.get(url, timeout=120, verify=certifi.where(), headers=_HEADERS)
    except requests.exceptions.SSLError:
        return requests.get(url, timeout=120, verify=False, headers=_HEADERS)


def fetch_target(name: str, uniprot: str) -> list[dict]:
    cache = CACHE / f"{name}.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
        print(f"  [{name}] cache: {len(rows)} affinities")
        return rows
    url = f"https://bindingdb.org/rest/getLigandsByUniprot?uniprot={uniprot}&cutoff={CUTOFF_NM}"
    rows, hit = [], None
    for attempt in range(5):
        r = _get(url)
        if r.status_code == 200 and r.content:
            try:
                resp = r.json().get("getLindsByUniprotResponse", {})
                rows = resp.get("bdb.affinities", []) or []
                hit = resp.get("bdb.hit")
                break
            except ValueError:
                pass
        wait = 5 * (2 ** attempt)  # exponential backoff for rate-limiting: 5,10,20,40,80s
        print(f"  [{name}] attempt {attempt + 1}: empty/non-JSON, waiting {wait}s")
        time.sleep(wait)
    if not rows:
        print(f"  [{name}] no data after retries (not cached, will refetch next run)")
        return rows
    cache.write_text(json.dumps(rows))
    print(f"  [{name}] fetched: {len(rows)} affinities (hit={hit})")
    return rows


def parse_affinity(raw: str) -> tuple[float | None, str]:
    """Convert a BindingDB affinity string in nM to (-log10(molar) potency, relation).

    BindingDB records censored measurements: ">10000" means no binding was detected up to 10 uM,
    "<1" means the affinity was beyond the assay's dynamic range. The relation is returned rather
    than discarded, because stripping it turns a bound into an exact value: ">10000" would become
    precisely 10 uM and "<1" precisely 1 nM, neither of which was measured.

    Returned relation is one of "=", ">", "<", "~". For a censored record the value is the bound,
    not an estimate, and the caller must treat it as such.
    """
    s = str(raw).strip()
    relation = "="
    if s[:1] in "><~":
        relation = s[0]
        s = s[1:]
    s = s.lstrip("=").strip()
    try:
        v = float(s)
    except ValueError:
        return None, relation
    if v <= 0:
        return None, relation
    return 9.0 - math.log10(v), relation  # nM -> pX


def label_from(pvalue: float) -> int:
    return 1 if pvalue >= 6.0 else (0 if pvalue < 5.0 else -1)


def build_bindingdb_compounds(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate BindingDB rows into labelled compounds keyed by InChIKey.

    Only exact measurements contribute to the pooled potency, because a median taken over a mixture
    of exact values and bounds is not a potency. Censored records are kept, separately, with the
    bound and the relation, so the negative class they carry is available to a later step rather
    than silently discarded here.

    Returns (exact_compounds, censored_records).
    """
    exact, censored = [], []
    for a in rows:
        p, relation = parse_affinity(a.get("bdb.affinity"))
        if p is None:
            continue
        csmi, ik = standardise(a.get("bdb.smile"))
        if ik is None:
            continue
        if relation == "=":
            exact.append({"inchikey": ik, "smiles": csmi, "pvalue": p})
        elif relation in "><":
            # ">10000 nM" bounds potency from above (the compound is weaker than the bound);
            # "<1 nM" bounds it from below. Decisive only when the bound falls outside the grey
            # zone: a ">" bound at or below the inactive cut means inactive whatever the true
            # value is, and a "<" bound at or above the active cut means active.
            decisive = (relation == ">" and p <= 5.0) or (relation == "<" and p >= 6.0)
            censored.append({"inchikey": ik, "smiles": csmi, "bound": p, "relation": relation,
                             "implied_label": (1 if relation == "<" else 0) if decisive else -1})
        # "~" is an approximation of unstated precision and is dropped.

    cen_df = (pd.DataFrame(censored) if censored else
              pd.DataFrame(columns=["inchikey", "smiles", "bound", "relation", "implied_label"]))
    if not exact:
        return pd.DataFrame(columns=["inchikey", "smiles", "pchembl", "label"]), cen_df
    df = pd.DataFrame(exact)
    g = df.groupby("inchikey").agg(smiles=("smiles", "first"),
                                   pchembl=("pvalue", "median")).reset_index()
    g["label"] = g["pchembl"].apply(label_from)
    return g[g["label"] >= 0].reset_index(drop=True), cen_df


def current_inchikeys(endpoint: str) -> set[str]:
    path = ROOT / "data" / "endpoints" / f"{endpoint}.csv"
    keys = set()
    if path.exists():
        for smi in pd.read_csv(path)["smiles"]:
            _, ik = standardise(smi)
            if ik:
                keys.add(ik)
    return keys


def main() -> None:
    summary = []
    for i, (name, uniprot) in enumerate(UNIPROT.items()):
        if i:
            time.sleep(6)  # be gentle with the BindingDB server between targets
        rows = fetch_target(name, uniprot)
        bdb, cen = build_bindingdb_compounds(rows)
        have = current_inchikeys(name)
        new = bdb[~bdb.inchikey.isin(have)]
        decisive = cen[cen.implied_label >= 0] if len(cen) else cen
        summary.append({
            "endpoint": name,
            "current_chembl_compounds": len(have),
            "bindingdb_labelled": len(bdb),
            "net_new_compounds": len(new),
            "net_new_active": int((new.label == 1).sum()),
            "net_new_inactive": int((new.label == 0).sum()),
            "combined_total": len(have) + len(new),
            # Censored records are counted rather than absorbed, so the size of the negative class
            # the query throws away is visible instead of implicit.
            "censored_records": len(cen),
            "censored_decisive": len(decisive),
            "censored_decisive_inactive": int((decisive.implied_label == 0).sum()) if len(decisive) else 0,
        })
        # persist the standardised BindingDB compounds for the later rebuild step
        bdb.to_csv(CACHE / f"{name}_labelled.csv", index=False)
        cen.to_csv(CACHE / f"{name}_censored.csv", index=False)
        print(f"  [{name}] current {len(have)} + net-new {len(new)} "
              f"-> combined {len(have)+len(new)}")
    out = pd.DataFrame(summary)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "results" / "tables" / "bindingdb_yield.csv", index=False)
    print("\n=== BindingDB net-new yield ===")
    print(out.to_string(index=False))
    print(f"\nTOTAL net-new compounds: {out.net_new_compounds.sum():,}")
    print("wrote results/tables/bindingdb_yield.csv")


if __name__ == "__main__":
    main()
