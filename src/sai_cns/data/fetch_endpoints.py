"""Retrieve the measured ChEMBL activities for the core target panel, and the B3DB BBB set.

This is the first step of the endpoint pipeline and the origin of the core training data. It was
written as BS_fetch_endpoints.py, archived in commit fea5029 and restored here, because
data/raw/measured_endpoints_SOURCE.md cites it as the retriever for both sources and the chain from
public database to training table was otherwise unreproducible.

What it does now, and what it deliberately no longer does. The original wrote
data/endpoints/<target>.csv directly, from ChEMBL alone. Those tables are now produced by
rebuild_endpoints.py, which pools ChEMBL with BindingDB at the compound level; writing them here
would silently replace the pooled tables with ChEMBL-only ones. This script therefore stops at the
cache, which is what rebuild_endpoints.py consumes. BBB is different and is still written here,
because B3DB is its only source and nothing else produces it.

Query and labelling are unchanged from the archived version, so a regenerated cache matches the one
the shipped tables were built from: activities with a non-null pchembl_value, standard_type in
IC50/Ki/Kd/EC50/Potency, per-compound median potency and earliest document_year taken downstream.
ChEMBL assigns a pchembl_value only where standard_relation is '=', so censored measurements are
excluded by that query rather than silently treated as exact. They are not discarded: a second query
retrieves the '>' records, which are the measured non-binders, and keeps those whose bound settles
the compound as inactive whatever the true value is. Excluding them by construction is why most of
this panel is over 90 per cent active.

Outputs:
  data/_chembl_cache/<target>_y.json          exact activities, consumed by rebuild_endpoints.py
  data/_chembl_cache/<target>_inactive.json   measured non-binders, pooled in as label 0
  data/endpoints/BBB.csv                      smiles,label from B3DB classification

Run:  python src/sai_cns/data/fetch_endpoints.py
      python src/sai_cns/data/fetch_endpoints.py --refresh   (ignore existing caches and refetch)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import time
import sys
from pathlib import Path

import pandas as pd
import requests

import _tls  # noqa: E402  (sibling module)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activity_labels import INACTIVE_CUT, bound_settles_inactive  # noqa: E402,F401
ROOT = Path(__file__).resolve().parents[3]
CHEMBL_CACHE = ROOT / "data" / "_chembl_cache"
ENDPOINTS = ROOT / "data" / "endpoints"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
# One verified session for the whole run. On a network that terminates TLS with a
# malformed but administrator-installed CA, _tls keeps chain, signature and hostname
# checks in force and clears only the RFC formatting strictness. See _tls.py.
_SESSION = None


def _http():
    global _SESSION
    if _SESSION is None:
        _SESSION = _tls.session()
    return _SESSION
B3DB_URL = "https://raw.githubusercontent.com/theochem/B3DB/main/B3DB/B3DB_classification.tsv"

TARGETS = {
    # core CNS targets
    "AChE":  ("CHEMBL220",  "Alzheimer's / cognition"),
    "BChE":  ("CHEMBL1914", "Alzheimer's / cholinergic"),
    "BACE1": ("CHEMBL4822", "Alzheimer's / amyloid"),
    "GSK3B": ("CHEMBL262",  "tau / neuroprotection"),
    "MAO_B": ("CHEMBL2039", "Parkinson's / dopamine"),
    "MAO_A": ("CHEMBL1951", "mood / depression"),
    # expanded panel
    "D2":    ("CHEMBL217",  "Parkinson's / psychosis (dopamine D2)"),
    "A2A":   ("CHEMBL251",  "Parkinson's (adenosine A2A)"),
    "HT2A":  ("CHEMBL224",  "mood / psychosis (5-HT2A)"),
    "SERT":  ("CHEMBL228",  "depression (serotonin transporter)"),
    # safety anti-target
    "hERG":  ("CHEMBL240",  "SAFETY: cardiotoxicity liability (hERG block)"),
}
KEEP_TYPES = ("IC50", "Ki", "Kd", "EC50", "Potency")
MAX_PAGES, PAGE = 16, 1000
# A B3DB download smaller than this is a failed or truncated fetch, not a result. The shipped set
# has 7,807 rows, so the floor is far below any real outcome.
MIN_BBB_ROWS = 1000
# Potency at or below this is inactive under the project label rule, so a ">" bound
# reaching it settles the compound whatever the true value is.


def chembl_version() -> dict:
    """Record which ChEMBL release answered, so a rebuild can be dated and compared."""
    try:
        s = _http().get(f"{CHEMBL}/status.json", timeout=30).json()
        return {"chembl_db_version": s.get("chembl_db_version"),
                "chembl_release_date": s.get("chembl_release_date")}
    except Exception as exc:
        print(f"  could not read ChEMBL status: {exc}")
        return {}


def fetch_target_activities(name: str, tid: str, refresh: bool = False) -> list[dict]:
    """Every kept activity for one target, from the cache when present."""
    cache = CHEMBL_CACHE / f"{name}_y.json"
    if cache.exists() and not refresh:
        rows = json.loads(cache.read_text())
        print(f"  [{name}] cache: {len(rows)}")
        return rows
    rows, offset, truncated = [], 0, False
    for page in range(MAX_PAGES):
        url = (f"{CHEMBL}/activity.json?target_chembl_id={tid}"
               f"&pchembl_value__isnull=false&limit={PAGE}&offset={offset}")
        try:
            j = _http().get(url, timeout=45).json()
        except Exception as exc:
            raise RuntimeError(f"[{name}] ChEMBL page {page} failed: {exc}") from exc
        for a in j.get("activities", []):
            smi, pv, st = a.get("canonical_smiles"), a.get("pchembl_value"), a.get("standard_type")
            if smi and pv and st in KEEP_TYPES:
                rows.append({"smiles": smi, "pchembl": float(pv), "year": a.get("document_year")})
        offset += PAGE
        if not j.get("page_meta", {}).get("next"):
            break
        if page == MAX_PAGES - 1:
            truncated = True
        time.sleep(0.25)
    if truncated:
        # The archived version stopped here in silence, so a target that outgrew the page cap would
        # be quietly under-sampled and nothing downstream would know.
        print(f"  [{name}] WARNING: hit the {MAX_PAGES}-page cap ({MAX_PAGES * PAGE} activities) "
              "and more remain. Raise MAX_PAGES and refetch before using this target.")
    CHEMBL_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    print(f"  [{name}] fetched: {len(rows)}")
    return rows


def fetch_inactive_activities(name: str, tid: str, refresh: bool = False) -> list[dict]:
    """Retrieve the measured non-binders, which the pchembl_value filter excludes by construction.

    ChEMBL assigns a pchembl_value only where standard_relation is '=', so a query filtered on
    pchembl_value__isnull=false silently drops every record of the form "no inhibition up to 10 uM".
    Those records are not missing data, they are the measured negative class, and dropping them is
    why most endpoints in this panel are over 90 per cent active and five have fewer than 25
    inactives. For AChE alone the filter discards 2,210 measured inactives.

    This asks for the complement: standard_relation '>' with a concentration in nM. The bound is
    converted to a potency bound, and only records whose bound is at or below the inactive cut are
    usable, because those are inactive whatever the true value is. A ">100 nM" record bounds potency
    at pX 7 and is consistent with active or inactive, so it is counted and discarded rather than
    guessed at.
    """
    cache = CHEMBL_CACHE / f"{name}_inactive.json"
    if cache.exists() and not refresh:
        rows = json.loads(cache.read_text())
        print(f"  [{name}] inactive cache: {len(rows)}")
        return rows
    rows, offset, undecidable = [], 0, 0
    for page in range(MAX_PAGES):
        url = (f"{CHEMBL}/activity.json?target_chembl_id={tid}"
               f"&standard_relation=%3E&standard_units=nM&limit={PAGE}&offset={offset}")
        try:
            j = _http().get(url, timeout=45).json()
        except Exception as exc:
            raise RuntimeError(f"[{name}] ChEMBL inactive page {page} failed: {exc}") from exc
        for a in j.get("activities", []):
            smi, val, st = a.get("canonical_smiles"), a.get("standard_value"), a.get("standard_type")
            if not (smi and val and st in KEEP_TYPES):
                continue
            try:
                nm = float(val)
            except (TypeError, ValueError):
                continue
            if nm <= 0:
                continue
            bound = 9.0 - math.log10(nm)          # potency is strictly below this
            if not bound_settles_inactive(bound):
                undecidable += 1                   # e.g. ">100 nM": could still be active
                continue
            rows.append({"smiles": smi, "pchembl_bound": bound, "relation": ">",
                         "year": a.get("document_year")})
        offset += PAGE
        if not j.get("page_meta", {}).get("next"):
            break
        time.sleep(0.25)
    CHEMBL_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    print(f"  [{name}] measured non-binders: {len(rows)} usable "
          f"({undecidable} bounds too weak to decide, discarded)")
    return rows


def fetch_b3db() -> None:
    """Write data/endpoints/BBB.csv from the B3DB classification table."""
    df = pd.read_csv(io.StringIO(_http().get(B3DB_URL, timeout=60).text), sep="\t")
    col = "BBB+/BBB-" if "BBB+/BBB-" in df.columns else [c for c in df.columns if "BBB" in c][0]
    smi = "SMILES" if "SMILES" in df.columns else [c for c in df.columns if c.lower() == "smiles"][0]
    out = pd.DataFrame({
        "smiles": df[smi],
        "label": df[col].astype(str).str.upper().str.contains(r"BBB\+").astype(int),
    }).dropna().drop_duplicates("smiles")

    if len(out) < MIN_BBB_ROWS:
        raise SystemExit(
            f"B3DB returned {len(out)} usable rows, below the {MIN_BBB_ROWS}-row floor. That is a "
            "failed or truncated download, not a result. data/endpoints/BBB.csv was not written."
        )
    existing = ENDPOINTS / "BBB.csv"
    if existing.exists():
        n_old = len(pd.read_csv(existing))
        if n_old and len(out) < 0.5 * n_old:
            raise SystemExit(
                f"B3DB returned {len(out)} rows against {n_old} already on disk. Refusing to "
                "overwrite data/endpoints/BBB.csv with less than half the data it holds."
            )
    ENDPOINTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(existing, index=False)
    print(f"  [BBB] {len(out)} rows (permeable={int(out.label.sum())}) -> data/endpoints/BBB.csv")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fetch the core ChEMBL activities and the B3DB BBB set.")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore existing caches and refetch from ChEMBL")
    args = ap.parse_args(argv)

    version = chembl_version()
    if version:
        print(f"ChEMBL {version.get('chembl_db_version')} "
              f"(released {version.get('chembl_release_date')})")

    print("\nB3DB (blood-brain barrier) ...")
    fetch_b3db()

    print("\nChEMBL target activities ...")
    summary = {}
    for name, (tid, meaning) in TARGETS.items():
        rows = fetch_target_activities(name, tid, refresh=args.refresh)
        inactive = fetch_inactive_activities(name, tid, refresh=args.refresh)
        years = pd.to_numeric(pd.DataFrame(rows).get("year"), errors="coerce") if rows else None
        summary[name] = {
            "target_chembl_id": tid, "meaning": meaning, "n_activities": len(rows),
            "n_measured_inactive": len(inactive),
            "year_min": int(years.min()) if years is not None and years.notna().any() else None,
            "year_max": int(years.max()) if years is not None and years.notna().any() else None,
            **version,
        }
    (CHEMBL_CACHE / "_fetch_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\ncached {len(summary)} targets -> data/_chembl_cache/<target>_y.json")
    print("next: python src/sai_cns/data/fetch_bindingdb.py")
    print("then: python src/sai_cns/data/rebuild_endpoints.py   (builds data/endpoints/<target>.csv)")


if __name__ == "__main__":
    main()
