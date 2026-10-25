"""Fetch the censored activity records the labelling rule discards, so they can be modelled.

WHY THIS EXISTS. The panel's labelling rule keeps a censored measurement only when its bound
settles a class on its own: "IC50 > 10 uM" bounds potency below pChEMBL 5.0, which is inactive
whatever the true value is, so it is kept as a negative. "IC50 > 100 nM" bounds potency below
pChEMBL 7.0, which is consistent with either class, so fetch_endpoints.py counts it and throws it
away. That discard happens at fetch time, which means the records are not merely unlabelled in the
training tables: they are absent from data/_chembl_cache entirely and cannot be recovered locally.

A censored-likelihood model does not need a bound to settle a class. It treats every measurement as
an interval the true potency lies in and asks the model to be consistent with all of them, so a
">100 nM" record is informative: it rules out the strong-potency region. Testing whether that
formulation beats the deployed binary one therefore requires re-fetching what the binary rule threw
out, and this script does only that.

WHAT IT FETCHES. Both censored relations, over the same standard types the main fetcher keeps, with
no bound filter:

  relation '>' on concentration  ->  potency is BELOW the bound   ->  interval (-inf, b]
  relation '<' on concentration  ->  potency is ABOVE the bound   ->  interval [b, +inf)

The relation inverts on the pChEMBL scale, which is the silent-mislabelling trap documented in
activity_labels.py, so the interval is stored on the pChEMBL scale as explicit pchembl_lower and
pchembl_upper fields with None for the open end. Nothing downstream has to re-derive the direction.

The main fetcher never asks for '<' at all, so those records have never been seen by this project.

This writes its own cache files and touches nothing the deployed pipeline reads. It is an
experiment's input, not a change to the labelling rule.

    python tools/fetch_censored_records.py [--endpoints AChE,hERG] [--refresh]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "_chembl_cache"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

# Matched to fetch_endpoints.py so the censored records describe the same assay types as the exact
# ones. Diverging here would make the censored arm's extra data a different experiment.
KEEP_TYPES = ("IC50", "Ki", "Kd", "EC50", "Potency")
PAGE = 1000
MAX_PAGES = 40          # higher than the main fetcher's 16: no bound filter, so more rows survive

TARGETS = {
    "AChE": "CHEMBL220", "BChE": "CHEMBL1914", "BACE1": "CHEMBL4822", "GSK3B": "CHEMBL262",
    "MAO_B": "CHEMBL2039", "MAO_A": "CHEMBL1951", "D2": "CHEMBL217", "A2A": "CHEMBL251",
    "HT2A": "CHEMBL224", "SERT": "CHEMBL228", "hERG": "CHEMBL240",
}

RELATIONS = {">": "weaker_than", "<": "stronger_than"}


def _get(url: str, tries: int = 4) -> dict:
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.loads(r.read())
        except Exception as exc:          # noqa: BLE001 - network, retried
            last = exc
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"{url} failed after {tries} tries: {last}")


def fetch_one(name: str, tid: str, relation: str) -> list[dict]:
    """Every censored record for one target and one relation, with no bound filter."""
    rows, offset, capped = [], 0, False
    for page in range(MAX_PAGES):
        url = (f"{CHEMBL}/activity.json?target_chembl_id={tid}"
               f"&standard_relation={urllib.parse.quote(relation)}"
               f"&standard_units=nM&limit={PAGE}&offset={offset}")
        j = _get(url)
        acts = j.get("activities", [])
        for a in acts:
            smi, val, st = a.get("canonical_smiles"), a.get("standard_value"), a.get("standard_type")
            if not (smi and val and st in KEEP_TYPES):
                continue
            try:
                nm = float(val)
            except (TypeError, ValueError):
                continue
            if nm <= 0 or not math.isfinite(nm):
                continue
            bound = 9.0 - math.log10(nm)       # the pChEMBL-scale bound
            # The relation refers to the concentration, so it inverts here. See activity_labels.py.
            if relation == ">":
                lo, hi = None, bound           # potency strictly below the bound
            else:
                lo, hi = bound, None           # potency strictly above the bound
            rows.append({"smiles": smi, "pchembl_lower": lo, "pchembl_upper": hi,
                         "relation": relation, "std": st, "year": a.get("document_year")})
        offset += PAGE
        if not j.get("page_meta", {}).get("next"):
            break
        if page == MAX_PAGES - 1:
            capped = True
        time.sleep(0.2)
    if capped:
        print(f"  [{name} {relation}] WARNING: hit the {MAX_PAGES}-page cap; more records remain",
              flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", default="")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    eps = [e.strip() for e in args.endpoints.split(",") if e.strip()] or sorted(TARGETS)
    CACHE.mkdir(parents=True, exist_ok=True)
    total = 0
    for ep in eps:
        out = CACHE / f"{ep}_censored.json"
        if out.exists() and not args.refresh:
            n = len(json.loads(out.read_text(encoding="utf-8")))
            print(f"[{ep:6}] cached: {n:,}", flush=True)
            total += n
            continue
        t0 = time.time()
        rows: list[dict] = []
        for rel in RELATIONS:
            got = fetch_one(ep, TARGETS[ep], rel)
            rows.extend(got)
            print(f"  [{ep:6} {rel}] {len(got):,}", flush=True)
        out.write_text(json.dumps(rows), encoding="utf-8")
        settles = sum(1 for r in rows
                      if (r["pchembl_upper"] is not None and r["pchembl_upper"] <= 5.0)
                      or (r["pchembl_lower"] is not None and r["pchembl_lower"] >= 6.0))
        print(f"[{ep:6}] {len(rows):,} censored records in {time.time() - t0:.0f}s "
              f"({settles:,} settle a class under the deployed rule, "
              f"{len(rows) - settles:,} are the ones it discards)", flush=True)
        total += len(rows)

    print(f"\ntotal censored records: {total:,}")
    print(f"wrote {CACHE.relative_to(ROOT).as_posix()}/<endpoint>_censored.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
