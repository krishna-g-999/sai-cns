"""Count the agonist/antagonist labels that actually exist, per receptor, as distinct molecules.

The manuscript states that the server cannot distinguish an agonist from an antagonist, because
the training label is a potency and ChEMBL's action_type field appears nowhere in this project's
data. Whether that is a permanent limitation or a gap waiting to be filled is a counting question.

A first pass using ChEMBL's `mechanism` endpoint suggested the limitation was permanent: 68
records for D2, 19 for A2A, 56 for HT2A, and only 13 agonists pooled across seven targets. That
was the wrong table. The `activity` endpoint carries action_type on far more records, in the
thousands rather than the tens, and this script counts what is there properly: distinct molecules
per action class, after collapsing the several activity rows a single compound usually has.

A compound is counted once per class. Compounds carrying conflicting labels at the same target,
which happens through assay heterogeneity and genuine partial agonism, are reported separately
rather than assigned arbitrarily, because forcing them into a class would inflate the apparent
training set with exactly the examples a classifier would most need to get right.

Run:
    python tools/modality_label_census.py
"""
from __future__ import annotations

import collections
import csv
import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables" / "modality_label_census.csv"
API = "https://www.ebi.ac.uk/chembl/api/data"

TARGETS = {
    "D2": "CHEMBL217", "A2A": "CHEMBL251", "HT2A": "CHEMBL224", "SERT": "CHEMBL228",
    "D3": "CHEMBL234", "HT1A": "CHEMBL214", "CB1": "CHEMBL218", "H3": "CHEMBL264",
    "GABA_A": "CHEMBL2093872", "A1": "CHEMBL226",
}

# Collapsed to the distinction the manuscript actually says cannot be made. BINDING AGENT,
# INHIBITOR and BLOCKER say nothing about direction of effect and are excluded rather than
# folded into "antagonist", which would be a guess dressed as a label.
AGONIST = {"AGONIST", "PARTIAL AGONIST"}
ANTAGONIST = {"ANTAGONIST", "INVERSE AGONIST"}

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def get(url, timeout=180):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=_CTX) as r:
                return json.load(r)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


def action_of(a):
    v = a.get("action_type")
    if isinstance(v, dict):
        return (v.get("action_type") or v.get("parent_type") or "").upper()
    return (v or "").upper()


def census(tid):
    """{molecule: set(classes)} for one target, over every labelled activity."""
    mol = collections.defaultdict(set)
    offset, page = 0, 1000
    while True:
        d = get(f"{API}/activity.json?target_chembl_id={tid}"
                f"&action_type__isnull=false&limit={page}&offset={offset}")
        acts = d.get("activities", [])
        for a in acts:
            at = action_of(a)
            m = a.get("molecule_chembl_id")
            if not m:
                continue
            if at in AGONIST:
                mol[m].add("agonist")
            elif at in ANTAGONIST:
                mol[m].add("antagonist")
        total = d["page_meta"]["total_count"]
        offset += page
        if offset >= total or not acts:
            return mol, total


def main() -> int:
    rows = []
    print(f"  {'target':8s} {'labelled acts':>14s} {'agonist':>9s} {'antagonist':>11s} "
          f"{'conflicting':>12s} {'usable':>8s}")
    print("  " + "-" * 70)
    for ep, tid in TARGETS.items():
        try:
            mol, total = census(tid)
        except Exception as e:
            print(f"  {ep:8s} fetch failed: {type(e).__name__}")
            continue
        ago = sum(1 for v in mol.values() if v == {"agonist"})
        ant = sum(1 for v in mol.values() if v == {"antagonist"})
        both = sum(1 for v in mol.values() if len(v) > 1)
        usable = min(ago, ant)
        print(f"  {ep:8s} {total:>14,} {ago:>9,} {ant:>11,} {both:>12,} {usable:>8,}")
        rows.append({"endpoint": ep, "chembl_target": tid, "labelled_activities": total,
                     "agonist_only": ago, "antagonist_only": ant, "conflicting": both,
                     "minority_class": usable})
    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  wrote {OUT.relative_to(ROOT)}")
        feasible = [r for r in rows if r["minority_class"] >= 50]
        print(f"\n  targets with at least 50 compounds in the smaller class: "
              f"{[r['endpoint'] for r in feasible] or 'none'}")
        print("  Fifty is the floor at which a scaffold-split evaluation can say anything at all;")
        print("  it is not a claim that fifty is enough to deploy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
