"""Measured natural-product activity against the panel targets, fetched and characterised.

Why this exists. A withanolide submitted to the server returns no engagement and a maximum Tanimoto
of 0.31 to 158,890 measured compounds, with the nearest neighbour a bile acid. That is honest
behaviour on chemistry the panel does not cover, and the coverage gap is measurable: the training
library has a median fraction-sp3 of 0.36 against 0.79 for withaferin A, and only 3.3 per cent of it
is both sp3-rich and free of aromatic rings.

The repository already holds 214,740 COCONUT natural products, but a decision recorded on 2026-07-20
placed them as coverage and never as training labels, because they carry structures and no measured
endpoint values; using them as labels would reintroduce the circularity the project removed. That
decision stands. This script does not touch them.

What it adds instead is natural-product chemistry that DOES carry measured activity against the
panel's own targets, drawn from ChEMBL, so it enters training under exactly the same rule as every
other row: a measurement, against a named target, with its relation preserved.

Two cautions are built in.

  the flag is not trusted   ChEMBL's natural_product field means natural-product-derived and is
                            over-inclusive: querying it returns prazosin, a synthetic quinazoline,
                            as its first record. Every fetched compound is therefore characterised
                            structurally (fraction sp3, aromatic ring count, ring system) and the
                            report separates genuinely terpenoid or steroidal chemistry from
                            flat semi-synthetic scaffolds that the panel already covers.
  nothing is merged here    this writes a review table and a coverage report. Merging into the
                            endpoint tables is a separate, deliberate act, because it changes every
                            model downstream and must be measured before and after.

Outputs:
  data/_chembl_cache/np_<target>.json          raw activities, as retrieved
  results/tables/natural_product_candidates.csv   one row per compound-target measurement
  results/tables/natural_product_coverage.csv     per-target summary, and what it would add

Run:  SAI_CNS_ALLOW_NONSTRICT_TLS=1 python src/sai_cns/data/fetch_natural_products.py
      ... --targets KEAP1 AChE MAO_B      restrict to named targets
      ... --limit-pages 4                 shorter run for a check
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tls  # noqa: E402
from activity_labels import ACTIVE_CUT, INACTIVE_CUT, label_from  # noqa: E402,F401

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
CACHE = ROOT / "data" / "_chembl_cache"
TAB = ROOT / "results" / "tables"
ENDPOINTS = ROOT / "data" / "endpoints"

KEEP_TYPES = ("IC50", "Ki", "Kd", "EC50", "Potency")
PAGE, MAX_PAGES = 1000, 12

# A compound counts as the chemistry this panel lacks when it is sp3-rich and not built on a flat
# aromatic core. These are the thresholds the coverage analysis used, and withaferin A sits well
# inside them at 0.79 and zero aromatic rings.
SP3_RICH, MAX_AROMATIC = 0.55, 1

_SESSION = None


def http():
    global _SESSION
    if _SESSION is None:
        _SESSION = _tls.session()
    return _SESSION


def target_ids() -> dict[str, str]:
    """Endpoint name to ChEMBL target id, read from the fetchers that defined them."""
    ids: dict[str, str] = {}
    for f in sorted((ROOT / "src" / "sai_cns" / "data").glob("fetch_*.py")):
        if f.name == Path(__file__).name:
            continue
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"[\"']([A-Za-z0-9_.]+)[\"']\s*:\s*\(?\s*[\"'](CHEMBL\d+)[\"']", text):
            ids.setdefault(m.group(1), m.group(2))
    return ids


def describe(smiles: str) -> dict | None:
    """Structural character, so the natural-product flag never has to be taken on trust."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    return {
        "fsp3": round(float(fsp3), 3),
        "aromatic_rings": int(arom),
        "rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "stereocentres": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "mw": round(float(rdMolDescriptors.CalcExactMolWt(mol)), 1),
        # the class the panel is short of, judged on structure rather than on a database field
        "fills_the_gap": bool(fsp3 >= SP3_RICH and arom <= MAX_AROMATIC),
    }


def fetch_target(name: str, tid: str, max_pages: int, refresh: bool) -> list[dict]:
    """Every measured activity at one target from a molecule ChEMBL flags natural-product."""
    cache = CACHE / f"np_{name}.json"
    if cache.exists() and not refresh:
        rows = json.loads(cache.read_text())
        print(f"  [{name:12s}] cache: {len(rows)}", flush=True)
        return rows

    rows, offset = [], 0
    for _ in range(max_pages):
        url = (f"{CHEMBL}/activity.json?target_chembl_id={tid}"
               f"&molecule_properties__natural_product=1"
               f"&standard_units=nM&limit={PAGE}&offset={offset}")
        try:
            j = http().get(url, timeout=45).json()
        except Exception as exc:
            print(f"  [{name:12s}] FAILED: {exc}", flush=True)
            break
        acts = j.get("activities", [])
        for a in acts:
            smi, val, typ = (a.get("canonical_smiles"), a.get("standard_value"),
                             a.get("standard_type"))
            rel = (a.get("standard_relation") or "=").strip()
            if not (smi and val and typ in KEEP_TYPES):
                continue
            try:
                nm = float(val)
            except (TypeError, ValueError):
                continue
            if nm <= 0:
                continue
            rows.append({"smiles": smi, "pchembl": round(9.0 - math.log10(nm), 3),
                         "relation": rel, "standard_type": typ,
                         "molecule_chembl_id": a.get("molecule_chembl_id"),
                         "year": a.get("document_year")})
        offset += PAGE
        if not j.get("page_meta", {}).get("next"):
            break
        time.sleep(0.2)

    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    print(f"  [{name:12s}] {len(rows)} measured activities", flush=True)
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Measured natural-product activity for the panel.")
    ap.add_argument("--targets", nargs="*", help="endpoint names (default: all with a ChEMBL id)")
    ap.add_argument("--limit-pages", type=int, default=MAX_PAGES)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)

    ids = target_ids()
    have = sorted(p.stem for p in ENDPOINTS.glob("*.csv"))
    todo = [e for e in have if e in ids]
    if args.targets:
        todo = [e for e in todo if e in set(args.targets)]
    print(f"{len(todo)} target(s) with a ChEMBL id\n")

    records, summary = [], []
    for name in todo:
        rows = fetch_target(name, ids[name], args.limit_pages, args.refresh)
        existing = set()
        path = ENDPOINTS / f"{name}.csv"
        if path.exists():
            existing = set(pd.read_csv(path, usecols=["smiles"]).smiles.astype(str))

        kept = gap = novel = 0
        for r in rows:
            lab = label_from(r["pchembl"], r["relation"])
            if lab is None:
                continue
            d = describe(r["smiles"])
            if d is None:
                continue
            kept += 1
            gap += int(d["fills_the_gap"])
            is_new = r["smiles"] not in existing
            novel += int(is_new)
            records.append({"endpoint": name, "label": lab, "already_in_table": not is_new,
                            **r, **d})
        summary.append({"endpoint": name, "fetched": len(rows), "usable": kept,
                        "new_to_this_endpoint": novel, "sp3_rich_non_aromatic": gap,
                        "rows_in_table_now": len(existing)})
        print(f"  [{name:12s}] usable {kept:5d} | new {novel:5d} | fills the gap {gap:5d}",
              flush=True)

    TAB.mkdir(parents=True, exist_ok=True)
    cand = pd.DataFrame(records)
    cand.to_csv(TAB / "natural_product_candidates.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(TAB / "natural_product_coverage.csv", index=False)

    print("\n=== what this would add ===")
    if len(cand):
        print(f"  measurements usable            {len(cand):,}")
        print(f"  new to their endpoint          {int((~cand.already_in_table).sum()):,}")
        gapc = cand[cand.fills_the_gap]
        print(f"  sp3-rich and non-aromatic      {len(gapc):,} "
              f"({100*len(gapc)/max(len(cand),1):.1f}% of the fetched set)")
        print(f"  median fsp3 of the fetched set {cand.fsp3.median():.2f} "
              f"(library median is 0.36, withaferin A is 0.79)")
        print(f"  distinct compounds             {cand.smiles.nunique():,}")
    else:
        print("  nothing usable was returned")
    print("\nwrote results/tables/natural_product_candidates.csv and "
          "natural_product_coverage.csv")
    print("Nothing was merged. Review the coverage table before adding any of it to training.")


if __name__ == "__main__":
    main(sys.argv[1:])
