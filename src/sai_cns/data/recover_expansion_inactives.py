"""Recover the measured negative class for the expansion endpoints, and merge it in.

BS-C-15 was fixed for the eleven core targets, whose tables are rebuilt from cache by
rebuild_endpoints.py. The other endpoints are written directly by the batch fetchers, so the same fix
did not reach them, and the panel was left on two bases: eleven endpoints averaging 71 per cent
active with their measured non-binders restored, and forty-nine averaging 86 per cent without. A
panel mean taken across both describes neither.

This closes that. For every expansion target it runs the query the pchembl_value filter excludes,
`standard_relation=">"` with a concentration in nM, keeps only the bounds that settle a compound as
inactive whatever the true value is, and merges those compounds into the endpoint table.

The merge is additive and conservative. A compound already in the table keeps the measurement it has,
because an exact value beats a bound; only compounds with no row at all are added, keyed on the
InChIKey of the desalted parent so a salt does not enter as a second copy of a compound already
present. Nothing is overwritten and no label is changed.

Outputs:
  data/_chembl_cache/<target>_inactive.json    raw bounds, as retrieved
  data/endpoints/<target>.csv                  extended with the recovered negatives
  results/tables/expansion_inactives.csv       what was added, per endpoint

Run:  SAI_CNS_ALLOW_NONSTRICT_TLS=1 python src/sai_cns/data/recover_expansion_inactives.py
      ... --fetch-only     retrieve the caches without touching any table
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tls  # noqa: E402
from build_compound_library import standardise  # noqa: E402
from activity_labels import INACTIVE_CUT, bound_settles_inactive  # noqa: E402,F401

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
CACHE = ROOT / "data" / "_chembl_cache"
ENDPOINTS = ROOT / "data" / "endpoints"
KEEP_TYPES = ("IC50", "Ki", "Kd", "EC50", "Potency")
MAX_PAGES, PAGE = 16, 1000
MIN_EXISTING = 50           # a table smaller than this is not one of ours; leave it alone

_SESSION = None


def _http():
    global _SESSION
    if _SESSION is None:
        _SESSION = _tls.session()
    return _SESSION


def target_ids() -> dict[str, str]:
    """Endpoint name to ChEMBL target id, read from the fetchers that defined them.

    Taken from the source rather than restated here, so this cannot drift from the queries that
    built the tables in the first place.
    """
    ids: dict[str, str] = {}
    for f in sorted((ROOT / "src" / "sai_cns" / "data").glob("fetch_*.py")):
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"[\"']([A-Za-z0-9_.]+)[\"']\s*:\s*\(?\s*[\"'](CHEMBL\d+)[\"']", text):
            ids.setdefault(m.group(1), m.group(2))
    return ids


def fetch_inactives(name: str, tid: str, refresh: bool = False) -> list[dict]:
    """The measured non-binders for one target, cached."""
    cache = CACHE / f"{name}_inactive.json"
    if cache.exists() and not refresh:
        rows = json.loads(cache.read_text())
        print(f"  [{name:12s}] cache: {len(rows)}", flush=True)
        return rows
    rows, offset, undecidable = [], 0, 0
    for page in range(MAX_PAGES):
        url = (f"{CHEMBL}/activity.json?target_chembl_id={tid}"
               f"&standard_relation=%3E&standard_units=nM&limit={PAGE}&offset={offset}")
        try:
            j = _http().get(url, timeout=45).json()
        except Exception as exc:
            raise RuntimeError(f"[{name}] page {page} failed: {exc}") from exc
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
            bound = 9.0 - math.log10(nm)
            if not bound_settles_inactive(bound):
                undecidable += 1          # consistent with either label; not guessed at
                continue
            rows.append({"smiles": smi, "pchembl_bound": bound, "relation": ">",
                         "year": a.get("document_year")})
        offset += PAGE
        if not j.get("page_meta", {}).get("next"):
            break
        time.sleep(0.25)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    print(f"  [{name:12s}] {len(rows)} usable ({undecidable} bounds too weak to decide)", flush=True)
    return rows


def merge_into_table(name: str, rows: list[dict]) -> dict:
    """Add the recovered negatives to an endpoint table, without disturbing what is there."""
    path = ENDPOINTS / f"{name}.csv"
    if not path.exists():
        return {"endpoint": name, "status": "no table", "added": 0}
    df = pd.read_csv(path)
    if "smiles" not in df.columns or "label" not in df.columns or len(df) < MIN_EXISTING:
        return {"endpoint": name, "status": "unexpected schema, skipped", "added": 0}

    existing = set()
    for smi in df["smiles"].astype(str):
        _, ik = standardise(smi)
        if ik:
            existing.add(ik)

    # weakest bound per compound: the safest summary of an upper limit on potency
    best: dict[str, dict] = {}
    for r in rows:
        csmi, ik = standardise(r["smiles"])
        if ik is None or ik in existing:
            continue
        prev = best.get(ik)
        if prev is None or r["pchembl_bound"] < prev["pchembl"]:
            best[ik] = {"smiles": csmi, "label": 0,
                        "pchembl": round(float(r["pchembl_bound"]), 3),
                        "year": pd.to_numeric(r.get("year"), errors="coerce"),
                        "source": "ChEMBL_inactive"}
    if not best:
        return {"endpoint": name, "status": "nothing new", "added": 0,
                "rows_before": len(df), "rows_after": len(df)}

    add = pd.DataFrame(list(best.values()))
    for col in df.columns:
        if col not in add.columns:
            add[col] = pd.NA
    out = pd.concat([df, add[df.columns]], ignore_index=True)
    before_act = float((df["label"] == 1).mean() * 100)
    out.to_csv(path, index=False)
    return {"endpoint": name, "status": "merged", "added": len(add),
            "rows_before": len(df), "rows_after": len(out),
            "pct_active_before": round(before_act, 1),
            "pct_active_after": round(float((out["label"] == 1).mean() * 100), 1)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Recover measured negatives for expansion endpoints.")
    ap.add_argument("--fetch-only", action="store_true", help="retrieve caches, change no table")
    ap.add_argument("--refresh", action="store_true", help="ignore existing inactive caches")
    args = ap.parse_args(argv)

    ids = target_ids()
    eps = sorted(p.stem for p in ENDPOINTS.glob("*.csv"))
    todo = [e for e in eps if e in ids and not e.endswith(".chembl_only")]
    already = [e for e in todo if (CACHE / f"{e}_inactive.json").exists() and not args.refresh]
    print(f"{len(todo)} endpoints with a ChEMBL id; {len(already)} already have an inactive cache")

    summary = []
    for ep in todo:
        try:
            rows = fetch_inactives(ep, ids[ep], refresh=args.refresh)
        except Exception as exc:
            print(f"  [{ep:12s}] FAILED: {exc}", flush=True)
            summary.append({"endpoint": ep, "status": f"fetch failed: {exc}", "added": 0})
            continue
        if args.fetch_only:
            summary.append({"endpoint": ep, "status": "fetched only", "added": len(rows)})
            continue
        summary.append(merge_into_table(ep, rows))

    out = pd.DataFrame(summary)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "results" / "tables" / "expansion_inactives.csv", index=False)
    merged = out[out.status == "merged"] if "status" in out.columns else out
    print(f"\nendpoints extended: {len(merged)}; compounds added: {int(out.added.sum()):,}")
    if len(merged):
        print(f"mean per cent active {merged.pct_active_before.mean():.1f} -> "
              f"{merged.pct_active_after.mean():.1f}")
    print("wrote results/tables/expansion_inactives.csv")


if __name__ == "__main__":
    main()
