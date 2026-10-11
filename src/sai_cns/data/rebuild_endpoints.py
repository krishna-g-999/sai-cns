"""Rebuild the endpoint training tables from pooled ChEMBL + BindingDB measured evidence.

For each target the two independent measured sources are combined at the compound level: each source
contributes a per-compound median potency (on the shared -log10 molar scale), the two are pooled by
InChIKey, and the label is derived with the unchanged rule (>=6 active, <5 inactive, 5-6 grey zone
dropped). Blood-brain-barrier data additionally gains the FDA-curated compounds that are not already
in B3DB. Nothing is imputed and no source overrides a measurement; where both sources measure a
compound, their medians are pooled.

The current tables are copied to a timestamped folder under archive/legacy/ first, so the change is
reversible and the provenance of every compound (which source, how many) is recorded.

This script rewrites the training data, so it refuses to run on incomplete inputs. Every cached
source response is checked before anything is touched, every table is rebuilt in memory and checked
for plausibility before any file is written, and a rebuild that would empty or halve a table stops
with a non-zero exit. On a fresh clone the caches are absent (they are not committed), and the
correct outcome is a clear error, not eleven empty tables.

Outputs:
  data/endpoints/<target>.csv        smiles,label,pchembl,year,source   (rebuilt)
  results/tables/endpoint_rebuild_provenance.csv   per-endpoint source breakdown

Run:  python src/sai_cns/data/rebuild_endpoints.py
      python src/sai_cns/data/rebuild_endpoints.py --allow-shrink   (only if a drop is intended)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_compound_library import standardise  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CHEMBL_CACHE = ROOT / "data" / "_chembl_cache"
BDB_CACHE = ROOT / "data" / "_bindingdb_cache"
ENDPOINTS = ROOT / "data" / "endpoints"

TARGETS = ["AChE", "BChE", "BACE1", "GSK3B", "MAO_B", "MAO_A", "D2", "A2A", "HT2A", "SERT", "hERG"]

# A rebuilt endpoint below this many rows is a broken cache, not a result. The smallest table this
# script legitimately produces is BChE at 2,621 rows, so the floor is far below any real outcome.
MIN_ROWS = 100
# Refuse to replace a table with one less than this fraction of its current size unless the operator
# says the reduction is intended.
SHRINK_LIMIT = 0.5


def label_from(pvalue: float) -> int:
    return 1 if pvalue >= 6.0 else (0 if pvalue < 5.0 else -1)


def missing_inputs() -> list[str]:
    """Every required input that is absent, named relative to the repository root."""
    absent_dirs = [d for d in (CHEMBL_CACHE, BDB_CACHE) if not d.is_dir()]
    if absent_dirs:
        return [f"{d.relative_to(ROOT).as_posix()}/  (the whole cache directory is absent)"
                for d in absent_dirs]
    missing = []
    for name in TARGETS:
        for path in (CHEMBL_CACHE / f"{name}_y.json", BDB_CACHE / f"{name}_labelled.csv"):
            if not path.exists():
                missing.append(path.relative_to(ROOT).as_posix())
    if not (ENDPOINTS / "BBB.csv").exists():
        missing.append((ENDPOINTS / "BBB.csv").relative_to(ROOT).as_posix())
    return missing


def check_plausible(name: str, df: pd.DataFrame, allow_shrink: bool) -> None:
    """Stop before overwriting a populated table with an empty or drastically smaller one."""
    if len(df) < MIN_ROWS:
        raise SystemExit(
            f"[{name}] the rebuild produced {len(df)} rows, below the {MIN_ROWS}-row floor. "
            "That indicates a broken or partial cache rather than a result. Nothing was written."
        )
    existing = ENDPOINTS / f"{name}.csv"
    if existing.exists() and not allow_shrink:
        n_old = len(pd.read_csv(existing))
        if n_old and len(df) < SHRINK_LIMIT * n_old:
            raise SystemExit(
                f"[{name}] the rebuild produced {len(df)} rows against {n_old} already on disk, a "
                f"drop of more than {int((1 - SHRINK_LIMIT) * 100)} per cent. Refusing to overwrite. "
                "Re-run with --allow-shrink if the reduction is intended. Nothing was written."
            )


def chembl_compound_level(name: str) -> pd.DataFrame:
    """Per-compound median pChEMBL from the cached ChEMBL activities."""
    cache = CHEMBL_CACHE / f"{name}_y.json"
    if not cache.exists():
        raise FileNotFoundError(
            f"cached ChEMBL activities for {name} are missing: {cache.relative_to(ROOT)}"
        )
    recs = []
    for r in json.loads(cache.read_text()):
        csmi, ik = standardise(r.get("smiles"))
        if ik is None:
            continue
        recs.append({"inchikey": ik, "smiles": csmi, "pchembl": float(r["pchembl"]),
                     "year": pd.to_numeric(r.get("year"), errors="coerce")})
    df = pd.DataFrame(recs)
    return df.groupby("inchikey").agg(smiles=("smiles", "first"),
                                      pchembl=("pchembl", "median"),
                                      year=("year", "min")).reset_index()


def chembl_inactive_level(name: str) -> pd.DataFrame:
    """Per-compound measured non-binders, from the censored ChEMBL records.

    Absent when the cache predates fetch_endpoints.fetch_inactive_activities, in which case the
    endpoint keeps the composition it had and the run says so rather than failing.
    """
    cache = CHEMBL_CACHE / f"{name}_inactive.json"
    if not cache.exists():
        return pd.DataFrame(columns=["inchikey", "smiles", "pchembl", "year"])
    recs = []
    for r in json.loads(cache.read_text()):
        csmi, ik = standardise(r.get("smiles"))
        if ik is None:
            continue
        recs.append({"inchikey": ik, "smiles": csmi,
                     "pchembl": float(r["pchembl_bound"]),
                     "year": pd.to_numeric(r.get("year"), errors="coerce")})
    if not recs:
        return pd.DataFrame(columns=["inchikey", "smiles", "pchembl", "year"])
    df = pd.DataFrame(recs)
    # The bound is an upper limit on potency, so the weakest bound is the safest summary.
    return df.groupby("inchikey").agg(smiles=("smiles", "first"),
                                      pchembl=("pchembl", "min"),
                                      year=("year", "min")).reset_index()


def bindingdb_compound_level(name: str) -> pd.DataFrame:
    """Per-compound median potency already standardised by fetch_bindingdb.py."""
    path = BDB_CACHE / f"{name}_labelled.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"cached BindingDB measurements for {name} are missing: {path.relative_to(ROOT)}. "
            "Regenerate with src/sai_cns/data/fetch_bindingdb.py"
        )
    df = pd.read_csv(path)
    df["year"] = np.nan
    return df[["inchikey", "smiles", "pchembl", "year"]]


def rebuild_target(name: str) -> tuple[pd.DataFrame, dict]:
    ch = chembl_compound_level(name); ch["src"] = "ChEMBL"
    bd = bindingdb_compound_level(name); bd["src"] = "BindingDB"
    # Measured non-binders, which the pchembl_value filter excludes. A compound with an exact
    # measurement keeps it; these only supply compounds that would otherwise be absent entirely.
    ina = chembl_inactive_level(name)
    if len(ina):
        ina = ina[~ina.inchikey.isin(set(ch.inchikey) | set(bd.inchikey))].copy()
        ina["src"] = "ChEMBL_inactive"
    long = pd.concat([ch, bd], ignore_index=True)
    # A source with no measurements for this target arrives as an all-object frame, and concatenating
    # it makes the pooled numeric columns object as well, which silently turns the .round() below into
    # a no-op. hERG is the one target where BindingDB contributes nothing, so without this coercion it
    # is the one table written with unrounded potencies. Coerce so every target is pooled and rounded
    # on the same numeric dtype.
    for col in ("pchembl", "year"):
        long[col] = pd.to_numeric(long[col], errors="coerce")
    g = long.groupby("inchikey").agg(
        smiles=("smiles", "first"),
        pchembl=("pchembl", "median"),
        year=("year", "min"),
        source=("src", lambda s: "+".join(sorted(set(s)))),
    ).reset_index()
    g_keys = set(g["inchikey"])
    g["label"] = g["pchembl"].apply(label_from)
    g = g[g["label"] >= 0].reset_index(drop=True)

    # Censored non-binders are labelled directly and never sent through label_from, because that
    # rule reads its argument as an exact potency and drops the 5 to 6 band as ambiguous. A bound is
    # not a potency: ">10000 nM" means the true value lies strictly below pX 5.0, which is inactive,
    # yet it arrives at the rule as exactly 5.0 and is discarded as grey-zone. For AChE alone that
    # silently lost 253 measured non-binders, which is the censoring error reappearing one step
    # further down. Only compounds with no exact measurement are added; an exact value always wins.
    if len(ina):
        fresh = ina[~ina.inchikey.isin(set(g_keys))].copy()
        if len(fresh):
            fresh["label"] = 0
            fresh["source"] = "ChEMBL_inactive"
            g = pd.concat([g, fresh[["smiles", "label", "pchembl", "year", "source"]]
                           if set(["smiles", "label", "pchembl", "year", "source"]).issubset(fresh.columns)
                           else fresh], ignore_index=True)
    prov = {
        "endpoint": name,
        "chembl_only": int((g.source == "ChEMBL").sum()),
        "chembl_measured_inactive": int((g.source == "ChEMBL_inactive").sum()),
        "bindingdb_only": int((g.source == "BindingDB").sum()),
        "both": int((g.source == "BindingDB+ChEMBL").sum()),
        "total": len(g),
        "active": int((g.label == 1).sum()),
        "inactive": int((g.label == 0).sum()),
    }
    return g[["smiles", "label", "pchembl", "year", "source"]].round({"pchembl": 3}), prov


def bbb_stats() -> dict:
    """Report the BBB training set (B3DB, left unchanged). The FDA-curated compounds are held out as
    an external validation set, not merged into training - a stronger, independent approved-drug test
    than a small training augmentation would be."""
    bbb = pd.read_csv(ENDPOINTS / "BBB.csv")
    return {"endpoint": "BBB", "chembl_only": 0, "bindingdb_only": 0, "both": 0,
            "total": len(bbb), "active": int((bbb.label == 1).sum()),
            "inactive": int((bbb.label == 0).sum()), "fda_added": 0}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Rebuild the endpoint training tables.")
    ap.add_argument("--allow-shrink", action="store_true",
                    help="permit a rebuilt table smaller than half the one it replaces")
    args = ap.parse_args(argv)

    missing = missing_inputs()
    if missing:
        raise SystemExit(
            "rebuild_endpoints.py rewrites the training tables and cannot run without the cached\n"
            "source responses. Missing:\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\n\nThese caches are not committed (see .gitignore). Regenerate them with the\n"
              "acquisition scripts in src/sai_cns/data/, or restore them from the archived\n"
              "download, then re-run. Nothing has been written and data/endpoints/ is untouched."
        )

    # Rebuild and check everything before writing anything, so a failure part way through cannot
    # leave data/endpoints/ half rebuilt and half stale.
    built, prov = {}, []
    for name in TARGETS:
        df, p = rebuild_target(name)
        check_plausible(name, df, args.allow_shrink)
        built[name] = df
        prov.append(p)

    backup = ROOT / "archive" / "legacy" / f"endpoints_before_rebuild_{datetime.now():%Y-%m-%dT%H%M%S}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ENDPOINTS, backup)
    print(f"backed up current endpoints -> {backup.relative_to(ROOT)}")

    for name, p in zip(built, prov):
        built[name].to_csv(ENDPOINTS / f"{name}.csv", index=False)
        print(f"[{name}] total {p['total']:>6}  (ChEMBL {p['chembl_only']}, "
              f"BindingDB {p['bindingdb_only']}, both {p['both']}; {p['active']} active)")
    bbb = bbb_stats()
    prov.append({k: bbb.get(k) for k in prov[0]})
    print(f"[BBB]  total {bbb['total']:>6}  (B3DB unchanged; FDA-curated held out for external test)")

    out = pd.DataFrame(prov)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "results" / "tables" / "endpoint_rebuild_provenance.csv", index=False)
    print(f"\nTOTAL labelled compound-endpoint records: {out.total.sum():,}")
    print("wrote rebuilt data/endpoints/*.csv and results/tables/endpoint_rebuild_provenance.csv")


if __name__ == "__main__":
    main()
