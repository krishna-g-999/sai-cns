"""Assemble the measured antioxidant regression set from ChEMBL DPPH radical-scavenging assays.

Written as BS_fetch_antioxidant.py, archived in commit fea5029 and restored here, because
data/raw/measured_endpoints_SOURCE.md cites it as the retriever for data/endpoints_reg/
antioxidant_dpph.csv and that table is a live training input (train_rf.py REGRESSION
"antioxidant_DPPH"). Without it the antioxidant endpoint had no reproducible origin.

Method, unchanged from the archived version. ChEMBL assays whose description mentions DPPH are
collected, their IC50 and EC50 activities pulled, concentrations converted to pIC50 = -log10(molar),
values outside 2 to 10 dropped as outside any sensible radical-scavenging range, and the remainder
aggregated per compound by median with the earliest document_year retained.

Note on what this endpoint is. DPPH is a chemical radical-scavenging assay, not a cellular or in vivo
antioxidant measurement, and results vary between protocols. The pooled median across assays is a
convenience, not an equivalence.

Outputs:
  data/_chembl_cache/antioxidant_dpph_raw.json   raw activities as retrieved
  data/endpoints_reg/antioxidant_dpph.csv        smiles,y,year

Run:  python src/sai_cns/data/fetch_antioxidant.py
      python src/sai_cns/data/fetch_antioxidant.py --refresh
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "_chembl_cache" / "antioxidant_dpph_raw.json"
OUT = ROOT / "data" / "endpoints_reg" / "antioxidant_dpph.csv"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

MAX_ASSAYS = 1600
UNIT_TO_M = {"pM": 1e-12, "nM": 1e-9, "uM": 1e-6, "µM": 1e-6, "mM": 1e-3, "M": 1.0}
PIC50_RANGE = (2, 10)
# The shipped table has 2,862 compounds. A run producing far fewer has failed rather than found less.
MIN_ROWS = 500


def assay_ids(term: str = "DPPH") -> list[str]:
    ids, offset = [], 0
    while len(ids) < MAX_ASSAYS:
        url = f"{CHEMBL}/assay.json?description__icontains={term}&limit=1000&offset={offset}"
        j = requests.get(url, timeout=60).json()
        ids.extend(a["assay_chembl_id"] for a in j.get("assays", []))
        if not j.get("page_meta", {}).get("next"):
            break
        offset += 1000
    if len(ids) >= MAX_ASSAYS:
        print(f"  WARNING: hit the {MAX_ASSAYS}-assay cap; more DPPH assays exist than were used.")
    return ids[:MAX_ASSAYS]


def fetch_activities(ids: list[str]) -> list[dict]:
    rows, skipped = [], 0
    for i in range(0, len(ids), 25):
        batch = ",".join(ids[i:i + 25])
        offset = 0
        while True:
            url = (f"{CHEMBL}/activity.json?assay_chembl_id__in={batch}"
                   f"&standard_type__in=IC50,EC50&limit=1000&offset={offset}")
            try:
                j = requests.get(url, timeout=60).json()
            except Exception as exc:
                # Counted rather than swallowed, so a partial harvest is visible in the run log.
                skipped += 1
                print(f"  batch {i // 25} offset {offset} failed: {exc}")
                break
            for a in j.get("activities", []):
                smi = a.get("canonical_smiles")
                val, unit = a.get("standard_value"), a.get("standard_units")
                if not (smi and val and unit in UNIT_TO_M):
                    continue
                try:
                    molar = float(val) * UNIT_TO_M[unit]
                except (TypeError, ValueError):
                    continue
                if molar > 0:
                    rows.append({"smiles": smi, "pIC50": -math.log10(molar),
                                 "year": a.get("document_year")})
            if not j.get("page_meta", {}).get("next"):
                break
            offset += 1000
        time.sleep(0.15)
    if skipped:
        print(f"  WARNING: {skipped} request batches failed and their activities are absent.")
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fetch the measured DPPH antioxidant set.")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and refetch")
    args = ap.parse_args(argv)

    if CACHE.exists() and not args.refresh:
        rows = json.loads(CACHE.read_text())
        print(f"cache: {len(rows)} raw activities")
    else:
        print("collecting DPPH assay ids ...")
        ids = assay_ids("DPPH")
        print(f"  {len(ids)} DPPH assays; fetching IC50/EC50 activities ...")
        rows = fetch_activities(ids)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(rows))
        print(f"  raw activities: {len(rows)}")

    df = pd.DataFrame(rows)
    df = df[(df.pIC50 > PIC50_RANGE[0]) & (df.pIC50 < PIC50_RANGE[1])]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    g = df.groupby("smiles").agg(y=("pIC50", "median"), year=("year", "min")).reset_index()

    if len(g) < MIN_ROWS:
        raise SystemExit(
            f"the harvest produced {len(g)} compounds, below the {MIN_ROWS}-compound floor. That is "
            f"a failed or partial retrieval, not a result. {OUT.relative_to(ROOT).as_posix()} "
            "was not written."
        )
    if OUT.exists():
        n_old = len(pd.read_csv(OUT))
        if n_old and len(g) < 0.5 * n_old:
            raise SystemExit(
                f"the harvest produced {len(g)} compounds against {n_old} already on disk. Refusing "
                "to overwrite with less than half the data."
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT, index=False)
    print(f"measured antioxidant (DPPH) set: {len(g)} unique compounds, "
          f"pIC50 {g.y.min():.1f} to {g.y.max():.1f} "
          f"(median {g.y.median():.1f}, sd {g.y.std():.2f}) -> {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
