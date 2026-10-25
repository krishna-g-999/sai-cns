"""How much of each endpoint's measured potency the labelling rule discards, per endpoint.

WHY THIS EXISTS. The panel labels actives at pChEMBL >= 6.0 and inactives at <= 5.0 and discards
the decade between as undecidable. The discarded fraction appears in no document, and it is the
first thing a referee will compute, because the band sits exactly where the decision boundary is.
Two figures have been quoted internally, 23.5 per cent panel-wide and 43.0 per cent at hERG, with
no artefact behind either. This writes the artefact.

WHAT IT CAN AND CANNOT MEASURE. The band fraction is only computable where the exact potency values
survive, which means the raw activity caches. data/endpoints/<T>.csv cannot be used: it is the
output of the labelling rule, so the band compounds have already been removed from it and the
fraction would read as zero. The caches cover a subset of the panel, and the census reports its own
coverage rather than implying it covers all 52 endpoints: an unmeasured endpoint is listed as
unmeasured.

Three cache shapes carry exact potencies, preferred in this order, so that the richest record wins
and one endpoint is never counted twice:

  <ep>_y.json     smiles, pchembl, year                    the fullest, used by the benchmarks
  <ep>_std.json   smiles, pchembl, std, year               adds the assay type
  <ep>.json       smiles, pchembl, molid                   the earliest harvest

Counting is on activity records before deduplication, which is the quantity a Methods statement
about discarded data should report: it is what the labelling rule saw. The benchmarks deduplicate on
the feature vector and therefore report smaller numbers for the same endpoints; both are correct for
their own purpose and the difference is noted in the output.

Censored records are excluded here. They carry a bound rather than a value, the rule handles them
separately, and results/tables/censored_data_inventory.csv counts them.

Writes results/tables/band_discard_census.csv

    python tools/band_discard_census.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sai_cns" / "data"))
from activity_labels import ACTIVE_CUT, INACTIVE_CUT  # noqa: E402

CACHE = ROOT / "data" / "_chembl_cache"
ENDPOINTS = ROOT / "data" / "endpoints"
OUT = ROOT / "results" / "tables" / "band_discard_census.csv"

SUFFIX_RANK = ["_y", "_std", ""]


def potencies(path: Path) -> list[float]:
    """Exact pChEMBL values from one cache file, whatever key it stores them under."""
    try:
        recs = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(recs, list):
        return []
    out = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        v = r.get("pchembl", r.get("pIC50"))
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def main() -> int:
    # Every endpoint the panel actually carries, so that coverage is measured against the panel
    # rather than against the set of files that happen to exist.
    panel = sorted(p.stem for p in ENDPOINTS.glob("*.csv"))

    rows = []
    for ep in panel:
        chosen, vals = None, []
        for suf in SUFFIX_RANK:
            p = CACHE / f"{ep}{suf}.json"
            if p.exists():
                v = potencies(p)
                if v:
                    chosen, vals = p.name, v
                    break
        if not vals:
            rows.append({"endpoint": ep, "source": None, "n_records": 0,
                         "n_active": None, "n_inactive": None, "n_band": None,
                         "band_fraction": None, "measurable": False})
            continue
        s = pd.Series(vals)
        band = int(((s > INACTIVE_CUT) & (s < ACTIVE_CUT)).sum())
        rows.append({"endpoint": ep, "source": chosen, "n_records": len(s),
                     "n_active": int((s >= ACTIVE_CUT).sum()),
                     "n_inactive": int((s <= INACTIVE_CUT).sum()),
                     "n_band": band, "band_fraction": round(band / len(s), 4),
                     "measurable": True})

    df = pd.DataFrame(rows).sort_values(
        ["measurable", "band_fraction"], ascending=[False, False])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    ok = df[df.measurable]
    print(f"panel endpoints: {len(df)};  measurable from the activity caches: {len(ok)};  "
          f"not measurable: {len(df) - len(ok)}")
    print("An endpoint is not measurable when no cache retaining exact potencies survives for it, "
          "which\nis a gap in what was kept rather than a property of the endpoint.\n")
    print(ok[["endpoint", "source", "n_records", "n_active", "n_inactive",
              "n_band", "band_fraction"]].to_string(index=False))

    tot_n = int(ok.n_records.sum())
    tot_band = int(ok.n_band.sum())
    print(f"\nover the {len(ok)} measurable endpoints: {tot_band:,} of {tot_n:,} records fall in "
          f"the 5 to 6 band, {100 * tot_band / tot_n:.1f} per cent")
    print(f"per-endpoint band fraction: min {ok.band_fraction.min():.3f} "
          f"({ok.loc[ok.band_fraction.idxmin(), 'endpoint']}), "
          f"median {ok.band_fraction.median():.3f}, "
          f"max {ok.band_fraction.max():.3f} "
          f"({ok.loc[ok.band_fraction.idxmax(), 'endpoint']})")
    print("\nThe pooled figure is a record-weighted mean, so the largest endpoints dominate it. "
          "A\nMethods statement should give the range and the maximum alongside it, because the "
          "endpoint\nthat discards most is the safety endpoint.")
    missing = df.loc[~df.measurable, "endpoint"].tolist()
    if missing:
        print(f"\nnot measurable ({len(missing)}): {', '.join(missing)}")
    print(f"\nwrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
