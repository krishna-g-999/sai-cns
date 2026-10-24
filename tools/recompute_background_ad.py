"""Recompute the applicability-domain columns of specificity_background_comparison.csv.

WHY THIS EXISTS. That table carries the headline caveat of the whole server: specificity is 0.925
on the curated non-CNS library and 0.8817 on 15,000 unseen PubChem compounds, and the gap is a
population effect rather than a novelty effect. Its two similarity columns were measured with the
stereochemistry-blind fingerprint. When ad_fingerprint.INCLUDE_CHIRALITY was switched on, the
manuscripts, the thesis and the viva notes were all updated to the re-measured values, median 0.400
and 49.4 per cent below Tanimoto 0.40, but the table they cite as the source was not. A reader
checking the citation found 0.417 and 0.433.

The table had no generator anywhere in the tree and was outside tools/check_freshness.py, which is
why the prose could move without it. This script is that generator, so the next change to the
fingerprint updates the table instead of stranding it.

The firing columns are not recomputed and do not need to be: whether an endpoint fires does not
depend on the reference, which is why specificity is unchanged at 0.8817 while the similarity
columns move.

    python tools/recompute_background_ad.py [--write]
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BACKGROUND = ROOT / "results" / "tables" / "background_pubchem_other.csv"
TABLE = ROOT / "results" / "tables" / "specificity_background_comparison.csv"
ROW_KEY = "pubchem_other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="update the table in place")
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.DataStructs import BulkTanimotoSimilarity
    from sai_cns import ad_fingerprint as adfp

    RDLogger.DisableLog("rdApp.*")

    ref_smiles, ref_fps = adfp.load_reference()
    print(f"reference: {len(ref_fps):,} compounds, spec {adfp.SPEC}")

    gen = adfp.generator()
    rows = list(csv.DictReader(BACKGROUND.open(newline="", encoding="utf-8")))
    print(f"background: {len(rows):,} compounds from {BACKGROUND.name}")

    sims, unparsed = [], 0
    for i, r in enumerate(rows, 1):
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            unparsed += 1
            continue
        sims.append(max(BulkTanimotoSimilarity(gen.GetFingerprint(mol), ref_fps)))
        if i % 2500 == 0:
            print(f"  {i:,} / {len(rows):,}")

    n = len(sims)
    median = statistics.median(sims)
    below = sum(1 for s in sims if s < 0.40) / n
    at_one = sum(1 for s in sims if s >= 1.0) / n
    print(f"\nscored {n:,} compounds ({unparsed} unparsable)")
    print(f"  median max Tanimoto   {median:.4f}")
    print(f"  fraction below 0.40   {below:.4f}")
    print(f"  fraction at 1.00      {at_one:.4f}")

    old = list(csv.DictReader(TABLE.open(newline="", encoding="utf-8")))
    target = [r for r in old if ROW_KEY in r["background"]]
    if len(target) != 1:
        print(f"\nexpected one row matching {ROW_KEY!r}, found {len(target)}; not writing")
        return 1
    row = target[0]
    print(f"\n  table currently says  median {row['median_ad_tanimoto']}, "
          f"below 0.40 {row['frac_ad_below_0.40']}, at 1.00 {row['frac_ad_at_1.00']}")

    if not args.write:
        print("\n(dry run; pass --write to update the table)")
        return 0

    row["median_ad_tanimoto"] = f"{median:.4f}"
    row["frac_ad_below_0.40"] = f"{below:.4f}"
    row["frac_ad_at_1.00"] = f"{at_one:.4f}"
    row["note"] = (
        "non-bioactive, non-drug, non-natural-product, MW 150-600, absent from the training index "
        "and from the stereochemistry-stripped applicability-domain reference, which is the "
        "stricter absence criterion. The similarity columns are measured against the "
        "chirality-aware reference (ad_fingerprint.SPEC) by tools/recompute_background_ad.py; the "
        "firing columns are unaffected by the reference and were not recomputed.")
    with TABLE.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(old[0].keys()))
        w.writeheader()
        w.writerows(old)
    print(f"\nupdated {TABLE.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
