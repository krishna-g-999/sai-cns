"""Does specificity actually degrade with distance, or does the background population explain it?

TWO MEASUREMENTS THAT LOOK LIKE THEY DISAGREE. The published specificity is 0.925, measured on a
1,000-compound non-CNS library in which every compound matches the applicability-domain reference
at a Tanimoto of exactly 1.00. On 15,000 unseen compounds drawn from PubChem entries that are
neither bioactive, drug nor natural product, specificity is 0.8817. Read carelessly, that pair says
specificity degrades on unfamiliar chemistry.

Falsification hypothesis H4 says the opposite. Scoring 600 random PubChem structures and
stratifying them by distance, the false-positive rate FELL as distance rose, from 0.1200 in domain
to 0.0164 beyond Tanimoto 0.30, the reverse of degradation, though on a single false positive in
the distant stratum and not separable from its comparator.

Both cannot be used as stated, and the reason they appear to conflict is that they are not the same
comparison. The 0.925 against 0.8817 contrast is BETWEEN two differently constituted libraries: a
curated non-CNS drug set against filtered PubChem entries. Novelty and chemical population move
together, so the gap cannot be assigned to either. H4 is a WITHIN-population comparison, which is
the right design, run at a sample size too small to conclude much.

This script runs H4's design at the larger sample size. It scores the 15,000-compound background
through the deployed pipeline and reports the fire rate by distance stratum, using H4's own strata
so the two tables can be read against each other. Within one population, distance is the only thing
varying, so the trend here is the answer to the question both measurements were reaching for.

A rising fire rate with distance would confirm degradation and refute H4 at a sample size H4 could
not reach. A falling or flat one would mean the 0.925 against 0.8817 gap is a population effect and
must not be described as a distance effect.

Run:
    python tools/background_specificity_by_distance.py [-n N]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))

BACKGROUND = ROOT / "results" / "tables" / "background_pubchem_other.csv"
OUT = ROOT / "results" / "tables" / "background_specificity_by_distance.csv"

# H4's strata, so the two tables are directly comparable.
STRATA = [("distant, T < 0.30", 0.0, 0.30),
          ("near domain, 0.30 to 0.50", 0.30, 0.50),
          ("in domain, T >= 0.50", 0.50, 1.01)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=0, help="score only the first N (0 = all)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    import app

    rows = list(csv.DictReader(BACKGROUND.open(encoding="utf-8")))
    if args.n:
        rows = rows[:args.n]
    smiles = [r["smiles"] for r in rows]
    print(f"  scoring {len(smiles):,} background compounds through the deployed pipeline")

    models = app.load_models()
    t0 = time.time()
    preds = app.predict_all_batch(smiles, models)
    print(f"  panel pass {time.time() - t0:.0f}s")

    t1 = time.time()
    recs = []
    for s, r in zip(smiles, preds):
        if r is None:
            continue
        dz = app.disease_scores(r)[2]
        top = dz[0] if dz else None
        fired = bool(top and top["gated"] >= app.MIN_ACTIONABLE_SCORE)
        try:
            sim = app.assess_domain(s).get("max_sim")
        except Exception:
            sim = None
        if sim is None:
            continue
        recs.append((float(sim), fired))
    print(f"  gating and domain {time.time() - t1:.0f}s; usable {len(recs):,}")

    out = []
    print(f"\n  {'stratum':28s} {'n':>6s} {'fired':>6s} {'FPR':>7s} {'95% Wilson':>18s}")
    print("  " + "-" * 70)
    for name, lo, hi in STRATA:
        sel = [f for s, f in recs if lo <= s < hi]
        if not sel:
            continue
        k, n = sum(sel), len(sel)
        a, b = wilson(k, n)
        print(f"  {name:28s} {n:>6,} {k:>6,} {k / n:>7.4f}   {a:.4f} to {b:.4f}")
        out.append({"stratum": name, "n": n, "fired": k, "fpr": round(k / n, 4),
                    "ci95_low": round(a, 4), "ci95_high": round(b, 4)})
    k, n = sum(f for _, f in recs), len(recs)
    a, b = wilson(k, n)
    print(f"  {'all':28s} {n:>6,} {k:>6,} {k / n:>7.4f}   {a:.4f} to {b:.4f}")
    out.append({"stratum": "all", "n": n, "fired": k, "fpr": round(k / n, 4),
                "ci95_low": round(a, 4), "ci95_high": round(b, 4)})

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    print(f"\n  wrote {OUT.relative_to(ROOT)}")
    print("  Rising FPR with distance confirms degradation. Falling or flat means the")
    print("  0.925 against 0.8817 gap is a population effect, not a distance effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
