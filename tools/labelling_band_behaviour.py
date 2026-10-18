"""What does the server actually do with a compound whose potency it refused to label?

The labelling rule calls pChEMBL at or above 6.0 active and at or below 5.0 inactive, and discards
the band between as undecidable. The gap is defensible: a single assay rarely separates 300 nM from
3 uM reliably, and forcing those compounds into a class would teach the model a distinction the
data does not support.

But the rule governs TRAINING, not QUERY. A user submitting a band compound still receives a
score, produced by a model that was never shown one, and nothing in the cross-validation figures
describes that case: the reported AUROC separates clear actives from clear inactives, which is the
easy half of the problem, and the hardest compounds are absent from it by construction.

This script measures the omitted case directly. It scores held-out compounds whose measured
potency falls in the discarded band through the deployed pipeline and reports, per half-log slice,
the mean probability, the fraction above 0.5, and the mean enrichment over the endpoint's base
rate. Enrichment is the quantity the interface actually shows for these endpoints, so it is the
quantity a user would read.

Two outcomes are worth distinguishing, and both are informative. If the slices are ordered by
measured potency, the model carries graded information across a band it never trained on, which is
more than the labelling rule entitles anyone to expect. If the upper slice nonetheless returns a
positive enrichment, then compounds few projects would call hits read as mildly promising, and
that is a reading users must be warned against rather than a defect to be hidden.

Run:
    python tools/labelling_band_behaviour.py
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))

HOLD = ROOT / "results" / "tables" / "chembl_holdout_set.csv"
OUT = ROOT / "results" / "tables" / "labelling_band_behaviour.csv"
SLICES = [(5.0, 5.5), (5.5, 6.0), (6.0, 99.0)]
MIN_N = 20


def main() -> int:
    warnings.filterwarnings("ignore")
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    import app as A

    rows = list(csv.DictReader(HOLD.open(encoding="utf-8")))
    models = A.load_models()
    out = []

    endpoints = sorted({r["endpoint"] for r in rows})
    print(f"  {'endpoint':9s} {'slice':11s} {'n':>5s} {'mean p':>8s} "
          f"{'frac>0.5':>9s} {'enrichment':>11s}")
    print("  " + "-" * 60)
    for ep in endpoints:
        br = A.base_rate(ep)
        if br is None:
            continue
        sub = [r for r in rows if r["endpoint"] == ep]
        preds = A.predict_all_batch([r["smiles"] for r in sub], models)
        pch = np.array([float(r["pchembl"]) for r in sub])
        p = np.array([(d or {}).get("targets", {}).get(ep, np.nan) for d in preds], dtype=float)
        ok = ~np.isnan(p)
        if ok.sum() == 0:
            continue
        printed = False
        for lo, hi in SLICES:
            m = ok & (pch >= lo) & (pch < hi)
            if m.sum() < MIN_N:
                continue
            q = p[m]
            e = float(np.mean([A.enrichment(ep, float(v)) for v in q]))
            lab = f"{lo:.1f}-{hi:.1f}" if hi < 90 else f">={lo:.1f}"
            print(f"  {ep if not printed else '':9s} {lab:11s} {int(m.sum()):>5,} "
                  f"{q.mean():>8.3f} {(q > 0.5).mean() * 100:>8.1f}% {e:>11.3f}")
            printed = True
            out.append({"endpoint": ep, "slice": lab, "base_rate": round(float(br), 4),
                        "n": int(m.sum()), "mean_probability": round(float(q.mean()), 4),
                        "frac_above_half": round(float((q > 0.5).mean()), 4),
                        "mean_enrichment": round(e, 4)})

    if out:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader(); w.writerows(out)
        print(f"\n  wrote {OUT.relative_to(ROOT)}")
        print("  Ordered slices mean graded information survives the gap. A positive enrichment")
        print("  in the upper slice means weak compounds read as mildly promising.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
