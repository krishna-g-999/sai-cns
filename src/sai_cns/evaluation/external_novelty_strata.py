"""Does prospective accuracy survive when the test compound does not resemble the training set?

A time split is the right way to build an external set from retrospective data, and it flatters any
field that publishes analogue series. Medicinal chemistry publishes little else. If the compounds
appearing after a cutoff are close relatives of those before it, a high prospective AUROC shows that
the panel interpolates within a series, which is worth something but is not the claim a server makes
to someone submitting a new scaffold.

The single number therefore has to be broken open. Every test compound in the prospective and
cross-provenance runs carries its maximum Tanimoto similarity to the training actives of its own
split. Binning on that distance and recomputing performance within each bin answers the question the
aggregate hides: how far from the training set can a compound sit before the panel stops working?

Both classes are binned, so each stratum compares actives against measured inactives at a comparable
distance from training. Stratifying only the actives would produce bins whose positives are novel and
whose negatives are not, and the AUROC of such a bin would measure that mismatch instead of the
model. Pooling across endpoints is necessary for power, and it is a real limitation: an endpoint with
many novel test compounds contributes more to the far bins than one with few.

The bins are the conventional medicinal-chemistry reading of an ECFP Tanimoto, and were fixed before
the numbers were computed:

  below 0.40   a different chemotype; the applicability-domain flag would call this out of domain
  0.40 - 0.55  a related series, recognisably not the same one
  0.55 - 0.70  the same series, different substitution
  0.70 and up  a close analogue of something already in the training set

Output: results/tables/external_novelty_strata.csv

Run:  python src/sai_cns/evaluation/external_novelty_strata.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
TAB = ROOT / "results" / "tables"

BINS = [(0.00, 0.40, "below 0.40 (different chemotype)"),
        (0.40, 0.55, "0.40 to 0.55 (related series)"),
        (0.55, 0.70, "0.55 to 0.70 (same series)"),
        (0.70, 1.01, "0.70 and above (close analogue)")]
MIN_ACTIVES = 30
MIN_NEGATIVES = 30


def _strata(d: pd.DataFrame, source: str, split: str) -> list[dict]:
    out = []
    for lo, hi, label in BINS:
        b = d[(d.max_tanimoto_to_training >= lo) & (d.max_tanimoto_to_training < hi)]
        act, ina = b[b.measured == 1], b[b.measured == 0]
        if len(act) < MIN_ACTIVES:
            out.append({"source": source, "split": split, "novelty_band": label,
                        "n_actives": len(act), "n_measured_inactives": len(ina),
                        "auroc": None, "recall_at_threshold": None,
                        "fpr_at_threshold": None, "n_endpoints": int(b.endpoint.nunique()),
                        "note": "too few actives in this band to report"})
            continue
        auroc = None
        if len(ina) >= MIN_NEGATIVES:
            auroc = round(float(roc_auc_score(b.measured.to_numpy(),
                                              b.probability.to_numpy())), 4)
        out.append({
            "source": source, "split": split, "novelty_band": label,
            "n_actives": len(act), "n_measured_inactives": len(ina),
            "auroc": auroc,
            "recall_at_threshold": round(float(act.called.mean()), 4),
            "fpr_at_threshold": (round(float(ina.called.mean()), 4) if len(ina) else None),
            "n_endpoints": int(b.endpoint.nunique()),
            "note": "" if auroc is not None else "no AUROC: too few measured inactives in band"})
    return out


def main() -> None:
    rows = []
    p = TAB / "external_prospective_compounds.csv"
    if p.exists():
        d = pd.read_csv(p)
        for split in sorted(d.split.unique()):
            rows += _strata(d[d.split == split], "prospective", split)
    c = TAB / "external_cross_source_compounds.csv"
    if c.exists():
        d = pd.read_csv(c)
        rows += _strata(d, "cross-provenance", "cross_source")

    if not rows:
        print("no compound-level results found; run external_prospective.py first")
        return
    r = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    r.to_csv(TAB / "external_novelty_strata.csv", index=False)

    pd.set_option("display.width", 200)
    print(r.to_string(index=False))

    t = r[(r.source == "prospective") & (r.split == "time") & r.recall_at_threshold.notna()]
    if len(t) >= 2:
        far, near = t.iloc[0], t.iloc[-1]
        print(f"\n  prospective recall falls from {near.recall_at_threshold:.3f} for close "
              f"analogues to {far.recall_at_threshold:.3f} for different chemotypes")
        a = t[t.auroc.notna()]
        if len(a) >= 2:
            print(f"  prospective AUROC falls from {a.iloc[-1].auroc:.3f} to {a.iloc[0].auroc:.3f} "
                  f"across the same range")
    print(f"\nwrote {(TAB / 'external_novelty_strata.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
