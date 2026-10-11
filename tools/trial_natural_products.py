"""Test the manuscript's natural-product claim against the chemistry it actually names.

The manuscript states: "Terpenoid and steroidal natural products are largely outside the training
library, whose median fraction-sp3 is 0.34, and are flagged as outside the domain."

An earlier run appeared to contradict this, showing natural products sitting CLOSER to training
chemistry than approved drugs. That run drew its natural products from flavonoid_panel.csv, and
every one of the 5,000 was a flavonoid core: aromatic, low-sp3, and well represented in the
antioxidant training set. It did not test the claim, because flavonoids are not what the claim is
about.

This run stratifies COCONUT by saturation, using the manuscript's own metric. Terpenoids and
steroids are highly saturated, so fraction_csp3 separates them from the aromatic natural products
the panel has seen. Four strata, equal size:

    flavonoid                 the earlier cohort, kept as the comparison
    saturated_fsp3_0.70+      terpenoid and steroid space
    saturated_fsp3_0.90+      almost fully saturated; the extreme of that space
    aromatic_fsp3_<0.30       aromatic natural products, nearer the training library

If the claim holds, applicability-domain similarity should fall as saturation rises, and the
saturated strata should sit well below the Tanimoto 0.40 band where the manuscript reports recall
near 0.16. If it does not hold, that is a finding and the manuscript sentence needs changing.

Run:  python tools/trial_natural_products.py --per 2500
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))

SRC = ROOT / "data" / "external" / "processed" / "natural_products_coverage.csv"
KEYS = ROOT / "results" / "tables" / "_train_keys.pkl"
OUT = ROOT / "results" / "tables" / "trial_np_predictions.csv"
SUMMARY = ROOT / "results" / "tables" / "trial_np_summary.csv"


def strata(per: int, seed: int = 42) -> pd.DataFrame:
    train = pickle.loads(KEYS.read_bytes())
    d = pd.read_csv(SRC, low_memory=False)
    d = d.dropna(subset=["inchikey", "canonical_smiles", "fraction_csp3"])
    d = d.drop_duplicates("inchikey")
    before = len(d)
    d = d[~d.inchikey.isin(train)]
    print(f"  COCONUT {before:,} -> {len(d):,} after removing training overlap "
          f"({len(train):,} training keys)")
    flav = d.is_flavonoid_core == True          # noqa: E712
    defs = {
        "flavonoid": d[flav],
        "saturated_fsp3_0.70+": d[(~flav) & (d.fraction_csp3 >= 0.70)],
        "saturated_fsp3_0.90+": d[(~flav) & (d.fraction_csp3 >= 0.90)],
        "aromatic_fsp3_<0.30": d[(~flav) & (d.fraction_csp3 < 0.30)],
    }
    rng = np.random.default_rng(seed)
    out = []
    for name, g in defs.items():
        take = min(per, len(g))
        s = g.iloc[rng.permutation(len(g))[:take]].copy()
        s["stratum"] = name
        out.append(s)
        print(f"  {name:22s} {take:6,} sampled of {len(g):7,} "
              f"(median fsp3 {s.fraction_csp3.median():.3f})")
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=2500)
    args = ap.parse_args()

    print("building strata")
    coh = strata(args.per)

    print("\nloading the deployed panel")
    import app
    models = app.load_models()

    print(f"\nscoring {len(coh):,} compounds")
    t0 = time.time()
    preds = app.predict_all_batch(coh.canonical_smiles.tolist(), models)
    print(f"  {time.time()-t0:.0f}s  ({(time.time()-t0)/len(coh)*1000:.1f} ms per compound)")

    print("applicability domain and gating")
    rows = []
    for rec, r in zip(coh.itertuples(), preds):
        row = {"inchikey": rec.inchikey, "smiles": rec.canonical_smiles,
               "stratum": rec.stratum, "fraction_csp3": rec.fraction_csp3,
               "mw": rec.mw, "clogp": rec.clogp}
        if r is None:
            row["status"] = "not_featurizable"; rows.append(row); continue
        row["status"] = "ok"
        bbb, neuro, dz = app.disease_scores(r)
        row["bbb"] = float(bbb)
        top = dz[0] if dz else None
        row["top_disease"] = top["disease"] if top else None
        row["top_disease_score"] = float(top["gated"]) if top else 0.0
        row["fired"] = bool(top and top["gated"] >= app.MIN_ACTIONABLE_SCORE)
        try:
            ad = app.assess_domain(rec.canonical_smiles)
            row["ad_max_tanimoto"] = ad.get("max_sim")
        except Exception:
            pass
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"  wrote {OUT.name} ({len(out):,} rows)")

    ok = out[out.status == "ok"]
    summ = []
    for name, g in ok.groupby("stratum"):
        a = g.ad_max_tanimoto.dropna()
        summ.append({
            "stratum": name, "n": len(g),
            "median_fraction_csp3": round(float(g.fraction_csp3.median()), 3),
            "median_ad_tanimoto": round(float(a.median()), 4),
            "frac_below_0.40": round(float((a < 0.40).mean()), 4),
            "frac_below_0.30": round(float((a < 0.30).mean()), 4),
            "median_bbb": round(float(g.bbb.median()), 4),
            "fire_rate": round(float(g.fired.mean()), 4),
        })
    s = pd.DataFrame(summ).sort_values("median_fraction_csp3")
    s.to_csv(SUMMARY, index=False)
    print()
    print(s.to_string(index=False))
    print()
    print("Reading: if the manuscript is right, median_ad_tanimoto should FALL and")
    print("frac_below_0.40 should RISE as median_fraction_csp3 rises.")


if __name__ == "__main__":
    main()
