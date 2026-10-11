"""Large-scale specificity test on ~1000 compounds with no known engagement of any modelled target.

Sensitivity is easy to inflate: a tool that fires on everything looks impressive on known drugs. The
complementary question is how often the tool raises a disease signal for a compound that has no
business producing one. This test answers it at a scale that supports a usable confidence interval,
which the eight-compound external set could not.

Construction of the negative set, in order:
  1. Start from the measured compound library (the same chemistry the models were trained on, so
     these are drug-like molecules rather than trivial rejects).
  2. Remove every compound that is recorded ACTIVE at any of the modelled targets, using each
     endpoint's own measured labels. What remains has no measured activity at anything the
     knowledge graph can route to a disease.
  3. Remove compounds that appear in the antioxidant table, since the neuroprotection axis is driven
     by that endpoint.
  4. Sample the target number at random with a fixed seed.

These compounds are "presumed inactive" rather than proven inactive: absence of a measured activity
is weaker evidence than a measured inactive. The false-positive rate reported here is therefore an
upper bound on the true rate, and is labelled as such.

Outputs:
  results/tables/noncns_specificity_predictions.csv   per-compound predictions
  results/tables/noncns_specificity_summary.csv       rates with Wilson intervals
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
TAB = ROOT / "results" / "tables"
N_SAMPLE = 1000
rng = np.random.default_rng(31415)


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m else None


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    import app

    actives, pool = set(), set()
    for f in glob.glob(str(ROOT / "data" / "endpoints" / "*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "label" in df.columns:
            for s in df.loc[df["label"] == 1, "smiles"].astype(str):
                c = canon(s)
                if c:
                    actives.add(c)
        if "pchembl" in df.columns:
            p = pd.to_numeric(df["pchembl"], errors="coerce")
            for s in df.loc[p >= 6, "smiles"].astype(str):
                c = canon(s)
                if c:
                    actives.add(c)
    for f in glob.glob(str(ROOT / "data" / "endpoints_reg" / "*.csv")):
        try:
            for s in pd.read_csv(f, usecols=["smiles"])["smiles"].astype(str):
                c = canon(s)
                if c:
                    actives.add(c)
        except Exception:
            continue
    for f in glob.glob(str(ROOT / "data" / "adme" / "*.csv")):
        try:
            for s in pd.read_csv(f, usecols=["smiles"])["smiles"].astype(str):
                c = canon(s)
                if c:
                    pool.add(c)
        except Exception:
            continue

    cand = sorted(pool - actives)
    print(f"library pool {len(pool):,}; known actives at a modelled target {len(actives):,}; "
          f"eligible negatives {len(cand):,}", flush=True)
    sel = [cand[i] for i in rng.choice(len(cand), size=min(N_SAMPLE, len(cand)), replace=False)]

    models = app.load_models()
    rows = []
    for i, smi in enumerate(sel):
        r = app.predict_all(smi, models)
        if r is None:
            continue
        bbb, neuro, dz = app.disease_scores(r)
        ranked = sorted(dz, key=lambda d: -d["gated"])
        top, score = ranked[0]["disease"], ranked[0]["gated"]
        fired = score >= app.MIN_ACTIONABLE_SCORE
        ad = app.assess_domain(smi)
        rows.append({"smiles": smi, "top_disease": top if fired else "NONE",
                     "top_score": round(score, 3), "fired": int(fired),
                     "n_diseases_fired": int(sum(d["gated"] >= app.MIN_ACTIONABLE_SCORE for d in dz)),
                     "bbb": round(bbb, 3), "herg": round(r["targets"]["hERG"], 3),
                     "ad_max_tanimoto": round(ad["max_sim"], 3) if ad else None})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sel)} scored, running false-positive rate "
                  f"{np.mean([x['fired'] for x in rows]):.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "noncns_specificity_predictions.csv", index=False)

    k, n = int(df.fired.sum()), len(df)
    lo, hi = wilson(k, n)
    summary = [{"metric": "False-positive rate (any actionable disease call)", "k": k, "n": n,
                "estimate": round(k / n, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                "note": "upper bound: compounds are presumed inactive, not proven inactive"}]
    ks, _ = int((df.fired == 0).sum()), None
    lo2, hi2 = wilson(ks, n)
    summary.append({"metric": "Specificity (no disease call)", "k": ks, "n": n,
                    "estimate": round(ks / n, 4), "ci95_low": round(lo2, 4),
                    "ci95_high": round(hi2, 4), "note": ""})
    for tier, sub in [("in domain (T>=0.5)", df[df.ad_max_tanimoto >= 0.5]),
                      ("near domain (0.3-0.5)", df[(df.ad_max_tanimoto >= 0.3) & (df.ad_max_tanimoto < 0.5)]),
                      ("out of domain (T<0.3)", df[df.ad_max_tanimoto < 0.3])]:
        if len(sub):
            kk = int(sub.fired.sum())
            l, h_ = wilson(kk, len(sub))
            summary.append({"metric": f"False-positive rate, {tier}", "k": kk, "n": len(sub),
                            "estimate": round(kk / len(sub), 4), "ci95_low": round(l, 4),
                            "ci95_high": round(h_, 4), "note": ""})
    fired = df[df.fired == 1]
    for d, sub in fired.groupby("top_disease"):
        summary.append({"metric": f"False positives assigned to: {d}", "k": len(sub), "n": n,
                        "estimate": round(len(sub) / n, 4), "ci95_low": "", "ci95_high": "",
                        "note": f"median score {sub.top_score.median():.2f}"})
    out = pd.DataFrame(summary)
    out.to_csv(TAB / "noncns_specificity_summary.csv", index=False)
    pd.set_option("display.width", 170)
    print()
    print(out.to_string(index=False))
    print(f"\nwrote {TAB / 'noncns_specificity_predictions.csv'}")


if __name__ == "__main__":
    main()
