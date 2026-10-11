"""Large-scale specificity test, efficient construction of the negative set.

Same question as noncns_specificity.py: how often does the tool raise a disease signal for a
compound that should produce none. The construction is identical in intent but avoids canonicalising
the whole corpus, which is the step that made the first version impractical.

Where the negatives come from, and why it matters. They used to be sampled from
models_rf/ad_reference.pkl, which is the library the applicability domain measures similarity
against. Every compound drawn from it is therefore its own nearest neighbour: all 1,000 sampled
compounds had a maximum Tanimoto of exactly 1.000, the in-domain stratum was numerically identical
to the unstratified row, and the near- and out-of-domain strata were empty and silently omitted. A
specificity measured that way describes compounds the model has already seen and cannot support a
claim about novel chemistry.

They are now drawn from the approved and experimental drugs that are absent from that reference
(models.pools.external_negatives), so the applicability-domain stratification means something and
the rate is about chemistry the model was not fitted to.

The exclusion set is built from the raw SMILES strings recorded as active at any modelled target.
The sampled compounds are canonicalised and re-checked against a canonicalised copy of the whole
exclusion set, and any that slip through are dropped and counted.

Compounds are presumed inactive rather than proven inactive, so the reported false-positive rate is
an upper bound on the true rate.

Outputs:
  results/tables/noncns_specificity_predictions.csv
  results/tables/noncns_specificity_summary.csv
"""
from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path

# app.py is a Streamlit script, and importing it outside `streamlit run` makes every session-state
# read log "missing ScriptRunContext" at WARNING. This scores a thousand compounds against roughly
# seventy models each, so that warning is emitted often enough to matter: an earlier run of this
# script wrote a 6.9 GB log and died when the disk filled. Silence the framework's own chatter and
# keep this script's output.
logging.getLogger("streamlit").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from models.pools import external_negatives  # noqa: E402

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

    # raw-string exclusion set: anything active at any modelled target, plus the antioxidant table
    active_raw = set()
    for f in glob.glob(str(ROOT / "data" / "endpoints" / "*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        s = df["smiles"].astype(str)
        keep = pd.Series(False, index=df.index)
        if "label" in df.columns:
            keep |= (df["label"] == 1)
        if "pchembl" in df.columns:
            keep |= (pd.to_numeric(df["pchembl"], errors="coerce") >= 6)
        active_raw.update(s[keep].tolist())
    for f in glob.glob(str(ROOT / "data" / "endpoints_reg" / "*.csv")):
        try:
            active_raw.update(pd.read_csv(f, usecols=["smiles"])["smiles"].astype(str).tolist())
        except Exception:
            continue
    print(f"exclusion set (active at a modelled target): {len(active_raw):,}", flush=True)

    lib = external_negatives()
    cand = [s for s in lib if s not in active_raw]
    print(f"external library {len(lib):,} (disjoint from the applicability-domain reference); "
          f"eligible negatives {len(cand):,}", flush=True)
    if len(cand) < N_SAMPLE:
        raise SystemExit(
            f"only {len(cand)} eligible negatives for a sample of {N_SAMPLE}. Widen the external "
            "pool rather than falling back to the reference library, which cannot support a "
            "specificity claim about novel chemistry."
        )

    take = rng.choice(len(cand), size=min(N_SAMPLE * 2, len(cand)), replace=False)
    # The whole exclusion set, canonicalised. Slicing an unordered set to 40,000, as this once did,
    # both truncated the check and made it depend on iteration order.
    active_canon = {c for c in (canon(s) for s in active_raw) if c}
    sel, slipped = [], 0
    for i in take:
        c = canon(cand[i])
        if c is None:
            continue
        if c in active_canon:
            slipped += 1
            continue
        sel.append(c)
        if len(sel) >= N_SAMPLE:
            break
    print(f"sampled {len(sel)} negatives ({slipped} dropped by the canonical re-check)", flush=True)

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
            print(f"  scored {len(rows)}, running false-positive rate "
                  f"{np.mean([x['fired'] for x in rows]):.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "noncns_specificity_predictions.csv", index=False)

    k, n = int(df.fired.sum()), len(df)
    lo, hi = wilson(k, n)
    lo2, hi2 = wilson(n - k, n)
    summary = [
        {"metric": "False-positive rate (any actionable disease call)", "k": k, "n": n,
         "estimate": round(k / n, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
         "note": "upper bound: compounds are presumed inactive, not proven inactive"},
        {"metric": "Specificity (no disease call)", "k": n - k, "n": n,
         "estimate": round((n - k) / n, 4), "ci95_low": round(lo2, 4), "ci95_high": round(hi2, 4),
         "note": ""}]
    for tier, sub in [("in domain (T>=0.5)", df[df.ad_max_tanimoto >= 0.5]),
                      ("near domain (0.3-0.5)", df[(df.ad_max_tanimoto >= 0.3) & (df.ad_max_tanimoto < 0.5)]),
                      ("out of domain (T<0.3)", df[df.ad_max_tanimoto < 0.3])]:
        if len(sub):
            kk = int(sub.fired.sum())
            l, h_ = wilson(kk, len(sub))
            summary.append({"metric": f"False-positive rate, {tier}", "k": kk, "n": len(sub),
                            "estimate": round(kk / len(sub), 4), "ci95_low": round(l, 4),
                            "ci95_high": round(h_, 4), "note": ""})
    for d, sub in df[df.fired == 1].groupby("top_disease"):
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
