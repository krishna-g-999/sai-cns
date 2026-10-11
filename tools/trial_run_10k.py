"""Score a diverse, genuinely held-out set of compounds and record every endpoint.

Purpose. The server's published validation is per-endpoint and retrospective. This is a single
large forward run over chemistry the panel has never seen, recording every value the tool produces
for every compound, so that its behaviour at scale can be inspected rather than inferred.

What this can and cannot measure. The compounds here carry no activity labels, so this run does
NOT measure accuracy. Accuracy was measured separately, on the only labelled held-out set that
exists (247 compounds, results/tables/trial_bbb_heldout_scored.csv). What this run measures is
behaviour: throughput, how often the server makes a call, how far the chemistry sits from training,
and whether the panel behaves differently across compound classes.

Exclusion. Every compound is standardised with the project's own standardiser and its InChIKey
checked against the full set of training keys built from data/endpoints/. Anything the panel was
fitted on is dropped before scoring, so nothing here is a compound the model has seen.

Diversity. Compounds are drawn from two populations with very different chemistry:
approved/investigational drugs from DrugBank, and natural products from COCONUT. The natural
products are deliberately included because the manuscript states they are largely outside the
training library, so this run should show that, and if it does not, that is a finding.

Run:  python tools/trial_run_10k.py --n 10000
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

SOURCES = [
    ("drug", ROOT / "data" / "external" / "processed" / "external_drugs.csv"),
    ("natural_product", ROOT / "data" / "external" / "processed" / "flavonoid_panel.csv"),
]
KEYS = ROOT / "results" / "tables" / "_train_keys.pkl"
OUT = ROOT / "results" / "tables" / "trial_10k_predictions.csv"
SUMMARY = ROOT / "results" / "tables" / "trial_10k_summary.csv"


def training_keys() -> set[str]:
    """The InChIKeys the panel was fitted on, cached because building it parses ~170k structures."""
    if KEYS.exists():
        keys = pickle.loads(KEYS.read_bytes())
        print(f"  training keys (cached): {len(keys):,}")
        return keys
    from data.build_compound_library import standardise
    import glob
    keys = set()
    for f in sorted(glob.glob(str(ROOT / "data" / "endpoints" / "*.csv"))):
        d = pd.read_csv(f, usecols=["smiles"], low_memory=False)
        for s in d.smiles.dropna().unique():
            try:
                r = standardise(str(s))
                if isinstance(r, tuple) and r[1]:
                    keys.add(r[1])
            except Exception:
                pass
    KEYS.write_bytes(pickle.dumps(keys))
    print(f"  training keys (built): {len(keys):,}")
    return keys


def build_cohort(n: int, seed: int = 42) -> pd.DataFrame:
    train = training_keys()
    rng = np.random.default_rng(seed)
    frames = []
    per = n // len(SOURCES)
    for kind, path in SOURCES:
        if not path.exists():
            print(f"  MISSING {path.name}, skipping {kind}")
            continue
        d = pd.read_csv(path, usecols=["inchikey", "canonical_smiles", "source"], low_memory=False)
        d = d.dropna(subset=["inchikey", "canonical_smiles"]).drop_duplicates("inchikey")
        before = len(d)
        d = d[~d.inchikey.isin(train)]
        print(f"  {kind:16s} {before:7,} available, {len(d):7,} after removing training overlap")
        take = min(per, len(d))
        d = d.iloc[rng.permutation(len(d))[:take]].copy()
        d["compound_class"] = kind
        frames.append(d)
    cohort = pd.concat(frames, ignore_index=True)
    print(f"  cohort: {len(cohort):,} compounds, "
          f"{cohort.compound_class.value_counts().to_dict()}")
    return cohort


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()

    print("building the cohort")
    cohort = build_cohort(args.n)

    print("\nloading the deployed panel")
    import app
    models = app.load_models()

    print(f"\nscoring {len(cohort):,} compounds through the deployed pipeline")
    t0 = time.time()
    preds = app.predict_all_batch(cohort.canonical_smiles.tolist(), models)
    t_score = time.time() - t0
    print(f"  scored in {t_score:.0f}s  ->  {t_score/len(cohort)*1000:.1f} ms per compound")

    print("gating, disease scoring and applicability domain")
    rows, t1 = [], time.time()
    for (rec, r) in zip(cohort.itertuples(), preds):
        row = {"inchikey": rec.inchikey, "smiles": rec.canonical_smiles,
               "compound_class": rec.compound_class, "source": rec.source}
        if r is None:
            row["status"] = "not_featurizable"
            rows.append(row); continue
        row["status"] = "ok"
        bbb, neuro, dz = app.disease_scores(r)
        row["bbb"] = float(bbb)
        # every endpoint the tool produces, one column each
        for grp in ("targets", "receptors", "receptor_binder", "adme", "descriptors"):
            for k, v in r[grp].items():
                row[f"{grp}:{k}"] = v
        row["antioxidant"] = r.get("antioxidant")
        for d in dz:
            row[f"disease:{d['disease']}"] = d["gated"]
        top = dz[0] if dz else None
        row["top_disease"] = top["disease"] if top else None
        row["top_disease_score"] = float(top["gated"]) if top else 0.0
        row["fired"] = bool(top and top["gated"] >= app.MIN_ACTIONABLE_SCORE)
        try:
            ad = app.assess_domain(rec.canonical_smiles)
            row["ad_max_tanimoto"] = ad.get("max_sim")
            row["ad_n_reference"] = ad.get("n_ref")
        except Exception:
            pass
        rows.append(row)
    t_post = time.time() - t1
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"  post-processing {t_post:.0f}s; wrote {OUT.name} "
          f"({len(out):,} rows x {out.shape[1]} columns)")

    ok = out[out.status == "ok"]
    summ = []

    def add(metric, value, note=""):
        summ.append({"metric": metric, "value": value, "note": note})

    add("compounds_scored", len(ok))
    add("not_featurizable", int((out.status != "ok").sum()))
    add("seconds_total", round(t_score + t_post, 1))
    add("ms_per_compound_scoring", round(t_score / max(len(cohort), 1) * 1000, 1),
        "vectorised panel pass only")
    add("projected_300_batch_seconds", round((t_score + t_post) / max(len(cohort), 1) * 300, 1))
    for cls, g in ok.groupby("compound_class"):
        add(f"n_{cls}", len(g))
        add(f"fire_rate_{cls}", round(float(g.fired.mean()), 4),
            "fraction returning an actionable disease call")
        add(f"median_bbb_{cls}", round(float(g.bbb.median()), 4))
        if "ad_max_tanimoto" in g:
            add(f"median_ad_tanimoto_{cls}", round(float(g.ad_max_tanimoto.median()), 4),
                "distance from training chemistry")
            add(f"frac_below_tanimoto_0.40_{cls}",
                round(float((g.ad_max_tanimoto < 0.40).mean()), 4),
                "recall is near 0.16 in this band")
    pd.DataFrame(summ).to_csv(SUMMARY, index=False)
    print(f"  wrote {SUMMARY.name}")
    print()
    print(pd.DataFrame(summ).to_string(index=False))


if __name__ == "__main__":
    main()
