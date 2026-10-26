"""Compute a second, high-precision threshold set for low-prevalence screening.

The integrity audit showed that the deployed operating point, sensitivity 0.90 at a background
false-positive rate of 0.125, gives a precision of only 7% when real actives are around one in a
hundred: roughly fourteen wasted experiments per true hit. That is the regime in which a diverse
library is triaged, and it is precisely where a tool meant to spare animals and reagents would
instead consume them.

Precision at fixed prevalence is governed by the false-positive rate, so a second threshold set is
calibrated against an independent background sample at SCREENING_FPR rather than the default. The
cost is sensitivity, which is measured here rather than assumed, so a user choosing the stricter
mode knows what fraction of true actives it forfeits.

Both threshold sets are stored. Neither is a correction of the other: triage thresholds suit a
focused or enriched set where recall matters, screening thresholds suit a large diverse library
where a false positive is expensive.

TWO DEFECTS THIS SCRIPT CARRIED, AND WHAT REPLACED THEM
-------------------------------------------------------
Its three sibling scripts had both corrections applied when the integrity audit found them.
This one was missed, and the figures it wrote reached a manuscript figure panel.

Sensitivity was computed over every active in data/endpoints/<T>.csv rather than over the
scaffold holdout, so roughly four fifths of what it scored was training data. The proof needed
no retraining: 28 of 52 endpoints reported a higher sensitivity at a higher threshold, which is
impossible on the same compounds. Across the fifteen endpoints whose two thresholds were
identical the inflation was mean +0.163 and maximum +0.523, with GABA-A published at 0.826
against a held-out 0.303. It now reads active_holdout from
models_rf/holdout/<T>_binder_holdout.json, the same set final_thresholds.py scores the triage
mode on. That is also what makes the two modes comparable: previously one was measured on
held-out actives and the other on all actives above pChEMBL 7, so the triage-versus-screening
sensitivity comparison was between two different compound sets.

The background sample came straight from the applicability-domain reference rather than from
the disjoint threshold pool built for the purpose, and the false-positive rate was then
reported on the same sample that had set the quantile. Such a rate is an arithmetic consequence
of the quantile and measures nothing: 35 of 52 endpoints reported exactly 0.0100, which is
SCREENING_FPR itself. The quantile is now taken on the threshold pool and the rate measured on
the disjoint evaluation pool, with the in-sample value kept beside it under an explicit name so
that the two cannot be confused again.

Writes the screening_threshold field into models_rf/binder_modes.json,
results/tables/screening_thresholds.csv and results/tables/screening_precision_curve.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize  # noqa: E402
from models.pools import background_pools  # noqa: E402

# Imported rather than re-declared. A near-copy of a four-line accessor is how the labelling
# rule drifted into six disagreeing versions; see activity_labels.py.
from models.final_thresholds import holdout_actives  # noqa: E402

M = ROOT / "models_rf"
SCREENING_FPR = 0.01
N_BACKGROUND = 3000
rng = np.random.default_rng(11)


def main():
    # The threshold pool sets the quantile, the disjoint evaluation pool measures the rate.
    # Drawing both from the reference, as this script used to, makes the second a restatement
    # of the first rather than a measurement of it.
    pools = background_pools()
    thr_pool, eval_pool = pools["threshold"], pools["evaluation"]
    idx = rng.choice(len(thr_pool), size=min(N_BACKGROUND, len(thr_pool)), replace=False)
    Xbg, _ = featurize([thr_pool[i] for i in idx])
    jdx = rng.choice(len(eval_pool), size=min(N_BACKGROUND, len(eval_pool)), replace=False)
    Xev, _ = featurize([eval_pool[j] for j in jdx])
    print(f"threshold pool sample: {Xbg.shape[0]} compounds; "
          f"disjoint evaluation sample: {Xev.shape[0]}", flush=True)
    missing_holdout = []

    modes = json.loads((M / "binder_modes.json").read_text())
    rows = []
    for ep, v in modes.items():
        mp = M / f"{ep}_binder.joblib"
        if not mp.exists():
            continue
        mdl = joblib.load(mp)
        pbg = mdl.predict_proba(Xbg)[:, 1]
        thr = float(np.clip(np.quantile(pbg, 1.0 - SCREENING_FPR), 0.05, 0.9995))
        # The screening cut must never be looser than the triage cut. For targets whose triage
        # threshold was set by the measured-inactive constraint rather than by the background
        # sample, the 1% background quantile can fall below it, which would make the mode advertised
        # as high-precision the less precise of the two against experimentally tested non-binders.
        thr = max(thr, float(v.get("threshold", 0.0)))

        # Only the actives withheld at training time. Reading data/endpoints/<T>.csv here, as
        # this script used to, scores the four fifths the model was fitted on.
        sens_tri = v.get("sensitivity_at_threshold")
        sens_scr = None
        act = holdout_actives(ep)
        if not act:
            missing_holdout.append(ep)
        else:
            Xa, _ = featurize(act)
            pa = mdl.predict_proba(Xa)[:, 1]
            sens_scr = float((pa >= thr).mean())

        # thr is a quantile of pbg, so the rate on pbg cannot exceed SCREENING_FPR whatever the
        # model does. The reportable rate is the one on the disjoint evaluation pool.
        pev = mdl.predict_proba(Xev)[:, 1]
        bgfpr = float((pev >= thr).mean())

        v["screening_threshold"] = round(thr, 4)
        v["screening_background_fpr"] = round(bgfpr, 4)
        v["screening_background_fpr_in_sample"] = round(float((pbg >= thr).mean()), 4)
        v["screening_background_basis"] = "threshold_pool_quantile_evaluation_pool_rate"
        if sens_scr is not None:
            v["screening_sensitivity"] = round(sens_scr, 3)
            v["screening_sensitivity_basis"] = "held_out_actives_by_scaffold"
            v["n_active_for_screening_sensitivity"] = int(len(act))
        modes[ep] = v
        rows.append({"target": ep, "triage_threshold": v.get("threshold"),
                     "triage_sensitivity": sens_tri,
                     "screening_threshold": round(thr, 4),
                     "screening_sensitivity": round(sens_scr, 3) if sens_scr is not None else None,
                     "screening_background_fpr": round(bgfpr, 4),
                     "screening_background_fpr_in_sample": v["screening_background_fpr_in_sample"],
                     "n_held_out_actives": int(len(act)) if act else 0})
        print(f"[{ep:8}] triage thr {v.get('threshold')} sens {sens_tri} -> "
              f"screening thr {thr:.3f} sens {sens_scr if sens_scr is None else round(sens_scr,3)}",
              flush=True)

    if missing_holdout:
        print(f"\nWARNING: no held-out actives for {len(missing_holdout)} endpoint(s): "
              f"{', '.join(missing_holdout[:8])}"
              + (" ..." if len(missing_holdout) > 8 else ""))
        print("Their screening sensitivity is left unset rather than filled in from the "
              "endpoint table, which is what inflated it before.")

    (M / "binder_modes.json").write_text(json.dumps(modes, indent=2))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tables" / "screening_thresholds.csv", index=False)
    ok = out.dropna(subset=["screening_sensitivity"])
    print(f"\nmean sensitivity on held-out actives  triage {ok.triage_sensitivity.mean():.3f}"
          f" -> screening {ok.screening_sensitivity.mean():.3f}")
    print(f"mean background FPR at screening threshold: "
          f"{out.screening_background_fpr.mean():.4f} on the evaluation pool, "
          f"{out.screening_background_fpr_in_sample.mean():.4f} in sample")
    higher = int((out.screening_sensitivity > out.triage_sensitivity).sum())
    print(f"endpoints reporting higher sensitivity at the stricter threshold: {higher} "
          f"(must be 0; both modes are now scored on the same held-out actives)")

    write_precision_curve(out, modes)


# The prevalence a triage tool is used at is the user's, not ours, and precision depends on it far
# more steeply than on anything the model does. Reporting precision at one assumed prevalence
# invites the reader to generalise from it; the curve makes the dependence the point. Both modes
# appear so the trade the stricter threshold buys is visible rather than asserted.
PREVALENCES = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.10, 0.30)


def write_precision_curve(out, modes) -> None:
    """Precision and false leads per true hit, per endpoint and mode, across the prevalence grid."""
    rows = []
    for r in out.to_dict("records"):
        ep = r["target"]
        v = modes.get(ep, {})
        for mode, sens, fpr in (
            ("triage", r.get("triage_sensitivity"), v.get("background_fpr_at_threshold")),
            ("screening", r.get("screening_sensitivity"), r.get("screening_background_fpr")),
        ):
            if sens is None or fpr is None:
                continue
            for prev in PREVALENCES:
                denom = sens * prev + fpr * (1 - prev)
                ppv = (sens * prev) / denom if denom > 0 else float("nan")
                rows.append({
                    "target": ep, "mode": mode, "prevalence": prev,
                    "sensitivity": round(float(sens), 4),
                    "background_fpr": round(float(fpr), 4),
                    "precision": round(ppv, 5) if ppv == ppv else None,
                    "false_leads_per_hit": (round((1 - ppv) / ppv, 2)
                                            if ppv == ppv and ppv > 0 else None),
                })
    if not rows:
        print("no endpoint had both a sensitivity and a background rate; curve not written")
        return
    curve = pd.DataFrame(rows)
    path = ROOT / "results" / "tables" / "screening_precision_curve.csv"
    curve.to_csv(path, index=False)

    print("\nprecision against prevalence, panel median of the per-endpoint values")
    print(f"{'prevalence':>11}  {'triage':>18}  {'screening':>18}")
    for prev in PREVALENCES:
        cell = {}
        for mode in ("triage", "screening"):
            sub = curve[(curve["mode"] == mode) & (curve["prevalence"] == prev)]
            sub = sub.dropna(subset=["precision"])
            if len(sub) == 0:
                cell[mode] = "       n/a        "
                continue
            # The false-lead count must be the median of the per-endpoint values, not a transform
            # of the mean precision. (1-p)/p is convex, so by Jensen's inequality the transform of
            # the mean sits below the mean of the transforms, and it understated the cost badly:
            # at 1 per cent prevalence it printed 2.0 false leads per hit where the per-endpoint
            # median is 3.4 and the mean 4.9, and at 1 in 10,000 it printed 59 against a median of
            # 344. The sign of the error reverses at high prevalence, which is the signature of
            # convexity rather than an arithmetic slip. The per-row column in the CSV was always
            # right; only this summary was wrong, and it is the line that got quoted.
            m = sub["precision"].median()
            fl = sub["false_leads_per_hit"].median()
            cell[mode] = f"{m:7.2%}  ({fl:6.1f} false)"
        print(f"{prev:>10.2%}   {cell['triage']}  {cell['screening']}")
    print(f"\nwrote {path.relative_to(ROOT).as_posix()}  ({len(curve):,} rows)")


if __name__ == "__main__":
    main()
