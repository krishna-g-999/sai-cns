"""The background false-positive rate at the CURRENTLY DEPLOYED threshold, measured honestly.

Three fields already exist in binder_modes.json for this quantity, and none of them is what the
manuscript's Methods paragraph claims to report, which is "the background false-positive rate,
measured on the pool it was not set on" at the threshold the server actually runs.

  background_fpr_at_threshold   Written last, by calibrate_background_specificity.py. Circular: the
                                 threshold there is a quantile of a single ad-hoc background sample
                                 (models_rf/ad_reference.pkl, resampled with a fixed seed), and the
                                 rate is then measured on that SAME sample. A threshold set as the
                                 95th percentile of a sample reports approximately 5 per cent on that
                                 sample whatever the model does; this is exactly the failure the
                                 manuscript's own paragraph describes calibrate_background_specificity
                                 as having fixed. It has not been fixed there. Every value under this
                                 key sits at or just under 0.05, which is the signature of the defect,
                                 not evidence the constraint holds.
  background_fpr_held_out       Written first, by train_binders_hybrid.py, correctly disjoint: the
                                 threshold is a quantile of the target's own held-out measured
                                 inactives, and the rate is measured on models_rf/models.pools's
                                 evaluation pool, disjoint from both the decoy and threshold pools.
                                 But this threshold is an EARLIER one. final_thresholds.py and
                                 calibrate_background_specificity.py run afterwards and can raise the
                                 deployed threshold specifically to bring the background rate down, so
                                 this field describes a threshold the server may no longer use. It is
                                 populated for only 44 of 47 deployed endpoints, is silent on the other
                                 three, and does not tell a reader anything about the endpoint the
                                 server actually runs today.

Neither field is dishonest by intent; each was correct for the pipeline stage that wrote it and went
stale as later stages moved the threshold without moving it. This script closes the gap directly:
score every deployed binder's CURRENT threshold (models_rf/binder_modes.json's own `threshold` field,
the one the server loads) against the evaluation pool from models.pools.background_pools(), which is
disjoint from the decoy pool the binder was trained against and from the threshold pool that set any
version of the cut. Every deployed endpoint is scored, not only the ones an earlier field happened to
cover.

Output: results/tables/background_specificity_disjoint.csv

Run:  python src/sai_cns/evaluation/background_specificity_disjoint.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize  # noqa: E402
from models.pools import background_pools  # noqa: E402

BACKGROUND_FPR_TARGET = 0.05


def main() -> None:
    modes = json.loads((ROOT / "models_rf" / "binder_modes.json").read_text(encoding="utf-8"))
    pools = background_pools()
    eval_pool = pools["evaluation"]
    Xev, mask = featurize(eval_pool)
    print(f"pool sizes: decoy {len(pools['decoy']):,}, threshold {len(pools['threshold']):,}, "
          f"evaluation {len(pools['evaluation']):,} (total {sum(len(p) for p in pools.values()):,})")
    print(f"evaluation pool featurised: {Xev.shape[0]:,} of {len(eval_pool):,}")

    rows = []
    for ep, v in modes.items():
        if not v.get("deployed", True):
            continue
        mp = ROOT / "models_rf" / f"{ep}_binder.joblib"
        if not mp.exists():
            continue
        mdl = joblib.load(mp)
        p = mdl.predict_proba(Xev)[:, 1]
        thr = float(v["threshold"])
        fpr = float((p >= thr).mean())
        rows.append({"target": ep, "deployed_threshold": round(thr, 4),
                     "background_fpr_disjoint": round(fpr, 4),
                     "exceeds_target": bool(fpr > BACKGROUND_FPR_TARGET)})

    d = pd.DataFrame(rows).sort_values("background_fpr_disjoint", ascending=False)
    out = ROOT / "results" / "tables" / "background_specificity_disjoint.csv"
    d.to_csv(out, index=False)

    print(f"\n{len(d)} deployed endpoints scored")
    print(f"median: {d.background_fpr_disjoint.median():.4f}")
    print(f"max: {d.background_fpr_disjoint.max():.4f} at {d.iloc[0]['target']}")
    print(f"exceeding {BACKGROUND_FPR_TARGET}: {int(d.exceeds_target.sum())} "
          f"({', '.join(d.loc[d.exceeds_target, 'target'])})")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
