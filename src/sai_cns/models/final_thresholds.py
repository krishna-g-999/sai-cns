"""Set every binder decision threshold from two independent requirements, computed from scratch.

Requirement A, selectivity against tested non-binders: at most TARGET_FPR of the target's measured
inactives may be called a binder. Only the half withheld at training time counts, read from
models_rf/holdout/<T>_binder_holdout.json. This step used to re-read every inactive in the endpoint
table, including the half the model was fitted on, which undid the holdout that training had
constructed and set the threshold on 40 of 49 deployed endpoints from data the model had seen.

Requirement B, specificity against unrelated chemistry: at most BACKGROUND_FPR of the threshold pool
may be called a binder. That pool is disjoint from the decoys the model trained on and from the
evaluation pool the reported rate is measured on, so neither the threshold nor the rate is a
restatement of the other. It exists because a prospective test found Nav1.1 assigning glucose a
binder probability of 0.806 while scoring 0.979 against its own assay's negatives: a model judged
only on its own chemistry can be confidently wrong about everything else.

The operating threshold is the stricter of the two, recomputed from scratch each run so repeated
calibration cannot ratchet thresholds upward. BACKGROUND_FPR is 0.05; 0.02 was tested and rejected
because it drove the mu-opioid threshold to 0.999 and caused morphine to be missed.

Updates models_rf/binder_modes.json and writes results/tables/final_thresholds.csv
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
import panel  # noqa: E402
from features.featurize import featurize  # noqa: E402
from models.pools import background_pools  # noqa: E402

M = ROOT / "models_rf"
TARGET_FPR = 0.10
BACKGROUND_FPR = 0.05
N_BACKGROUND = 3000
MIN_SENS = panel.MIN_SENSITIVITY   # the gate lives in panel.py; see the note there on why
rng = np.random.default_rng(7)


def holdout_inactives(ep: str) -> list[str] | None:
    """The measured inactives withheld at training time, or None if no record was kept."""
    p = M / "holdout" / f"{ep}_binder_holdout.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("measured_inactive_holdout") or None


def holdout_actives(ep: str) -> list[str] | None:
    p = M / "holdout" / f"{ep}_binder_holdout.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("active_holdout") or None


def main():
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
        thr_bg = float(np.quantile(pbg, 1.0 - BACKGROUND_FPR))

        thr_in, sens, n_in = None, None, 0
        pa = None
        # Only what training withheld. Reading the endpoint table here would pull back the half of
        # the inactives the model was fitted on, and the threshold would be set on seen data.
        ina = holdout_inactives(ep)
        act = holdout_actives(ep)
        if ina is None:
            missing_holdout.append(ep)
        if ina and len(ina) >= 15:
            Xi, _ = featurize(ina)
            pi = mdl.predict_proba(Xi)[:, 1]
            thr_in = float(np.quantile(pi, 1.0 - TARGET_FPR))
            n_in = len(pi)
        if act:
            Xa, _ = featurize(act)
            pa = mdl.predict_proba(Xa)[:, 1]

        thr = max([t for t in (thr_in, thr_bg) if t is not None])
        thr = float(np.clip(thr, 0.05, 0.999))
        if pa is not None and len(pa):
            sens = float((pa >= thr).mean())
        # Measured on the evaluation pool, not on pbg. thr is a quantile of pbg, so the rate on pbg
        # cannot exceed BACKGROUND_FPR whatever the model does, and reporting it says nothing.
        pev = mdl.predict_proba(Xev)[:, 1]
        bgfpr = float((pev >= thr).mean())
        bgfpr_in_sample = float((pbg >= thr).mean())

        v["threshold"] = round(thr, 4)
        v["threshold_basis"] = ("held_out_measured_inactives_and_background" if thr_in is not None
                                else "background_only")
        v["background_fpr_at_threshold"] = round(bgfpr, 4)
        v["background_fpr_in_sample"] = round(bgfpr_in_sample, 4)
        v["n_measured_inactive_for_threshold"] = int(n_in)
        v["n_measured_inactive_basis"] = "held_out_only"
        if sens is not None:
            v["sensitivity_basis"] = "held_out_actives_by_scaffold"
            v["sensitivity_at_threshold"] = round(sens, 3)
            v["reliable_call"] = panel.passes_gate(sens, v.get("auroc_vs_measured_inactives"))
        modes[ep] = v
        rows.append({"target": ep, "threshold": round(thr, 4),
                     "from_measured_inactives": round(thr_in, 4) if thr_in else None,
                     "from_background": round(thr_bg, 4),
                     "binding_constraint": ("background" if thr_in is None or thr_bg >= thr_in
                                            else "measured inactives"),
                     "background_fpr": round(bgfpr, 4),
                     "sensitivity": round(sens, 3) if sens is not None else None,
                     "reliable": v.get("reliable_call")})
        print(f"[{ep:8}] thr {thr:.3f} ({rows[-1]['binding_constraint']}) | bgFPR {bgfpr:.3f} | "
              f"sens {sens if sens is None else round(sens, 3)}", flush=True)

    if missing_holdout:
        print(f"\nWARNING: no holdout record for {len(missing_holdout)} endpoint(s): "
              f"{', '.join(missing_holdout[:8])}"
              + (" ..." if len(missing_holdout) > 8 else ""))
        print("Their thresholds rest on the background pool alone. Retrain them with "
              "train_binders_hybrid.py so a held-out set exists.")

    (M / "binder_modes.json").write_text(json.dumps(modes, indent=2))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tables" / "final_thresholds.csv", index=False)
    print(f"\nmedian background FPR {out.background_fpr.median():.3f} | "
          f"median sensitivity {out.sensitivity.median():.3f}")
    print("unreliable:", out.loc[out.reliable == False, 'target'].tolist())  # noqa: E712
    print("wrote", ROOT / "results" / "tables" / "final_thresholds.csv")


if __name__ == "__main__":
    main()
