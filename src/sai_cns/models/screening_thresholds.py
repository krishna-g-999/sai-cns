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

Writes the screening_threshold field into models_rf/binder_modes.json and
results/tables/screening_thresholds.csv
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize  # noqa: E402

M = ROOT / "models_rf"
SCREENING_FPR = 0.01
N_BACKGROUND = 3000
rng = np.random.default_rng(11)


def main():
    with (M / "ad_reference.pkl").open("rb") as fh:
        bg, _ = pickle.load(fh)
    idx = rng.choice(len(bg), size=min(N_BACKGROUND, len(bg)), replace=False)
    Xbg, _ = featurize([str(bg[i]) for i in idx])
    print(f"independent background sample: {Xbg.shape[0]} compounds", flush=True)

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

        f = ROOT / "data" / "endpoints" / f"{ep}.csv"
        sens_tri = v.get("sensitivity_at_threshold")
        sens_scr = None
        if f.exists():
            df = pd.read_csv(f).dropna(subset=["smiles"])
            p = pd.to_numeric(df.get("pchembl"), errors="coerce")
            act = df.loc[p >= 7, "smiles"].astype(str).tolist()
            if len(act) < 30:
                act = df.loc[df["label"] == 1, "smiles"].astype(str).tolist()
            if act:
                Xa, _ = featurize(act)
                pa = mdl.predict_proba(Xa)[:, 1]
                sens_scr = float((pa >= thr).mean())

        v["screening_threshold"] = round(thr, 4)
        v["screening_background_fpr"] = round(float((pbg >= thr).mean()), 4)
        if sens_scr is not None:
            v["screening_sensitivity"] = round(sens_scr, 3)
        modes[ep] = v
        rows.append({"target": ep, "triage_threshold": v.get("threshold"),
                     "triage_sensitivity": sens_tri,
                     "screening_threshold": round(thr, 4),
                     "screening_sensitivity": round(sens_scr, 3) if sens_scr is not None else None,
                     "screening_background_fpr": v["screening_background_fpr"]})
        print(f"[{ep:8}] triage thr {v.get('threshold')} sens {sens_tri} -> "
              f"screening thr {thr:.3f} sens {sens_scr if sens_scr is None else round(sens_scr,3)}",
              flush=True)

    (M / "binder_modes.json").write_text(json.dumps(modes, indent=2))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tables" / "screening_thresholds.csv", index=False)
    ok = out.dropna(subset=["screening_sensitivity"])
    print(f"\nmean sensitivity  triage {ok.triage_sensitivity.mean():.3f} -> "
          f"screening {ok.screening_sensitivity.mean():.3f}")
    print(f"mean background FPR at screening threshold: {out.screening_background_fpr.mean():.4f}")
    s, f_ = ok.screening_sensitivity.mean(), out.screening_background_fpr.mean()
    for prev in (0.10, 0.01):
        ppv = (s * prev) / (s * prev + f_ * (1 - prev))
        print(f"  precision at {prev:.0%} prevalence: {ppv:.1%} "
              f"({(1-ppv)/max(ppv,1e-9):.1f} false leads per true hit)")


if __name__ == "__main__":
    main()
