"""Calibrate each binder model's decision threshold against EXPERIMENTALLY MEASURED inactives.

Validation against measured inactives (compounds tested against the same target and found inactive)
shows that the binder models rank correctly, with a mean AUROC of 0.936 and no endpoint below 0.70.
It also shows that the default 0.5 probability threshold is far too permissive on real chemistry:
for several targets a majority of measured inactives exceed it, because the property-matched decoys
used in training are easier to reject than genuine tested non-binders.

Ranking is therefore trustworthy while the absolute cut is not. This script replaces the fixed cut
with a per-target threshold chosen so that a stated fraction of measured inactives (default 10%) is
called a binder. Targets without enough measured inactives keep the global fallback and are marked
as such, so the interface can say which basis was used.

Writes models_rf/binder_modes.json with, per target: mode, base rate where applicable, the
calibrated threshold, the measured-inactive AUROC and the achieved false-positive rate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
import panel  # noqa: E402
from features.featurize import featurize  # noqa: E402

M = ROOT / "models_rf"
TARGET_FPR = 0.10           # tolerated false-positive rate on measured inactives
MIN_INACTIVES = 25          # below this, no per-target calibration is attempted
GLOBAL_FALLBACK = 0.40


def main():
    modes = {}
    for mp in sorted(M.glob("*_binder.joblib")):
        ep = mp.name.replace("_binder.joblib", "")
        meta_p = M / f"{ep}_binder_meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        mode = meta.get("mode", "decoy_aware")
        rec = {"mode": mode}
        if mode == "measured_labels":
            pos, neg = meta.get("n_positive", 0), meta.get("n_negative", 0)
            if pos + neg:
                rec["base_rate"] = round(pos / (pos + neg), 4)

        f = ROOT / "data" / "endpoints" / f"{ep}.csv"
        thr = GLOBAL_FALLBACK
        rec["threshold_basis"] = "global_fallback"
        if f.exists():
            df = pd.read_csv(f).dropna(subset=["smiles"])
            p = pd.to_numeric(df.get("pchembl"), errors="coerce")
            act = df.loc[p >= 7, "smiles"].astype(str).tolist()
            ina = df.loc[df["label"] == 0, "smiles"].astype(str).tolist()
            if len(ina) >= MIN_INACTIVES and len(act) >= MIN_INACTIVES:
                mdl = joblib.load(mp)
                Xa, _ = featurize(act)
                Xi, _ = featurize(ina)
                pa = mdl.predict_proba(Xa)[:, 1]
                pi = mdl.predict_proba(Xi)[:, 1]
                # Operating point: the probability at which TARGET_FPR of measured inactives are
                # called binders. No arbitrary upper cap is applied, because capping the threshold
                # silently inflates the false-positive rate for the targets that need it most.
                thr = float(np.quantile(pi, 1.0 - TARGET_FPR))
                thr = float(min(max(thr, 0.05), 0.999))
                rec.update({
                    "threshold": round(thr, 3),
                    "threshold_basis": "measured_inactives",
                    "n_measured_inactive": int(len(pi)),
                    "auroc_vs_measured_inactives": round(float(
                        roc_auc_score(np.r_[np.ones(len(pa)), np.zeros(len(pi))], np.r_[pa, pi])), 3),
                    "fpr_at_threshold": round(float((pi >= thr).mean()), 3),
                    "sensitivity_at_threshold": round(float((pa >= thr).mean()), 3),
                })
                # A target is only useful if it still recovers a reasonable share of true binders
                # once the false-positive rate is held at the target level.
                rec["reliable_call"] = panel.passes_gate(
                    rec["sensitivity_at_threshold"], rec["auroc_vs_measured_inactives"])
                print(f"[{ep:10}] thr={thr:.2f} (was 0.50) | AUROC vs measured inactives "
                      f"{rec['auroc_vs_measured_inactives']:.3f} | FPR {rec['fpr_at_threshold']:.2f} "
                      f"| sensitivity {rec['sensitivity_at_threshold']:.2f} | n_inact={len(pi)}", flush=True)
        if "threshold" not in rec:
            rec["threshold"] = GLOBAL_FALLBACK
            print(f"[{ep:10}] no measured inactives, global fallback {GLOBAL_FALLBACK}", flush=True)
        modes[ep] = rec

    (M / "binder_modes.json").write_text(json.dumps(modes, indent=2))
    cal = [v for v in modes.values() if v["threshold_basis"] == "measured_inactives"]
    print(f"\ncalibrated on measured inactives: {len(cal)} / {len(modes)}")
    if cal:
        print(f"mean AUROC vs measured inactives : {np.mean([c['auroc_vs_measured_inactives'] for c in cal]):.3f}")
        print(f"mean sensitivity at threshold    : {np.mean([c['sensitivity_at_threshold'] for c in cal]):.3f}")
        print(f"mean FPR at threshold            : {np.mean([c['fpr_at_threshold'] for c in cal]):.3f}")
    print("wrote", M / "binder_modes.json")


if __name__ == "__main__":
    main()
