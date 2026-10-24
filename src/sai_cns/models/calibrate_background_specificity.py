"""Add a background-specificity requirement to every binder threshold.

A prospective test exposed a failure mode that per-target validation had missed. Nav1.1 was trained
on its own measured actives and inactives, both of which are sodium-channel screening chemistry, and
scored an AUROC of 0.979 against that narrow negative set. Presented with arbitrary chemistry it
assigned glucose a binder probability of 0.806, because nothing in its training ever showed it what
an unrelated molecule looks like. A model validated only against its own assay's negatives can be
confidently wrong about everything else.

The threshold is therefore required to satisfy two conditions simultaneously:

  1. no more than TARGET_FPR of the target's held-out measured inactives is called a binder, and
  2. no more than BACKGROUND_FPR of a random sample of the measured compound library is called a
     binder, which bounds how often the model fires on unrelated chemistry. The level is set at 5%:
     tightening it to 2% was tested and rejected because it cost too much sensitivity, pushing the
     mu-opioid threshold to 0.999 and causing morphine to be missed.

The operating threshold is the stricter of the two. Sensitivity is then re-measured at that
threshold and a target is marked unreliable if too few genuine binders survive. Targets that cannot
satisfy both conditions while retaining useful sensitivity are reported so they can be excluded
rather than silently trusted.

Updates models_rf/binder_modes.json and writes results/tables/background_specificity.csv
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

# The applicability-domain reference has one guarded accessor; see ad_fingerprint.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "sai_cns"))
import ad_fingerprint as _adfp  # noqa: E402

M = ROOT / "models_rf"
TARGET_FPR = 0.10        # on the target's own measured inactives
BACKGROUND_FPR = 0.05    # on random library chemistry
N_BACKGROUND = 3000
MIN_SENS = panel.MIN_SENSITIVITY   # the gate lives in panel.py; see the note there on why
rng = np.random.default_rng(7)


def main():
    bg_smiles, _ = _adfp.load_reference(M / "ad_reference.pkl")
    idx = rng.choice(len(bg_smiles), size=min(N_BACKGROUND, len(bg_smiles)), replace=False)
    bg = [str(bg_smiles[i]) for i in idx]
    Xbg, _ = featurize(bg)
    print(f"background sample: {Xbg.shape[0]} compounds", flush=True)

    modes = json.loads((M / "binder_modes.json").read_text())
    rows = []
    for ep, v in modes.items():
        mp = M / f"{ep}_binder.joblib"
        if not mp.exists():
            continue
        if not v.get("deployed", True):
            # A withdrawn endpoint has no operating point to calibrate, and leaving it in the output
            # table means every aggregate computed from that table describes a panel that is not the
            # one running. The provenance audit caught exactly this.
            print(f"[{ep:8}] withdrawn, not calibrated", flush=True)
            continue
        mdl = joblib.load(mp)
        pbg = mdl.predict_proba(Xbg)[:, 1]
        thr_bg = float(np.quantile(pbg, 1.0 - BACKGROUND_FPR))
        thr_old = float(v.get("threshold", 0.40))

        f = ROOT / "data" / "endpoints" / f"{ep}.csv"
        sens = None
        pa = None
        if f.exists():
            df = pd.read_csv(f).dropna(subset=["smiles"])
            p = pd.to_numeric(df.get("pchembl"), errors="coerce")
            # Score the actives the model did NOT see. This line selected every active in the
            # endpoint table, so roughly four fifths of the scoring set were compounds the model had
            # been fitted on, and the sensitivity it wrote over the registry was in-sample. The
            # registry's own sensitivity_basis field, written earlier by final_thresholds.py and
            # never updated here, went on claiming "held_out_actives_by_scaffold". Across the panel
            # the published mean was 0.898 where the held-out mean is 0.764. reliable_call is
            # derived from this figure a few lines below, so five endpoints were marked reliable on
            # an in-sample number.
            held = ROOT / "models_rf" / "holdout" / f"{ep}_binder_holdout.json"
            act = []
            if held.exists():
                try:
                    act = list(json.loads(held.read_text(encoding="utf-8")).get("active_holdout", []))
                except Exception:
                    act = []
            if len(act) < 30:
                # No usable hold-out. Fall back to the whole table, but say so, rather than
                # silently reporting an in-sample figure as though it were held out.
                act = df.loc[p >= 7, "smiles"].astype(str).tolist()
                if len(act) < 30:
                    act = df.loc[df["label"] == 1, "smiles"].astype(str).tolist()
                v["sensitivity_basis"] = "whole_table_no_usable_holdout"
            else:
                v["sensitivity_basis"] = "held_out_actives_by_scaffold"
            if act:
                Xa, _ = featurize(act)
                pa = mdl.predict_proba(Xa)[:, 1]

        thr_new = max(thr_old, thr_bg)
        if pa is not None and len(pa):
            sens = float((pa >= thr_new).mean())
        bgfpr_old = float((pbg >= thr_old).mean())
        bgfpr_new = float((pbg >= thr_new).mean())

        v["threshold"] = round(thr_new, 4)
        v["background_fpr_at_old_threshold"] = round(bgfpr_old, 4)
        v["background_fpr_at_threshold"] = round(bgfpr_new, 4)
        if sens is not None:
            v["sensitivity_at_threshold"] = round(sens, 3)
            v["reliable_call"] = panel.passes_gate(sens, v.get("auroc_vs_measured_inactives"))
        modes[ep] = v
        rows.append({"target": ep, "old_threshold": round(thr_old, 4),
                     "background_fpr_before": round(bgfpr_old, 4),
                     "new_threshold": round(thr_new, 4),
                     "background_fpr_after": round(bgfpr_new, 4),
                     "sensitivity_after": round(sens, 3) if sens is not None else None,
                     "reliable": v.get("reliable_call")})
        flag = "  <-- was firing on background" if bgfpr_old > 0.10 else ""
        print(f"[{ep:8}] bgFPR {bgfpr_old:.3f} -> {bgfpr_new:.3f} | thr {thr_old:.3f} -> {thr_new:.3f}"
              f" | sens {sens if sens is None else round(sens,3)}{flag}", flush=True)

    (M / "binder_modes.json").write_text(json.dumps(modes, indent=2))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tables" / "background_specificity.csv", index=False)
    bad = out[(out.sensitivity_after.fillna(0) < MIN_SENS)]
    print(f"\nmedian background FPR before {out.background_fpr_before.median():.3f}, "
          f"after {out.background_fpr_after.median():.3f}")
    print(f"targets failing the sensitivity floor after tightening: {bad.target.tolist()}")
    print("wrote", ROOT / "results" / "tables" / "background_specificity.csv")


if __name__ == "__main__":
    main()
