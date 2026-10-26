"""Average precision on the measured-inactive validation set, the basis the headline AUROC uses.

WHY THIS EXISTS. The panel's headline classification figure is a mean AUROC of 0.917 against
compounds experimentally tested at the same target and found inactive, which is a far harder and
more honest comparison than one against synthetic decoys. It has no precision-recall counterpart.
AUROC is insensitive to class imbalance by construction, and this validation set is strongly
imbalanced: most endpoints hold thousands of actives against a few hundred measured inactives, so a
reviewer asking "what does the ranking cost in precision" currently has nothing to read. Average
precision is computed elsewhere in the project, but only under cross-validation with decoys in the
negative class, which is a different question on a different set.

Note which way the imbalance runs. Actives are the majority here, so average precision against a
majority positive class is generous and its baseline is high; the useful quantity is the lift over
that baseline, which is reported. For the screening question, where actives are rare, the
prevalence-dependent precision curve in results/tables/screening_precision_curve.csv is the figure
to read instead. This table answers "how good is the ranking on measured data"; that one answers
"what precision would a user see".

THE CONTROL. AUROC is recomputed here from the same scored compounds and compared against the
registry's stored auroc_vs_measured_inactives. If the two agree, the average precision is computed
on the compounds the headline figure describes and can be reported beside it. If they do not, this
script is scoring a different set and says so rather than publishing the number.

Both classes come from models_rf/holdout/<T>_binder_holdout.json, so nothing scored here was
trained on.

Writes results/tables/measured_inactive_pr_auc.csv

    python tools/measured_inactive_pr_auc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
sys.path.insert(0, str(ROOT / "tools"))
from features.featurize import featurize  # noqa: E402

M = ROOT / "models_rf"
OUT = ROOT / "results" / "tables" / "measured_inactive_pr_auc.csv"
MIN_INACTIVE = 15
AUROC_TOLERANCE = 0.0006     # three-decimal rounding bounds the error at 0.0005; the extra
                             # 1e-4 is float slack. At the previous 0.005 a genuine 0.004
                             # discrepancy, which would mean a different compound set, passed
                             # the control the docstring promises would catch it.


def holdout(ep, key):
    p = M / "holdout" / f"{ep}_binder_holdout.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get(key) or None


def main() -> int:
    from benchmark_label_formulations import auroc, average_precision

    modes = json.loads((M / "binder_modes.json").read_text(encoding="utf-8"))
    rows = []
    for ep, v in sorted(modes.items()):
        if not v.get("deployed", True):
            continue
        mp = M / f"{ep}_binder.joblib"
        act, ina = holdout(ep, "active_holdout"), holdout(ep, "measured_inactive_holdout")
        if not mp.exists() or not act or not ina or len(ina) < MIN_INACTIVE:
            rows.append({"target": ep, "n_active": len(act or []), "n_measured_inactive":
                         len(ina or []), "scorable": False})
            continue
        mdl = joblib.load(mp)
        Xa, _ = featurize(act)
        Xi, _ = featurize(ina)
        pa = mdl.predict_proba(Xa)[:, 1]
        pi = mdl.predict_proba(Xi)[:, 1]
        y = np.concatenate([np.ones(len(pa), int), np.zeros(len(pi), int)])
        s = np.concatenate([pa, pi])
        base = float(y.mean())
        ap = average_precision(y, s)
        au = auroc(y, s)
        stored = v.get("auroc_vs_measured_inactives")
        rows.append({
            "target": ep, "n_active": len(pa), "n_measured_inactive": len(pi),
            "positive_rate": round(base, 4),
            "auroc_recomputed": round(au, 4),
            "auroc_in_registry": stored,
            "auroc_agrees": (None if stored is None
                             else bool(abs(au - float(stored)) <= AUROC_TOLERANCE)),
            "average_precision": round(ap, 4),
            # Lift over prevalence is kept because it is the familiar quantity, but it is NOT
            # comparable across endpoints: its ceiling is 1/prevalence, which runs from 1.13 to
            # 25.91 here, so it ranks COX2 (AUROC 0.783) above CGRP (AUROC 0.985) purely because
            # COX2's actives are rarer. The range-normalised form divides the gain over prevalence
            # by the gain that was available, so 1.0 is a perfect ranker at any prevalence, and it
            # orders the panel the way AUROC does. Quote the normalised column, not the lift.
            "ap_lift_over_prevalence": round(ap / base, 3) if base > 0 else None,
            "ap_normalised": round((ap - base) / (1 - base), 4) if base < 1 else None,
            "scorable": True,
        })
        print(f"[{ep:10}] AUROC {au:.4f} (registry {stored}) | AP {ap:.4f} "
              f"| prevalence {base:.3f} | lift {ap / base:.3f}", flush=True)

    d = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)

    ok = d[d.scorable == True]  # noqa: E712
    checked = ok[ok.auroc_agrees.notna()]
    disagree = checked[checked.auroc_agrees == False]  # noqa: E712
    print(f"\n{len(ok)} of {len(d)} deployed endpoints scorable "
          f"({len(d) - len(ok)} lack a usable measured-inactive holdout)")
    print(f"AUROC control: {len(checked) - len(disagree)} of {len(checked)} reproduce the registry "
          f"within {AUROC_TOLERANCE}")
    if len(disagree):
        print("  DISAGREE: " + ", ".join(
            f"{r.target} {r.auroc_recomputed} vs {r.auroc_in_registry}"
            for r in disagree.itertuples()))
        print("  The average precision below is therefore NOT on the same compounds as the "
              "headline AUROC for those endpoints.")
    print(f"\nmean AUROC            {ok.auroc_recomputed.mean():.4f}")
    print(f"mean average precision {ok.average_precision.mean():.4f}  "
          f"(mean prevalence {ok.positive_rate.mean():.4f})")
    print(f"mean range-normalised AP {ok.ap_normalised.mean():.3f}  "
          f"({ok.ap_normalised.min():.3f} to {ok.ap_normalised.max():.3f}); "
          f"this is the comparable figure, not the raw lift")
    print(f"average precision range {ok.average_precision.min():.3f} "
          f"({ok.loc[ok.average_precision.idxmin(), 'target']}) to "
          f"{ok.average_precision.max():.3f} "
          f"({ok.loc[ok.average_precision.idxmax(), 'target']})")
    # Previously thresholded the raw lift at 1.05, which is prevalence-dependent: above a
    # prevalence of 0.952 even a perfect ranker scores a lift below 1.05 and would have been
    # flagged as worthless. The normalised form has no such dependence.
    low = ok[ok.ap_normalised < 0.50]
    print(f"endpoints recovering under half the average precision available above their own "
          f"prevalence: {len(low)}"
          + (f" ({', '.join(low.target)})" if len(low) else ""))
    print(f"\nwrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
