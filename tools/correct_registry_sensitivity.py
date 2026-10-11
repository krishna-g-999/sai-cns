"""Replace the registry's in-sample sensitivity with the held-out figure, and nothing else.

The registry reports sensitivity_at_threshold labelled "held_out_actives_by_scaffold". It is not.
calibrate_background_specificity.py overwrote the held-out figure computed by final_thresholds.py
with one measured over the whole endpoint table, roughly four fifths of which the model was fitted
on, and left the label untouched. The panel mean is 0.898 published against 0.764 held out.

Three fields are corrected, from results/tables/sensitivity_reconciliation.csv, which recomputes the
held-out figure from models_rf/holdout and is verified against final_thresholds.csv:

    sensitivity_at_threshold  the held-out value
    sensitivity_basis         a label that matches what was measured
    reliable_call             recomputed from the held-out value under the gate
                              defined in panel.py, the one definition every writer imports

Nothing else is touched, and that restraint is the point. Every threshold, every background
false-positive rate and every AUROC is left byte-identical, because thresholds are the operating
parameters of a deployed panel and re-running the threshold sequence caused a regression in this
project once already. Sensitivity is a reported quantity, not an operating one: correcting it
changes what the server says about itself, not what it does.

The consequence is visible and intended. Endpoints passing the gate go from 46 to 41, and the About
page's list of endpoints below it, which is selected on the same field, goes from two to six.

Run:  python tools/correct_registry_sensitivity.py --check
      python tools/correct_registry_sensitivity.py --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "models_rf" / "binder_modes.json"
RECON = ROOT / "results" / "tables" / "sensitivity_reconciliation.csv"

sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
import panel  # noqa: E402

UNTOUCHED = ["threshold", "background_fpr_at_threshold", "auroc_vs_measured_inactives",
             "scaffold_cv_auroc", "screening_threshold", "deployed"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    min_sens = panel.MIN_SENSITIVITY
    print(f"gate: sensitivity {min_sens:.2f}, AUROC {panel.MIN_AUROC:.2f}\n")

    modes = json.loads(REG.read_text(encoding="utf-8"))
    rec = pd.read_csv(RECON).set_index("target")
    before = {k: {f: v.get(f) for f in UNTOUCHED} for k, v in modes.items()}

    changed, rel_flips = [], []
    for ep, v in modes.items():
        if ep not in rec.index or not v.get("deployed", True):
            continue
        held = float(rec.loc[ep, "sensitivity_heldout"])
        old_s = v.get("sensitivity_at_threshold")
        old_r = bool(v.get("reliable_call"))
        new_r = panel.passes_gate(held, v.get("auroc_vs_measured_inactives"))
        changed.append((ep, old_s, round(held, 3), old_r, new_r))
        if old_r != new_r:
            rel_flips.append(ep)
        if args.apply:
            v["sensitivity_at_threshold"] = round(held, 3)
            v["sensitivity_basis"] = "held_out_actives_by_scaffold"
            # Record the superseded figure once. Re-running must not overwrite it with a value this
            # script itself wrote, which would erase the evidence of what was corrected.
            v.setdefault("sensitivity_in_sample_superseded", old_s)
            v["reliable_call"] = new_r

    s_old = pd.Series({e: o for e, o, _n, _a, _b in changed})
    s_new = pd.Series({e: n for e, _o, n, _a, _b in changed})
    print(f"{len(changed)} deployed endpoints")
    print(f"  mean sensitivity published : {s_old.mean():.4f}")
    print(f"  mean sensitivity held out  : {s_new.mean():.4f}")
    print(f"  mean inflation removed     : {(s_old - s_new).mean():.4f}")
    print(f"  reliable_call flips to False: {len(rel_flips)} -> {sorted(rel_flips)}")

    if not args.apply:
        print("\nnothing written; pass --apply to write")
        return

    backup = REG.with_suffix(f".json.pre_sensitivity_fix_{datetime.now():%Y%m%d_%H%M}")
    shutil.copy2(REG, backup)
    REG.write_text(json.dumps(modes, indent=2), encoding="utf-8")

    after = json.loads(REG.read_text(encoding="utf-8"))
    drift = [(ep, f) for ep, fields in before.items() for f, val in fields.items()
             if after.get(ep, {}).get(f) != val]
    print(f"\nbacked up to {backup.name}")
    print(f"operating parameters changed: {len(drift)} (must be 0)")
    if drift:
        for ep, f in drift[:10]:
            print(f"   DRIFT {ep}.{f}")
        raise SystemExit("aborting: an operating parameter moved")
    print("every threshold, background rate and AUROC is byte-identical")


if __name__ == "__main__":
    main()
