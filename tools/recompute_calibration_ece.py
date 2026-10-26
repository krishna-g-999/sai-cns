"""Recompute results/tables/calibration.csv from the saved out-of-fold predictions.

WHY THIS EXISTS. calibrate.py produces this table AND refits the eight calibrated deployment
models in the same pass. The expected-calibration-error function was corrected on 2026-10-24: it
had dropped every prediction of exactly 1.0, which are the most over-confident ones a saturated
forest emits. The table on disk predates that fix, so the figure quoted in the README was computed
with the broken metric, and check_freshness.py did not notice because it declared the table's input
as models_rf/BBB.joblib rather than as its own generator.

Re-running calibrate.py would fix the table and also rewrite models_rf/*_calibrated.joblib, which
would move eight checksums in models_manifest.json while the per-file delta is still unpublished.
All four columns are computable from data/processed/cv_predictions alone, with no refitting, so
this recomputes them and leaves the deployment models untouched.

The Brier columns do not depend on the corrected function. They are the control: if they reproduce
the stored values exactly, the reproduction is faithful and the ECE columns can be trusted.

    python tools/recompute_calibration_ece.py [--write]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sai_cns.models.calibrate import expected_calibration_error  # noqa: E402
from sai_cns.models.train_rf import CLASSIFICATION  # noqa: E402

OOF = ROOT / "data" / "processed" / "cv_predictions"
TABLE = ROOT / "results" / "tables" / "calibration.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    stored = pd.read_csv(TABLE).set_index("endpoint")
    rows, scheme_rows = [], []
    for ep in CLASSIFICATION:
        oof = pd.read_csv(OOF / f"{ep}_random_oof.csv")
        y = oof["y_true"].to_numpy().astype(int)
        p = oof["prediction"].to_numpy().astype(float)

        # All three inner folds, so the choice is auditable rather than asserted. "stratified" is
        # the one written out, because it is what CalibratedClassifierCV(cv=5) does and therefore
        # what the deployed calibrator is. "contiguous" is what cv=5 silently gave a regressor and
        # what the previously published figure measured. "shuffled" is the robustness check: if it
        # disagreed with stratified, the gain would be a fold-ordering artefact rather than the
        # class-balance mismatch it is.
        cals = {
            "stratified": cross_val_predict(IsotonicRegression(out_of_bounds="clip"), p, y,
                                            cv=StratifiedKFold(5)),
            "contiguous": cross_val_predict(IsotonicRegression(out_of_bounds="clip"), p, y,
                                            cv=KFold(5)),
            "shuffled": cross_val_predict(IsotonicRegression(out_of_bounds="clip"), p, y,
                                          cv=StratifiedKFold(5, shuffle=True, random_state=0)),
        }
        prevalences = [round(float(y[te].mean()), 3) for _, te in KFold(5).split(p.reshape(-1, 1))]
        for name, pc in cals.items():
            scheme_rows.append({"endpoint": ep, "scheme": name,
                                "ece_calibrated": round(expected_calibration_error(y, pc), 4),
                                "brier_calibrated": round(brier_score_loss(y, pc), 4)})
        p_cal = cals["stratified"]
        rows.append({
            "endpoint": ep,
            "brier_raw": round(brier_score_loss(y, p), 4),
            "brier_calibrated": round(brier_score_loss(y, p_cal), 4),
            "ece_raw": round(expected_calibration_error(y, p), 4),
            "ece_calibrated": round(expected_calibration_error(y, p_cal), 4),
            "n": len(y),
            "n_at_one_raw": int((p >= 1.0).sum()),
            "n_at_one_cal": int((p_cal >= 1.0).sum()),
            "contiguous_fold_prevalences": "/".join(f"{v:.3f}" for v in prevalences),
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    # brier_raw depends on neither the metric fix nor the fold scheme, so it is the control that
    # the out-of-fold predictions being read are the ones the stored table was built from.
    # brier_calibrated legitimately moves with the fold scheme and is no longer a control.
    brier_ok = True
    for r in rows:
        s = stored.loc[r["endpoint"]]
        if abs(float(s["brier_raw"]) - r["brier_raw"]) > 1e-9:
            print(f"  CONTROL FAILED {r['endpoint']} brier_raw: "
                  f"stored {s['brier_raw']} vs {r['brier_raw']}")
            brier_ok = False
    print(f"\nbrier_raw control (must match the stored table exactly): "
          f"{'PASS' if brier_ok else 'FAIL'}")

    sch = pd.DataFrame(scheme_rows)
    piv = sch.pivot(index="endpoint", columns="scheme", values="ece_calibrated")
    piv = piv[["contiguous", "stratified", "shuffled"]]
    print("\n=== calibrated ECE by inner fold scheme ===")
    print(piv.to_string())
    print("\npanel mean:")
    print(piv.mean().round(4).to_string())
    print("\ncontiguous is what cv=5 gave IsotonicRegression and what was published; "
          "stratified is what CalibratedClassifierCV(cv=5) deploys.")
    harm = [r["endpoint"] for r in rows
            if r["brier_calibrated"] > float(stored.loc[r["endpoint"], "brier_raw"])]
    print(f"endpoints where calibration now worsens the Brier score: {harm or 'none'}")

    print(f"\nmean ECE raw        {out.ece_raw.mean():.4f}   "
          f"stored {stored.ece_raw.mean():.4f}")
    print(f"mean ECE calibrated {out.ece_calibrated.mean():.4f}   "
          f"stored {stored.ece_calibrated.mean():.4f}")
    print(f"\npredictions at exactly 1.0: raw {out.n_at_one_raw.sum():,} "
          f"calibrated {out.n_at_one_cal.sum():,} of {out.n.sum():,} "
          f"({100 * out.n_at_one_cal.sum() / out.n.sum():.2f}% calibrated)")

    if not brier_ok:
        print("\nnot writing: the control failed, so this is not a faithful reproduction")
        return 1
    if not args.write:
        print("\n(dry run; pass --write to update the table)")
        return 0

    # Written only under --write. It used to be written unconditionally, so a dry run
    # modified a table check_freshness.py tracks, which is the opposite of a dry run.
    sch.to_csv(ROOT / "results" / "tables" / "calibration_fold_schemes.csv", index=False)
    print(f"wrote results/tables/calibration_fold_schemes.csv ({len(sch)} rows)")
    out[["endpoint", "brier_raw", "brier_calibrated", "ece_raw",
         "ece_calibrated"]].to_csv(TABLE, index=False)
    print(f"\nwrote {TABLE.relative_to(ROOT).as_posix()} "
          f"(deployment models untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
