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
from sklearn.model_selection import cross_val_predict

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
    rows = []
    for ep in CLASSIFICATION:
        oof = pd.read_csv(OOF / f"{ep}_random_oof.csv")
        y = oof["y_true"].to_numpy().astype(int)
        p = oof["prediction"].to_numpy().astype(float)
        p_cal = cross_val_predict(IsotonicRegression(out_of_bounds="clip"), p, y, cv=5)
        rows.append({
            "endpoint": ep,
            "brier_raw": round(brier_score_loss(y, p), 4),
            "brier_calibrated": round(brier_score_loss(y, p_cal), 4),
            "ece_raw": round(expected_calibration_error(y, p), 4),
            "ece_calibrated": round(expected_calibration_error(y, p_cal), 4),
            "n": len(y),
            "n_at_one_raw": int((p >= 1.0).sum()),
            "n_at_one_cal": int((p_cal >= 1.0).sum()),
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    brier_ok = True
    for r in rows:
        s = stored.loc[r["endpoint"]]
        for col in ("brier_raw", "brier_calibrated"):
            if abs(float(s[col]) - r[col]) > 1e-9:
                print(f"  CONTROL FAILED {r['endpoint']} {col}: stored {s[col]} vs {r[col]}")
                brier_ok = False
    print(f"\nBrier control (must match the stored table exactly): "
          f"{'PASS' if brier_ok else 'FAIL'}")

    print(f"\nmean ECE raw        {out.ece_raw.mean():.4f}   "
          f"stored {stored.ece_raw.mean():.4f}")
    print(f"mean ECE calibrated {out.ece_calibrated.mean():.4f}   "
          f"stored {stored.ece_calibrated.mean():.4f}   (README quotes 0.0147)")
    print(f"\npredictions at exactly 1.0: raw {out.n_at_one_raw.sum():,} "
          f"calibrated {out.n_at_one_cal.sum():,} of {out.n.sum():,} "
          f"({100 * out.n_at_one_cal.sum() / out.n.sum():.2f}% calibrated)")

    if not brier_ok:
        print("\nnot writing: the control failed, so this is not a faithful reproduction")
        return 1
    if not args.write:
        print("\n(dry run; pass --write to update the table)")
        return 0

    out[["endpoint", "brier_raw", "brier_calibrated", "ece_raw",
         "ece_calibrated"]].to_csv(TABLE, index=False)
    print(f"\nwrote {TABLE.relative_to(ROOT).as_posix()} "
          f"(deployment models untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
