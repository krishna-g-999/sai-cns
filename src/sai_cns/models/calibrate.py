"""Probability calibration for the classification endpoints.

Random-forest class probabilities are systematically miscalibrated (they cluster toward the middle),
so a raw 0.72 is not a true 72% chance. This fits an isotonic calibration per endpoint and reports
the honest improvement, then saves calibrated deployment models.

Method. The calibration quality is measured on the already-saved out-of-fold predictions (random
10-fold), so nothing is evaluated on data it was fit on: the raw out-of-fold probabilities are
recalibrated by isotonic regression under an inner 5-fold, and the Brier score and expected
calibration error are reported before and after. Deployment models wrap the random forest in
CalibratedClassifierCV (isotonic, 5-fold) fit on all data.

Outputs:
  results/tables/calibration.csv          Brier and ECE, raw vs calibrated, per endpoint
  models_rf/<endpoint>_calibrated.joblib  calibrated deployment model
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.featurize import featurize  # noqa: E402
from models.train_rf import CLASSIFICATION, RF_COMMON, SEED, _load  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OOF = ROOT / "data" / "processed" / "cv_predictions"


def expected_calibration_error(y, p, n_bins=10):
    """Mean gap between confidence and accuracy across equal-width probability bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def main() -> None:
    rows = []
    for ep in CLASSIFICATION:
        oof = pd.read_csv(OOF / f"{ep}_random_oof.csv")
        y = oof["y_true"].to_numpy().astype(int)
        p = oof["prediction"].to_numpy().astype(float)

        # honest recalibration on the out-of-fold probabilities (inner 5-fold)
        p_cal = cross_val_predict(IsotonicRegression(out_of_bounds="clip"), p, y, cv=5)
        rows.append({
            "endpoint": ep,
            "brier_raw": round(brier_score_loss(y, p), 4),
            "brier_calibrated": round(brier_score_loss(y, p_cal), 4),
            "ece_raw": round(expected_calibration_error(y, p), 4),
            "ece_calibrated": round(expected_calibration_error(y, p_cal), 4),
        })
        print(f"[{ep}] Brier {rows[-1]['brier_raw']} -> {rows[-1]['brier_calibrated']}   "
              f"ECE {rows[-1]['ece_raw']} -> {rows[-1]['ece_calibrated']}")

        # calibrated deployment model on all data
        df = _load(ep).dropna(subset=["smiles", "label"]).reset_index(drop=True)
        X, mask = featurize(df["smiles"].tolist())
        yv = df.loc[mask, "label"].to_numpy().astype(int)
        base = RandomForestClassifier(class_weight="balanced", **RF_COMMON)
        cal = CalibratedClassifierCV(base, method="isotonic", cv=5)
        cal.fit(X, yv)
        joblib.dump(cal, ROOT / "models_rf" / f"{ep}_calibrated.joblib")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tables" / "calibration.csv", index=False)
    print("\n=== calibration (lower is better) ===")
    print(out.to_string(index=False))
    print(f"\nmean Brier {out.brier_raw.mean():.3f} -> {out.brier_calibrated.mean():.3f}; "
          f"mean ECE {out.ece_raw.mean():.3f} -> {out.ece_calibrated.mean():.3f}")
    print("wrote results/tables/calibration.csv and models_rf/*_calibrated.joblib")


if __name__ == "__main__":
    main()
