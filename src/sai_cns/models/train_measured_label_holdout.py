"""Retrain the remaining measured-label models with a held-out inactive split.

SIRT1 and Nav1.5 have too few binders at pChEMBL >= 7 for the hybrid binder scheme, so they are
trained on the measured activity label instead (active at pChEMBL >= 6, inactive below 5). In the
earlier pass these models were evaluated on the full set of measured inactives, including the ones
used to fit them, which inflates their apparent performance. Half the measured inactives are now
withheld from training and used for validation and threshold calibration.

That alone was not enough, and this docstring previously claimed more than the code delivered. Every
positive still entered training and sensitivity was still reported on all of them, so the figure
measured recall of memorised compounds. A fifth of the active scaffold groups is now withheld too,
and sensitivity is reported on those, matching the hybrid binder panel so the numbers are
comparable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GroupKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize  # noqa: E402
from models.train_rf import _scaffold_groups  # noqa: E402
import panel  # noqa: E402

M = ROOT / "models_rf"
# Seed list, used only to bootstrap a panel that has no registry yet. Everything after that first
# build comes from panel.py: this list still said ["SIRT1", "Nav1_5"] when eight endpoints used this
# mode, so `make train` refitted two of them and left six carrying weights from an older
# featurisation while the rest of the panel moved, without any error.
SEED_TARGETS = ["SIRT1", "Nav1_5"]
MODE = panel.MEASURED_LABEL
TARGET_FPR = 0.10
ACTIVE_HOLDOUT = 0.20
RF = dict(n_estimators=300, min_samples_leaf=4, n_jobs=-1, random_state=42, class_weight="balanced")
rng = np.random.default_rng(42)


def _targets() -> list[str]:
    """Every endpoint this script is responsible for, read from the panel registry.

    Derived rather than declared so the list cannot fall behind the panel. Falls back to the seed
    list only when there is no registry yet, which is the first-ever build.
    """
    if not panel.MODES.exists():
        return list(SEED_TARGETS)
    return panel.names(mode=MODE)


def main():
    # Targets may be named on the command line, so an addition can be trained without refitting the
    # rest. _train_small.py existed only to carry a different TARGETS list and is superseded by this.
    TARGETS = sys.argv[1:] if len(sys.argv) > 1 else _targets()
    if len(sys.argv) > 1:
        print(f"training only: {', '.join(TARGETS)}", flush=True)
    else:
        print(f"training all {len(TARGETS)} endpoints in mode {MODE}: "
              f"{', '.join(TARGETS)}", flush=True)
    modes = json.loads((M / "binder_modes.json").read_text())
    for ep in TARGETS:
        f = ROOT / "data" / "endpoints" / f"{ep}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f).dropna(subset=["smiles", "label"])
        pos = df.loc[df.label == 1, "smiles"].astype(str).tolist()
        neg_all = list(rng.permutation(df.loc[df.label == 0, "smiles"].astype(str).tolist()))
        half = len(neg_all) // 2
        neg_train, neg_hold = neg_all[:half], neg_all[half:]

        # Withhold a fifth of the active scaffold groups as well. Holding out only the negatives and
        # then reporting sensitivity on every positive measures recall of the training set, which is
        # the thing this file's own docstring claimed to have fixed.
        pos_groups = _scaffold_groups(pos)
        uniq = np.unique(pos_groups)
        held = set(rng.permutation(uniq)[:max(1, int(round(ACTIVE_HOLDOUT * len(uniq))))])
        pos_hold_mask = np.array([g in held for g in pos_groups])
        if pos_hold_mask.all() or not pos_hold_mask.any():
            pos_hold_mask = np.zeros(len(pos), dtype=bool)
        pos_train = [s for s, h in zip(pos, pos_hold_mask) if not h]
        pos_hold = [s for s, h in zip(pos, pos_hold_mask) if h]

        smiles = pos_train + neg_train
        y = np.array([1] * len(pos_train) + [0] * len(neg_train))
        X, mask = featurize(smiles)
        y = y[mask]
        smiles = [s for s, k in zip(smiles, mask) if k]
        groups = _scaffold_groups(smiles)

        oof = cross_val_predict(RandomForestClassifier(**RF), X, y, groups=groups,
                                cv=GroupKFold(10), method="predict_proba", n_jobs=-1)[:, 1]
        auroc_cv = float(roc_auc_score(y, oof))

        Xt, Xc, yt, yc = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        forest = RandomForestClassifier(**RF).fit(Xt, yt)
        cal = CalibratedClassifierCV(FrozenEstimator(forest), method="sigmoid").fit(Xc, yc)

        hX, _ = featurize(neg_hold)
        pX, _ = featurize(pos_hold)
        ph = cal.predict_proba(hX)[:, 1]
        pa = cal.predict_proba(pX)[:, 1]
        thr = float(np.clip(np.quantile(ph, 1.0 - TARGET_FPR), 0.05, 0.999))
        rec = {"mode": "measured_labels_holdout",
               "n_positive": int(len(pos_train)), "n_active_holdout": len(pos_hold),
               "n_measured_inactive_train": len(neg_train),
               "n_measured_inactive": len(neg_hold),
               "scaffold_cv_auroc": round(auroc_cv, 3),
               "threshold": round(thr, 4), "threshold_basis": "held_out_measured_inactives",
               "auroc_vs_measured_inactives": round(float(roc_auc_score(
                   np.r_[np.ones(len(pa)), np.zeros(len(ph))], np.r_[pa, ph])), 3),
               "fpr_in_sample_on_threshold_set": round(float((ph >= thr).mean()), 3),
               "sensitivity_at_threshold": round(float((pa >= thr).mean()), 3),
               "sensitivity_basis": "held_out_actives_by_scaffold"}
        rec["reliable_call"] = panel.passes_gate(
            rec["sensitivity_at_threshold"], rec["auroc_vs_measured_inactives"])
        old = modes.get(ep, {})
        print(f"[{ep:8}] AUROC {rec['auroc_vs_measured_inactives']:.3f} "
              f"(previously {old.get('auroc_vs_measured_inactives','-')}) | "
              f"sens {rec['sensitivity_at_threshold']:.3f} | thr {thr:.3f} | "
              f"held-out inactives {len(neg_hold)}", flush=True)
        (M / "holdout").mkdir(parents=True, exist_ok=True)
        (M / "holdout" / f"{ep}_binder_holdout.json").write_text(json.dumps({
            "endpoint": ep, "active_holdout": pos_hold,
            "measured_inactive_holdout": neg_hold,
            "active_train": pos_train, "measured_inactive_train": neg_train}),
            encoding="utf-8")
        joblib.dump(cal, M / f"{ep}_binder.joblib", compress=3)
        modes[ep] = rec

    (M / "binder_modes.json").write_text(json.dumps(modes, indent=2))
    good = [v for v in modes.values() if "auroc_vs_measured_inactives" in v]
    print(f"\npanel mean AUROC {np.mean([g['auroc_vs_measured_inactives'] for g in good]):.3f} | "
          f"mean sensitivity {np.mean([g['sensitivity_at_threshold'] for g in good]):.3f} | n={len(good)}")
    print("DONE")


if __name__ == "__main__":
    main()
