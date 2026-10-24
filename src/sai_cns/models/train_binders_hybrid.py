"""Hybrid retraining: property-matched decoys PLUS experimentally measured inactives.

Decoy-only training has a specific failure mode. Because the negatives are chosen to be structurally
dissimilar, the model becomes over-confident and saturates: for melatonin MT1 the measured inactives
score a median binder probability of 0.972 against an active median of 0.997, so the decision
boundary sits in a region where tiny probability differences swing the result. Sensitivity at a
controlled false-positive rate then collapses (MT1 0.52, KEAP1 0.33).

The fix is to give the model the hard negatives it never saw. Measured inactives (compounds tested
against the target and found inactive) are split in half: one half joins the training negatives
alongside the decoys, the other half is held out and used for unbiased validation and for setting
the decision threshold. Positives remain measured binders at pChEMBL >= 7.

Every model is scaffold-grouped 10-fold cross-validated, then refit and calibrated.

What is withheld, and why each number comes from a different place. Actives are split by scaffold and
a fifth of the scaffold groups never enter training, so sensitivity is measured on structurally
distinct compounds rather than on recall of the training set. Measured inactives are split in half,
one half training as hard negatives and the other setting the threshold. Decoys are drawn only from
the decoy pool, and the target's own measured inactives are excluded from decoy eligibility, so a
compound cannot be both a trained negative and an unseen one. The false-positive rate is reported on
the evaluation pool, which is disjoint from both: measuring it on the sample the threshold quantile
came from returns that quantile whatever the model does.

Both halves of every split are written to models_rf/holdout/<T>_binder_holdout.json, so a later
threshold step cannot reach past the holdout by re-reading the endpoint table.

Outputs models_rf/<T>_binder.joblib and refreshed binder_modes.json entries.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GroupKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize  # noqa: E402
from models.train_rf import _scaffold_groups  # noqa: E402
from models.pools import background_pools  # noqa: E402
import panel  # noqa: E402

M = ROOT / "models_rf"
ACTIVE_P, DECOY_RATIO, TAN_MAX = 7.0, 3, 0.35
ACTIVE_HOLDOUT = 0.20   # share of active scaffold groups withheld from training
TARGET_FPR = 0.10
# One definition of the applicability-domain fingerprint. A private copy here compared
# actives against reference fingerprints built with a different definition; see
# src/sai_cns/ad_fingerprint.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "sai_cns"))
import ad_fingerprint as _adfp  # noqa: E402

_GEN = _adfp.generator()
rng = np.random.default_rng(42)
RF = dict(n_estimators=300, min_samples_leaf=4, n_jobs=-1, random_state=42, class_weight="balanced")

# Seed list, used only to bootstrap a panel with no registry yet. Everything after that first build
# comes from panel.py, because a hardcoded list stops covering the panel the moment the panel grows
# and does so silently: this one held 37 names while 44 endpoints used this mode, so `make train`
# refitted 39 of 52 binders and said DONE.
SEED_TARGETS = ["HT1A", "HT6", "HT7", "H3", "DAT", "NET", "Sigma1", "CB1", "OPRK1", "OPRM1", "D3",
                "A1", "a7nAChR", "LRRK2", "NLRP3", "P2X7", "COX2", "CSF1R", "PDE10A", "HDAC1",
                "HDAC6", "GluN2B", "mGluR5", "GABA_A", "OX1", "OX2", "MT1", "mTOR", "SIRT1",
                "KEAP1", "GBA1", "PDE4B", "Nav1_5", "D2", "A2A", "HT2A", "SERT"]
MODE = panel.HYBRID


def _targets() -> list[str]:
    """Every endpoint this script is responsible for, from the panel registry."""
    if not panel.MODES.exists():
        return list(SEED_TARGETS)
    return panel.names(mode=MODE)


def _fp(s):
    m = Chem.MolFromSmiles(s)
    return _GEN.GetFingerprint(m) if m else None


def main():
    # Targets may be named on the command line to train an addition without refitting the panel.
    # Existing entries in binder_modes.json are merged, not replaced, so a partial run leaves every
    # untouched endpoint exactly as it was validated. Retraining a deployed model silently would
    # invalidate every number already reported for it.
    TARGETS = sys.argv[1:] if len(sys.argv) > 1 else _targets()
    if len(sys.argv) > 1:
        print(f"training only: {', '.join(TARGETS)}", flush=True)
    else:
        print(f"training all {len(TARGETS)} endpoints in mode {MODE}", flush=True)
    # Decoys come from the decoy pool only. Drawing them from the whole background library put the
    # same compounds into training and into the sample the false-positive rate is later measured on.
    pools = background_pools(with_fingerprints=True)
    dec_smiles, dec_fps = pools["decoy"]
    eval_smiles, _ = pools["evaluation"]
    print(f"background pools: {len(dec_smiles)} decoy, {len(pools['threshold'][0])} threshold, "
          f"{len(eval_smiles)} evaluation (disjoint)", flush=True)

    bg_smiles = np.array(dec_smiles)
    bgX, bgm = featurize(list(bg_smiles))
    bg_ok = bg_smiles[bgm]
    bg_fp_ok = [f for f, k in zip(dec_fps, bgm) if k]
    bg_mw, bg_lp = bgX[:, -12], bgX[:, -11]

    evalX, evalm = featurize(eval_smiles)
    HOLDOUT_DIR = M / "holdout"
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    modes = json.loads((M / "binder_modes.json").read_text()) if (M / "binder_modes.json").exists() else {}
    for ep in TARGETS:
        f = ROOT / "data" / "endpoints" / f"{ep}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f).dropna(subset=["smiles"])
        p = pd.to_numeric(df.get("pchembl"), errors="coerce")
        act = sorted(set(df.loc[p >= ACTIVE_P, "smiles"].astype(str)))
        ina_all = sorted(set(df.loc[df["label"] == 0, "smiles"].astype(str)) - set(act))
        if len(act) < 100:
            print(f"[{ep}] too few binders ({len(act)}), skipped", flush=True)
            continue

        # split measured inactives: half to train as hard negatives, half held out
        ina_all = list(rng.permutation(ina_all))
        half = len(ina_all) // 2
        ina_train, ina_hold = ina_all[:half], ina_all[half:]

        aX, am = featurize(act)
        act = [s for s, k in zip(act, am) if k]
        afps = [_fp(s) for s in act]

        # Hold out actives too, by scaffold, and never train on them. Reporting sensitivity on the
        # positives the model was fitted to measures recall of memorised compounds, which is not
        # what the number is read as. The split is by scaffold rather than at random so a held-out
        # active is a structurally different compound, not a close analogue of a training one.
        a_groups = _scaffold_groups(act)
        uniq = np.unique(a_groups)
        held_groups = set(rng.permutation(uniq)[:max(1, int(round(ACTIVE_HOLDOUT * len(uniq))))])
        a_hold_mask = np.array([g in held_groups for g in a_groups])
        if a_hold_mask.all() or not a_hold_mask.any():   # degenerate scaffold structure
            a_hold_mask = np.zeros(len(act), dtype=bool)
        act_train = [s for s, h in zip(act, a_hold_mask) if not h]
        act_hold = [s for s, h in zip(act, a_hold_mask) if h]
        aX_hold = aX[a_hold_mask]

        mw_lo, mw_hi = np.quantile(aX[:, -12], .02), np.quantile(aX[:, -12], .98)
        lp_lo, lp_hi = np.quantile(aX[:, -11], .02), np.quantile(aX[:, -11], .98)
        cand = (bg_mw >= mw_lo) & (bg_mw <= mw_hi) & (bg_lp >= lp_lo) & (bg_lp <= lp_hi)
        # Exclude the target's measured inactives as well as its actives. Half of those inactives
        # are held out to set the threshold, and a compound cannot be both a training decoy and an
        # unseen negative.
        aset = set(act) | set(ina_all)
        idx = [i for i in np.where(cand)[0] if bg_ok[i] not in aset]
        rng.shuffle(idx)
        need = max(DECOY_RATIO * len(act) - len(ina_train), len(act))
        dec = []
        for i in idx:
            fp = bg_fp_ok[i]
            if fp is None:
                continue
            if max(DataStructs.BulkTanimotoSimilarity(fp, afps)) < TAN_MAX:
                dec.append(str(bg_ok[i]))
            if len(dec) >= need:
                break

        smiles = act_train + dec + ina_train
        y = np.array([1] * len(act_train) + [0] * (len(dec) + len(ina_train)))
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

        rec = {"mode": "hybrid_decoys_plus_measured_inactives",
               "n_positive": int((y == 1).sum()), "n_decoy": len(dec),
               "n_active_train": len(act_train), "n_active_holdout": len(act_hold),
               "n_measured_inactive_train": len(ina_train),
               "n_measured_inactive_holdout": len(ina_hold),
               "scaffold_cv_auroc": round(auroc_cv, 3)}

        # Persist what was withheld, so the threshold step cannot silently reach past it. Without
        # this record final_thresholds.py re-reads every measured inactive, including the half that
        # was trained on, and the holdout constructed here is undone.
        (HOLDOUT_DIR / f"{ep}_binder_holdout.json").write_text(json.dumps({
            "endpoint": ep, "active_holdout": act_hold, "measured_inactive_holdout": ina_hold,
            "active_train": act_train, "measured_inactive_train": ina_train}), encoding="utf-8")

        if len(ina_hold) >= 15 and len(act_hold) >= 10:
            hX, _ = featurize(ina_hold)
            ph = cal.predict_proba(hX)[:, 1]
            pa = cal.predict_proba(aX_hold)[:, 1]
            thr = float(np.clip(np.quantile(ph, 1.0 - TARGET_FPR), 0.05, 0.999))
            # The threshold is a quantile of ph, so the rate on ph is that quantile by construction
            # and measures nothing. The evaluation pool is disjoint from both the decoys and the
            # threshold pool, so the rate on it is an observation.
            p_eval = cal.predict_proba(evalX)[:, 1]
            rec.update({
                "threshold": round(thr, 4), "threshold_basis": "held_out_measured_inactives",
                "n_measured_inactive": len(ina_hold),
                "auroc_vs_measured_inactives": round(float(roc_auc_score(
                    np.r_[np.ones(len(pa)), np.zeros(len(ph))], np.r_[pa, ph])), 3),
                "fpr_in_sample_on_threshold_set": round(float((ph >= thr).mean()), 3),
                "background_fpr_held_out": round(float((p_eval >= thr).mean()), 4),
                "n_background_evaluation": int(len(p_eval)),
                "sensitivity_at_threshold": round(float((pa >= thr).mean()), 3),
                "sensitivity_basis": "held_out_actives_by_scaffold"})
            rec["reliable_call"] = panel.passes_gate(
                rec["sensitivity_at_threshold"], rec["auroc_vs_measured_inactives"])
            old = modes.get(ep, {})
            print(f"[{ep:8}] AUROC {rec['auroc_vs_measured_inactives']:.3f} "
                  f"(was {old.get('auroc_vs_measured_inactives','-')}) | sens "
                  f"{rec['sensitivity_at_threshold']:.3f} (was {old.get('sensitivity_at_threshold','-')}) "
                  f"| thr {thr:.3f}", flush=True)
        else:
            rec.update({"threshold": 0.40, "threshold_basis": "global_fallback"})
            print(f"[{ep:8}] too few held-out inactives, fallback threshold", flush=True)

        joblib.dump(cal, M / f"{ep}_binder.joblib", compress=3)
        modes[ep] = rec

    (M / "binder_modes.json").write_text(json.dumps(modes, indent=2))
    good = [v for v in modes.values() if "auroc_vs_measured_inactives" in v]
    if good:
        print(f"\nmean AUROC {np.mean([g['auroc_vs_measured_inactives'] for g in good]):.3f} | "
              f"mean sensitivity {np.mean([g['sensitivity_at_threshold'] for g in good]):.3f} | "
              f"unreliable {[k for k, v in modes.items() if v.get('reliable_call') is False]}")
    print("DONE")


if __name__ == "__main__":
    main()
