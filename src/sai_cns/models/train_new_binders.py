"""Decoy-aware binder classifiers for the 14 new brain-target endpoints.

Same method as train_receptor_binders.py (these ChEMBL sets are 82-99% actives, so only a
binder-vs-decoy classifier is well posed): positives are measured binders (pChEMBL >= 7),
negatives are property-matched decoys from the 74k background library with max ECFP Tanimoto
< 0.35 to any positive. Each model is a single compact random forest with prefit sigmoid
calibration, saved compressed to models_rf/<TARGET>_binder.joblib, with honest metrics
(hard-decoy AUROC, background false-positive rate). Run:
  python src/sai_cns/models/train_new_binders.py
"""
from __future__ import annotations

import json
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
from models.pools import background_pools, scaffold_holdout, write_holdout  # noqa: E402

TARGETS = ["HT1A", "HT6", "HT7", "H3", "DAT", "NET", "Sigma1", "CB1",
           "OPRK1", "OPRM1", "D3", "A1", "a7nAChR", "LRRK2"]
ACTIVE_PCHEMBL = 7.0
DECOY_RATIO = 3
ACTIVE_HOLDOUT = 0.20
TANIMOTO_MAX = 0.35
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
rng = np.random.default_rng(42)


def _fp(smi):
    m = Chem.MolFromSmiles(smi)
    return _GEN.GetFingerprint(m) if m else None


def main():
    # Decoys from the decoy pool; the background rate below is measured on a disjoint pool, so it
    # is not a restatement of what the model was trained to score as zero.
    pools = background_pools(with_fingerprints=True)
    bg_smiles, bg_fps = pools["decoy"]
    eval_pool, _ = pools["evaluation"]   # (smiles, fingerprints); only smiles needed here
    bg_smiles = np.array(bg_smiles)
    bgX, bg_mask = featurize(list(bg_smiles))
    bg_smiles_ok = bg_smiles[bg_mask]
    bg_fps_ok = [f for f, ok in zip(bg_fps, bg_mask) if ok]
    bg_mw, bg_clogp = bgX[:, -12], bgX[:, -11]

    active_sets = {}
    for ep in TARGETS:
        df = pd.read_csv(ROOT / "data" / "endpoints" / f"{ep}.csv")
        p = pd.to_numeric(df["pchembl"], errors="coerce")
        active_sets[ep] = set(df.loc[p >= ACTIVE_PCHEMBL, "smiles"].astype(str))

    summary = {}
    for ep in TARGETS:
        actives = sorted(active_sets[ep])
        aX, amask = featurize(actives)
        actives = [s for s, ok in zip(actives, amask) if ok]
        afps = [_fp(s) for s in actives]
        a_mw, a_clogp = aX[:, -12], aX[:, -11]
        mw_lo, mw_hi = np.quantile(a_mw, 0.02), np.quantile(a_mw, 0.98)
        lp_lo, lp_hi = np.quantile(a_clogp, 0.02), np.quantile(a_clogp, 0.98)

        cand = ((bg_mw >= mw_lo) & (bg_mw <= mw_hi) & (bg_clogp >= lp_lo) & (bg_clogp <= lp_hi))
        cand_idx = [i for i in np.where(cand)[0] if bg_smiles_ok[i] not in active_sets[ep]]
        rng.shuffle(cand_idx)
        need = DECOY_RATIO * len(actives)
        decoys, hard = [], []
        for i in cand_idx:
            f = bg_fps_ok[i]
            if f is None:
                continue
            mt = max(DataStructs.BulkTanimotoSimilarity(f, afps))
            if mt < TANIMOTO_MAX and len(decoys) < need:
                decoys.append(str(bg_smiles_ok[i]))
            elif TANIMOTO_MAX <= mt < 0.55 and len(hard) < len(actives):
                hard.append(str(bg_smiles_ok[i]))
            if len(decoys) >= need and len(hard) >= len(actives):
                break

        # Whole active scaffold groups withheld and never trained on.
        a_hold = scaffold_holdout(_scaffold_groups(actives), ACTIVE_HOLDOUT, rng)
        act_train = [s for s, h in zip(actives, a_hold) if not h]
        act_hold = [s for s, h in zip(actives, a_hold) if h]
        write_holdout(ep, active_holdout=act_hold, active_train=act_train)

        smiles = act_train + decoys
        y = np.array([1] * len(act_train) + [0] * len(decoys))
        X, mask = featurize(smiles)
        y = y[mask]; smiles = [s for s, ok in zip(smiles, mask) if ok]
        groups = _scaffold_groups(smiles)

        oof = cross_val_predict(
            RandomForestClassifier(200, min_samples_leaf=6, n_jobs=-1, random_state=42, class_weight="balanced"),
            X, y, groups=groups, cv=GroupKFold(10), method="predict_proba", n_jobs=-1)[:, 1]
        auroc_easy = roc_auc_score(y, oof)

        Xt, Xc, yt, yc = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        forest = RandomForestClassifier(200, min_samples_leaf=6, n_jobs=-1, random_state=42,
                                        class_weight="balanced").fit(Xt, yt)
        cal = CalibratedClassifierCV(FrozenEstimator(forest), method="sigmoid").fit(Xc, yc)
        joblib.dump(cal, ROOT / "models_rf" / f"{ep}_binder.joblib", compress=3)

        auroc_hard, sens = None, None
        paX, _ = featurize(act_hold) if act_hold else (None, None)
        if len(hard) >= 30 and paX is not None and len(paX):
            hX, hm = featurize(hard)
            # Withheld actives, not Xc[yc == 1]: cal was fitted on Xc, so those were not held out.
            pa = cal.predict_proba(paX)[:, 1]
            ph = cal.predict_proba(hX)[:, 1]
            auroc_hard = float(roc_auc_score(np.r_[np.ones(len(pa)), np.zeros(len(ph))], np.r_[pa, ph]))
            sens = float((pa >= 0.5).mean())
        bg_idx = rng.choice(len(eval_pool), size=min(2500, len(eval_pool)), replace=False)
        bg_samp = [eval_pool[i] for i in bg_idx if eval_pool[i] not in active_sets[ep]]
        bX, bm = featurize(bg_samp)
        fpr = float((cal.predict_proba(bX)[:, 1] >= 0.5).mean())

        meta = {"endpoint": ep, "task": "binder_vs_decoy", "n_active": int((y == 1).sum()),
                "n_decoy": int((y == 0).sum()), "auroc_easy_decoys": round(float(auroc_easy), 3),
                "auroc_hard_decoys": round(auroc_hard, 3) if auroc_hard else None,
                "background_fpr@0.5": round(fpr, 3),
                "n_active_holdout": len(act_hold),
                "sensitivity_at_0.5": round(sens, 3) if sens is not None else None,
                "sensitivity_basis": "held_out_actives_by_scaffold" if sens is not None else None,
                "background_fpr_basis": "held_out_evaluation_pool"}
        (ROOT / "models_rf" / f"{ep}_binder_meta.json").write_text(json.dumps(meta, indent=2))
        summary[ep] = meta
        print(f"[{ep}] act={meta['n_active']} dec={meta['n_decoy']} | AUROC easy={auroc_easy:.3f} "
              f"hard={auroc_hard if auroc_hard is None else round(auroc_hard, 3)} | bg-FPR={fpr:.3f}", flush=True)
    print("DONE\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
