"""Train the second expansion batch, choosing the appropriate task per target.

Many of these targets are enzymes rather than receptors, and enzyme assays in ChEMBL often contain a
genuine measured-inactive population. Where measured inactives are sufficient (at least 15% of the
set and at least 150 compounds) we train a standard classifier on measured labels, which is
preferable because the negatives are real. Where the set is binder-dominated we fall back to the
decoy-aware scheme used for the receptor panel. The choice is recorded in the metadata so the
manuscript can report it per endpoint.

Validation: scaffold-grouped 10-fold out-of-fold AUROC for measured-label classifiers; hard
(near-miss) decoy AUROC plus background false-positive rate for decoy-aware classifiers.
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

TARGETS = ["NLRP3", "P2X7", "COX2", "CSF1R", "PDE10A", "HDAC1", "HDAC6", "GluN2B", "mGluR5",
           "GABA_A", "OX1", "OX2", "MT1", "mTOR", "SIRT1", "KEAP1", "GBA1", "PDE4B", "Nav1_5"]
ACTIVE_P, DECOY_RATIO, TAN_MAX = 7.0, 3, 0.35
MIN_INACT_FRAC, MIN_INACT_N = 0.15, 150
ACTIVE_HOLDOUT = 0.20
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
rng = np.random.default_rng(42)
RF = dict(n_estimators=200, min_samples_leaf=6, n_jobs=-1, random_state=42, class_weight="balanced")


def _fp(s):
    m = Chem.MolFromSmiles(s)
    return _GEN.GetFingerprint(m) if m else None


def main():
    # Decoys from the decoy pool; the background rate is measured on the disjoint evaluation pool.
    pools = background_pools(with_fingerprints=True)
    bg_smiles, bg_fps = pools["decoy"]
    eval_pool, _ = pools["evaluation"]   # (smiles, fingerprints); only smiles needed here
    bg_smiles = np.array(bg_smiles)
    bgX, bgm = featurize(list(bg_smiles))
    bg_ok = bg_smiles[bgm]
    bg_fp_ok = [f for f, k in zip(bg_fps, bgm) if k]
    bg_mw, bg_lp = bgX[:, -12], bgX[:, -11]

    summary = {}
    for ep in TARGETS:
        f = ROOT / "data" / "endpoints" / f"{ep}.csv"
        if not f.exists():
            print(f"[{ep}] no data file, skipping", flush=True)
            continue
        df = pd.read_csv(f).dropna(subset=["smiles", "label"])
        n_inact = int((df.label == 0).sum())
        frac = n_inact / max(len(df), 1)
        measured = (frac >= MIN_INACT_FRAC and n_inact >= MIN_INACT_N)

        act_hold = []
        if measured:
            all_smiles = df["smiles"].astype(str).tolist()
            all_y = df["label"].to_numpy().astype(int)
            pos = [s for s, lab in zip(all_smiles, all_y) if lab == 1]
            hold = scaffold_holdout(_scaffold_groups(pos), ACTIVE_HOLDOUT, rng)
            act_hold = [s for s, h in zip(pos, hold) if h]
            held = set(act_hold)
            keep = [i for i, s in enumerate(all_smiles) if s not in held]
            smiles = [all_smiles[i] for i in keep]
            y = all_y[keep]
            write_holdout(ep, active_holdout=act_hold)
            mode = "measured_labels"
        else:
            act = sorted(set(df.loc[pd.to_numeric(df.pchembl, errors="coerce") >= ACTIVE_P, "smiles"].astype(str)))
            if len(act) < 120:
                print(f"[{ep}] only {len(act)} binders, skipping", flush=True)
                continue
            aX, am = featurize(act)
            act = [s for s, k in zip(act, am) if k]
            afps = [_fp(s) for s in act]
            mw_lo, mw_hi = np.quantile(aX[:, -12], .02), np.quantile(aX[:, -12], .98)
            lp_lo, lp_hi = np.quantile(aX[:, -11], .02), np.quantile(aX[:, -11], .98)
            cand = (bg_mw >= mw_lo) & (bg_mw <= mw_hi) & (bg_lp >= lp_lo) & (bg_lp <= lp_hi)
            aset = set(act)
            idx = [i for i in np.where(cand)[0] if bg_ok[i] not in aset]
            rng.shuffle(idx)
            need = DECOY_RATIO * len(act)
            dec, hard = [], []
            for i in idx:
                fp = bg_fp_ok[i]
                if fp is None:
                    continue
                mt = max(DataStructs.BulkTanimotoSimilarity(fp, afps))
                if mt < TAN_MAX and len(dec) < need:
                    dec.append(str(bg_ok[i]))
                elif TAN_MAX <= mt < 0.55 and len(hard) < len(act):
                    hard.append(str(bg_ok[i]))
                if len(dec) >= need and len(hard) >= len(act):
                    break
            hold = scaffold_holdout(_scaffold_groups(act), ACTIVE_HOLDOUT, rng)
            act_train = [s for s, h in zip(act, hold) if not h]
            act_hold = [s for s, h in zip(act, hold) if h]
            write_holdout(ep, active_holdout=act_hold, active_train=act_train)
            smiles = act_train + dec
            y = np.array([1] * len(act_train) + [0] * len(dec))
            mode = "decoy_aware"

        X, mask = featurize(smiles)
        y = y[mask]
        smiles = [s for s, k in zip(smiles, mask) if k]
        if len(set(y)) < 2 or len(y) < 200:
            print(f"[{ep}] insufficient usable data, skipping", flush=True)
            continue
        groups = _scaffold_groups(smiles)

        oof = cross_val_predict(RandomForestClassifier(**RF), X, y, groups=groups,
                                cv=GroupKFold(10), method="predict_proba", n_jobs=-1)[:, 1]
        auroc_scaffold = float(roc_auc_score(y, oof))

        Xt, Xc, yt, yc = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        forest = RandomForestClassifier(**RF).fit(Xt, yt)
        cal = CalibratedClassifierCV(FrozenEstimator(forest), method="sigmoid").fit(Xc, yc)
        joblib.dump(cal, ROOT / "models_rf" / f"{ep}_binder.joblib", compress=3)

        meta = {"endpoint": ep, "mode": mode, "n_positive": int((y == 1).sum()),
                "n_negative": int((y == 0).sum()),
                "measured_inactive_fraction": round(float(frac), 3),
                "scaffold_cv_auroc": round(auroc_scaffold, 3)}
        if mode == "decoy_aware":
            paX, _ = featurize(act_hold) if act_hold else (None, None)
            if len(hard) >= 30 and paX is not None and len(paX):
                hX, _ = featurize(hard)
                # Withheld actives. Xc[yc == 1] is the calibration split, which cal was fitted on.
                pa = cal.predict_proba(paX)[:, 1]
                ph = cal.predict_proba(hX)[:, 1]
                meta["auroc_hard_decoys"] = round(float(roc_auc_score(
                    np.r_[np.ones(len(pa)), np.zeros(len(ph))], np.r_[pa, ph])), 3)
                meta["sensitivity_at_0.5"] = round(float((pa >= 0.5).mean()), 3)
                meta["sensitivity_basis"] = "held_out_actives_by_scaffold"
            meta["n_active_holdout"] = len(act_hold)
            meta["background_fpr_basis"] = "held_out_evaluation_pool"
            bi = rng.choice(len(eval_pool), size=min(2000, len(eval_pool)), replace=False)
            bs = [eval_pool[i] for i in bi]
            bX, _ = featurize(bs)
            meta["background_fpr@0.5"] = round(float((cal.predict_proba(bX)[:, 1] >= 0.5).mean()), 3)
        (ROOT / "models_rf" / f"{ep}_binder_meta.json").write_text(json.dumps(meta, indent=2))
        summary[ep] = meta
        print(f"[{ep}] {mode:15} pos={meta['n_positive']:5d} neg={meta['n_negative']:5d} "
              f"scaffoldAUROC={auroc_scaffold:.3f} hard={meta.get('auroc_hard_decoys','-')} "
              f"bgFPR={meta.get('background_fpr@0.5','-')}", flush=True)
    print("DONE\n" + json.dumps(summary, indent=2)[:800])


if __name__ == "__main__":
    main()
