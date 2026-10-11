"""Scaffold-held-out retraining of the whole binder panel, for a properly powered prospective test.

The earlier external test was underpowered (n=8) because the ChEMBL corpus already contains almost
every characterised CNS drug, and the few drugs it does not contain are mostly ones whose targets
this tool does not model. The fix is to create genuine prospective conditions rather than hunt for
unseen drugs: hold out entire Bemis-Murcko scaffolds, retrain every binder model on the remaining
scaffolds, and then predict the held-out compounds. A held-out compound shares no scaffold with
anything the model saw, which is the same generalisation demand a new chemical series imposes.

For each target 20% of scaffolds are withheld. Every binder model is retrained on the remaining 80%
and written to models_rf/holdout/<target>_binder.joblib, together with a threshold recalibrated on
the held-out negatives and an independent background sample, so the held-out evaluation never uses a
threshold informed by the compounds it is scoring.

Output: models_rf/holdout/*.joblib, models_rf/holdout/binder_modes.json
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
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize  # noqa: E402
from models.train_rf import _scaffold_groups  # noqa: E402

M = ROOT / "models_rf"
OUT = M / "holdout"
OUT.mkdir(exist_ok=True)
ACTIVE_P, DECOY_RATIO, TAN_MAX = 7.0, 3, 0.35
HOLDOUT_FRAC = 0.20
TARGET_FPR, BACKGROUND_FPR = 0.10, 0.05
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
rng = np.random.default_rng(2026)

TARGETS = ["HT1A", "HT6", "HT7", "H3", "DAT", "NET", "Sigma1", "CB1", "OPRK1", "OPRM1", "D3", "A1",
           "a7nAChR", "LRRK2", "NLRP3", "P2X7", "COX2", "CSF1R", "PDE10A", "HDAC1", "HDAC6",
           "GluN2B", "mGluR5", "GABA_A", "OX1", "OX2", "MT1", "mTOR", "SIRT1", "KEAP1", "GBA1",
           "PDE4B", "Nav1_5", "Nav1_7", "Nav1_1", "TAAR1", "GluA2", "D2", "A2A", "HT2A", "SERT"]


def _fp(s):
    m = Chem.MolFromSmiles(str(s))
    return _GEN.GetFingerprint(m) if m else None


def main():
    # Targets may be named on the command line so an addition to the panel can be given its
    # hold-out twin without refitting, and thereby perturbing, the twins already evaluated.
    global TARGETS
    if len(sys.argv) > 1:
        TARGETS = sys.argv[1:]
        print(f"retraining hold-out twins for: {', '.join(TARGETS)}", flush=True)

    with (M / "ad_reference.pkl").open("rb") as fh:
        bg_smiles, bg_fps = pickle.load(fh)
    bg_smiles = np.array(bg_smiles)
    bgX, bgm = featurize(list(bg_smiles))
    bg_ok = bg_smiles[bgm]
    bg_fp_ok = [f for f, k in zip(bg_fps, bgm) if k]
    bg_mw, bg_lp = bgX[:, -12], bgX[:, -11]
    sub = rng.choice(len(bg_ok), size=min(2500, len(bg_ok)), replace=False)
    Xbg_eval, _ = featurize([str(bg_ok[i]) for i in sub])

    # Merge rather than replace. A subset run must not discard the hold-out models and held-out
    # compound lists that inversion H1 and H7 already scored: regenerating them under a different
    # random draw would silently move every number those tests reported.
    modes = json.loads((OUT / "binder_modes.json").read_text()) \
        if (OUT / "binder_modes.json").exists() else {}
    heldout = json.loads((OUT / "heldout_actives.json").read_text()) \
        if (OUT / "heldout_actives.json").exists() else {}
    if modes:
        print(f"merging into {len(modes)} existing hold-out entries", flush=True)
    for ep in TARGETS:
        f = ROOT / "data" / "endpoints" / f"{ep}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f).dropna(subset=["smiles"])
        p = pd.to_numeric(df.get("pchembl"), errors="coerce")
        act = sorted(set(df.loc[p >= ACTIVE_P, "smiles"].astype(str)))
        if len(act) < 60:
            act = sorted(set(df.loc[df["label"] == 1, "smiles"].astype(str)))
        ina = sorted(set(df.loc[df["label"] == 0, "smiles"].astype(str)) - set(act))
        if len(act) < 60:
            print(f"[{ep}] too few actives, skipped", flush=True)
            continue

        # split ACTIVES by scaffold so a held-out compound shares no scaffold with training
        g = _scaffold_groups(act)
        uniq = np.unique(g)
        rng.shuffle(uniq)
        n_hold = max(1, int(round(HOLDOUT_FRAC * len(uniq))))
        hold_groups = set(uniq[:n_hold].tolist())
        act_hold = [s for s, gg in zip(act, g) if gg in hold_groups]
        act_train = [s for s, gg in zip(act, g) if gg not in hold_groups]
        if len(act_hold) < 15 or len(act_train) < 40:
            print(f"[{ep}] scaffold split too uneven ({len(act_train)}/{len(act_hold)}), skipped", flush=True)
            continue

        # negatives: split measured inactives too, plus property-matched decoys for training only
        ina = list(rng.permutation(ina))
        h = len(ina) // 2
        ina_train, ina_hold = ina[:h], ina[h:]

        aX, am = featurize(act_train)
        act_train = [s for s, k in zip(act_train, am) if k]
        afps = [_fp(s) for s in act_train]
        mw_lo, mw_hi = np.quantile(aX[:, -12], .02), np.quantile(aX[:, -12], .98)
        lp_lo, lp_hi = np.quantile(aX[:, -11], .02), np.quantile(aX[:, -11], .98)
        cand = (bg_mw >= mw_lo) & (bg_mw <= mw_hi) & (bg_lp >= lp_lo) & (bg_lp <= lp_hi)
        aset = set(act_train) | set(act_hold)
        idxs = [i for i in np.where(cand)[0] if bg_ok[i] not in aset]
        rng.shuffle(idxs)
        need = max(DECOY_RATIO * len(act_train) - len(ina_train), len(act_train))
        dec = []
        for i in idxs:
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
        forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, n_jobs=-1,
                                        random_state=42, class_weight="balanced").fit(X, y)
        cal = CalibratedClassifierCV(FrozenEstimator(forest), method="sigmoid").fit(X, y)
        joblib.dump(cal, OUT / f"{ep}_binder.joblib", compress=3)

        thr_bg = float(np.quantile(cal.predict_proba(Xbg_eval)[:, 1], 1 - BACKGROUND_FPR))
        thr_in = None
        if len(ina_hold) >= 15:
            hX, _ = featurize(ina_hold)
            thr_in = float(np.quantile(cal.predict_proba(hX)[:, 1], 1 - TARGET_FPR))
        thr = float(np.clip(max([t for t in (thr_in, thr_bg) if t is not None]), 0.05, 0.999))

        hoX, hom = featurize(act_hold)
        ph = cal.predict_proba(hoX)[:, 1]
        rec_rate = float((ph >= thr).mean())
        modes[ep] = {"mode": "scaffold_holdout", "threshold": round(thr, 4),
                     "threshold_basis": "held_out_inactives_and_background",
                     "n_train_actives": len(act_train), "n_holdout_actives": len(act_hold),
                     "n_holdout_scaffolds": int(n_hold),
                     "holdout_recall_at_threshold": round(rec_rate, 3)}
        heldout[ep] = act_hold
        print(f"[{ep:8}] train {len(act_train):5} / holdout {len(act_hold):4} actives "
              f"({n_hold} scaffolds) | thr {thr:.3f} | holdout recall {rec_rate:.3f}", flush=True)

    (OUT / "binder_modes.json").write_text(json.dumps(modes, indent=2))
    (OUT / "heldout_actives.json").write_text(json.dumps(heldout))
    rr = [v["holdout_recall_at_threshold"] for v in modes.values()]
    print(f"\nretrained {len(modes)} targets | mean scaffold-holdout recall {np.mean(rr):.3f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
