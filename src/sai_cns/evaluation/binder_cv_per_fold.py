"""Per-fold ten-fold cross-validation for the binder panel.

The binder models were cross-validated at training time, ten-fold, grouped on Bemis-Murcko scaffold,
and each endpoint's out-of-fold AUROC is recorded in models_rf/binder_modes.json. What was not kept
was the detail behind that single number: the individual folds, their sizes, and the conventional
random split for comparison. A reviewer asking to see the ten-fold results is asking for exactly that
detail, so this recomputes it and writes every fold out.

Reconstructing the training set
-------------------------------
A binder endpoint's training set is not its CSV. It is the measured actives (pChEMBL >= 7), a decoy
set drawn from background chemistry, and half the measured inactives; the other half is held out for
threshold calibration and never trained on. This script rebuilds that composition by the same rule,
with a seed derived from the endpoint name so each is independent and reproducible.

Because the decoys are redrawn rather than restored, these folds are an independent re-run of the
training protocol, not a replay of the original run. That is the stronger test: if the recorded
training-time AUROC and this re-run disagree, the endpoint's score depends on which decoys it drew,
which is worth knowing. The comparison is written out as `recorded_scaffold_cv_auroc` alongside the
re-run mean so the two can be read against each other.

Endpoints trained on measured labels alone are cross-validated on those labels directly, with no
decoys involved.

Outputs
  results/tables/binder_cv_folds.csv     one row per endpoint x split x fold
  results/tables/binder_cv_summary.csv   mean and sd per endpoint x split, with the recorded value

Resumable: endpoints already present in the folds file are skipped, so an interrupted run continues.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize  # noqa: E402

M = ROOT / "models_rf"
EP = ROOT / "data" / "endpoints"
OUT_F = ROOT / "results" / "tables" / "binder_cv_folds.csv"
OUT_S = ROOT / "results" / "tables" / "binder_cv_summary.csv"

ACTIVE_P, DECOY_RATIO, TAN_MAX, N_SPLITS = 7.0, 3, 0.35, 10
RF = dict(n_estimators=300, min_samples_leaf=4, n_jobs=-1, random_state=42,
          class_weight="balanced")
_GEN = None


def _fpgen():
    global _GEN
    if _GEN is None:
        from rdkit.Chem import rdFingerprintGenerator
        # 2048 bits, matching models_rf/ad_reference.pkl. This is the similarity fingerprint used to
        # keep decoys away from the actives, and is deliberately not the 1024-bit model input.
        _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return _GEN


def _fp(s):
    m = Chem.MolFromSmiles(s)
    return _fpgen().GetFingerprint(m) if m else None


def _scaffold_groups(smiles):
    codes, mapping = [], {}
    for smi in smiles:
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(str(smi)),
                                                      includeChirality=False)
        except Exception:
            scaf = ""
        codes.append(mapping.setdefault(scaf or f"_none_{len(mapping)}", len(mapping)))
    return np.asarray(codes)


def _metrics(y, proba):
    pred = (proba >= 0.5).astype(int)
    one_class = len(set(y)) < 2
    return {
        "roc_auc": np.nan if one_class else roc_auc_score(y, proba),
        "pr_auc": np.nan if one_class else average_precision_score(y, proba),
        "mcc": matthews_corrcoef(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
    }


def build_set(ep, mode, bg_ok, bg_fp_ok, bg_mw, bg_lp):
    """Rebuild the endpoint's training set exactly as the training protocol specifies."""
    df = pd.read_csv(EP / f"{ep}.csv").dropna(subset=["smiles"])
    rng = np.random.default_rng(zlib.crc32(ep.encode()) % (2 ** 31))

    if mode == "measured_labels_holdout":
        df = df.drop_duplicates("smiles")
        smiles = df["smiles"].astype(str).tolist()
        y = df["label"].astype(int).to_numpy()
        return smiles, y, dict(n_positive=int((y == 1).sum()), n_decoy=0,
                               n_measured_inactive_train=int((y == 0).sum()))

    p = pd.to_numeric(df.get("pchembl"), errors="coerce")
    act = sorted(set(df.loc[p >= ACTIVE_P, "smiles"].astype(str)))
    ina_all = sorted(set(df.loc[df["label"] == 0, "smiles"].astype(str)) - set(act))
    ina_all = list(rng.permutation(ina_all))
    ina_train = ina_all[:len(ina_all) // 2]

    aX, am = featurize(act)
    act = [s for s, k in zip(act, am) if k]
    afps = [_fp(s) for s in act]
    mw_lo, mw_hi = np.quantile(aX[:, -12], .02), np.quantile(aX[:, -12], .98)
    lp_lo, lp_hi = np.quantile(aX[:, -11], .02), np.quantile(aX[:, -11], .98)
    cand = (bg_mw >= mw_lo) & (bg_mw <= mw_hi) & (bg_lp >= lp_lo) & (bg_lp <= lp_hi)
    aset = set(act)
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

    smiles = act + dec + ina_train
    y = np.array([1] * len(act) + [0] * (len(dec) + len(ina_train)))
    return smiles, y, dict(n_positive=len(act), n_decoy=len(dec),
                           n_measured_inactive_train=len(ina_train))


def main():
    modes = json.loads((M / "binder_modes.json").read_text())
    todo = sys.argv[1:] or sorted(modes)

    # Resuming is only safe while the models have not moved. The skip list is keyed on endpoint
    # name alone, so after a retrain every row in the folds file describes estimators that no
    # longer exist and every endpoint would be skipped: the script would exit reporting a complete
    # cross-validation of the previous panel, with nothing in the output saying so. Refuse instead,
    # and say what to do. `--fresh` starts over, keeping the old file beside it for comparison.
    fresh = "--fresh" in sys.argv
    todo = [a for a in todo if not a.startswith("--")] or sorted(modes)
    done = set()
    if OUT_F.exists():
        newest_model = max((p.stat().st_mtime for p in M.glob("*_binder.joblib")), default=0.0)
        if newest_model > OUT_F.stat().st_mtime and not fresh:
            raise SystemExit(
                f"{OUT_F.name} predates the fitted binder models, so resuming would skip every\n"
                f"endpoint and report the previous panel's cross-validation as though it were this\n"
                f"one. Re-run with --fresh to recompute all endpoints.")
        if fresh:
            keep = OUT_F.with_suffix(".pre_retrain.csv")
            OUT_F.replace(keep)
            print(f"starting fresh; previous folds kept at {keep.name}", flush=True)
        else:
            done = set(pd.read_csv(OUT_F)["endpoint"].astype(str))

    with (M / "ad_reference.pkl").open("rb") as fh:
        bg_smiles, bg_fps = pickle.load(fh)
    bg_smiles = np.array(bg_smiles)
    bgX, bgm = featurize(list(bg_smiles))
    bg_ok = bg_smiles[bgm]
    bg_fp_ok = [f for f, k in zip(bg_fps, bgm) if k]
    bg_mw, bg_lp = bgX[:, -12], bgX[:, -11]

    for ep in todo:
        if ep in done:
            print(f"[{ep}] already done, skipped", flush=True)
            continue
        if not (EP / f"{ep}.csv").exists():
            print(f"[{ep}] no training table, skipped", flush=True)
            continue
        rec = modes[ep]
        smiles, y, comp = build_set(ep, rec.get("mode"), bg_ok, bg_fp_ok, bg_mw, bg_lp)
        X, mask = featurize(smiles)
        y = y[mask]
        smiles = [s for s, k in zip(smiles, mask) if k]
        groups = _scaffold_groups(smiles)
        if len(set(y)) < 2 or min(np.bincount(y)) < N_SPLITS:
            print(f"[{ep}] too few in one class for ten folds, skipped", flush=True)
            continue

        rows = []
        for split, splitter in (("random", StratifiedKFold(N_SPLITS, shuffle=True, random_state=42)),
                                ("scaffold", GroupKFold(N_SPLITS))):
            it = splitter.split(X, y, groups) if split == "scaffold" else splitter.split(X, y)
            for fold, (tr, te) in enumerate(it, 1):
                model = RandomForestClassifier(**RF).fit(X[tr], y[tr])
                proba = model.predict_proba(X[te])[:, 1]
                rows.append({
                    "endpoint": ep, "panel": "Binder classifier",
                    "mode": rec.get("mode"), "split": split, "fold": fold,
                    "n_train": len(tr), "n_test": len(te),
                    "n_positive_test": int(y[te].sum()),
                    "n_scaffolds_test": int(len(set(groups[te]))) if split == "scaffold" else None,
                    **_metrics(y[te], proba),
                    "n_total": len(y), **comp,
                })
        d = pd.DataFrame(rows)
        d.to_csv(OUT_F, mode="a", header=not OUT_F.exists(), index=False)
        sc = d[d.split == "scaffold"]["roc_auc"].mean()
        rd = d[d.split == "random"]["roc_auc"].mean()
        print(f"[{ep}] n={len(y):6,}  random {rd:.3f}  scaffold {sc:.3f}  "
              f"(recorded at training {rec.get('scaffold_cv_auroc')})", flush=True)

    if OUT_F.exists():
        f = pd.read_csv(OUT_F)
        g = f.groupby(["endpoint", "split"]).agg(
            n=("n_total", "first"), folds=("fold", "count"),
            n_positive=("n_positive", "first"), n_decoy=("n_decoy", "first"),
            n_measured_inactive_train=("n_measured_inactive_train", "first"),
            roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
            mcc_mean=("mcc", "mean"), f1_mean=("f1", "mean"),
            balanced_acc_mean=("balanced_acc", "mean")).reset_index()
        g["recorded_scaffold_cv_auroc"] = g["endpoint"].map(
            lambda e: modes.get(e, {}).get("scaffold_cv_auroc"))
        g["rerun_minus_recorded"] = np.where(
            g.split == "scaffold", g.roc_auc_mean - g.recorded_scaffold_cv_auroc, np.nan)
        g.to_csv(OUT_S, index=False)
        agree = g[g.split == "scaffold"]["rerun_minus_recorded"].abs()
        print(f"\nwrote {OUT_S}  ({g.endpoint.nunique()} endpoints)")
        print(f"re-run vs recorded scaffold AUROC: median |difference| {agree.median():.4f}, "
              f"max {agree.max():.4f}")


if __name__ == "__main__":
    main()
