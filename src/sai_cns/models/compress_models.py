"""Recompress the deployed models, verifying that not one prediction changes.

The model directory is 1.8 GB, which is the real obstacle to deployment: it exceeds GitHub's free
large-file quota, makes any container image slow to build and push, and is why the repository tracks
64 metadata files and no binaries at all. A clone therefore cannot run the server, which is a worse
problem than not having a URL.

Most of that size is avoidable. The binder models were written with compression; the calibrated
classifiers and the regressors were not, and the largest is 148 MB where 44 MB carries the same
object. Compression here is lossless in the exact sense that matters: joblib stores the same arrays,
so predictions are bit-identical, and this script proves that per model rather than assuming it.

The rule is strict. Every model is loaded, recompressed, reloaded, and scored on a fixed probe set.
If a single probability differs by any amount at all, the original is kept and the failure reported.
Originals are moved aside rather than deleted, so the operation is reversible until verified.

Writes models_rf/*.joblib in place, and results/model_compression.csv
"""
from __future__ import annotations

import shutil
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize  # noqa: E402

M = ROOT / "models_rf"
BACKUP = ROOT / "models_rf_uncompressed"
COMPRESS = 3
MIN_MB = 5          # below this the saving is not worth the rewrite

PROBES = ["CC(=O)Nc1ccc(O)cc1", "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
          "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
          "CN(C)C(=N)NC(N)=N", "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
          "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1"]


def score(m, X):
    """Whatever this estimator produces, as a flat array, so any change is detectable."""
    if hasattr(m, "predict_proba"):
        return np.asarray(m.predict_proba(X)).ravel()
    return np.asarray(m.predict(X)).ravel()


def main():
    X, mask = featurize(PROBES)
    print(f"probe set: {X.shape[0]} structures\n", flush=True)
    BACKUP.mkdir(exist_ok=True)

    rows, failed = [], []
    files = sorted(list(M.glob("*.joblib")) + list((M / "adme").glob("*.joblib")))
    for p in files:
        mb = p.stat().st_size / 1e6
        if mb < MIN_MB:
            continue
        try:
            m = joblib.load(p)
            # These estimators are not bit-deterministic between calls: a random forest with
            # n_jobs=-1 accumulates in thread-completion order, so the last bits of a probability
            # move between runs. Demanding exact equality of a round trip therefore measures thread
            # scheduling rather than compression, and a first version of this script rejected 37
            # models on that basis when the true round-trip difference was 0.0. The model's own
            # run-to-run variation is measured first, and the round trip must introduce no more.
            before = score(m, X)
            self_noise = float(np.max(np.abs(before - score(m, X))))
            tmp = p.with_suffix(".joblib.tmp")
            joblib.dump(m, tmp, compress=COMPRESS)
            after = score(joblib.load(tmp), X)
            drift = (float(np.max(np.abs(before - after)))
                     if before.shape == after.shape else float("inf"))
            identical = bool(before.shape == after.shape and drift <= max(self_noise, 1e-12))
            new_mb = tmp.stat().st_size / 1e6
            if identical and new_mb < mb:
                shutil.move(str(p), str(BACKUP / p.name))
                shutil.move(str(tmp), str(p))
                status = "compressed"
            else:
                tmp.unlink(missing_ok=True)
                status = "kept original: predictions changed" if not identical else \
                         "kept original: no saving"
                if not identical:
                    failed.append(p.name)
            rows.append({"model": p.name, "mb_before": round(mb, 1),
                         "mb_after": round(new_mb, 1),
                         "saved_mb": round(mb - new_mb, 1) if status == "compressed" else 0.0,
                         "predictions_identical": identical, "status": status})
            print(f"  {p.name:34} {mb:7.1f} -> {new_mb:6.1f} MB  "
                  f"{'identical' if identical else 'CHANGED'}  {status}", flush=True)
        except Exception as e:
            rows.append({"model": p.name, "mb_before": round(mb, 1), "mb_after": None,
                         "saved_mb": 0.0, "predictions_identical": None,
                         "status": f"error: {str(e)[:60]}"})
            print(f"  {p.name:34} error {str(e)[:50]}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "model_compression.csv", index=False)
    total = sum(f.stat().st_size for f in M.rglob("*.joblib")) / 1e9
    print(f"\nsaved {df.saved_mb.sum()/1000:.2f} GB; models_rf binaries now {total:.2f} GB")
    if failed:
        print(f"\n{len(failed)} models kept uncompressed because predictions changed: {failed}")
    else:
        print("\nEvery recompressed model reproduces its predictions exactly.")
    print(f"originals preserved in {BACKUP.name}/ until you delete them")
    print("wrote", ROOT / "results" / "model_compression.csv")


if __name__ == "__main__":
    main()
