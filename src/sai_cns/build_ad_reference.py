"""Precompute the applicability-domain reference fingerprints once.

Gathers every unique measured training SMILES (target + ADME endpoints), computes a
2048-bit ECFP-4 fingerprint per compound, and pickles (smiles, fps) to
models_rf/ad_reference.pkl. The app loads this to report, per prediction, the maximum
Tanimoto similarity to measured training chemistry (applicability domain) and the nearest
measured analogue. Run once:  python src/sai_cns/build_ad_reference.py
"""
from __future__ import annotations

import glob
import pickle
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models_rf" / "ad_reference.pkl"
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def main():
    smiles = set()
    for f in glob.glob(str(ROOT / "data" / "endpoints" / "*.csv")) + \
             glob.glob(str(ROOT / "data" / "adme" / "*.csv")):
        try:
            df = pd.read_csv(f, usecols=["smiles"])
            smiles.update(df["smiles"].dropna().astype(str))
        except Exception:
            continue
    smiles = sorted(smiles)
    keep, fps = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        keep.append(s)
        fps.append(_GEN.GetFingerprint(m))
    with OUT.open("wb") as fh:
        pickle.dump((keep, fps), fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {OUT} with {len(keep)} reference fingerprints")


if __name__ == "__main__":
    main()
