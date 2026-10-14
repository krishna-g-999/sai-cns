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
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

# The one definition of the applicability-domain fingerprint lives beside this file. The path
# insert has to precede the import, which is the sort of thing that works on the author's machine
# and fails on a clean checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ad_fingerprint as adfp  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models_rf" / "ad_reference.pkl"
_GEN = adfp.generator()


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
    # The spec travels with the fingerprints. A reference built with one fingerprint and queried
    # with another returns similarities that are wrong rather than missing, and wrong silently.
    with OUT.open("wb") as fh:
        pickle.dump((keep, fps, adfp.SPEC), fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {OUT} with {len(keep)} reference fingerprints, spec {adfp.SPEC}")


if __name__ == "__main__":
    main()
