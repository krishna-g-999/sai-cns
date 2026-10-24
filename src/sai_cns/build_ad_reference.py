"""Precompute the applicability-domain reference fingerprints.

Gathers every unique measured training SMILES (target + ADME endpoints), computes a
2048-bit ECFP-4 fingerprint per compound, and pickles (smiles, fps, spec) to
models_rf/ad_reference.pkl. The app loads this to report, per prediction, the maximum
Tanimoto similarity to measured training chemistry (applicability domain) and the nearest
measured analogue.

    python src/sai_cns/build_ad_reference.py              rebuild from the training CSVs
    python src/sai_cns/build_ad_reference.py --same-set    rebuild over the existing reference's
                                                           compound set, changing only the
                                                           fingerprint

WHY --same-set EXISTS. Enabling chirality and refreshing the compound set are two separate
changes, and the default path does both at once. The CSVs now yield about 185,800 compounds
against the 158,890 the published reference holds. Enlarging the set is not a free improvement:
expected_recall() maps similarity to recall using bands derived against the smaller set, so a
larger reference raises reported similarity and the server then claims more recall than the
evidence supports. The present shortfall makes the domain conservative, which is the safe
direction. Correcting it properly means re-deriving the recall bands and every published
applicability-domain figure, which is follow-up work.

--same-set holds the compound set fixed and changes only the fingerprint, which is what shipping
the stereochemistry fix requires and nothing more. See src/sai_cns/ad_fingerprint.py.
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


def existing_reference_smiles():
    """The compound set the published reference already holds, in its stored order."""
    if not OUT.exists():
        raise SystemExit(f"{OUT} is absent, so there is no compound set to preserve; "
                         f"run without --same-set to build one from the training CSVs")
    with OUT.open("rb") as fh:
        obj = pickle.load(fh)
    return list(obj[0])


def main():
    argv = sys.argv[1:]
    out = OUT
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1])

    if "--same-set" in sys.argv[1:]:
        smiles = existing_reference_smiles()
        print(f"rebuilding over the existing compound set: {len(smiles):,} compounds, "
              f"fingerprint only")
    else:
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
    with out.open("wb") as fh:
        pickle.dump((keep, fps, adfp.SPEC), fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {out} with {len(keep)} reference fingerprints, spec {adfp.SPEC}")


if __name__ == "__main__":
    main()
