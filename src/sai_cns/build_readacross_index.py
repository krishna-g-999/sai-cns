"""Precompute the read-across index: measured compounds, their fingerprints, and what they hit.

When no model fires, the measured library still holds evidence about the query. This index maps
every compound that is a measured active at one or more modelled targets to the set of those
targets, alongside an ECFP-4 fingerprint, so the server can answer "the closest measured compounds
to your query are actives at X and Y" for any structure.

This is read-across, which is weaker evidence than a validated model and fails at activity cliffs
where a single substitution abolishes binding. The similarity is therefore always carried alongside
the claim so the user can judge it, and the interface must phrase the result as a statement about
neighbours rather than about the query.

Writes models_rf/readacross_index.pkl
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
OUT = ROOT / "models_rf" / "readacross_index.pkl"
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# BBB is an ADME property rather than a therapeutic target, so it is excluded from the
# pharmacology statement; it is reported separately by the exposure panel.
SKIP = {"BBB"}


def main():
    activity: dict[str, set[str]] = {}
    for f in glob.glob(str(ROOT / "data" / "endpoints" / "*.csv")):
        target = Path(f).stem
        if target in SKIP:
            continue
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "label" not in d.columns:
            continue
        sel = d["label"] == 1
        if "pchembl" in d.columns:
            # prefer clearly potent actives where a potency value exists
            p = pd.to_numeric(d["pchembl"], errors="coerce")
            sel = sel & (p.isna() | (p >= 6))
        for s in d.loc[sel, "smiles"].astype(str):
            activity.setdefault(s, set()).add(target)

    # read-across-only targets: measured actives for proteins this project does not model.
    # They widen what the index can recognise without implying any prediction, and are marked
    # with a trailing asterisk so the interface can label them as similarity evidence only.
    for f in glob.glob(str(ROOT / "data" / "readacross" / "*.csv")):
        target = Path(f).stem + "*"
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "smiles" not in d.columns:
            continue
        for s in d["smiles"].astype(str):
            activity.setdefault(s, set()).add(target)

    smiles, fps, targets = [], [], []
    for s, tg in activity.items():
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        smiles.append(s)
        fps.append(GEN.GetFingerprint(m))
        targets.append(sorted(tg))

    with OUT.open("wb") as fh:
        pickle.dump({"smiles": smiles, "fps": fps, "targets": targets}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    n_t = len({t for tg in targets for t in tg})
    print(f"indexed {len(smiles):,} measured actives across {n_t} targets")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
