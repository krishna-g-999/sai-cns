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
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models_rf" / "readacross_index.pkl"
# One definition of the applicability-domain fingerprint. A private copy here was built
# stereochemistry-blind while app.py queried it chirality-aware, which returned
# similarities that were wrong rather than missing. See src/sai_cns/ad_fingerprint.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ad_fingerprint as _adfp  # noqa: E402

GEN = _adfp.generator()

# BBB is an ADME property rather than a therapeutic target, so it is excluded from the
# pharmacology statement; it is reported separately by the exposure panel.
SKIP = {"BBB"}

# The target name is the CSV's filename stem, so any variant or intermediate table dropped in
# data/endpoints/ becomes a target in the index and is shown to users as pharmacology. Two already
# had: OX1.chembl_only.csv and OX2.chembl_only.csv are single-source variants of OX1 and OX2, and
# the index carried "OX1.chembl_only" and "OX2.chembl_only" as if they were distinct receptors. A
# stem containing a dot is a variant table by this project's naming, never a target.
def _is_variant(stem: str) -> bool:
    return "." in stem


def main():
    activity: dict[str, set[str]] = {}
    for f in glob.glob(str(ROOT / "data" / "endpoints" / "*.csv")):
        target = Path(f).stem
        if target in SKIP or _is_variant(target):
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
        pickle.dump({"smiles": smiles, "fps": fps, "targets": targets,
                     "spec": _adfp.SPEC}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    n_t = len({t for tg in targets for t in tg})
    print(f"indexed {len(smiles):,} measured actives across {n_t} targets")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
