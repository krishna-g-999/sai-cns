"""How many distinct compounds the panel is trained on, counted the way the manuscript claims.

The manuscript states two different counts of the same population in two different paragraphs:
"169,341 unique compounds keyed by the InChIKey of the desalted parent" in the training-data section,
and "170,617 unique structures in the panel" in the neutralisation section three paragraphs later.
Both cannot describe compounds keyed the same way. 170,617/170,619 is independently reproduced twice
elsewhere (this session's `library_sp3_coverage.py`, and `parent_mol()`'s own docstring) by counting
distinct SMILES strings across the endpoint tables. 169,341 is not reproduced anywhere: it exists only
as a typed figure, repeated verbatim across the manuscript, the web app's About page, the thesis and
the submission package, and the thesis's own outstanding-items list (chapters 1, 2 and 10) already
flagged it as unverified and due to be re-derived before further use.

This computes the number the training-data sentence actually describes: parent_mol() every distinct
SMILES string in the endpoint tables exactly as training does, take the InChIKey of the result, and
count distinct keys. Two SMILES strings that desalt and neutralise to the same parent collapse to one
compound here even though they were two rows in the raw tables; two SMILES strings that are the same
salt or the same tautomer written two different ways likewise collapse. This is expected to be at or
below the distinct-SMILES count (170,619), never above it, since every merge only removes duplicates.

Output: results/tables/unique_compound_count.csv

Run:  python src/sai_cns/evaluation/unique_compound_count.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import parent_mol  # noqa: E402

OUT = ROOT / "results" / "tables" / "unique_compound_count.csv"


def main() -> None:
    smiles: set[str] = set()
    for f in sorted(glob.glob(str(ROOT / "data" / "endpoints" / "*.csv"))):
        smiles |= set(pd.read_csv(f, usecols=["smiles"], low_memory=False).smiles.astype(str))

    n_smiles = len(smiles)
    keys: set[str] = set()
    unparsed = 0
    for i, s in enumerate(smiles, 1):
        mol = parent_mol(s)
        if mol is None:
            unparsed += 1
            continue
        try:
            keys.add(Chem.MolToInchiKey(mol))
        except Exception:
            unparsed += 1
        if i % 20000 == 0:
            print(f"  {i:,}/{n_smiles:,} processed", flush=True)

    rows = [
        ("distinct SMILES strings across the endpoint tables", n_smiles, "raw text, no standardisation"),
        ("of which parent_mol() could not parse", unparsed, "excluded from the InChIKey count below"),
        ("distinct InChIKeys of the desalted, neutralised parent", len(keys),
         "the population the training-data paragraph describes; this is the compound count"),
        ("SMILES strings collapsed by this standardisation", n_smiles - unparsed - len(keys),
         "salts, tautomers and charge states of a compound already counted"),
    ]
    d = pd.DataFrame(rows, columns=["measure", "value", "note"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)

    print(f"\n{n_smiles:,} distinct SMILES, {unparsed:,} unparsed, "
          f"{len(keys):,} distinct InChIKeys of the desalted parent")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
