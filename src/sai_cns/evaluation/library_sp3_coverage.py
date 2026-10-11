"""The two library-composition figures behind the natural-product limitation.

The technical report's natural-product paragraph rests on two numbers about the training library's
own chemistry: the median fraction of sp3-hybridised carbon, and the share of the library that is
both sp3-rich and largely non-aromatic. Both are measured rather than asserted, which is right, but
they are measured inside `build_technical_report.py` and written to no file. A thesis audit made the
consequence concrete: no other document can cite them, nothing checks them, and the generator's own
comment records that when they were carried as prose they had already drifted once, the median stated
as 0.36 against a measured 0.34 and the sp3-rich share as 3.3 per cent against a measured 9.2.

This script measures the same two quantities and writes them to an artefact so they can be cited and
checked like every other figure in the project.

Definitions, stated because both thresholds are choices:

  fraction sp3   RDKit CalcFractionCSP3 on the desalted, neutralised parent, the same structure the
                 featuriser sees, so the figure describes what the models were actually shown
  sp3-rich       fraction sp3 of at least 0.55 and at most one aromatic ring. This is a coarse proxy
                 for terpenoid and steroidal chemistry and is not a definition of natural-product
                 likeness; it is the same rule the report uses, kept identical so the artefact can
                 replace the in-generator computation without changing the published number

Run:  brainsafe_env/Scripts/python.exe src/sai_cns/evaluation/library_sp3_coverage.py
Out:  results/tables/library_sp3_coverage.csv
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import parent_mol  # noqa: E402

RDLogger.DisableLog("rdApp.*")
OUT = ROOT / "results" / "tables" / "library_sp3_coverage.csv"
SP3_RICH = 0.55
MAX_AROMATIC_RINGS = 1


def main() -> None:
    smiles: set[str] = set()
    for f in sorted(glob.glob(str(ROOT / "data" / "endpoints" / "*.csv"))):
        smiles |= set(pd.read_csv(f, usecols=["smiles"]).smiles.astype(str))
    print(f"distinct SMILES across the endpoint tables: {len(smiles):,}", flush=True)

    frac: list[float] = []
    rich = 0
    for s in smiles:
        m = parent_mol(s)
        if m is None:
            continue
        try:
            f3 = rdMolDescriptors.CalcFractionCSP3(m)
            ar = rdMolDescriptors.CalcNumAromaticRings(m)
        except Exception:
            continue
        frac.append(float(f3))
        if f3 >= SP3_RICH and ar <= MAX_AROMATIC_RINGS:
            rich += 1

    n = len(frac)
    rows = [
        {"metric": "distinct SMILES in the endpoint tables", "value": len(smiles),
         "note": "before parsing; a structure RDKit cannot read is excluded below"},
        {"metric": "structures parsed as a desalted parent", "value": n,
         "note": "the population every figure below is computed on"},
        {"metric": "median fraction sp3", "value": round(float(np.median(frac)), 4),
         "note": "RDKit CalcFractionCSP3 on the parent the featuriser sees"},
        {"metric": "mean fraction sp3", "value": round(float(np.mean(frac)), 4), "note": ""},
        {"metric": "fraction sp3, 25th percentile",
         "value": round(float(np.percentile(frac, 25)), 4), "note": ""},
        {"metric": "fraction sp3, 75th percentile",
         "value": round(float(np.percentile(frac, 75)), 4), "note": ""},
        {"metric": "sp3-rich and at most one aromatic ring, count", "value": rich,
         "note": f"fraction sp3 >= {SP3_RICH} and aromatic rings <= {MAX_AROMATIC_RINGS}"},
        {"metric": "sp3-rich and at most one aromatic ring, per cent",
         "value": round(100.0 * rich / max(n, 1), 2),
         "note": "the share of the library in which terpenoid and steroidal chemistry would sit"},
    ]
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 160)
    print()
    print(out.to_string(index=False))
    print(f"\nwrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
