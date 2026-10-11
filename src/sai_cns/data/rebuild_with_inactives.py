"""Merge the PubChem measured inactives into the protein-target classifier sets.

Rules (scientifically conservative):
  - Only compounds not already present (by InChIKey) are added, as measured inactives (label 0). An
    existing dose-response active is never overridden by a single-concentration HTS inactive.
  - The number added is capped to balance the set toward roughly 50% active, not to flip it into an
    inactive majority (which would create the opposite bias and invite easy-negative inflation).
  - BBB and the receptor regressors are untouched (BBB has no protein target; regressors do not use a
    class label).

The previous (pooled ChEMBL+BindingDB) tables are backed up first. Every added compound keeps a
`source = PubChem_inactive` tag for provenance.

Output: rebuilt data/endpoints/<target>.csv, results/tables/inactives_merge_provenance.csv,
and data/_pubchem_cache/<target>_added.csv (the InChIKeys actually added, for the audit).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_compound_library import standardise  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ENDPOINTS = ROOT / "data" / "endpoints"
CACHE = ROOT / "data" / "_pubchem_cache"
TARGETS = ["AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG"]


def existing_inchikeys(df):
    keys = {}
    for smi in df["smiles"]:
        _, ik = standardise(smi)
        if ik:
            keys[ik] = True
    return set(keys)


def main():
    backup = ROOT / "archive" / "legacy" / "endpoints_pre_inactives_2026-07-21"
    if not backup.exists():
        shutil.copytree(ENDPOINTS, backup)
        print(f"backed up pre-inactives endpoints -> {backup.relative_to(ROOT)}")

    prov = []
    for name in TARGETS:
        ep_path = ENDPOINTS / f"{name}.csv"
        inact_path = CACHE / f"{name}_inactives.csv"
        df = pd.read_csv(ep_path)
        n_act = int((df.label == 1).sum()); n_inact = int((df.label == 0).sum())
        added = 0
        if inact_path.exists() and inact_path.stat().st_size > 0:
            try:
                pub = pd.read_csv(inact_path)
            except pd.errors.EmptyDataError:
                pub = pd.DataFrame(columns=["inchikey", "smiles"])
            have = existing_inchikeys(df)
            new = pub[~pub.inchikey.isin(have)].drop_duplicates("inchikey")
            # cap: balance toward ~50% active, never exceed making inactive the majority
            cap = max(0, n_act - n_inact)
            new = new.head(cap)
            if len(new):
                add = pd.DataFrame({"smiles": new["smiles"], "label": 0,
                                    "pchembl": np.nan, "year": np.nan, "source": "PubChem_inactive"})
                df = pd.concat([df, add], ignore_index=True)
                added = len(new)
                new[["inchikey", "smiles"]].to_csv(CACHE / f"{name}_added.csv", index=False)
        df.to_csv(ep_path, index=False)
        new_act = int((df.label == 1).sum()); new_inact = int((df.label == 0).sum())
        prov.append({"endpoint": name, "active": new_act, "inactive": new_inact,
                     "added_inactive": added, "active_frac_before": round(n_act / (n_act + n_inact), 3),
                     "active_frac_after": round(new_act / (new_act + new_inact), 3)})
        print(f"[{name}] +{added} inactives -> {new_act} act / {new_inact} inact "
              f"(active frac {prov[-1]['active_frac_before']} -> {prov[-1]['active_frac_after']})")

    pd.DataFrame(prov).to_csv(ROOT / "results" / "tables" / "inactives_merge_provenance.csv", index=False)
    print("\nwrote rebuilt endpoints and results/tables/inactives_merge_provenance.csv")


if __name__ == "__main__":
    main()
