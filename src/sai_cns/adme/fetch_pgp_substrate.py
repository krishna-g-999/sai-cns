"""Fetch a measured P-glycoprotein *substrate* dataset from ChEMBL (distinct from inhibition).

Substrate status is measured as the efflux ratio in bidirectional transport assays: a compound pumped
by P-gp shows a basolateral-to-apical / apical-to-basolateral flux ratio well above 1. The standard
binary cut is efflux ratio >= 2 = substrate. Values are pooled per compound by median (robust to the
occasional artefactual extreme ratio) and thresholded.

This is the endpoint that most directly predicts brain efflux, and it is the gap the inhibition model
could not fill (it is why the exposure readout mis-called loperamide).

Caveat: efflux-ratio cut-offs are assay-dependent (Caco-2 vs MDCK-MDR1); >= 2 is a widely used, but
approximate, threshold. Documented in data/adme/SOURCE.md.

Output: data/adme/pgp_substrate.csv (inchikey, smiles, efflux_ratio, label)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
import requests
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from build_compound_library import standardise  # noqa: E402

urllib3.disable_warnings()
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "adme"
CH = "https://www.ebi.ac.uk/chembl/api/data"
TARGET = "CHEMBL4302"  # ATP-dependent translocase ABCB1 (P-glycoprotein)
SUBSTRATE_ER = 2.0


def _get(u):
    try:
        return requests.get(u, timeout=90, verify=certifi.where())
    except Exception:
        return requests.get(u, timeout=90, verify=False)


def main():
    rows, offset = [], 0
    for _ in range(6):
        j = _get(f"{CH}/activity.json?target_chembl_id={TARGET}&standard_type=Ratio"
                 f"&limit=1000&offset={offset}").json()
        rows += j.get("activities", [])
        if not j.get("page_meta", {}).get("next"):
            break
        offset += 1000
        time.sleep(0.2)
    recs = []
    for a in rows:
        v, smi = a.get("standard_value"), a.get("canonical_smiles")
        if v is None or smi is None:
            continue
        try:
            er = float(v)
        except ValueError:
            continue
        if er <= 0 or er > 1000:  # drop non-physical / artefactual ratios
            continue
        csmi, ik = standardise(smi)
        if ik is not None:
            recs.append({"inchikey": ik, "smiles": csmi, "efflux_ratio": er})
    df = pd.DataFrame(recs)
    g = df.groupby("inchikey").agg(smiles=("smiles", "first"),
                                   efflux_ratio=("efflux_ratio", "median")).reset_index()
    g["label"] = (g["efflux_ratio"] >= SUBSTRATE_ER).astype(int)
    g.to_csv(OUT / "pgp_substrate.csv", index=False)
    print(f"P-gp substrate: {len(g)} unique compounds "
          f"({int(g.label.sum())} substrate / {int((g.label == 0).sum())} non-substrate); "
          f"median efflux ratio {np.median(g.efflux_ratio):.2f}")


if __name__ == "__main__":
    main()
