"""Per-endpoint applicability-domain references.

For each of the 13 endpoints, store fingerprints of ALL its measured training compounds, so a
prediction can carry an endpoint-specific in/near/out-of-domain flag (max Tanimoto to that
endpoint's chemistry) rather than only a single global one. Writes models_rf/ad_per_endpoint.pkl.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models_rf" / "ad_per_endpoint.pkl"
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

ENDPOINTS = {ep: "endpoints" for ep in [
    "BBB", "AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG",
    "D2", "A2A", "HT2A", "SERT",
    "HT1A", "HT6", "HT7", "H3", "DAT", "NET", "Sigma1", "CB1",
    "OPRK1", "OPRM1", "D3", "A1", "a7nAChR", "LRRK2",
    "NLRP3", "P2X7", "COX2", "CSF1R", "PDE10A", "HDAC1", "HDAC6", "GluN2B",
    "mGluR5", "GABA_A", "OX1", "OX2", "MT1", "mTOR", "SIRT1", "KEAP1",
    "GBA1", "PDE4B", "Nav1_5", "Nav1_7", "Nav1_1", "TAAR1", "GluA2",
]}


def _fps(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            out.append(_GEN.GetFingerprint(m))
    return out


def main():
    ref = {}
    for ep in ENDPOINTS:
        f = ROOT / "data" / "endpoints" / f"{ep}.csv"
        if f.exists():
            smi = pd.read_csv(f, usecols=["smiles"])["smiles"].dropna().astype(str).tolist()
            ref[ep] = _fps(smi)
            print(f"{ep}: {len(ref[ep])} fps")
    af = ROOT / "data" / "endpoints_reg" / "antioxidant_dpph.csv"
    if af.exists():
        smi = pd.read_csv(af, usecols=["smiles"])["smiles"].dropna().astype(str).tolist()
        ref["antioxidant"] = _fps(smi)
        print(f"antioxidant: {len(ref['antioxidant'])} fps")
    with OUT.open("wb") as fh:
        pickle.dump(ref, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
