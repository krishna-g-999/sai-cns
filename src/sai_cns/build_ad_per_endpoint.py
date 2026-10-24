"""Per-endpoint applicability-domain references.

For each of the 13 endpoints, store fingerprints of ALL its measured training compounds, so a
prediction can carry an endpoint-specific in/near/out-of-domain flag (max Tanimoto to that
endpoint's chemistry) rather than only a single global one. Writes models_rf/ad_per_endpoint.pkl.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models_rf" / "ad_per_endpoint.pkl"
# One definition of the applicability-domain fingerprint. A private copy here was built
# stereochemistry-blind while app.py queried it chirality-aware, which returned
# similarities that were wrong rather than missing. See src/sai_cns/ad_fingerprint.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ad_fingerprint as _adfp  # noqa: E402

_GEN = _adfp.generator()

# The endpoint list comes from the registry, never from a copy kept here. The copy this
# replaced held 49 names: it still listed withdrawn Nav1_1 and GluA2, and omitted 11 registered
# endpoints, 8 of them deployed, which therefore had no endpoint-specific domain at all while the
# script printed success. That is the defect src/sai_cns/panel.py was written to abolish.
def _endpoint_list():
    try:
        import panel as _panel
        names = sorted(set(_panel.names(deployed=True)))
    except Exception:
        names = []
    core = ["BBB", "AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG"]
    tables = {p.stem for p in (ROOT / "data" / "endpoints").glob("*.csv")}
    return {ep: "endpoints" for ep in sorted(set(names) | set(core)) if ep in tables}


ENDPOINTS = _endpoint_list()


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
        # The spec travels with the fingerprints so a mismatch can be refused at load time
        # rather than silently answering with wrong similarities.
        pickle.dump({"spec": _adfp.SPEC, "reference": ref}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
