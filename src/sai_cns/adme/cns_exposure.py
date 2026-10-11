"""Combine the target, BBB and ADME models into an interpretable CNS free-exposure readout.

This is the point of ladder B: not any single property, but the chain that decides whether an
achievable dose puts *free* drug on a brain target. For a molecule it reports the measured-model
predictions for BBB penetration, passive permeability (Caco-2), lipophilicity, solubility, free
fraction (from plasma-protein binding) and a P-gp inhibition flag, then a qualitative free-brain-
exposure call.

Important honesty notes:
  - This is a heuristic *proxy* for K_p,uu (unbound brain-to-plasma), not a measured K_p,uu; no large
    public K_p,uu dataset exists to train on directly.
  - The P-gp model is inhibition, not substrate; it is shown as a related flag, not as the efflux call.
  - Each number carries the reliability of its own model (see docs/ADME_RESULTS.md); clearance and
    protein binding are the weakest.

Run:  python src/sai_cns/adme/cns_exposure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.featurize import featurize_one  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models_rf"

# High-permeability Caco-2 cut (~8e-6 cm/s) and a workable free-fraction floor.
CACO2_HIGH = -5.15
FREE_FRACTION_FLOOR = 0.05


def _load():
    m = {"BBB": joblib.load(MODELS / "BBB.joblib")}
    for ep in ("caco2_permeability", "lipophilicity", "solubility",
               "plasma_protein_binding", "pgp_inhibition", "pgp_substrate", "logbb", "kpuu"):
        m[ep] = joblib.load(MODELS / "adme" / f"{ep}.joblib")
    return m


def predict(smiles, m):
    x = featurize_one(smiles)
    if x is None:
        return None
    x = x.reshape(1, -1)
    bbb = float(m["BBB"].predict_proba(x)[0, 1])
    caco2 = float(m["caco2_permeability"].predict(x)[0])
    logd = float(m["lipophilicity"].predict(x)[0])
    logs = float(m["solubility"].predict(x)[0])
    ppb = float(m["plasma_protein_binding"].predict(x)[0])
    pgp_inhib = float(m["pgp_inhibition"].predict_proba(x)[0, 1])
    pgp_sub = float(m["pgp_substrate"].predict_proba(x)[0, 1])
    logbb = float(m["logbb"].predict(x)[0])  # total brain/plasma (NOT free exposure)
    kpuu = float(10 ** m["kpuu"].predict(x)[0])  # directly-modelled unbound brain/plasma
    free_frac = max(0.0, min(1.0, (100.0 - ppb) / 100.0))
    # Primary call from the directly-measured Kp,uu model (>=0.3 = meaningful free brain exposure).
    if kpuu >= 0.3:
        call = "favourable"
    elif kpuu >= 0.1:
        call = "borderline"
    else:
        call = "limited (low free brain exposure)"
    return {"kpuu_pred": round(kpuu, 3), "bbb_prob": round(bbb, 2), "logBB_total": round(logbb, 2),
            "caco2_logPapp": round(caco2, 2), "logD": round(logd, 2), "percent_bound": round(ppb, 1),
            "free_fraction": round(free_frac, 3), "pgp_substrate_prob": round(pgp_sub, 2),
            "free_brain_exposure": call}


DEMO = {
    "Diazepam (CNS, high penetration)": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "Donepezil (CNS, AD drug)": "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
    "Caffeine (CNS stimulant)": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Atenolol (peripheral beta-blocker)": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "Loperamide (peripheral; P-gp substrate)": "CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1",
}


def main():
    m = _load()
    rows = []
    for name, smi in DEMO.items():
        p = predict(smi, m)
        if p:
            rows.append({"compound": name, **p})
    out = pd.DataFrame(rows)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "results" / "tables" / "cns_exposure_demo.csv", index=False)
    print("=== CNS free-exposure readout (heuristic K_p,uu proxy) ===")
    print(out.to_string(index=False))
    print("\nwrote results/tables/cns_exposure_demo.csv")


if __name__ == "__main__":
    main()
