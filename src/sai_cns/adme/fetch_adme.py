"""Fetch measured ADME / exposure datasets for ladder B (the properties that govern brain exposure).

These are established, measured public benchmark sets (TDC / Harvard Dataverse and MoleculeNet). They
answer the question the target-engagement panel cannot: not "does it bind" but "does an achievable
dose put enough free drug on the target in the brain". Each set is standardised the same way as the
rest of the project (largest fragment, canonical SMILES, InChIKey dedup).

Endpoints:
  solubility            AqSolDB       regression   aqueous solubility, logS (mol/L)   developability
  lipophilicity         AstraZeneca   regression   logD7.4                             distribution
  caco2_permeability    Wang          regression   log Papp (cm/s)                     passive permeability
  pgp_inhibition        Broccatelli   classification  P-gp inhibitor (0/1)             brain efflux
  plasma_protein_binding AZ           regression   percent bound                       free fraction
  clearance_hepatocyte  AZ            regression   uL/min/1e6 cells                    metabolic stability

Output: data/adme/<endpoint>.csv  (smiles, y[, label]) and data/adme/SOURCE.md
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import certifi
import pandas as pd
import requests
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from build_compound_library import standardise  # noqa: E402

urllib3.disable_warnings()
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "adme"
OUT.mkdir(parents=True, exist_ok=True)
DV = "https://dataverse.harvard.edu/api/access/datafile"

# name -> (url, task, unit/description)
SOURCES = {
    "solubility": (f"{DV}/4259610", "regression", "aqueous solubility logS (mol/L), AqSolDB"),
    "caco2_permeability": (f"{DV}/4259569", "regression", "log Papp (cm/s), Wang et al."),
    "pgp_inhibition": (f"{DV}/4259597", "classification", "P-glycoprotein inhibitor 0/1, Broccatelli"),
    "plasma_protein_binding": (f"{DV}/6413140", "regression", "percent bound, AstraZeneca"),
    "clearance_hepatocyte": (f"{DV}/4266187", "regression", "uL/min/1e6 cells, AstraZeneca"),
    "lipophilicity": ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
                      "regression", "logD7.4, AstraZeneca (MoleculeNet)"),
}


def _get(url):
    for kw in ({"verify": certifi.where()}, {"verify": False}):
        try:
            r = requests.get(url, timeout=120, **kw)
            if r.status_code < 500:
                return r
        except Exception:
            continue
    raise RuntimeError(f"failed to fetch {url}")


def _read(url):
    txt = _get(url).text
    sep = "\t" if "\t" in txt.split("\n", 1)[0] else ","
    return pd.read_csv(io.StringIO(txt), sep=sep)


def _smiles_col(df):
    for c in ("Drug", "smiles", "SMILES", "X", "mol"):
        if c in df.columns:
            return c
    raise KeyError(f"no SMILES column in {list(df.columns)}")


def _value_col(df, smiles_col):
    for c in ("Y", "exp", "y", "label", "value"):
        if c in df.columns:
            return c
    # otherwise first numeric column that is not the SMILES column
    for c in df.columns:
        if c != smiles_col and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise KeyError(f"no value column in {list(df.columns)}")


def main():
    lines = ["# Source: measured ADME / exposure datasets (ladder B)", "",
             "Downloaded 2026-07-21 from TDC / Harvard Dataverse and MoleculeNet; standardised by",
             "`src/sai_cns/adme/fetch_adme.py`. All values are measured.", "",
             "| Endpoint | Task | n (standardised) | Description |", "|---|---|---|---|"]
    for name, (url, task, desc) in SOURCES.items():
        try:
            df = _read(url)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            continue
        sc = _smiles_col(df); vc = _value_col(df, sc)
        rows = []
        for smi, val in zip(df[sc], df[vc]):
            if pd.isna(smi) or pd.isna(val):
                continue
            csmi, ik = standardise(smi)
            if ik is None:
                continue
            rows.append({"inchikey": ik, "smiles": csmi, "y": float(val)})
        out = pd.DataFrame(rows).drop_duplicates("inchikey").reset_index(drop=True)
        if task == "classification":
            out["label"] = out["y"].round().astype(int)
        out.to_csv(OUT / f"{name}.csv", index=False)
        print(f"[{name}] {task}: {len(out)} standardised compounds")
        lines.append(f"| {name} | {task} | {len(out)} | {desc} |")
    (OUT / "SOURCE.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote data/adme/*.csv and data/adme/SOURCE.md")


if __name__ == "__main__":
    main()
