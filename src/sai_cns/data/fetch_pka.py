"""Fetch measured pKa from ChEMBL, and separate basic from acidic ionisation.

The CNS MPO desirability score uses the MOST BASIC pKa. ChEMBL records pKa as a single
standard_type without stating whether the measured constant describes protonation of a base or
deprotonation of an acid, and inspection shows the field mixes both: carboxyl dissociation,
alpha-CH acidity and amine protonation all appear under the same label, and a minority of values lie
outside any chemically plausible range. Training on the pooled field and calling the result a basic
pKa would be wrong.

Records are therefore filtered to a plausible range and classified from two independent signals: the
wording of the assay description, and whether RDKit finds a basic centre (aliphatic or aromatic
amine, amidine, guanidine) in the structure. A record is treated as basic only when the description
does not indicate acid dissociation AND the molecule contains a basic centre. The remainder is kept
separately so the pooled model can still be trained and compared.

Output: data/endpoints_reg/pka_basic.csv, data/endpoints_reg/pka_all.csv
"""
from __future__ import annotations

import re
import time
import sys
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
requests.packages.urllib3.disable_warnings()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_compound_library import add_parent_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "endpoints_reg"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://www.ebi.ac.uk/chembl/api/data"

PLAUSIBLE = (0.0, 14.0)
ACID_WORDS = re.compile(r"carboxyl|acid dissociation|deprotonation|alpha-ch|phenol|acidic|"
                        r"cooh|sulfon|hydroxyl", re.I)
BASE_WORDS = re.compile(r"basic|conjugate acid|protonation|amine|amino|nitrogen|pyridin", re.I)
BASIC_SMARTS = [
    Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(N[!#6]);!$(N=*);!$(N#*);!$(N[a])]"),  # aliphatic amine
    Chem.MolFromSmarts("[NX3][CX3]=[NX2]"),                                     # amidine
    Chem.MolFromSmarts("[NX3][CX3](=[NX2])[NX3]"),                              # guanidine
    Chem.MolFromSmarts("n1ccccc1"),                                             # pyridine-like
]


def has_basic_centre(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return any(m.HasSubstructMatch(p) for p in BASIC_SMARTS if p is not None)


def fetch():
    rows, url = [], f"{BASE}/activity.json?standard_type=pKa&limit=1000"
    page = 0
    while url:
        for _ in range(4):
            try:
                r = requests.get(url, timeout=90, verify=False)
                r.raise_for_status()
                break
            except Exception:
                time.sleep(3)
        else:
            break
        j = r.json()
        for a in j["activities"]:
            smi, v = a.get("canonical_smiles"), a.get("standard_value")
            if not smi or v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            rows.append({"smiles": smi, "pka": v,
                         "assay": (a.get("assay_description") or "")[:200],
                         "year": a.get("document_year")})
        page += 1
        print(f"  page {page}: {len(rows)} records", flush=True)
        nxt = j["page_meta"]["next"]
        url = ("https://www.ebi.ac.uk" + nxt) if nxt else None
    return pd.DataFrame(rows)


def main():
    df = fetch()
    print(f"fetched {len(df)} raw pKa records")
    df = df[(df.pka >= PLAUSIBLE[0]) & (df.pka <= PLAUSIBLE[1])]
    print(f"  {len(df)} within the plausible range {PLAUSIBLE}")

    df["basic_centre"] = [has_basic_centre(s) for s in df.smiles]
    df = df[df.basic_centre.notna()]
    df["says_acid"] = df.assay.str.contains(ACID_WORDS, na=False)
    df["says_base"] = df.assay.str.contains(BASE_WORDS, na=False)

    # median per compound, so a compound measured repeatedly does not dominate
    allp = add_parent_key(df).groupby("inchikey").agg(
                                    smiles=("smiles", "first"),
                                    pka=("pka", "median"), year=("year", "max"),
                                    basic_centre=("basic_centre", "first"),
                                    says_acid=("says_acid", "any"),
                                    says_base=("says_base", "any")).reset_index(drop=True)
    allp.to_csv(OUT / "pka_all.csv", index=False)

    basic = allp[(allp.basic_centre) & (~allp.says_acid)]
    basic[["smiles", "pka", "year"]].to_csv(OUT / "pka_basic.csv", index=False)

    print(f"\npooled pKa compounds        : {len(allp):,}")
    print(f"with an RDKit basic centre  : {int(allp.basic_centre.sum()):,}")
    print(f"description indicates acid  : {int(allp.says_acid.sum()):,}")
    print(f"BASIC subset kept for MPO   : {len(basic):,}  "
          f"(median pKa {basic.pka.median():.2f}, range {basic.pka.min():.1f}-{basic.pka.max():.1f})")
    print("wrote", OUT / "pka_basic.csv", "and", OUT / "pka_all.csv")


if __name__ == "__main__":
    main()
