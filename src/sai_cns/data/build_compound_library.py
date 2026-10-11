"""Build the BrainSafe compound library from measured bioactivity sources.

The library is a single, deduplicated table of every compound that carries at least one
measured endpoint label, plus the clinical-precedent reference compounds. Each compound is
standardised to a canonical parent structure, keyed by InChIKey, and annotated with its
measured labels, a natural-product (flavonoid-core) flag, and interpretable physicochemical
descriptors. Nothing is imputed: a missing label is left empty, not guessed.

Measured sources (labels):
  data/endpoints/*.csv            classification labels per target (SMILES, label, pChEMBL, year)
  data/endpoints_reg/antioxidant_dpph.csv   antioxidant regression (SMILES, pIC50, year)
Reference (no training label):
  data/clinical_cns_reference.csv approved / clinical CNS compounds (name, SMILES, phase, ATC, disease)

Outputs:
  data/processed/compound_library.csv     one row per unique compound (wide)
  data/processed/endpoint_labels_long.csv  one row per compound-endpoint measurement (long)
  results/tables/library_summary.csv       record counts by source and endpoint
  docs/DATA_DICTIONARY.md                  definition of every column

Run:  python src/sai_cns/data/build_compound_library.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[3]

# Endpoint files and the disease/role context used in the manuscript.
CLASSIFICATION = {
    "BBB": "blood-brain-barrier permeability",
    "AChE": "acetylcholinesterase (Alzheimer)",
    "BChE": "butyrylcholinesterase (Alzheimer)",
    "BACE1": "beta-secretase 1 (Alzheimer)",
    "GSK3B": "glycogen synthase kinase-3 beta (neuroprotection)",
    "MAO_A": "monoamine oxidase A (depression)",
    "MAO_B": "monoamine oxidase B (Parkinson)",
    "D2": "dopamine D2 receptor",
    "A2A": "adenosine A2A receptor",
    "HT2A": "serotonin 5-HT2A receptor",
    "SERT": "serotonin transporter",
    "hERG": "hERG channel (cardiac safety)",
}

# Flavonoid / polyphenol core scaffolds, matched as substructures. Documented in the data
# dictionary so the flag is fully traceable.
FLAVONOID_SMARTS = {
    "flavone/flavonol": "O=c1cc(-c2ccccc2)oc2ccccc12",
    "isoflavone": "O=c1c(-c2ccccc2)coc2ccccc12",
    "flavanone": "O=C1CC(c2ccccc2)Oc2ccccc21",
    "flavan-3-ol": "OC1Cc2ccccc2OC1c1ccccc1",
    "chalcone": "O=C(C=Cc1ccccc1)c1ccccc1",
}
_FLAVO = {k: Chem.MolFromSmarts(v) for k, v in FLAVONOID_SMARTS.items()}


def parent_mol(smiles: str):
    """Return the largest organic fragment of a SMILES as an RDKit Mol, or None."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def standardise(smiles: str):
    """Return (canonical_smiles, inchikey) for the parent structure, or (None, None)."""
    mol = parent_mol(smiles)
    if mol is None:
        return None, None
    try:
        return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol)
    except Exception:
        return None, None


def descriptors(mol) -> dict:
    """Interpretable physicochemical descriptors used as model features and for reporting."""
    return {
        "mw": round(Descriptors.MolWt(mol), 2),
        "clogp": round(Crippen.MolLogP(mol), 3),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "fraction_csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "qed": round(QED.qed(mol), 3),
    }


def flavonoid_flag(mol) -> bool:
    return any(mol.HasSubstructMatch(p) for p in _FLAVO.values() if p is not None)


def add_parent_key(df, smiles_col="smiles"):
    """Add an `inchikey` column and canonicalise `smiles` to the desalted parent.

    Measurement tables must be collapsed per compound before they become training rows, and the key
    for that has to be the InChIKey of the parent structure. Grouping on the raw SMILES string, which
    the expansion fetchers all did, counts a salt and its free base as two compounds and lets both
    reach the training table. They are one compound, and once the featuriser strips the counter-ion
    they are one input, so no split can keep them apart. Collapsing them here, while the measurement
    is still attached and a median still means something, is the only place it can be done properly.

    Note what this deliberately does not merge. The InChIKey separates stereoisomers, and it should:
    enantiomers are different compounds and can differ in potency by orders of magnitude, so a data
    step has no business averaging them. That they collide under a stereo-blind fingerprint is a
    property of the model rather than the chemistry, and is handled where it belongs, in
    train_rf._dedup_features.

    Rows whose structure cannot be parsed are dropped, and the count is returned to the caller
    through the row difference rather than being silently absorbed.
    """
    keys, parents = [], []
    for smi in df[smiles_col].astype(str):
        csmi, ik = standardise(smi)
        keys.append(ik)
        parents.append(csmi)
    out = df.copy()
    out["inchikey"] = keys
    out[smiles_col] = parents
    return out.dropna(subset=["inchikey"]).reset_index(drop=True)


def collect_measurements() -> pd.DataFrame:
    """Read every measured source into a long table of standardised measurements."""
    records = []
    for ep in CLASSIFICATION:
        path = ROOT / "data" / "endpoints" / f"{ep}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            csmi, ik = standardise(r["smiles"])
            if ik is None:
                continue
            src = r.get("source") if pd.notna(r.get("source")) else ("B3DB" if ep == "BBB" else "ChEMBL_37")
            records.append({
                "inchikey": ik, "canonical_smiles": csmi, "endpoint": ep,
                "label": int(r["label"]) if pd.notna(r.get("label")) else None,
                "value_pchembl": float(r["pchembl"]) if pd.notna(r.get("pchembl")) else None,
                "year": r.get("year"), "source": src,
                "role": "training_label",
            })
    anti = ROOT / "data" / "endpoints_reg" / "antioxidant_dpph.csv"
    if anti.exists():
        for _, r in pd.read_csv(anti).iterrows():
            csmi, ik = standardise(r["smiles"])
            if ik is None:
                continue
            records.append({
                "inchikey": ik, "canonical_smiles": csmi, "endpoint": "antioxidant_DPPH",
                "label": None, "value_pchembl": float(r["y"]) if pd.notna(r.get("y")) else None,
                "year": r.get("year"), "source": "ChEMBL_DPPH", "role": "training_label",
            })
    return pd.DataFrame(records)


def build_library() -> None:
    long = collect_measurements()
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    long.to_csv(ROOT / "data" / "processed" / "endpoint_labels_long.csv", index=False)

    # one canonical structure per InChIKey
    struct = long.drop_duplicates("inchikey").set_index("inchikey")["canonical_smiles"].to_dict()

    # clinical reference compounds (structure/name only; no training label)
    clin = {}
    clin_path = ROOT / "data" / "clinical_cns_reference.csv"
    if clin_path.exists():
        for _, r in pd.read_csv(clin_path).iterrows():
            csmi, ik = standardise(r["smiles"])
            if ik is None:
                continue
            struct.setdefault(ik, csmi)
            clin[ik] = {"name": r.get("name"), "clinical_max_phase": r.get("max_phase"),
                        "atc": r.get("atc"), "disease": r.get("disease")}

    rows = []
    for ik, csmi in struct.items():
        mol = Chem.MolFromSmiles(csmi)
        if mol is None:
            continue
        sub = long[long.inchikey == ik]
        row = {"inchikey": ik, "canonical_smiles": csmi,
               "name": clin.get(ik, {}).get("name")}
        for ep in CLASSIFICATION:
            e = sub[sub.endpoint == ep]
            row[f"{ep}_label"] = int(e["label"].dropna().median()) if e["label"].notna().any() else None
            row[f"{ep}_pchembl"] = round(e["value_pchembl"].median(), 3) if e["value_pchembl"].notna().any() else None
        anti = sub[sub.endpoint == "antioxidant_DPPH"]
        row["antioxidant_pIC50"] = round(anti["value_pchembl"].median(), 3) if anti["value_pchembl"].notna().any() else None
        row["n_endpoints_measured"] = int(sub["endpoint"].nunique())
        row["role"] = "training_label" if len(sub) else "reference"
        row["sources"] = ";".join(sorted(set(sub["source"]))) or "ChEMBL_ATC-N"
        c = clin.get(ik, {})
        row["clinical_max_phase"] = c.get("clinical_max_phase")
        row["atc"] = c.get("atc")
        row["disease"] = c.get("disease")
        row["is_flavonoid_core"] = flavonoid_flag(mol)
        row.update(descriptors(mol))
        rows.append(row)

    lib = pd.DataFrame(rows).sort_values("inchikey").reset_index(drop=True)
    lib.to_csv(ROOT / "data" / "processed" / "compound_library.csv", index=False)

    # summary
    summ = [{"metric": "unique_compounds", "value": len(lib)},
            {"metric": "with_training_label", "value": int((lib.role == "training_label").sum())},
            {"metric": "reference_only", "value": int((lib.role == "reference").sum())},
            {"metric": "flavonoid_core", "value": int(lib.is_flavonoid_core.sum())},
            {"metric": "measured_records_total", "value": len(long)}]
    for ep in list(CLASSIFICATION) + ["antioxidant_DPPH"]:
        col = f"{ep}_label" if ep in CLASSIFICATION else "antioxidant_pIC50"
        summ.append({"metric": f"compounds_with_{ep}", "value": int(lib[col].notna().sum())})
    pd.DataFrame(summ).to_csv(ROOT / "results" / "tables" / "library_summary.csv", index=False)

    write_data_dictionary(lib)
    print(f"compound_library.csv: {len(lib)} unique compounds "
          f"({int((lib.role=='training_label').sum())} labelled, "
          f"{int(lib.is_flavonoid_core.sum())} flavonoid-core)")
    print("wrote data/processed/{compound_library,endpoint_labels_long}.csv, "
          "results/tables/library_summary.csv, docs/DATA_DICTIONARY.md")


def write_data_dictionary(lib: pd.DataFrame) -> None:
    lines = [
        "# Data dictionary: compound_library.csv", "",
        "One row per unique compound (keyed by InChIKey of the standardised parent structure). "
        "Empty cells mean *not measured*, never imputed.", "",
        "| Column | Type | Definition |", "|---|---|---|",
        "| inchikey | text | Standard InChIKey of the canonical parent (salt-stripped, largest fragment). Primary key. |",
        "| canonical_smiles | text | RDKit canonical SMILES of the parent structure. |",
        "| name | text | Compound name where known (from the clinical reference set). |",
    ]
    for ep, desc in CLASSIFICATION.items():
        lines.append(f"| {ep}_label | 0/1 | Measured active(1)/inactive(0) call for {desc}; pChEMBL>=6 active, <5 inactive. |")
        lines.append(f"| {ep}_pchembl | float | Median measured pChEMBL for {desc}. |")
    lines += [
        "| antioxidant_pIC50 | float | Median measured DPPH radical-scavenging pIC50. |",
        "| n_endpoints_measured | int | Number of endpoints with a measured value for this compound. |",
        "| role | text | training_label (has a measured endpoint) or reference (clinical structure only). |",
        "| sources | text | Semicolon-separated measured-data sources contributing to this compound. |",
        "| clinical_max_phase | float | Highest clinical phase reached (ChEMBL ATC-N reference), if applicable. |",
        "| atc | text | Anatomical Therapeutic Chemical code (reference set). |",
        "| disease | text | Disease grouping from the ATC-N reference set. |",
        "| is_flavonoid_core | bool | True if the structure matches a flavonoid/polyphenol core substructure "
        f"({', '.join(FLAVONOID_SMARTS)}). |",
        "| mw | float | Molecular weight. |",
        "| clogp | float | Crippen calculated logP. |",
        "| tpsa | float | Topological polar surface area. |",
        "| hbd | int | Hydrogen-bond donors. |",
        "| hba | int | Hydrogen-bond acceptors. |",
        "| rotatable_bonds | int | Rotatable bond count. |",
        "| aromatic_rings | int | Aromatic ring count. |",
        "| fraction_csp3 | float | Fraction of sp3-hybridised carbons. |",
        "| qed | float | Quantitative estimate of drug-likeness. |",
        "",
        "## Standardisation", "",
        "SMILES are parsed with RDKit; the largest organic fragment is kept (salt/counter-ion "
        "removal), the structure is sanitised, and the canonical SMILES and InChIKey are computed. "
        "Deduplication is by full standard InChIKey.", "",
        "## Flavonoid-core substructures", "",
    ]
    for k, v in FLAVONOID_SMARTS.items():
        lines.append(f"- {k}: `{v}`")
    (ROOT / "docs" / "DATA_DICTIONARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build_library()
