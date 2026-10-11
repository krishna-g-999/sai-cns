"""Integrate the external compound libraries pulled from the HPC cluster.

These libraries provide *structures*, not measured endpoint labels, so they are never used as
training labels (that would reintroduce the circularity we already removed). They are folded in
with explicit roles:

  evaluation : approved / experimental drugs held out for external testing (DrugBank; the
               FDA-curated BBB set, which additionally carries a measured BBB label)
  coverage   : natural products and broad chemical space used to characterise the applicability
               domain and to provide the flavonoid / natural-product panel requested in review

The measured training core (data/processed/compound_library.csv) is left untouched; every output
here is written under data/external/processed/ so the two never mix.

Sources (data/external/), verified 2026-07-20:
  drugbank_smiles.tsv                 12,313  DrugBank structures (src,id,smiles)
  BBB_Final_GNN_Dataset_..._v5.csv     1,690  FDA-curated BBB set with measured bbb_status
  coconut_smiles.tsv                 695,119  COCONUT natural products (flavonoid source)
  natural_all_CNS.tsv                217,480  COCONUT CNS-like natural products (+descriptors)
  pubchem_natural_smiles_clean.tsv     2,450  PubChem natural products
  chembl34_smiles_clean.tsv        1,673,554  ChEMBL 34 structures (coverage pool; counted only)

Outputs (data/external/processed/):
  external_drugs.csv          standardised DrugBank small molecules (role=evaluation)
  external_bbb_test.csv       FDA-curated BBB set with measured label + overlap flag vs B3DB
  flavonoid_panel.csv         flavonoid-core compounds found across the natural-product sources
  natural_products_coverage.csv   deduplicated natural-product coverage set
  external_summary.csv        record counts per source and role

Run:  python src/sai_cns/data/integrate_external.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_compound_library import standardise, descriptors, flavonoid_flag, _FLAVO  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
EXT = ROOT / "data" / "external"
OUT = EXT / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# DrugBank carries biologics (peptides, proteins) with very large SMILES; keep small molecules.
SMALL_MOLECULE_MW_MAX = 900.0


def _progress(source: str, i: int, every: int = 50_000) -> None:
    if i and i % every == 0:
        print(f"  {source}: {i:,} rows scanned")


def standardise_source(path: Path, source: str, role: str, sep: str,
                       smiles_col: str, id_col: str | None,
                       mw_max: float | None = None) -> pd.DataFrame:
    """Standardise every structure in a source file into a uniform table.

    Returns one row per successfully parsed structure with its canonical parent, InChIKey,
    interpretable descriptors and a flavonoid-core flag. Structures failing to parse, or above
    mw_max (used to drop biologics), are skipped and counted in the run log.
    """
    df = pd.read_csv(path, sep=sep)
    kept, skipped = [], 0
    for i, r in enumerate(df.itertuples(index=False), 1):
        _progress(source, i)
        row = r._asdict()
        csmi, ik = standardise(row[smiles_col])
        if ik is None:
            skipped += 1
            continue
        mol = Chem.MolFromSmiles(csmi)
        if mol is None:
            skipped += 1
            continue
        d = descriptors(mol)
        if mw_max is not None and d["mw"] > mw_max:
            skipped += 1
            continue
        kept.append({
            "inchikey": ik,
            "canonical_smiles": csmi,
            "source": source,
            "source_id": row.get(id_col) if id_col else None,
            "role": role,
            "is_flavonoid_core": flavonoid_flag(mol),
            **d,
        })
    out = pd.DataFrame(kept).drop_duplicates("inchikey").reset_index(drop=True)
    print(f"{source}: {len(out):,} unique structures kept, {skipped:,} skipped "
          f"(unparsable or > {mw_max} Da)" if mw_max else
          f"{source}: {len(out):,} unique structures kept, {skipped:,} skipped (unparsable)")
    return out


def scan_flavonoids(path: Path, source: str, sep: str, smiles_col: str, id_col: str) -> pd.DataFrame:
    """Two-pass flavonoid extraction for large files.

    Pass 1 parses each SMILES and tests the flavonoid-core substructures only (cheap); pass 2
    fully standardises just the matches. This avoids computing descriptors for hundreds of
    thousands of non-matching natural products.
    """
    df = pd.read_csv(path, sep=sep)
    hits = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        _progress(source, i)
        row = r._asdict()
        mol = Chem.MolFromSmiles(str(row[smiles_col]))
        if mol is None:
            continue
        if any(mol.HasSubstructMatch(p) for p in _FLAVO.values() if p is not None):
            csmi, ik = standardise(row[smiles_col])
            if ik is None:
                continue
            m2 = Chem.MolFromSmiles(csmi)
            if m2 is None:
                continue
            rec = {"inchikey": ik, "canonical_smiles": csmi, "source": source,
                   "source_id": row.get(id_col), "role": "coverage",
                   "is_flavonoid_core": True, **descriptors(m2)}
            hits.append(rec)
    out = pd.DataFrame(hits).drop_duplicates("inchikey").reset_index(drop=True)
    print(f"{source}: {len(out):,} flavonoid-core compounds")
    return out


def count_rows(path: Path, sep: str) -> int:
    """Fast line count for a coverage pool we record but do not fully standardise."""
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        return sum(1 for _ in fh) - 1


def training_bbb_inchikeys() -> set[str]:
    """InChIKeys of the B3DB BBB training compounds, to flag overlap with the external set."""
    keys = set()
    bbb = ROOT / "data" / "endpoints" / "BBB.csv"
    if bbb.exists():
        for smi in pd.read_csv(bbb)["smiles"]:
            _, ik = standardise(smi)
            if ik:
                keys.add(ik)
    return keys


def training_bbb_feature_vectors() -> set[bytes]:
    """Feature vectors of the BBB training compounds.

    The InChIKey is the wrong granularity for deciding whether the model has seen a compound. It
    separates stereoisomers, salts and protonation states; the featuriser folds a stereo-blind
    fingerprint over the desalted parent and does not. A compound can therefore pass the InChIKey
    check and still be, to the model, a training compound it has memorised. Measured on the shipped
    set: 65 of the 306 compounds flagged novel by InChIKey are byte-identical in feature space to a
    training compound, so a fifth of the external test was not external.
    """
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
    from features.featurize import featurize as _featurize
    bbb = ROOT / "data" / "endpoints" / "BBB.csv"
    if not bbb.exists():
        return set()
    X, _ = _featurize(pd.read_csv(bbb)["smiles"].astype(str).tolist())
    return {row.tobytes() for row in X}


def main() -> None:
    summary = []

    # 1. DrugBank -> external drug evaluation set (small molecules only)
    print("[1/5] DrugBank")
    drugs = standardise_source(EXT / "drugbank_smiles.tsv", "DrugBank", "evaluation",
                               sep="\t", smiles_col="smiles", id_col="id",
                               mw_max=SMALL_MOLECULE_MW_MAX)
    drugs.to_csv(OUT / "external_drugs.csv", index=False)
    summary.append({"source": "DrugBank", "role": "evaluation", "unique_structures": len(drugs),
                    "flavonoid_core": int(drugs.is_flavonoid_core.sum())})

    # 2. FDA-curated BBB set -> external BBB test (keeps the measured label)
    print("[2/5] FDA-curated BBB set")
    bbb = pd.read_csv(EXT / "BBB_Final_GNN_Dataset_2026_CLEAN_v5.csv")
    train_keys = training_bbb_inchikeys()
    train_vectors = training_bbb_feature_vectors()
    recs = []
    for r in bbb.itertuples(index=False):
        row = r._asdict()
        csmi, ik = standardise(row["smiles"])
        if ik is None:
            continue
        recs.append({"inchikey": ik, "canonical_smiles": csmi, "name": row.get("name"),
                     "bbb_status": int(row["bbb_status"]), "orig_source": row.get("source"),
                     "in_b3db_training": ik in train_keys, "role": "evaluation"})
    bbb_out = pd.DataFrame(recs).drop_duplicates("inchikey").reset_index(drop=True)

    # Second, stricter flag: is this compound distinguishable from a training compound by the model?
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
    from features.featurize import featurize as _featurize
    Xe, me = _featurize(bbb_out["canonical_smiles"].astype(str).tolist())
    seen, j = [], 0
    for ok in me:
        if not ok:
            seen.append(True)          # unparseable: treat as not novel rather than as evidence
            continue
        seen.append(Xe[j].tobytes() in train_vectors)
        j += 1
    bbb_out["feature_identical_to_training"] = seen
    bbb_out["novel_to_model"] = (~bbb_out.in_b3db_training) & (~bbb_out.feature_identical_to_training)
    bbb_out.to_csv(OUT / "external_bbb_test.csv", index=False)

    novel = int((~bbb_out.in_b3db_training).sum())
    novel_model = int(bbb_out.novel_to_model.sum())
    print(f"BBB set: {len(bbb_out):,} unique ({novel:,} not in B3DB training by InChIKey; "
          f"{novel_model:,} also distinguishable from training in feature space; "
          f"{int(bbb_out.bbb_status.sum())} permeable / {int((bbb_out.bbb_status==0).sum())} non-permeable)")
    summary.append({"source": "BBB_FDA_curated", "role": "evaluation",
                    "unique_structures": len(bbb_out), "flavonoid_core": None})

    # 3. Flavonoid panel across the natural-product sources (the review request)
    print("[3/5] Flavonoid extraction (COCONUT + PubChem natural)")
    flavo_parts = [
        scan_flavonoids(EXT / "coconut_smiles.tsv", "COCONUT", "\t", "smiles", "id"),
        scan_flavonoids(EXT / "pubchem_natural_smiles_clean.tsv", "PubChem_Natural", "\t", "smiles", "id"),
    ]
    flavo = pd.concat(flavo_parts, ignore_index=True).drop_duplicates("inchikey").reset_index(drop=True)
    flavo.to_csv(OUT / "flavonoid_panel.csv", index=False)
    summary.append({"source": "flavonoid_panel", "role": "coverage",
                    "unique_structures": len(flavo), "flavonoid_core": len(flavo)})

    # 4. Natural-product coverage (CNS-like natural products)
    print("[4/5] Natural-product coverage")
    nat = standardise_source(EXT / "natural_all_CNS.tsv", "COCONUT_CNS", "coverage",
                             sep="\t", smiles_col="smiles", id_col="id")
    nat.to_csv(OUT / "natural_products_coverage.csv", index=False)
    summary.append({"source": "COCONUT_CNS", "role": "coverage", "unique_structures": len(nat),
                    "flavonoid_core": int(nat.is_flavonoid_core.sum())})

    # 5. ChEMBL 34 broad coverage pool (counted only; standardised on demand for AD analysis)
    print("[5/5] ChEMBL 34 coverage pool (count only)")
    n_chembl = count_rows(EXT / "chembl34_smiles_clean.tsv", "\t")
    summary.append({"source": "ChEMBL34_pool", "role": "coverage_pool",
                    "unique_structures": n_chembl, "flavonoid_core": None})

    pd.DataFrame(summary).to_csv(OUT / "external_summary.csv", index=False)
    print("\n=== external integration summary ===")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nWrote {OUT}/external_drugs.csv, external_bbb_test.csv, flavonoid_panel.csv, "
          "natural_products_coverage.csv, external_summary.csv")


if __name__ == "__main__":
    main()
