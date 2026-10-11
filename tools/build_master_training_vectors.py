"""The actual numeric feature vectors used to train every model, and how each was used.

Every endpoint table (data/endpoints/*.csv, data/adme/*.csv) stores a SMILES string and a label; the
model never sees the SMILES itself, it sees the 1,036-column vector `featurize_one()` computes from
it (a 1,024-bit folded ECFP-4 fingerprint plus twelve physicochemical descriptors,
`models_rf/feature_names.json` names each column). That vector was not previously written to a
single file anywhere: it is computed on the fly at training and inference time and never persisted,
so a reviewer asking "what did the model actually see" could not be shown it directly. This computes
it once, for every distinct compound across every endpoint and ADME table, and writes two files that
between them answer both halves of the question:

  master_feature_vectors.csv   one row per distinct compound (by InChIKey of the desalted,
                                neutralised parent), the 1,036 feature columns exactly as the
                                deployed models receive them.
  master_training_usage.csv    one row per (endpoint, compound) pair actually used, with its label,
                                source and the original as-deposited SMILES, joinable to the vectors
                                file on inchikey. This is "how each vector was used": which of the 75
                                estimators trained or tested on it, as a positive, a negative, or a
                                regression target.

Both keyed on inchikey rather than duplicated per row, since a compound reused across several
endpoints (a genuine and common case) would otherwise repeat 1,036 columns once per endpoint for no
reason.

Run:  python tools/build_master_training_vectors.py
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize, feature_names, parent_mol  # noqa: E402

OUT_VECTORS = ROOT / "results" / "tables" / "master_feature_vectors.csv"
OUT_USAGE = ROOT / "results" / "tables" / "master_training_usage.csv"


def endpoint_files() -> list[Path]:
    # *.chembl_only.csv files (OX1, OX2) are an earlier, superseded pull kept on disk for audit,
    # from before BindingDB was pooled in; they are not read by any training script and including
    # them here would misrepresent what a deployed model was actually trained on.
    files = sorted(Path(p) for p in glob.glob(str(ROOT / "data" / "endpoints" / "*.csv"))
                   if ".chembl_only" not in Path(p).name)
    files += sorted(Path(p) for p in glob.glob(str(ROOT / "data" / "adme" / "*.csv")))
    return files


def main() -> None:
    t_start = time.time()
    files = endpoint_files()
    print(f"{len(files)} endpoint/ADME tables")

    usage_rows = []
    all_smiles: set[str] = set()
    for f in files:
        endpoint = f.stem
        df = pd.read_csv(f, low_memory=False)
        if "smiles" not in df.columns:
            print(f"  skip (no smiles column): {f.name}")
            continue
        label_col = "label" if "label" in df.columns else ("y" if "y" in df.columns else None)
        value_col = "pchembl" if "pchembl" in df.columns else ("y" if "y" in df.columns else None)
        source_col = "source" if "source" in df.columns else None
        year_col = "year" if "year" in df.columns else None
        for row in df.itertuples(index=False):
            smi = str(getattr(row, "smiles"))
            all_smiles.add(smi)
            usage_rows.append({
                "endpoint": endpoint,
                "smiles_as_deposited": smi,
                "label": getattr(row, label_col) if label_col else None,
                "value": getattr(row, value_col) if value_col else None,
                "source": getattr(row, source_col) if source_col else None,
                "year": getattr(row, year_col) if year_col else None,
            })
        print(f"  {endpoint:28s} {len(df):>6,} rows")

    print(f"\n{len(all_smiles):,} distinct as-deposited SMILES strings across all tables")

    # ---- standardise once per distinct string, compute the InChIKey each collapses to ----------
    smi_list = sorted(all_smiles)
    smi_to_key: dict[str, str | None] = {}
    key_to_parent_smiles: dict[str, str] = {}
    unparsed = 0
    for i, s in enumerate(smi_list, 1):
        mol = parent_mol(s)
        if mol is None:
            smi_to_key[s] = None
            unparsed += 1
            continue
        try:
            key = Chem.MolToInchiKey(mol)
        except Exception:
            smi_to_key[s] = None
            unparsed += 1
            continue
        smi_to_key[s] = key
        key_to_parent_smiles.setdefault(key, Chem.MolToSmiles(mol))
        if i % 20000 == 0:
            print(f"  standardised {i:,}/{len(smi_list):,}", flush=True)

    print(f"{unparsed:,} strings could not be parsed and are excluded")
    print(f"{len(key_to_parent_smiles):,} distinct compounds (InChIKeys) remain")

    # ---- featurize each distinct compound exactly once ------------------------------------------
    keys = sorted(key_to_parent_smiles)
    parent_smiles = [key_to_parent_smiles[k] for k in keys]
    t0 = time.time()
    X, mask = featurize(parent_smiles)
    print(f"featurised {X.shape[0]:,} compounds x {X.shape[1]} columns in "
          f"{time.time() - t0:.0f}s")

    cols = feature_names()
    vec_df = pd.DataFrame(X, columns=cols)
    vec_df.insert(0, "canonical_smiles", [parent_smiles[i] for i in range(len(keys)) if mask[i]])
    vec_df.insert(0, "inchikey", [keys[i] for i in range(len(keys)) if mask[i]])
    vec_df.to_csv(OUT_VECTORS, index=False)
    print(f"wrote {OUT_VECTORS.relative_to(ROOT)} "
          f"({vec_df.shape[0]:,} rows x {vec_df.shape[1]} columns, "
          f"{OUT_VECTORS.stat().st_size / 1e6:.0f} MB)")

    # ---- usage table, joined to the same inchikeys ----------------------------------------------
    usage_df = pd.DataFrame(usage_rows)
    usage_df["inchikey"] = usage_df["smiles_as_deposited"].map(smi_to_key)
    usage_df = usage_df.dropna(subset=["inchikey"])
    usage_df = usage_df[["endpoint", "inchikey", "smiles_as_deposited", "label", "value",
                         "source", "year"]]
    usage_df.to_csv(OUT_USAGE, index=False)
    print(f"wrote {OUT_USAGE.relative_to(ROOT)} "
          f"({usage_df.shape[0]:,} rows, {OUT_USAGE.stat().st_size / 1e6:.0f} MB)")

    print(f"\ntotal time {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
