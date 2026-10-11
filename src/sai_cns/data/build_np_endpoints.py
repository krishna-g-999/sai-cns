"""Build endpoint tables for targets added to close the natural-product gap.

Three targets are added, chosen on mechanism rather than on data volume. The survey found 58 new
trainable targets, most of which have nothing to do with the brain: Geminin, apurinic site lyase and
prelamin-A/C all carry more measured natural-product activity than any of the three taken here, and
adding them would dilute a CNS panel with replication and DNA-repair biology to make a count larger.

  NRF2      Q16236   the effector of the KEAP1-NRF2 axis already modelled. KEAP1 is the sensor and
                     was in the panel alone, so the arm that natural products are most often reported
                     to act on had no readout. This is the withanolide mechanism.
  NFKB1     P19838   the canonical inflammatory transcription factor, joining NLRP3 and RIPK1 on a
                     neuroinflammation axis that already exists in the disease graph.
  NR3C1     P04150   the glucocorticoid receptor, on the stress and depression axis. The smallest of
                     the three but the best balanced, and the richest in the sp3 chemistry the panel
                     lacks.

Labels come from the project's existing rule, censored bounds included, so these tables are built the
same way as every other endpoint and carry the same meaning. Structures are deduplicated on the
InChIKey of the desalted parent, because two records of one compound written differently would
otherwise enter as two.

This writes tables only. Training is a separate command, and the tables should be inspected first:
each one is a new scientific claim about what the panel measures.

Output: data/endpoints/<NAME>.csv, in the schema every other endpoint uses
        results/tables/np_endpoints_built.csv

Run:  python src/sai_cns/data/build_np_endpoints.py
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_compound_library import standardise  # noqa: E402
from activity_labels import ACTIVE_CUT, INACTIVE_CUT, label_from  # noqa: E402,F401

ENDPOINTS = ROOT / "data" / "endpoints"
TAB = ROOT / "results" / "tables"

KEEP_TYPES = {"IC50", "Ki", "Kd", "EC50", "Potency"}
TO_NM = {"nM": 1.0, "uM": 1e3, "mM": 1e6, "M": 1e9, "pM": 1e-3}

# endpoint name -> (UniProt, why it is here)
SELECTED = {
    "NRF2": ("Q16236", "effector of the KEAP1-NRF2 axis; the mechanism most often ascribed to "
                       "withanolides and other electrophilic natural products"),
    "NFKB1": ("P19838", "canonical inflammatory transcription factor; joins NLRP3 and RIPK1 on the "
                        "neuroinflammation axis"),
    "NR3C1": ("P04150", "glucocorticoid receptor; stress and depression axis, best class balance "
                        "and richest in sp3 chemistry of the candidates"),
}


def assay_composition(m: pd.DataFrame) -> pd.DataFrame:
    """What kind of measurement each endpoint's labels are actually made of.

    Written because the first explanation offered for why these three endpoints failed, that their
    activity is recorded in cell-based reporter assays, was checked against the NPASS metadata and
    was not supported: a cell type is named for only 3.0, 30.3 and 6.7 per cent of NFKB1, NR3C1 and
    NRF2 records. What the metadata does support is this table. A direct binding constant, Ki or Kd,
    is what a fingerprint-based binder classifier is fitted to reproduce, and for two of the three
    there is essentially none: the labels are almost entirely `Potency`, a pooled functional readout
    that mixes assay formats and does not define a binding class.
    """
    rows = []
    for ep, g in m.groupby("endpoint"):
        vc = g.activity_type_grouped.value_counts()
        binding = int(vc.get("Ki", 0) + vc.get("Kd", 0))
        cells = g.assay_cell_type.fillna("n.a.").ne("n.a.").mean() if "assay_cell_type" in g else 0.0
        rows.append({"endpoint": ep, "labelled_records": len(g),
                     "potency": int(vc.get("Potency", 0)), "ic50": int(vc.get("IC50", 0)),
                     "ec50": int(vc.get("EC50", 0)), "ki_kd": binding,
                     "pct_direct_binding_constant": round(100 * binding / max(len(g), 1), 1),
                     "pct_naming_a_cell_type": round(100 * float(cells), 1)})
    out = pd.DataFrame(rows).sort_values("endpoint")
    TAB.mkdir(parents=True, exist_ok=True)
    out.to_csv(TAB / "np_endpoint_assay_composition.csv", index=False)
    print("assay composition of the labels:")
    print(out.to_string(index=False))
    print("wrote results/tables/np_endpoint_assay_composition.csv\n")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build endpoint tables for the added NP targets.")
    ap.add_argument("--npass-dir", default="NPASS data")
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite a table that already exists")
    args = ap.parse_args(argv)
    npass = ROOT / args.npass_dir
    if not npass.exists():
        raise SystemExit(f"NPASS directory not found: {npass}")

    T = "\t"
    tgt = pd.read_csv(npass / "NPASS3.0_target.txt", sep=T, low_memory=False)
    tgt["acc"] = tgt.uniprot_id.astype(str).str.upper().str.strip()
    want = {acc: name for name, (acc, _why) in SELECTED.items()}
    tgt = tgt[tgt.acc.isin(want)]
    tgt["endpoint"] = tgt.acc.map(want)
    print(f"{len(tgt)} NPASS target records for {tgt.endpoint.nunique()} selected endpoints")

    acts = pd.read_csv(npass / "NPASS3.0_activities.txt", sep=T, low_memory=False)
    acts = acts[acts.target_id.isin(set(tgt.target_id))]
    acts = acts[acts.activity_type_grouped.isin(KEEP_TYPES) & acts.activity_units.isin(TO_NM)]
    acts["v"] = pd.to_numeric(acts.activity_value, errors="coerce")
    acts = acts[acts.v > 0]
    acts["pchembl"] = 9.0 - acts.v.mul(acts.activity_units.map(TO_NM)).map(math.log10)
    acts["label"] = [label_from(p, r) for p, r in zip(acts.pchembl, acts.activity_relation)]
    acts = acts[acts.label.notna()]

    struct = pd.read_csv(npass / "NPASS3.0_naturalproducts_structure.txt", sep=T,
                         low_memory=False)[["np_id", "SMILES"]]
    m = (acts.merge(tgt[["target_id", "endpoint"]], on="target_id", how="inner")
             .merge(struct, on="np_id", how="inner"))
    m = m[m.SMILES.notna()]
    print(f"{len(m):,} labelled measurements joined to a structure\n")
    assay_composition(m)

    built = []
    ENDPOINTS.mkdir(parents=True, exist_ok=True)
    for ep, g in m.groupby("endpoint"):
        path = ENDPOINTS / f"{ep}.csv"
        if path.exists() and not args.overwrite:
            print(f"[{ep:6s}] exists, skipped (use --overwrite)")
            continue

        # One row per compound. Where a compound is measured more than once, the most potent
        # exact value wins; a bound never displaces an exact value, because a bound is weaker
        # evidence about the same molecule.
        best: dict[str, dict] = {}
        for r in g.itertuples():
            csmi, key = standardise(str(r.SMILES))
            if key is None:
                continue
            cur = best.get(key)
            exact = str(r.activity_relation).strip() in ("=", "", "nan", "n.a.")
            cand = {"smiles": csmi, "label": int(r.label),
                    "pchembl": round(float(r.pchembl), 3),
                    "year": pd.NA, "source": "NPASS3.0",
                    "_exact": exact}
            if cur is None:
                best[key] = cand
            elif exact and not cur["_exact"]:
                best[key] = cand
            elif exact == cur["_exact"] and cand["pchembl"] > cur["pchembl"]:
                best[key] = cand

        df = pd.DataFrame(list(best.values())).drop(columns=["_exact"])
        df = df[["smiles", "label", "pchembl", "year", "source"]]
        df.to_csv(path, index=False)
        n_act = int((df.label == 1).sum())
        built.append({"endpoint": ep, "uniprot": SELECTED[ep][0], "compounds": len(df),
                      "actives": n_act, "inactives": len(df) - n_act,
                      "pct_active": round(100 * n_act / max(len(df), 1), 1),
                      "rationale": SELECTED[ep][1]})
        print(f"[{ep:6s}] {len(df):5d} compounds  {n_act:4d} active  "
              f"{len(df)-n_act:4d} inactive  -> data/endpoints/{ep}.csv")

    if built:
        out = pd.DataFrame(built)
        TAB.mkdir(parents=True, exist_ok=True)
        out.to_csv(TAB / "np_endpoints_built.csv", index=False)
        print(f"\n{len(out)} endpoint table(s) written, "
              f"{int(out.compounds.sum()):,} compounds total")
        print("wrote results/tables/np_endpoints_built.csv")
    print("\nNothing trained. Inspect the tables, then run the training command.")


if __name__ == "__main__":
    main(sys.argv[1:])
