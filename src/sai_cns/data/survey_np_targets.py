"""Which targets could actually carry a natural-product endpoint, measured rather than guessed.

Adding targets to the panel because they sound relevant is how a panel acquires endpoints that
cannot be trained. This asks the data instead: across NPASS 3.0, which targets have enough measured
natural-product activity, on a human protein, with both classes present, to support a model at all.

Four filters, in order, each of which removes a specific way of being wrong:

  a protein          NPASS records activity against cell lines and whole organisms as well as
                     proteins. HeLa, MCF7 and A549 are not targets, and cytotoxicity is not
                     engagement; a UniProt accession is required.
  human              a plant enzyme or a bacterial target is measurable and irrelevant here.
  enough compounds   below MIN_COMPOUNDS a model cannot be fitted honestly, whatever the biology.
  both classes       a target with only actives teaches a classifier nothing, and the project's
                     censored-bound rule is what supplies most of the inactives.

The survey also reports whether a target is already in the panel, and how much of its natural-product
chemistry is the sp3-rich non-aromatic kind the panel is short of, because a new target that brings
only more flat aromatic chemistry does not close the gap that motivated this.

Nothing is fetched or trained here. The output is a ranked candidate list for a decision.

Output: results/tables/np_target_survey.csv

Run:  python src/sai_cns/data/survey_np_targets.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activity_labels import ACTIVE_CUT, INACTIVE_CUT, label_from  # noqa: E402,F401

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[3]
TAB = ROOT / "results" / "tables"
ENDPOINTS = ROOT / "data" / "endpoints"

KEEP_TYPES = {"IC50", "Ki", "Kd", "EC50", "Potency"}
TO_NM = {"nM": 1.0, "uM": 1e3, "mM": 1e6, "M": 1e9, "pM": 1e-3}
MIN_COMPOUNDS = 60          # below this a per-target model is not defensible
MIN_PER_CLASS = 15
SP3_RICH, MAX_AROMATIC = 0.55, 1

# Words that mark a target as plausibly CNS-relevant. Deliberately broad: this is a shortlist for a
# human decision, not an automatic inclusion rule, and a false positive here costs a glance.
CNS_HINTS = re.compile(
    r"receptor|transporter|channel|kinase|phosphodiesterase|monoamine|cholinesterase|"
    r"gaba|glutamate|dopamin|seroton|opioid|cannabinoid|adenosin|histamin|sigma|"
    r"secretase|synuclein|amyloid|tau|deacetylase|sirtuin|nitric oxide|cyclooxygenase|"
    r"aromatase|carbonic anhydrase|topoisomerase|tyrosinase|xanthine|aldose|glycogen",
    re.I)


def panel_accessions() -> dict[str, str]:
    """UniProt to endpoint name, for everything already modelled."""
    out: dict[str, str] = {}
    for src in (ROOT / "data" / "core_panel_target_ids.json",
                ROOT / "data" / "_chembl_cache" / "uniprot_map.json"):
        if not src.exists():
            continue
        d = json.loads(src.read_text(encoding="utf-8"))
        for ep, rec in d.items():
            acc = rec.get("uniprot") if isinstance(rec, dict) else rec
            if acc:
                out[str(acc).upper()] = ep
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Rank targets that could carry an NP endpoint.")
    ap.add_argument("--npass-dir", default="NPASS data")
    ap.add_argument("--min-compounds", type=int, default=MIN_COMPOUNDS)
    args = ap.parse_args(argv)
    npass = ROOT / args.npass_dir
    if not npass.exists():
        raise SystemExit(f"NPASS directory not found: {npass}")

    T = "\t"
    print("reading NPASS ...", flush=True)
    tgt = pd.read_csv(npass / "NPASS3.0_target.txt", sep=T, low_memory=False)
    before = len(tgt)
    tgt = tgt[tgt.uniprot_id.notna() & (tgt.uniprot_id.astype(str).str.strip() != "n.a.")]
    print(f"  targets: {before:,} -> {len(tgt):,} carrying a UniProt accession "
          f"(cell lines and organisms dropped)")
    human = tgt[tgt.target_organism.astype(str).str.contains("Homo sapiens", case=False, na=False)]
    print(f"  human targets: {len(human):,}")

    acts = pd.read_csv(npass / "NPASS3.0_activities.txt", sep=T, low_memory=False)
    acts = acts[acts.target_id.isin(set(human.target_id))]
    acts = acts[acts.activity_type_grouped.isin(KEEP_TYPES)
                & acts.activity_units.isin(TO_NM)]
    acts["v"] = pd.to_numeric(acts.activity_value, errors="coerce")
    acts = acts[acts.v > 0]
    import math
    acts["pchembl"] = 9.0 - acts.v.mul(acts.activity_units.map(TO_NM)).map(math.log10)
    acts["label"] = [label_from(p, r) for p, r in zip(acts.pchembl, acts.activity_relation)]
    acts = acts[acts.label.notna()]
    print(f"  usable measurements at human protein targets: {len(acts):,}\n")

    struct = pd.read_csv(npass / "NPASS3.0_naturalproducts_structure.txt", sep=T,
                         low_memory=False)[["np_id", "SMILES"]]
    m = acts.merge(struct, on="np_id", how="inner")
    m = m[m.SMILES.notna()]

    # sp3-richness once per structure, not once per measurement
    desc: dict[str, bool] = {}
    for smi in m.SMILES.unique():
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            desc[smi] = False
            continue
        desc[smi] = bool(rdMolDescriptors.CalcFractionCSP3(mol) >= SP3_RICH
                         and rdMolDescriptors.CalcNumAromaticRings(mol) <= MAX_AROMATIC)
    m["fills_gap"] = m.SMILES.map(desc)

    known = panel_accessions()
    m = m.merge(human[["target_id", "target_name", "uniprot_id"]], on="target_id", how="left")
    m["acc"] = m.uniprot_id.astype(str).str.upper().str.strip()

    rows = []
    for acc, g in m.groupby("acc"):
        gg = g.drop_duplicates("SMILES")
        n_act = int((gg.label == 1).sum())
        n_inact = int((gg.label == 0).sum())
        name = str(g.target_name.iloc[0])
        rows.append({
            "uniprot": acc, "target_name": name[:70],
            "already_in_panel": known.get(acc, ""),
            "compounds": len(gg), "actives": n_act, "inactives": n_inact,
            "measurements": len(g),
            "sp3_rich_non_aromatic": int(gg.fills_gap.sum()),
            "pct_gap_chemistry": round(100 * float(gg.fills_gap.mean()), 1),
            "cns_keyword": bool(CNS_HINTS.search(name)),
        })
    surv = pd.DataFrame(rows)
    surv["trainable"] = ((surv.compounds >= args.min_compounds)
                         & (surv.actives >= MIN_PER_CLASS)
                         & (surv.inactives >= MIN_PER_CLASS))
    surv = surv.sort_values(["trainable", "compounds"], ascending=[False, False])
    TAB.mkdir(parents=True, exist_ok=True)
    surv.to_csv(TAB / "np_target_survey.csv", index=False)

    new = surv[surv.trainable & (surv.already_in_panel == "")]
    print("=== targets that could carry a natural-product endpoint ===")
    print(f"  human protein targets with any usable NP data   {len(surv):,}")
    print(f"  trainable (>={args.min_compounds} compounds, >={MIN_PER_CLASS} per class)  "
          f"{int(surv.trainable.sum())}")
    print(f"    of those, already in the panel                "
          f"{int((surv.trainable & (surv.already_in_panel != '')).sum())}")
    print(f"    of those, NEW                                 {len(new)}")
    print()
    if len(new):
        print(new.head(25)[["uniprot", "target_name", "compounds", "actives", "inactives",
                            "pct_gap_chemistry", "cns_keyword"]].to_string(index=False))
    print("\nwrote results/tables/np_target_survey.csv")
    print("Nothing fetched or trained. This is a shortlist for a decision.")


if __name__ == "__main__":
    main(sys.argv[1:])
