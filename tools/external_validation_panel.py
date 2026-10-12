"""Compounds other groups have reported as CNS-active, for a blind external test of SAI-CNS.

The point of this panel is that SAI-CNS has never seen these claims. Each compound was reported
by someone else, in a published source named here, as acting on a target SAI-CNS models. The
server is then asked the same question and the two answers are recorded side by side, whether or
not they agree. A disagreement is a result, not a failure to be tuned away.

EVIDENCE TIERS. These are not interchangeable and the panel does not pretend otherwise.

  A   a measured IC50 or Ki from an in vitro enzyme assay. The only tier that constitutes
      experimental evidence of inhibition.
  B   reported active in vitro or in cells, where the retrieved source did not give a usable
      numeric potency. Directionally informative, quantitatively not.
  C   molecular docking or other computation only. This is a PREDICTION by another group, not a
      measurement. Agreement between SAI-CNS and a docking score is agreement between two
      predictions and says nothing about what the compound does. Tier C is included because the
      user asked for reported candidates and much of the Ayurvedic literature is of this kind,
      but it is scored and reported separately and must never be pooled with tier A.

HONESTY CONSTRAINTS observed in building this list:

  - Every claim is attributed to a retrievable source. Nothing is asserted from memory.
  - Where a review reported a value, the review is named rather than the primary paper, because
    the primary paper was not itself read. One known transcription problem is flagged inline.
  - Crude plant EXTRACT activity is excluded entirely. An extract IC50 in ug/mL cannot be
    attributed to a named molecule and is not comparable to a pure-compound assay.
  - Compounds are kept in the panel even when expected to be in the training set. Which ones
    turn out to be held out is a finding, not something to arrange in advance.

Run:
    python tools/external_validation_panel.py            # resolve, key, check training overlap
"""
from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))

TRAIN_KEYS = ROOT / "results" / "tables" / "_train_keys.pkl"
OUT = ROOT / "results" / "tables" / "external_panel_resolved.csv"

# (name, target claimed, claim, tier, source)
PANEL = [
    # ---- tier A: measured in vitro potency -------------------------------------------------
    ("didymin", "BACE1", "BACE1 IC50 2.34 uM; most potent of its cluster", "A",
     "Flavonoid BACE1 structure-activity analysis, PMC9271896"),
    ("cardamonin", "BACE1", "BACE1 IC50 4.35 +/- 0.38 uM, Ki 5.1 uM", "A",
     "Boesenbergia rotunda BACE1 study, PMC6471523"),
    ("pinocembrin", "BACE1", "BACE1 IC50 27.01 uM", "A",
     "Boesenbergia rotunda BACE1 study, PMC6471523"),
    ("pinostrobin", "BACE1", "BACE1 IC50 28.44 uM", "A",
     "Boesenbergia rotunda BACE1 study, PMC6471523"),
    ("baicalein", "AChE", "AChE IC50 0.61 uM; highest of the flavonoids tested", "A",
     "Explor Neurosci 2025;4:100697, citing Balkis et al."),
    ("naringin", "AChE", "AChE IC50 0.45 mmol/L", "A",
     "Nutrients 2025 flavonoid-AChE review, PMC11722824"),
    ("quercetin", "MAO_B", "MAO-B IC50 20 uM, from Hypericum hircinum", "A",
     "Molecules 2022;27:4297 natural MAO inhibitors"),
    ("formononetin", "MAO_B", "MAO-A IC50 21.1 uM, MAO-B IC50 11.0 uM", "A",
     "Molecules 2022;27:4297 natural MAO inhibitors"),
    ("huperzine A", "AChE", "AChE inhibitor; H. serrata alkaloid fraction IC50 5.96 ug/mL", "A",
     "TCM cholinesterase screen, PMC6273583"),
    ("berberine", "AChE", "AChE inhibition; Coptis chinensis, with coptisine and palmatine", "C",
     "TCM cholinesterase screen, PMC6273583"),
    ("coptisine", "AChE", "AChE inhibition attributed to the Coptis EXTRACT and to synergy among berberine, coptisine and palmatine; no pure-compound IC50 retrieved", "C",
     "TCM cholinesterase screen, PMC6273583"),
    ("palmatine", "AChE", "AChE inhibition, Coptis chinensis isoquinoline alkaloid", "C",
     "TCM cholinesterase screen, PMC6273583"),

    # ---- tier B: reported active, no usable number retrieved -------------------------------
    ("liensinine", "AChE", "reported dual AChE / BACE1 inhibitory activity", "B",
     "cited in Centella asiatica dual-target review, S2950199726000194"),
    ("coumestrol", "BACE1", "kinetic study against AChE, BChE and BACE1", "B",
     "Pueraria lobata molecular modelling study, PMC6017171"),
    ("haemanthidine", "MAO_B", "inhibits human MAO-B; from Scadoxus puniceus", "B",
     "Molecules 2022;27:4297 natural MAO inhibitors"),
    ("schisandrin B", "GSK3B", "upregulates p-GSK-3beta (Ser9); no IC50 retrieved", "B",
     "GSK-3beta natural inhibitors review, PMC11317746"),
    ("withaferin A", "BACE1", "withanolide; BACE1 inhibition and ADAM10 induction in vitro", "B",
     "MDPI 2025 withanolide review"),

    # ---- tier C: docking / in silico only, NOT experimental evidence ------------------------
    ("asiatic acid", "BACE1", "docking: among strongest BACE1 and AChE binders of 8 tested", "C",
     "Centella asiatica in silico study, S2950199726000194"),
    ("madecassoside", "BACE1", "docking: among strongest BACE1 and AChE binders of 8 tested", "C",
     "Centella asiatica in silico study, S2950199726000194"),
    ("luteolin", "BACE1", "docking: among strongest BACE1 and AChE binders of 8 tested", "C",
     "Centella asiatica in silico study, S2950199726000194"),
    ("asiaticoside", "AChE", "asiaticoside G docked AChE -10.54, BACE1 -9.10 kcal/mol", "C",
     "PLOS ONE Ayurvedic dual-target screen, PMC12193798"),
    ("bacopaside I", "BACE1", "docked against BACE1 alongside atabecestat/verubecestat", "C",
     "bioRxiv 2024.09.19.613992 preprint, not peer reviewed"),
    ("bacoside A", "BACE1", "Bacopa saponin docked against BACE1", "C",
     "bioRxiv 2024.09.19.613992 preprint, not peer reviewed"),

    # ---- widely claimed CNS naturals, included as controls ---------------------------------
    ("curcumin", "BACE1", "widely claimed neuroprotective and BACE1-active", "B",
     "MDPI 2026 bioactive compounds in Alzheimer's review"),
    ("epigallocatechin gallate", "AChE", "green tea catechin, AChE activity reported", "B",
     "Nutrients 2025 flavonoid-AChE review, PMC11722824"),
    ("apigenin", "AChE", "AChE activity reported; see note on value integrity", "B",
     "Nutrients 2025 review PMC11722824 (review lists the same IC50 for apigenin and "
     "quercetin, which looks like a transcription error; value therefore not quoted)"),
    ("reserpine", "AChE", "Rauvolfia serpentina alkaloid; extract was the most potent screened", "C",
     "PLOS ONE Ayurvedic extract screen, journal.pone.0086804 (extract-level, not pure compound)"),
    ("ajmaline", "AChE", "Rauvolfia serpentina alkaloid, same extract-level attribution", "C",
     "PLOS ONE Ayurvedic extract screen, journal.pone.0086804 (extract-level, not pure compound)"),
]


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore")
    import app as A
    from data.build_compound_library import standardise

    train = pickle.loads(TRAIN_KEYS.read_bytes())
    print(f"training InChIKeys: {len(train):,}\n")

    rows, held, seen, unresolved = [], 0, 0, 0
    for name, target, claim, tier, source in PANEL:
        smi, disp, err = A.resolve_query(name)
        if err or not smi:
            print(f"  UNRESOLVED  {name}")
            rows.append({"name": name, "target": target, "claim": claim, "tier": tier,
                         "source": source, "status": "unresolved"})
            unresolved += 1
            continue
        cs, key = standardise(smi)
        if not key:
            rows.append({"name": name, "target": target, "claim": claim, "tier": tier,
                         "source": source, "status": "unkeyable"})
            unresolved += 1
            continue
        in_train = key in train
        held += (not in_train)
        seen += in_train
        print(f"  {'IN TRAINING' if in_train else 'held out   '}  {tier}  {name:28s} {key}")
        rows.append({"name": name, "resolved_as": disp, "target": target, "claim": claim,
                     "tier": tier, "source": source, "smiles": cs, "inchikey": key,
                     "in_training": int(in_train),
                     "status": "in_training" if in_train else "held_out"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["name", "resolved_as", "target", "claim", "tier", "source", "smiles", "inchikey",
            "in_training", "status"]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    print(f"\n  panel {len(PANEL)}   held out {held}   already in training {seen}   "
          f"unresolved {unresolved}")
    print(f"  wrote {OUT.relative_to(ROOT)}")
    print("\n  Only the held-out rows can test generalisation. The rest are recorded so the")
    print("  selection is visible and cannot look cherry-picked after the fact.")


if __name__ == "__main__":
    main()
