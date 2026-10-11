"""Fetch the fifth expansion: migraine, multiple sclerosis, and a genuinely ALS-linked mechanism.

The coverage audit (results/unmodelled_target_data.csv) found that two conditions listed as
unmodelled are not unmodelled for want of data but for want of an endpoint: migraine has a defined
target in the calcitonin-gene-related-peptide receptor and multiple sclerosis has one in
dihydroorotate dehydrogenase, both comfortably above the volume and source-diversity bars.

Amyotrophic lateral sclerosis is already scored, through seven targets, but all seven are general
mechanisms shared with other conditions: neuroinflammation, oxidative stress, excitotoxicity and
proteostasis. None is specific to the disease. Receptor-interacting serine/threonine-protein kinase 1
is added to close that gap. It sits on the necroptosis pathway, is the most heavily measured
candidate found (5,291 activities across 55 sources), and its inhibitors have been taken into
clinical trials for both amyotrophic lateral sclerosis and multiple sclerosis, so it also
strengthens the latter.

Two candidates were checked and rejected on data rather than judgement: SARM1, the axon-degeneration
NAD hydrolase, has 85 activities from 3 sources, and KCNQ2 has 103 from 12. A third, S1P1, could not
be resolved reliably: a name search for the Edg-1 receptor returned sphingosine 1-phosphate receptor
4, a different protein, which is the same class of resolution error that this project has been bitten
by before, so it is left out rather than guessed at.

Identifiers are re-verified against the live ChEMBL target record before any activity is downloaded
and a mismatch aborts the fetch.

Writes data/endpoints/<TARGET>.csv in the existing schema.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_compound_library import add_parent_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from data.fetch_batch4 import OUT, fetch, verify  # noqa: E402

import pandas as pd  # noqa: E402

TARGETS = {
    "CGRP":  ("CHEMBL3798", "Calcitonin gene-related peptide type 1 receptor",
              "the migraine mechanism; the gepant antagonists and the antibody class act here"),
    "DHODH": ("CHEMBL1966", "Dihydroorotate dehydrogenase (quinone), mitochondrial",
              "pyrimidine synthesis in proliferating lymphocytes; the teriflunomide mechanism in "
              "multiple sclerosis"),
    "RIPK1": ("CHEMBL5464", "Receptor-interacting serine/threonine-protein kinase 1",
              "necroptosis and neuroinflammation; inhibitors are in trials for amyotrophic lateral "
              "sclerosis and multiple sclerosis"),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, (cid, pref, why) in TARGETS.items():
        dest = OUT / f"{name}.csv"
        if dest.exists():
            print(f"[{name}] exists, skipping", flush=True)
            continue
        print(f"[{name}] verifying {cid} ...", flush=True)
        try:
            verify(cid, pref)
        except RuntimeError as e:
            print(f"[{name}] ABORTED: {e}", flush=True)
            continue
        print(f"[{name}] {pref}: {why}", flush=True)
        df = fetch(cid)
        if df.empty:
            print(f"[{name}] no data", flush=True)
            continue
        med = add_parent_key(df).groupby("inchikey").agg(
            smiles=("smiles", "first"), pchembl=("pchembl", "median"),
            year=("year", "max")).reset_index(drop=True)
        med["label"] = med["pchembl"].apply(lambda p: 1 if p >= 6 else (0 if p < 5 else None))
        med = med.dropna(subset=["label"])
        med["label"] = med["label"].astype(int)
        med["source"] = "ChEMBL"
        med[["smiles", "label", "pchembl", "year", "source"]].to_csv(dest, index=False)
        summary.append({"target": name, "chembl_id": cid, "compounds": len(med),
                        "actives_p6": int(med.label.sum()),
                        "binders_p7": int((med.pchembl >= 7).sum()),
                        "measured_inactives": int((med.label == 0).sum()),
                        "distinct_documents": int(df["doc"].nunique())})
        print(f"[{name}] wrote {len(med)} compounds: {int(med.label.sum())} active at >=6, "
              f"{int((med.pchembl >= 7).sum())} binders at >=7, "
              f"{int((med.label == 0).sum())} measured inactives, "
              f"{df['doc'].nunique()} documents", flush=True)
    if summary:
        s = pd.DataFrame(summary)
        s.to_csv(ROOT / "results" / "batch5_fetch_summary.csv", index=False)
        pd.set_option("display.width", 200)
        print()
        print(s.to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    main()
