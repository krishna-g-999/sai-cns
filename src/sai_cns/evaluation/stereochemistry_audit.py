"""What does ignoring chirality actually cost this panel?

The featuriser excludes stereochemistry, so two enantiomers produce byte-identical rows and receive
identical predictions. For a CNS tool this is the sharpest criticism available: escitalopram and
its distomer, dexmethylphenidate and its, levodopa and its inactive antipode. The limitation is
real and is disclosed, but "we ignore chirality" is not an answer. The answer is how much of this
panel's data could distinguish stereoisomers at all, and how often it says they differ.

Three questions, in the order that decides whether a chirality-aware fingerprint would help:

  1. How much training chemistry even carries an assigned stereocentre? A chirality-aware
     fingerprint can only separate what the labels distinguish.
  2. How often does one flat skeleton appear as more than one stereoisomer at the same endpoint?
     Without such pairs there is nothing for chirality to resolve.
  3. When such a pair exists, do its members actually disagree? If the measured labels agree, adding
     stereochemistry adds parameters and sparsity for no signal.

The result decides the recommendation rather than the other way round.

Output: results/tables/stereochemistry_audit.csv
        results/tables/stereochemistry_pairs.csv

Run:  python src/sai_cns/evaluation/stereochemistry_audit.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import parent_mol  # noqa: E402

TAB = ROOT / "results" / "tables"


def main() -> None:
    n_total = n_stereo = 0
    pair_rows, same_lab, diff_lab, gaps = [], 0, 0, []

    for f in sorted(glob.glob(str(ROOT / "data" / "endpoints" / "*.csv"))):
        ep = Path(f).stem
        d = pd.read_csv(f)
        if "label" not in d.columns:
            continue
        has_p = "pchembl" in d.columns
        groups: dict[str, list] = {}
        for i in range(len(d)):
            m = parent_mol(str(d["smiles"].iloc[i]))
            if m is None:
                continue
            n_total += 1
            iso = Chem.MolToSmiles(m)
            flat = Chem.MolToSmiles(m, isomericSmiles=False)
            if iso == flat:                       # no stereocentre survives standardisation
                continue
            n_stereo += 1
            pv = (float(d["pchembl"].iloc[i])
                  if has_p and pd.notna(d["pchembl"].iloc[i]) else np.nan)
            groups.setdefault(flat, []).append((iso, int(d["label"].iloc[i]), pv))

        for flat, rows in groups.items():
            if len({r[0] for r in rows}) < 2:     # one stereoisomer only: nothing to resolve
                continue
            labels = {r[1] for r in rows}
            ps = [r[2] for r in rows if not np.isnan(r[2])]
            gap = (max(ps) - min(ps)) if len(ps) > 1 else np.nan
            if len(labels) > 1:
                diff_lab += 1
            else:
                same_lab += 1
            if len(ps) > 1:
                gaps.append(gap)
            pair_rows.append({"endpoint": ep, "skeleton": flat[:70],
                              "n_stereoisomers": len({r[0] for r in rows}),
                              "labels_disagree": len(labels) > 1,
                              "potency_gap_log_units": None if np.isnan(gap) else round(gap, 2)})

    pairs = pd.DataFrame(pair_rows)
    n_pairs = same_lab + diff_lab
    g = np.array([x for x in gaps if not np.isnan(x)])

    summary = pd.DataFrame([
        {"question": "training structures parsed", "value": n_total},
        {"question": "carrying an assigned stereocentre", "value": n_stereo},
        {"question": "percent carrying a stereocentre", "value": round(100 * n_stereo / max(n_total, 1), 1)},
        {"question": "skeletons present as 2+ stereoisomers at one endpoint", "value": n_pairs},
        {"question": "those pairs whose labels AGREE", "value": same_lab},
        {"question": "those pairs whose labels DISAGREE", "value": diff_lab},
        {"question": "percent of pairs that disagree", "value": round(100 * diff_lab / max(n_pairs, 1), 1)},
        {"question": "pairs with a measurable potency gap", "value": int(len(g))},
        {"question": "median potency gap, log units", "value": round(float(np.median(g)), 2) if len(g) else None},
        {"question": "pairs differing by more than 1 log unit", "value": int((g > 1).sum()) if len(g) else 0},
        {"question": "percent of pairs differing by more than 1 log unit",
         "value": round(100 * float((g > 1).mean()), 1) if len(g) else 0.0},
        {"question": "percent of the whole panel where chirality could change a call",
         "value": round(100 * diff_lab / max(n_total, 1), 2)},
    ])

    TAB.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TAB / "stereochemistry_audit.csv", index=False)
    pairs.to_csv(TAB / "stereochemistry_pairs.csv", index=False)

    print(summary.to_string(index=False))
    if len(pairs):
        top = pairs[pairs.labels_disagree].endpoint.value_counts().head(6)
        print("\nendpoints where stereoisomers most often disagree on class:")
        for k, v in top.items():
            print(f"  {k:12s} {v}")
    print(f"\nwrote {(TAB / 'stereochemistry_audit.csv').relative_to(ROOT)} and "
          f"{(TAB / 'stereochemistry_pairs.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
