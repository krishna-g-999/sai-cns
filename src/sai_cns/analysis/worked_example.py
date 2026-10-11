"""A real query, taken through the deployed pipeline, so the report can explain an actual output.

The technical report describes what each quantity means in the abstract. A reader whose background is
bench pharmacology rather than statistics asked, reasonably, what the numbers look like on a compound
they know and what they would do next. Inventing a plausible-looking output for that purpose would be
fabrication, so this script runs the deployed server code on real compounds and writes what it
returns.

Three compounds, chosen before the outputs were seen, because each is supposed to exercise a
different part of the pipeline:

  donepezil     an approved acetylcholinesterase inhibitor that penetrates the brain. The panel
                should engage the target it was designed for and pass the exposure gate.
  atenolol      a beta-blocker deliberately optimised NOT to enter the brain. The exposure layer
                should say so, and the disease layer should therefore stay quiet whatever any target
                model reports.
  withanolide A a steroidal natural product, outside the chemistry this library covers. The domain
                distance should be low and the report should say the silence is uninformative.

Nothing is selected after the fact: whatever these three return is what the report prints.

Output: results/tables/worked_example.csv

Run:  python src/sai_cns/analysis/worked_example.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "tables" / "worked_example.csv"

COMPOUNDS = [
    ("donepezil", "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
     "approved AChE inhibitor, brain-penetrant"),
    ("atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
     "beta-blocker optimised against brain entry"),
    ("withanolide A", "CC1C(C)C(=O)OC1C(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3C(O)CC12C",
     "steroidal natural product, outside the covered chemistry"),
]

KEEP = {"BBB", "Kp_uu_brain", "hERG", "AChE", "max_tanimoto", "expected_recall",
        "top_disease", "n_engaged_targets"}


def main() -> None:
    import app as A

    models = A.load_models()
    frames = []
    for name, smi, why in COMPOUNDS:
        r = A.predict_all(smi, models)
        if r is None:
            print(f"  {name}: could not be scored", flush=True)
            continue
        bbb, neuro, dz = A.disease_scores(r)
        df = A.result_frame(smi, name, r)
        df["role_in_the_example"] = why
        top = dz[0] if dz else None
        df.loc[len(df)] = {
            "compound": name, "smiles": smi, "section": "Disease layer",
            "endpoint": "top_disease",
            "description": "highest-ranked condition after exposure gating",
            "value": (round(float(top["gated"]), 4) if top else None), "unit": "score",
            "context": (f"{top['disease']}, driven by {top['driver'][0]}"
                        if top and top.get("driver")
                        else "nothing cleared the reporting threshold"),
            "role_in_the_example": why}
        frames.append(df)
        print(f"  {name:14s} BBB {bbb:.3f}  top condition "
              f"{(top['disease'] if top else 'none'):24s} gated {(top['gated'] if top else 0):.3f}",
              flush=True)

    if not frames:
        print("nothing scored; models unavailable")
        return
    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(out)} rows, {out.compound.nunique()} compounds)")


if __name__ == "__main__":
    main()
