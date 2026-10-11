"""Gather the clinical CNS precedent reference: ATC class N molecules that reached a clinical phase.

Written as BS_fetch_clinical.py, archived in commit fea5029 and restored here, because
data/clinical_cns_reference.csv is a live input (build_compound_library.py, build_endpoint_context.py)
and had no reproducible origin once the script was removed.

What this is, and what it is not. Every row is a molecule that a regulator or sponsor advanced to a
declared clinical phase, with its ATC codes mapped to the disease area the panel scores. It records
clinical precedent, which is measured. It is not an efficacy label and no model is trained on it; it
exists so that a predicted profile can be set beside what has actually been taken into humans for
that indication. Target engagement is not efficacy, and this file is the reminder of the gap rather
than a bridge across it.

Outputs:
  data/clinical_cns_reference.csv    name,smiles,max_phase,atc,disease

Run:  python src/sai_cns/data/fetch_clinical.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "clinical_cns_reference.csv"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

# ATC subcode -> the disease class used by the panel. The prefixes are mutually disjoint, so a code
# matches at most one of them and the order of this mapping does not affect the result; where a
# molecule carries several ATC codes, the first code decides.
ATC_DISEASE = {
    "N06D": "Alzheimer's / dementia",
    "N06A": "Depression",
    "N06B": "Cognition/ADHD",
    "N05A": "Psychosis",
    "N05B": "Anxiety",
    "N04": "Parkinson's",
    "N03": "Epilepsy",
    "N07": "Other CNS",
}
# The shipped table holds several hundred molecules. Far fewer means the query failed part way.
MIN_ROWS = 100


def disease_for(codes: list[str]) -> str:
    for code in codes:
        for prefix, disease in ATC_DISEASE.items():
            if code.startswith(prefix):
                return disease
    return "CNS (other)"


def main() -> None:
    rows, offset, failed = [], 0, 0
    while True:
        url = (f"{CHEMBL}/molecule.json?atc_classifications__level1=N&max_phase__gte=1"
               f"&limit=1000&offset={offset}")
        try:
            j = requests.get(url, timeout=60).json()
        except Exception as exc:
            # Counted and reported: a dropped page means a short table, and that must not look
            # like a smaller clinical landscape.
            failed += 1
            print(f"  page at offset {offset} failed: {exc}")
            break
        for m in j.get("molecules", []):
            smi = (m.get("molecule_structures") or {}).get("canonical_smiles")
            if not smi:
                continue
            atc = [a for a in (m.get("atc_classifications") or []) if isinstance(a, str)]
            rows.append({"name": m.get("pref_name"), "smiles": smi,
                         "max_phase": m.get("max_phase"), "atc": ";".join(atc),
                         "disease": disease_for(atc)})
        if not j.get("page_meta", {}).get("next"):
            break
        offset += 1000
        time.sleep(0.2)

    df = pd.DataFrame(rows).dropna(subset=["smiles"]).drop_duplicates("smiles")

    if failed or len(df) < MIN_ROWS:
        raise SystemExit(
            f"the query returned {len(df)} molecules"
            + (f" after {failed} failed page(s)" if failed else "")
            + f", below the {MIN_ROWS}-molecule floor or incomplete. "
            f"{OUT.relative_to(ROOT).as_posix()} was not written."
        )
    if OUT.exists():
        n_old = len(pd.read_csv(OUT))
        if n_old and len(df) < 0.5 * n_old:
            raise SystemExit(
                f"the query returned {len(df)} molecules against {n_old} already on disk. Refusing "
                "to overwrite with less than half the data."
            )

    df.to_csv(OUT, index=False)
    print(f"clinical CNS reference: {len(df)} molecules with phase and structure "
          f"-> {OUT.relative_to(ROOT).as_posix()}")
    print(df["disease"].value_counts().to_string())
    print("phase 4 (approved):", int((pd.to_numeric(df.max_phase, errors="coerce") == 4).sum()))


if __name__ == "__main__":
    main()
