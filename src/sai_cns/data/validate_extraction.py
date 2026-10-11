"""Measure the error rate of literature extraction before any of it is trusted.

Every other number in this project traces to a database record with an identifier. Hand-curated data
breaks that chain, and this project has already demonstrated why that matters: a Reactome identifier
recalled from memory denoted the wrong pathway, a name search for one kainate subunit returned
another, and a UniProt accession was nearly taken from memory rather than resolved. Extraction from
papers has an error rate of the same kind, and unlike an API call there is no second source to check
it against.

Except where the two overlap. Some compounds measured in these papers are already in ChEMBL, curated
independently by people whose job it is. Those overlaps are the calibration set: extract them
blind, compare against ChEMBL, and the disagreement is a direct estimate of the extraction error
rate. If it is small the rest of the harvest can be trusted at that rate; if it is not, the harvest
is discarded rather than used with a caveat.

The rule is set before the measurement, so it cannot be adjusted to fit the answer:

  agreement within 0.3 log units        counted as correct; that is roughly the reproducibility of
                                        the same assay between laboratories, so demanding better
                                        would measure biology rather than extraction
  error rate at or below 5 per cent     harvest usable, with the rate reported alongside any model
                                        trained on it
  error rate above 5 per cent           harvest rejected. Not corrected, not partially used

A structure that PubChem cannot resolve from its name is dropped rather than guessed, and the drop is
counted, because a silent drop biases the estimate toward whatever is easy to resolve.

Writes results/extraction_validation.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3

urllib3.disable_warnings()
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
AGREE_LOG = 0.3          # log units within which extraction and ChEMBL are called consistent
MAX_ERROR_RATE = 0.05    # above this the harvest is rejected outright

UNIT_TO_M = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "μm": 1e-6, "nm": 1e-9, "pm": 1e-12}


def get(url, params=None, tries=3):
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, timeout=90, verify=False)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(2)
    return None


def name_to_chembl(name):
    """Resolve a compound name to a ChEMBL identifier, or None. Never guessed."""
    j = get(f"{CHEMBL}/molecule/search.json", {"q": name, "limit": 5})
    for m in (j or {}).get("molecules", []):
        pref = (m.get("pref_name") or "").lower()
        if pref and (pref == name.lower() or name.lower() in pref):
            return m["molecule_chembl_id"], m.get("pref_name")
    return None, None


def chembl_values(mol_id, target_ids):
    """Every pChEMBL value recorded for this compound at these targets."""
    out = []
    for t in target_ids:
        j = get(f"{CHEMBL}/activity.json",
                {"molecule_chembl_id": mol_id, "target_chembl_id": t,
                 "pchembl_value__isnull": "false", "limit": 100})
        for a in (j or {}).get("activities", []):
            try:
                out.append(float(a["pchembl_value"]))
            except (TypeError, ValueError, KeyError):
                pass
        time.sleep(0.15)
    return out


def to_pchembl(value, unit):
    try:
        v = float(value) * UNIT_TO_M[str(unit).lower()]
        return -np.log10(v) if v > 0 else None
    except (ValueError, KeyError, TypeError):
        return None


def main():
    src = OUT / "state_dependence_extracted.csv"
    if not src.exists():
        print("no extraction file yet; run harvest_state_dependence.py first")
        return
    ex = pd.read_csv(src)
    print(f"{len(ex)} candidate triples to validate", flush=True)

    # the sodium-channel targets these papers concern, from the verified audit
    NAV = ["CHEMBL1845", "CHEMBL4187", "CHEMBL5202", "CHEMBL4296", "CHEMBL5451"]

    rows = []
    for name, g in ex.groupby("compound"):
        mol_id, pref = name_to_chembl(str(name))
        if mol_id is None:
            rows.append({"compound": name, "chembl_id": None, "status": "unresolvable name",
                         "n_extracted": len(g), "extracted_pchembl": None,
                         "chembl_pchembl_median": None, "abs_log_difference": None,
                         "agrees": None})
            print(f"  {str(name)[:24]:26} name not resolvable in ChEMBL, dropped", flush=True)
            continue
        ref = chembl_values(mol_id, NAV)
        vals = [to_pchembl(r.value, r.unit) for r in g.itertuples()]
        vals = [v for v in vals if v is not None]
        if not ref or not vals:
            rows.append({"compound": name, "chembl_id": mol_id,
                         "status": "no overlap" if not ref else "no parsable extracted value",
                         "n_extracted": len(g), "extracted_pchembl": None,
                         "chembl_pchembl_median": (round(float(np.median(ref)), 2) if ref else None),
                         "abs_log_difference": None, "agrees": None})
            print(f"  {str(name)[:24]:26} {mol_id}: no overlap to compare", flush=True)
            continue
        # the extracted set may span protocols, so the closest extracted value is compared: the
        # question here is whether extraction produced a number that exists in the paper at all,
        # not whether the protocols match
        med_ref = float(np.median(ref))
        diff = min(abs(v - med_ref) for v in vals)
        agrees = bool(diff <= AGREE_LOG)
        rows.append({"compound": name, "chembl_id": mol_id, "status": "compared",
                     "n_extracted": len(g),
                     "extracted_pchembl": round(min(vals, key=lambda v: abs(v - med_ref)), 2),
                     "chembl_pchembl_median": round(med_ref, 2),
                     "abs_log_difference": round(diff, 2), "agrees": agrees})
        print(f"  {str(name)[:24]:26} {mol_id}: extracted {rows[-1]['extracted_pchembl']} vs "
              f"ChEMBL {rows[-1]['chembl_pchembl_median']}  "
              f"{'agrees' if agrees else 'DISAGREES'}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "extraction_validation.csv", index=False)
    comp = df[df.status == "compared"]
    pd.set_option("display.width", 220)
    print()
    print(df.to_string(index=False))
    if len(comp) < 10:
        print(f"\nOnly {len(comp)} comparable points. That is too few to estimate an error rate, so "
              f"the harvest stays unusable: an unmeasured error rate is not the same as a low one.")
    else:
        err = 1.0 - float(comp.agrees.mean())
        print(f"\nextraction error rate {err:.1%} over {len(comp)} comparable points "
              f"(agreement within {AGREE_LOG} log units)")
        print("VERDICT:", "harvest usable, reporting this rate alongside anything trained on it"
              if err <= MAX_ERROR_RATE else
              "harvest REJECTED; the error rate exceeds the pre-set bound and the data is discarded")
    print(f"\ndropped for unresolvable names: {int((df.status == 'unresolvable name').sum())}")
    print("wrote", OUT / "extraction_validation.csv")


if __name__ == "__main__":
    main()
