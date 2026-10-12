"""Build a large external test set of MEASURED activities on compounds the model never saw.

Why this exists. The hand-curated literature panel was honest but small: four compounds with a
measured potency is not enough to support any quantitative claim. This builds the same kind of
test systematically and at scale, from measured dose-response data rather than from review
articles, so the result can carry weight.

WHAT THIS IS, STATED PRECISELY.

The panel was trained on ChEMBL_37, and ChEMBL_37 is still the current release. There is
therefore NO prospective temporal hold-out available here, and this script does not pretend
otherwise. What it builds is a *disjoint* hold-out: compound-target pairs with measured
pChEMBL values whose compounds are absent from both the training key index and the
applicability-domain reference set.

THE CONFOUND THIS CREATES, AND WHY IT IS CHECKED.

If training consumed essentially all of ChEMBL for these targets, then whatever remains unseen
is not a random sample: it is the residue that the training pipeline filtered out, which may be
systematically harder, more ambiguous, or differently distributed. A hold-out made of other
people's rejects would make the model look worse than it is, and reporting it without saying so
would be misleading in the opposite direction from the usual.

The script therefore reports, per target: how many measured compounds exist, how many are unseen,
and the publication-year and potency distributions of seen versus unseen. If the unseen set is a
biased residue, that will show, and it is reported either way.

MEMBERSHIP IS CHECKED TWICE, because once was not enough. The earlier literature panel found
coptisine absent from the training key index yet present in the AD reference, and found that the
AD fingerprint is stereo-blind, so a different stereoisomer registers as identical chemistry.
A compound is treated as unseen only if its InChIKey is absent from the training index AND its
stereochemistry-stripped structure is absent from the AD reference.

LABELS use the project's own rule from data/activity_labels.py: active at pChEMBL >= 6.0,
inactive at <= 5.0, and the band between is dropped as genuinely ambiguous rather than forced
to one side.

Run:
    python tools/external_chembl_holdout.py --targets BACE1,AChE --probe
    python tools/external_chembl_holdout.py                       # all targets, full build
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import ssl
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))

API = "https://www.ebi.ac.uk/chembl/api/data"
TRAIN_KEYS = ROOT / "results" / "tables" / "_train_keys.pkl"
AD_REF = ROOT / "models_rf" / "ad_reference.pkl"
OUT = ROOT / "results" / "tables" / "chembl_holdout_set.csv"
REPORT = ROOT / "results" / "tables" / "chembl_holdout_composition.csv"

ACTIVE_CUT, INACTIVE_CUT = 6.0, 5.0

# The same target identifiers the training fetchers used, so this asks about the same proteins.
TARGETS = {
    "AChE":  "CHEMBL220",
    "BChE":  "CHEMBL1914",
    "BACE1": "CHEMBL4822",
    "GSK3B": "CHEMBL262",
    "MAO_B": "CHEMBL2039",
    "MAO_A": "CHEMBL1951",
    "hERG":  "CHEMBL240",
    "D2":    "CHEMBL217",
    "A2A":   "CHEMBL251",
    "HT2A":  "CHEMBL224",
    "SERT":  "CHEMBL228",
}

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def fetch_activities(target_id: str, page_size: int = 1000, cap: int | None = None):
    """All human binding-assay activities carrying a pChEMBL value for one target."""
    got, offset = [], 0
    while True:
        u = (f"{API}/activity.json?target_chembl_id={target_id}"
             f"&pchembl_value__isnull=false&assay_type=B&target_organism=Homo%20sapiens"
             f"&limit={page_size}&offset={offset}")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(u, timeout=180, context=_CTX) as r:
                    d = json.load(r)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(4 * (attempt + 1))
        rows = d.get("activities", [])
        got.extend(rows)
        total = d["page_meta"]["total_count"]
        offset += page_size
        if offset >= total or not rows or (cap and len(got) >= cap):
            return got, total


def collapse(rows):
    """One record per molecule: median pChEMBL, earliest and latest document year.

    Median rather than max, because taking the strongest of several measurements biases every
    compound towards 'active' and would inflate the apparent positive rate.
    """
    by = defaultdict(list)
    for a in rows:
        smi, p = a.get("canonical_smiles"), a.get("pchembl_value")
        if not smi or p is None:
            continue
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        by[smi].append((p, a.get("document_year"), a.get("standard_type")))
    out = {}
    for smi, vals in by.items():
        ps = [v[0] for v in vals]
        yrs = [v[1] for v in vals if v[1]]
        out[smi] = {"pchembl": round(median(ps), 3), "n_meas": len(ps),
                    "year_min": min(yrs) if yrs else None,
                    "year_max": max(yrs) if yrs else None,
                    "types": ",".join(sorted({v[2] for v in vals if v[2]}))}
    return out


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore")
    from rdkit import Chem
    from data.build_compound_library import standardise

    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="")
    ap.add_argument("--probe", action="store_true", help="cap each target, report composition only")
    a = ap.parse_args()
    want = [t.strip() for t in a.targets.split(",") if t.strip()] or list(TARGETS)
    cap = 3000 if a.probe else None

    train = pickle.loads(TRAIN_KEYS.read_bytes())
    ref_smiles = list(pickle.loads(AD_REF.read_bytes())[0])
    print(f"training keys {len(train):,}   AD reference {len(ref_smiles):,}")
    print("flattening the AD reference so a stereoisomer cannot pass as unseen ...")
    ref_flat = set()
    for s in ref_smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            ref_flat.add(Chem.MolToSmiles(m, isomericSmiles=False))
    print(f"  {len(ref_flat):,} distinct stereochemistry-stripped reference structures\n")

    kept, comp = [], []
    for ep in want:
        tid = TARGETS[ep]
        try:
            rows, total = fetch_activities(tid, cap=cap)
        except Exception as e:
            print(f"  {ep}: fetch failed, {type(e).__name__}")
            continue
        mols = collapse(rows)

        seen_p, unseen_p, seen_y, unseen_y, n_unseen = [], [], [], [], 0
        for smi, rec in mols.items():
            cs, key = standardise(smi)
            if not key:
                continue
            m = Chem.MolFromSmiles(cs)
            flat = Chem.MolToSmiles(m, isomericSmiles=False) if m else None
            unseen = (key not in train) and (flat not in ref_flat)
            (unseen_p if unseen else seen_p).append(rec["pchembl"])
            if rec["year_max"]:
                (unseen_y if unseen else seen_y).append(rec["year_max"])
            if not unseen:
                continue
            n_unseen += 1
            p = rec["pchembl"]
            # EVERY unseen compound is kept, with its measured potency. The binary label is
            # recorded where the project's rule gives one and left empty otherwise.
            #
            # Keeping only the labelled ones destroyed this experiment once already. The rule
            # discards 5.0 < pChEMBL < 6.0, so the compounds training never used are very
            # largely that discarded band: filtering the hold-out by the same rule removed
            # about 97% of it and left a residue of compounds with discordant measurements.
            # The continuous potency has no such circularity, and it is what the correlation
            # analysis uses.
            label = 1 if p >= ACTIVE_CUT else (0 if p <= INACTIVE_CUT else None)
            kept.append({"endpoint": ep, "chembl_target": tid, "smiles": cs, "inchikey": key,
                         "pchembl": p, "label": "" if label is None else label,
                         "in_ambiguous_band": int(label is None),
                         "n_measurements": rec["n_meas"],
                         "year_min": rec["year_min"], "year_max": rec["year_max"],
                         "types": rec["types"]})

        def ms(x):
            return f"{median(x):.2f}" if x else "n/a"
        print(f"  {ep:6s} activities {total:>7,}  molecules {len(mols):>6,}  "
              f"unseen {n_unseen:>5,} ({n_unseen/max(1,len(mols)):5.1%})  "
              f"median pChEMBL seen {ms(seen_p)} / unseen {ms(unseen_p)}  "
              f"median year seen {ms(seen_y)} / unseen {ms(unseen_y)}")
        comp.append({"endpoint": ep, "activities": total, "molecules": len(mols),
                     "unseen": n_unseen,
                     "unseen_frac": round(n_unseen / max(1, len(mols)), 4),
                     "median_pchembl_seen": ms(seen_p),
                     "median_pchembl_unseen": ms(unseen_p),
                     "median_year_seen": ms(seen_y), "median_year_unseen": ms(unseen_y)})

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if comp:
        with REPORT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(comp[0]))
            w.writeheader()
            w.writerows(comp)
    if kept and not a.probe:
        with OUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(kept[0]))
            w.writeheader()
            w.writerows(kept)
        pos = sum(r["label"] for r in kept)
        print(f"\n  held-out labelled pairs: {len(kept):,}  ({pos:,} active, {len(kept)-pos:,} inactive)")
        print(f"  wrote {OUT.relative_to(ROOT)}")
    print(f"  wrote {REPORT.relative_to(ROOT)}")
    print("\n  Read the composition line before the results: if the unseen median pChEMBL or")
    print("  year differs markedly from the seen set, the hold-out is a filtered residue and")
    print("  any performance figure on it must be qualified accordingly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
