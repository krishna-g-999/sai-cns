"""Independent external validation against Tox21 / NCATS, a source the panel was not trained on.

WHY THIS IS THE ONLY INDEPENDENT SOURCE FOUND.

The obvious candidate, PubChem BioAssay, turns out not to be independent at all for these
targets. Every assay PubChem annotates to the BACE1, AChE and hERG protein accessions was
deposited by ChEMBL or BindingDB, both of which this project trained on:

    BACE1   1,278 assays   1,098 ChEMBL + 180 BindingDB    0 independent
    AChE    2,496 assays   2,439 ChEMBL +  57 BindingDB    0 independent
    hERG    4,896 assays   4,762 ChEMBL + 134 BindingDB    0 independent

Querying by GENE rather than by protein accession changes the picture, because high-throughput
deposits are annotated to the gene and not always to the accession. For ACHE that surfaces 24
assays from NCGC, NCATS, GNF, Abbott and Harvard ICCB covering 77,666 rows, none of them from a
training source.

This script uses the cleanest of those: a cell-free biochemical screen, so the comparison is
against the same kind of measurement the panel was trained on, without the permeability
confound a cell-based assay would add.

WHAT MAKES THIS DIFFERENT FROM THE ChEMBL HOLD-OUT, AND BETTER.

  independent      a different laboratory, a different assay protocol, a different compound
                   library, and no shared provenance with any training source.
  real labels      the depositor's own Active / Inactive call, made by their criteria, not
                   reconstructed by applying this project's own labelling rule to leftovers.
  no circularity   the ChEMBL hold-out was overwhelmingly the ambiguous band that the training
                   rule discards. Nothing of that kind applies here.

WHAT IS WEAKER ABOUT IT, AND MUST BE SAID.

  qHTS is noisy. Single-concentration-series screens carry well-known false-positive modes:
  compound aggregation, assay interference, fluorescence quenching. A compound this panel
  predicts active and Tox21 calls inactive is not automatically a model error.

  The library differs. Tox21 is environmental chemicals, food additives and drugs, not the
  medicinal-chemistry series that dominate ChEMBL. Many will be far outside the applicability
  domain, which is itself worth measuring and is reported.

  Active rates are low, a few percent, so AUROC is the sensible metric and raw accuracy is not.

Run:
    python tools/external_tox21_independent.py
"""
from __future__ import annotations

import csv
import io
import json
import pickle
import random
import ssl
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))

OUT = ROOT / "results" / "tables" / "tox21_independent_set.csv"
PERF = ROOT / "results" / "tables" / "tox21_independent_performance.csv"
TRAIN_KEYS = ROOT / "results" / "tables" / "_train_keys.pkl"
AD_REF = ROOT / "models_rf" / "ad_reference.pkl"

# Cell-free and enzyme-based only. Cell-based antagonism would confound inhibition with
# permeability and cytotoxicity, which is not what the AChE classifier predicts.
ASSAYS = {
    2284396: ("AChE", "Tox21 cell-free AChE antagonism, colorimetric"),
    1347397: ("AChE", "AChE antagonists, enzyme-based qHTS"),
    2283129: ("AChE", "NovaScreen enzymatic assay, human AChE"),
}

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def fetch(url, timeout=300, binary=False):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=_CTX) as r:
                return r.read() if binary else r.read().decode("utf-8", "replace")
        except Exception:
            if attempt == 3:
                raise
            time.sleep(4 * (attempt + 1))


def assay_table(aid: int):
    txt = fetch(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV")
    return list(csv.DictReader(io.StringIO(txt)))


def cids_to_smiles(cids):
    """Canonical SMILES for a list of CIDs, in batches PubChem will accept."""
    out = {}
    cids = [c for c in cids if c]
    for i in range(0, len(cids), 200):
        chunk = cids[i:i + 200]
        # IsomericSMILES to keep stereochemistry: the applicability domain is stereo-blind, so
        # discarding stereo here would make the unseen check weaker than it already is.
        #
        # The column name is read from the header rather than assumed. Asking for
        # CanonicalSMILES returns a column called ConnectivitySMILES, and looking it up by the
        # requested name silently yields nothing, which is how this failed the first time.
        txt = fetch("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                    + ",".join(map(str, chunk)) + "/property/IsomericSMILES/CSV")
        rdr = csv.DictReader(io.StringIO(txt))
        col = next((c for c in (rdr.fieldnames or []) if "SMILES" in c.upper()), None)
        if not col:
            continue
        for r in rdr:
            s = r.get(col)
            if s:
                out[r["CID"]] = s
        time.sleep(0.2)
    return out


def auroc(y, p):
    pairs = sorted(zip(p, y))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    pos = sum(1 for _, yy in pairs if yy == 1)
    neg = len(pairs) - pos
    if not pos or not neg:
        return float("nan")
    s = sum(r for r, (_, yy) in zip(ranks, pairs) if yy == 1)
    return (s - pos * (pos + 1) / 2.0) / (pos * neg)


def boot_ci(y, p, n=2000, seed=0):
    rng = random.Random(seed)
    m, vals = len(y), []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        yy = [y[i] for i in idx]
        if 0 < sum(yy) < len(yy):
            vals.append(auroc(yy, [p[i] for i in idx]))
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))])


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore")
    from rdkit import Chem
    import app as A
    from data.build_compound_library import standardise

    train = pickle.loads(TRAIN_KEYS.read_bytes())
    ref = list(pickle.loads(AD_REF.read_bytes())[0])
    print(f"training keys {len(train):,}   AD reference {len(ref):,}")
    ref_flat = set()
    for s in ref:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            ref_flat.add(Chem.MolToSmiles(m, isomericSmiles=False))
    print(f"flattened reference skeletons {len(ref_flat):,}\n")

    models = A.load_models.__wrapped__() if hasattr(A.load_models, "__wrapped__") else A.load_models()

    allrows, perf = [], []
    for aid, (ep, label) in ASSAYS.items():
        try:
            tbl = assay_table(aid)
        except Exception as e:
            print(f"  AID {aid}: fetch failed, {type(e).__name__}")
            continue
        # The depositor's own call. Only unambiguous calls are used; Inconclusive and
        # Unspecified are dropped rather than coerced.
        keep = [r for r in tbl if (r.get("PUBCHEM_ACTIVITY_OUTCOME") or "") in ("Active", "Inactive")
                and r.get("PUBCHEM_CID")]
        cids = sorted({r["PUBCHEM_CID"] for r in keep})
        print(f"  AID {aid} {label}")
        print(f"    rows {len(tbl):,}  with a definite call {len(keep):,}  unique CIDs {len(cids):,}")
        smi = cids_to_smiles(cids)
        print(f"    structures resolved {len(smi):,}")

        unseen = []
        for r in keep:
            s = smi.get(r["PUBCHEM_CID"])
            if not s:
                continue
            cs, key = standardise(s)
            if not key:
                continue
            m = Chem.MolFromSmiles(cs)
            flat = Chem.MolToSmiles(m, isomericSmiles=False) if m else None
            if key in train or flat in ref_flat:
                continue
            unseen.append({"aid": aid, "endpoint": ep, "assay": label,
                           "cid": r["PUBCHEM_CID"], "smiles": cs, "inchikey": key,
                           "outcome": r["PUBCHEM_ACTIVITY_OUTCOME"],
                           "label": 1 if r["PUBCHEM_ACTIVITY_OUTCOME"] == "Active" else 0,
                           "ac50_uM": r.get("PUBCHEM_ACTIVITY_SCORE") or ""})
        # de-duplicate within the assay
        seen_keys, ded = set(), []
        for r in unseen:
            if r["inchikey"] in seen_keys:
                continue
            seen_keys.add(r["inchikey"])
            ded.append(r)
        npos = sum(r["label"] for r in ded)
        print(f"    UNSEEN by this panel: {len(ded):,}  ({npos:,} active, {len(ded)-npos:,} inactive)")

        if len(ded) < 50 or not (0 < npos < len(ded)):
            print("    too few, or single class; not scored\n")
            allrows.extend(ded)
            continue

        res = A.predict_all_batch([r["smiles"] for r in ded], models)
        y, p, ads = [], [], []
        for r, d in zip(ded, res):
            if d is None:
                continue
            prob = (d.get("targets") or {}).get(ep)
            if prob is None:
                continue
            y.append(r["label"])
            p.append(float(prob))
            r["predicted_probability"] = round(float(prob), 6)
            # The applicability domain is NOT part of predict_all_batch: it is a separate
            # similarity search against the reference set. Reading it off the prediction dict
            # produced no AD at all on the first run, silently.
            try:
                ad = (A.assess_domain(r["smiles"]) or {}).get("max_sim")
            except Exception:
                ad = None
            r["ad_max_tanimoto"] = None if ad is None else round(float(ad), 4)
            if ad is not None:
                ads.append(float(ad))
        allrows.extend(ded)
        a = auroc(y, p)
        lo, hi = boot_ci(y, p)
        med_ad = sorted(ads)[len(ads) // 2] if ads else None
        frac_far = sum(1 for x in ads if x < 0.40) / len(ads) if ads else None
        print(f"    AUROC {a:.4f}  95% CI [{lo:.3f}, {hi:.3f}]   "
              f"{'no AD computed' if med_ad is None else f'median AD {med_ad:.3f}   below AD 0.40: {frac_far:.1%}'}\n")
        perf.append({"aid": aid, "assay": label, "endpoint": ep, "n": len(y),
                     "n_active": sum(y), "active_rate": round(sum(y) / len(y), 4),
                     "auroc": round(a, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "median_ad": med_ad,
                     "frac_below_ad_0.40": None if frac_far is None else round(frac_far, 4)})

    if allrows:
        cols = ["aid", "endpoint", "assay", "cid", "smiles", "inchikey", "outcome", "label",
                "ac50_uM", "predicted_probability", "ad_max_tanimoto"]
        with OUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in allrows:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"  wrote {OUT.relative_to(ROOT)} ({len(allrows):,} rows)")
    if perf:
        with PERF.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(perf[0]))
            w.writeheader()
            w.writerows(perf)
        print(f"  wrote {PERF.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
