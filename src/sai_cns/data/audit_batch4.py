"""Audit the six new endpoints before any of them is trained.

The fetch summary answers "is there data". It does not answer the questions that decide whether a
model built on that data will mean anything, and those are the ones that have caught real errors in
this project before:

  scaffold count            a scaffold-grouped split needs many distinct scaffolds. Six hundred
                            compounds drawn from forty analogue series is forty independent facts,
                            and a model trained on it reports a validation number it has not earned
  measured inactives        the deployed training scheme splits measured inactives in half, one half
                            joining the negatives and the other reserved for threshold setting and
                            unbiased validation. An endpoint with a handful of them cannot have a
                            threshold set honestly, whatever its active count
  overlap with the panel    sodium-channel and nicotinic ligands are routinely cross-reactive. If a
                            new endpoint's actives are largely the same molecules as an existing
                            endpoint's, it adds correlated false positives rather than information,
                            and the multiple-comparison cost is paid for nothing
  contamination of controls if these compounds sit in the non-CNS set used to measure the background
                            false-positive rate, then the specificity that sets every threshold is
                            measured on training data and is not a specificity at all

A target that fails any of these is reported as unfit here rather than discovered to be useless after
it has been wired into the interface.

Read-only. Writes results/batch4_audit.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
EP = ROOT / "data" / "endpoints"
OUT = ROOT / "results"
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

NEW = ["a4b2nAChR", "a3b4nAChR", "Nav1_6", "Nav1_8", "Cav3_2", "GABAA_a5",
       "CGRP", "DHODH", "RIPK1"]
MIN_INACTIVES = 30      # below this a held-out half cannot set a threshold with any precision
MIN_SCAFFOLDS = 100     # below this a scaffold-grouped split has too few groups to be meaningful
MAX_OVERLAP = 0.60      # above this the endpoint is largely a restatement of an existing one


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def scaffold(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return None


def load(name):
    p = EP / f"{name}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["canon"] = [canon(s) for s in d.smiles]
    return d.dropna(subset=["canon"])


def internal_redundancy(smis, cap=400):
    """Median nearest-neighbour similarity among actives. A set of near-duplicates from one series
    scores high, and its apparent size overstates the information it carries."""
    smis = list(smis)[:cap]
    fps = [GEN.GetFingerprint(m) for m in (Chem.MolFromSmiles(s) for s in smis) if m]
    if len(fps) < 10:
        return None
    best = []
    for i, f in enumerate(fps):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(f, fps))
        sims[i] = -1
        best.append(float(sims.max()))
    return float(np.median(best))


def main():
    existing = [p.stem for p in EP.glob("*.csv") if p.stem not in NEW]
    print(f"comparing against {len(existing)} existing endpoints", flush=True)
    ex_actives = {}
    for e in existing:
        d = load(e)
        if d is not None and len(d):
            ex_actives[e] = set(d[d.label == 1].canon)

    # the non-CNS control set whose false-positive rate sets every deployed threshold
    control = set()
    for cand in [OUT / "noncns_specificity_fast.csv", OUT / "noncns_specificity.csv",
                 ROOT / "inversion" / "results" / "H4_distant_predictions.csv"]:
        if cand.exists():
            c = pd.read_csv(cand)
            col = next((x for x in c.columns if x.lower() in ("smiles", "canonical_smiles")), None)
            if col:
                control |= {canon(s) for s in c[col].dropna()}
    control.discard(None)
    print(f"control compounds available for contamination check: {len(control):,}", flush=True)

    rows = []
    for name in NEW:
        d = load(name)
        if d is None:
            print(f"[{name}] no file", flush=True)
            continue
        act = d[d.label == 1]
        inact = d[d.label == 0]
        binders = d[d.pchembl >= 7]
        scafs = {scaffold(s) for s in act.canon}
        scafs.discard(None)
        red = internal_redundancy(act.canon)

        # worst overlap with any single existing endpoint, measured on actives
        ov = {e: len(set(act.canon) & s) / max(1, len(act)) for e, s in ex_actives.items() if s}
        worst_e, worst_v = (max(ov.items(), key=lambda kv: kv[1]) if ov else ("", 0.0))
        contam = len(set(d.canon) & control)

        fails = []
        if len(inact) < MIN_INACTIVES:
            fails.append(f"only {len(inact)} measured inactives, cannot set a threshold honestly")
        if len(scafs) < MIN_SCAFFOLDS:
            fails.append(f"only {len(scafs)} distinct scaffolds among actives")
        if worst_v > MAX_OVERLAP:
            fails.append(f"{worst_v:.0%} of actives already in {worst_e}")
        if contam:
            fails.append(f"{contam} compounds also in the specificity control set")

        rows.append({"target": name, "compounds": len(d), "actives": len(act),
                     "binders_p7": len(binders), "measured_inactives": len(inact),
                     "scaffolds_in_actives": len(scafs),
                     "compounds_per_scaffold": round(len(act) / max(1, len(scafs)), 2),
                     "median_nn_similarity": round(red, 3) if red else None,
                     "worst_overlap_endpoint": worst_e,
                     "worst_overlap_fraction": round(worst_v, 3),
                     "in_control_set": contam,
                     "verdict": "FIT" if not fails else "UNFIT",
                     "problems": "; ".join(fails)})
        print(f"  {name:11} {len(act):4} actives  {len(scafs):4} scaffolds  "
              f"{len(inact):4} inactives  overlap {worst_v:.0%} with {worst_e or 'none'}  "
              f"{'FIT' if not fails else 'UNFIT'}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "batch4_audit.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print()
    print(df.drop(columns=["problems"]).to_string(index=False))
    bad = df[df.verdict == "UNFIT"]
    if len(bad):
        print(f"\n{len(bad)} of {len(df)} endpoints are unfit to train as they stand:")
        for _, r in bad.iterrows():
            print(f"   {r.target:11} {r.problems}")
    else:
        print("\nAll endpoints pass. None is a restatement of an existing one, none contaminates "
              "the specificity controls, and each has enough scaffolds and measured inactives.")
    print("\nwrote", OUT / "batch4_audit.csv")


if __name__ == "__main__":
    main()
