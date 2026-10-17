"""Can agonist be told from antagonist at all? A scaffold-split prototype, not a deployed model.

The manuscript records as a limitation that the server predicts engagement, not modulation: an
agonist and an antagonist at the same receptor can share a potency, so a model trained on potency
cannot separate them. That is true of the panel as built. It is a different claim from "the
distinction cannot be learned", and the second claim should be tested rather than assumed.

A census of ChEMBL action_type labels found eight receptors with at least fifty compounds in the
smaller class, two of them close to balanced: D2 at 289 agonists against 339 antagonists, and HT2A
at 378 against 367. That is enough to ask the question.

EVALUATION. Bemis-Murcko scaffold split, not random. This project has already shown why: a random
split of medicinal-chemistry data holds out 83 per cent close analogues of its own training set,
because the published record is organised as series, and a random-split score on such data is
mostly a measure of how well a model recognises the series it was trained on. Compounds carrying
conflicting labels at the same target are dropped rather than assigned, since forcing them into a
class would both inflate the training set and remove the hardest cases from the test.

WHAT A POSITIVE RESULT WOULD AND WOULD NOT MEAN. A scaffold AUROC meaningfully above 0.5 would
show the distinction is learnable from 2D structure for these receptors, which is the question the
limitation turns on. It would not produce a deployable endpoint: a few hundred compounds per class
at two of fifty-four targets is a demonstration, not a panel, and the modality label itself is
assay-dependent and noisy.

Run:
    python tools/modality_prototype.py
"""
from __future__ import annotations

import collections
import csv
import json
import ssl
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))

OUT = ROOT / "results" / "tables" / "modality_prototype.csv"
API = "https://www.ebi.ac.uk/chembl/api/data"
TARGETS = {"D2": "CHEMBL217", "HT2A": "CHEMBL224", "CB1": "CHEMBL218", "HT1A": "CHEMBL214"}
AGONIST = {"AGONIST", "PARTIAL AGONIST"}
ANTAGONIST = {"ANTAGONIST", "INVERSE AGONIST"}

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def get(url, timeout=180):
    for a in range(4):
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=_CTX) as r:
                return json.load(r)
        except Exception:
            if a == 3:
                raise
            time.sleep(3 * (a + 1))


def action_of(a):
    v = a.get("action_type")
    if isinstance(v, dict):
        return (v.get("action_type") or v.get("parent_type") or "").upper()
    return (v or "").upper()


def labelled(tid):
    """{smiles: label} for one target, 1 = agonist, 0 = antagonist, conflicts dropped."""
    cls = collections.defaultdict(set)
    smi = {}
    offset, page = 0, 1000
    while True:
        d = get(f"{API}/activity.json?target_chembl_id={tid}"
                f"&action_type__isnull=false&limit={page}&offset={offset}")
        acts = d.get("activities", [])
        for a in acts:
            at, m, s = action_of(a), a.get("molecule_chembl_id"), a.get("canonical_smiles")
            if not m or not s:
                continue
            if at in AGONIST:
                cls[m].add(1)
            elif at in ANTAGONIST:
                cls[m].add(0)
            else:
                continue
            smi[m] = s
        offset += page
        if offset >= d["page_meta"]["total_count"] or not acts:
            break
    return {smi[m]: next(iter(v)) for m, v in cls.items() if len(v) == 1 and m in smi}


def auroc(y, p):
    pr = sorted(zip(p, y))
    rk, i = [0.0] * len(pr), 0
    while i < len(pr):
        j = i
        while j + 1 < len(pr) and pr[j + 1][0] == pr[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rk[k] = r
        i = j + 1
    pos = sum(1 for _, v in pr if v == 1)
    neg = len(pr) - pos
    if not pos or not neg:
        return float("nan")
    s = sum(r for r, (_, v) in zip(rk, pr) if v == 1)
    return (s - pos * (pos + 1) / 2.0) / (pos * neg)


def main() -> int:
    warnings.filterwarnings("ignore")
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from sklearn.ensemble import RandomForestClassifier
    RDLogger.DisableLog("rdApp.*")
    import app as A

    rows = []
    print(f"  {'target':7s} {'n':>5s} {'ago':>5s} {'ant':>5s} {'scaffolds':>10s} "
          f"{'test n':>7s} {'scaffold AUROC':>15s}")
    print("  " + "-" * 64)
    for ep, tid in TARGETS.items():
        try:
            data = labelled(tid)
        except Exception as e:
            print(f"  {ep:7s} fetch failed: {type(e).__name__}")
            continue

        X, y, scaf = [], [], []
        for s, lab in data.items():
            f = A.featurize_one(s)
            if f is None:
                continue
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            try:
                sk = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
            except Exception:
                continue
            X.append(f)
            y.append(lab)
            scaf.append(sk or "")
        if len(y) < 80 or not (0 < sum(y) < len(y)):
            print(f"  {ep:7s} {len(y):>5} too few after featurisation")
            continue
        X = np.vstack(X)
        y = np.asarray(y)

        # scaffold split: whole scaffolds go to one side, largest first, until the test
        # holds about 30 per cent of compounds
        groups = collections.defaultdict(list)
        for i, s in enumerate(scaf):
            groups[s].append(i)
        order = sorted(groups.values(), key=len, reverse=True)
        test_idx, target_n = [], int(0.3 * len(y))
        for g in order[::-1]:          # smallest scaffolds into test first, as is conventional
            if len(test_idx) >= target_n:
                break
            test_idx.extend(g)
        test = np.zeros(len(y), bool)
        test[test_idx] = True
        tr, te = ~test, test
        if not (0 < y[te].sum() < te.sum()) or not (0 < y[tr].sum() < tr.sum()):
            print(f"  {ep:7s} split left a single class; skipped")
            continue

        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                     class_weight="balanced", n_jobs=-1, random_state=0)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        a = auroc(y[te].tolist(), p.tolist())
        print(f"  {ep:7s} {len(y):>5,} {int(y.sum()):>5,} {int((1-y).sum()):>5,} "
              f"{len(groups):>10,} {int(te.sum()):>7,} {a:>15.3f}")
        rows.append({"endpoint": ep, "n": len(y), "n_agonist": int(y.sum()),
                     "n_antagonist": int((1 - y).sum()), "n_scaffolds": len(groups),
                     "n_test": int(te.sum()), "scaffold_auroc": round(a, 4)})

    if rows:
        with OUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  wrote {OUT.relative_to(ROOT)}")
        print("  A scaffold AUROC near 0.5 would mean the distinction is not learnable from 2D")
        print("  structure at these sample sizes. Well above 0.5 means it is, and the limitation")
        print("  is a gap in the panel rather than a property of the problem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
