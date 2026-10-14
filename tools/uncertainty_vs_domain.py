"""Does ensemble disagreement predict error better than the applicability domain does?

THE QUESTION. The server's applicability domain is the maximum Tanimoto similarity to measured
training chemistry. Its own reported discrimination is weak: it separates genuinely absent
chemistry from unseen approved drugs at median 0.47 against 0.57, and at a threshold rejecting a
tenth of genuine drugs it catches only a fifth of distant chemistry.

Maximum similarity to a single neighbour is a thin statistic. It uses one point of the training
distribution and nothing about the model fitted on it. A random forest carries a second, free
signal: how much its trees disagree. Each deployed classifier is a CalibratedClassifierCV over
five folds, 300 trees each, so 1,500 trees vote on every query and their spread is available at
no cost.

WHAT IS MEASURED. On ground truth from an independent laboratory (the NCATS AChE screens, which
are not a training source), three candidate signals are compared on their ability to identify
predictions that are wrong:

    distance    1 - max Tanimoto to the reference set        (the current domain flag)
    spread      standard deviation across all 1,500 trees    (ensemble disagreement)
    margin      |p - 0.5|, inverted                          (the model's own confidence)

Error is |y - p|, the absolute deviation of the calibrated probability from the measured
outcome. A signal is useful exactly insofar as it ranks high-error predictions above low-error
ones, so the comparison is Spearman correlation with error, plus the AUROC of each signal
treating "the call was wrong" as the positive class.

WHY THIS MATTERS. If spread beats distance, the server can report a better-founded confidence
without any refitting, without new data, and without touching the models: the trees are already
there. If distance wins, that is worth knowing too, and the current flag is vindicated.

Run:
    python tools/uncertainty_vs_domain.py
"""
from __future__ import annotations

import csv
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))

IN = ROOT / "results" / "tables" / "tox21_independent_set.csv"
OUT = ROOT / "results" / "tables" / "uncertainty_vs_domain.csv"


def tree_spread(model, X):
    """Standard deviation of P(active) across every fitted tree in the calibrated ensemble.

    CalibratedClassifierCV holds one fitted forest per fold, and the unfitted template sits on
    .estimator at the top level, which is why the trees have to be reached through
    calibrated_classifiers_. The spread is taken over the pooled trees of all folds, since a
    disagreement between folds is as informative as one within a fold.
    """
    per_tree = []
    for cc in getattr(model, "calibrated_classifiers_", []):
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est is None or not hasattr(est, "estimators_"):
            continue
        for t in est.estimators_:
            p = t.predict_proba(X)
            per_tree.append(p[:, 1] if p.shape[1] > 1 else p[:, 0])
    if not per_tree:
        return None
    return np.vstack(per_tree).std(axis=0)


def spearman(x, y):
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            a = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[o[k]] = a
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


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
    import app as A

    rows = [r for r in csv.DictReader(IN.open(encoding="utf-8")) if r.get("predicted_probability")]
    by = defaultdict(list)
    for r in rows:
        by[r["aid"]].append(r)
    print(f"independent compounds with a prediction: {len(rows):,}\n")

    models = A.load_models.__wrapped__() if hasattr(A.load_models, "__wrapped__") else A.load_models()
    model = models["AChE"]

    out = []
    for aid, rs in sorted(by.items()):
        feats, keep = [], []
        for r in rs:
            x = A.featurize_one(r["smiles"])
            if x is None:
                continue
            feats.append(x)
            keep.append(r)
        X = np.vstack(feats)
        sd = tree_spread(model, X)
        if sd is None:
            print(f"  AID {aid}: could not reach the trees")
            continue

        y, p, dist, spread, margin = [], [], [], [], []
        for r, s in zip(keep, sd):
            ad = r.get("ad_max_tanimoto")
            if ad in (None, ""):
                continue
            prob = float(r["predicted_probability"])
            y.append(int(r["label"]))
            p.append(prob)
            dist.append(1.0 - float(ad))        # larger = further from known chemistry
            spread.append(float(s))
            margin.append(0.5 - abs(prob - 0.5))  # larger = less decisive
        err = [abs(a - b) for a, b in zip(y, p)]
        # "the call was wrong" at the prevalence-matched operating point, so the binary view does
        # not depend on an arbitrary 0.5 cut on a heavily imbalanced set
        k = max(1, sum(y))
        cut = sorted(p, reverse=True)[k - 1]
        wrong = [int((pp >= cut) != (yy == 1)) for pp, yy in zip(p, y)]

        rec = {"aid": aid, "n": len(y), "n_active": sum(y)}
        print(f"  AID {aid}  n={len(y):,}  actives={sum(y)}  wrong={sum(wrong)}")
        print(f"    {'signal':9s} {'Spearman vs |y-p|':>18s} {'AUROC vs wrong':>16s}")
        for name, sig in (("distance", dist), ("spread", spread), ("margin", margin)):
            rho = spearman(sig, err)
            au = auroc(wrong, sig)
            rec[f"rho_{name}"] = round(rho, 4)
            rec[f"auroc_{name}"] = round(au, 4)
            print(f"    {name:9s} {rho:>18.3f} {au:>16.3f}")
        # the obvious combination, standardised so neither term dominates by scale alone
        def z(v):
            mu = sum(v) / len(v)
            sdv = (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5 or 1.0
            return [(x - mu) / sdv for x in v]
        comb = [a + b for a, b in zip(z(dist), z(spread))]
        rec["rho_combined"] = round(spearman(comb, err), 4)
        rec["auroc_combined"] = round(auroc(wrong, comb), 4)
        print(f"    {'combined':9s} {rec['rho_combined']:>18.3f} {rec['auroc_combined']:>16.3f}")
        print()
        out.append(rec)

    if out:
        with OUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"  wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
