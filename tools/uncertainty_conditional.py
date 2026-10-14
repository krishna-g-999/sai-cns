"""Does any uncertainty signal add information about error ONCE the prediction is controlled for?

WHY THE NAIVE VERSION IS WRONG. Correlating a candidate signal with |y - p| looks like the
obvious test and is not. These sets are 2 to 6 per cent active, so for the overwhelming majority
y = 0 and |y - p| is exactly p. Any signal that is a monotone function of p therefore correlates
with "error" almost perfectly while carrying no information at all. Tree spread is near zero when
p is near zero and rises towards p = 0.5; the margin 0.5 - |p - 0.5| simply equals p below 0.5.
Run naively, spread scores Spearman 0.95 and margin 0.98, and both numbers are restatements of p.

THE CORRECT TEST. Hold the prediction fixed and ask whether the signal still separates right from
wrong. Two ways, reported together because they fail differently:

  stratified      bin compounds by predicted probability, then inside each bin compute the AUROC
                  of the signal against "this call was wrong". Within a bin p is nearly constant,
                  so any discrimination left is genuine. Bins are pooled by sample size.

  logistic        fit wrong ~ logit(p) + signal and report the signal's coefficient with a
                  likelihood-ratio test against the model with p alone. This asks directly
                  whether the signal earns its place beside the prediction.

Both are run for the applicability-domain distance and for ensemble spread, so the comparison is
like for like.

Run:
    python tools/uncertainty_conditional.py
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
OUT = ROOT / "results" / "tables" / "uncertainty_conditional.csv"


def auroc(y, s):
    pr = sorted(zip(s, y))
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
        return None
    t = sum(r for r, (_, v) in zip(rk, pr) if v == 1)
    return (t - pos * (pos + 1) / 2.0) / (pos * neg)


def tree_spread(model, X):
    per = []
    for cc in getattr(model, "calibrated_classifiers_", []):
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est is None or not hasattr(est, "estimators_"):
            continue
        for t in est.estimators_:
            pp = t.predict_proba(X)
            per.append(pp[:, 1] if pp.shape[1] > 1 else pp[:, 0])
    return None if not per else np.vstack(per).std(axis=0)


def logistic_lrt(y, base, extra):
    """Fit wrong ~ base (+ extra) by IRLS and return the likelihood-ratio p-value for extra."""
    def fit(cols):
        X = np.column_stack([np.ones(len(y))] + cols)
        b = np.zeros(X.shape[1])
        yv = np.asarray(y, float)
        for _ in range(60):
            eta = np.clip(X @ b, -30, 30)
            mu = 1 / (1 + np.exp(-eta))
            w = np.clip(mu * (1 - mu), 1e-8, None)
            z = eta + (yv - mu) / w
            XtW = X.T * w
            try:
                b_new = np.linalg.solve(XtW @ X + 1e-8 * np.eye(X.shape[1]), XtW @ z)
            except np.linalg.LinAlgError:
                return b, -np.inf
            if np.max(np.abs(b_new - b)) < 1e-9:
                b = b_new
                break
            b = b_new
        eta = np.clip(X @ b, -30, 30)
        mu = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
        ll = float(np.sum(yv * np.log(mu) + (1 - yv) * np.log(1 - mu)))
        return b, ll
    b0, ll0 = fit([base])
    b1, ll1 = fit([base, extra])
    stat = 2 * (ll1 - ll0)
    # one degree of freedom; survival function of chi-square(1) is erfc(sqrt(stat/2))
    p = math.erfc(math.sqrt(max(stat, 0) / 2))
    return b1[-1], stat, p


def main() -> int:
    warnings.filterwarnings("ignore")
    import app as A

    rows = [r for r in csv.DictReader(IN.open(encoding="utf-8"))
            if r.get("predicted_probability") and r.get("ad_max_tanimoto") not in (None, "")]
    by = defaultdict(list)
    for r in rows:
        by[r["aid"]].append(r)

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
        sd = tree_spread(model, np.vstack(feats))
        y = np.array([int(r["label"]) for r in keep])
        p = np.array([float(r["predicted_probability"]) for r in keep])
        dist = np.array([1.0 - float(r["ad_max_tanimoto"]) for r in keep])
        spread = np.asarray(sd, float)

        k = max(1, int(y.sum()))
        cut = np.sort(p)[::-1][k - 1]
        wrong = ((p >= cut).astype(int) != y).astype(int)

        print(f"  AID {aid}   n={len(y):,}  actives={int(y.sum())}  wrong={int(wrong.sum())}")

        # --- stratified: within narrow bands of p, is there any discrimination left?
        qs = np.quantile(p, np.linspace(0, 1, 11))
        qs[-1] += 1e-9
        print(f"    {'signal':9s} {'pooled within-bin AUROC':>24s}  {'bins used':>10s}")
        rec = {"aid": aid, "n": len(y), "n_active": int(y.sum()), "n_wrong": int(wrong.sum())}
        for name, sig in (("distance", dist), ("spread", spread)):
            num = den = 0.0
            nb = 0
            for lo, hi in zip(qs[:-1], qs[1:]):
                m = (p >= lo) & (p < hi)
                if m.sum() < 30:
                    continue
                a = auroc(wrong[m].tolist(), sig[m].tolist())
                if a is None:
                    continue
                num += a * m.sum()
                den += m.sum()
                nb += 1
            pooled = num / den if den else float("nan")
            rec[f"within_auroc_{name}"] = round(pooled, 4)
            print(f"    {name:9s} {pooled:>24.3f}  {nb:>10d}")

        # --- logistic: does the signal earn a place beside the prediction?
        logit_p = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        print(f"    {'signal':9s} {'coef':>10s} {'LR stat':>10s} {'p-value':>12s}")
        for name, sig in (("distance", dist), ("spread", spread)):
            z = (sig - sig.mean()) / (sig.std() or 1.0)
            coef, stat, pv = logistic_lrt(wrong, logit_p, z)
            rec[f"coef_{name}"] = round(float(coef), 4)
            rec[f"lrt_p_{name}"] = f"{pv:.2e}"
            print(f"    {name:9s} {coef:>10.3f} {stat:>10.2f} {pv:>12.2e}")
        print()
        out.append(rec)

    if out:
        with OUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"  wrote {OUT.relative_to(ROOT)}")
        print("\n  An AUROC near 0.5 within bins means the signal adds nothing once the")
        print("  prediction is known, however well it appears to do when p is left free.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
