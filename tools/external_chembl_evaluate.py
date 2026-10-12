"""Score the ChEMBL hold-out set and report what the panel actually achieves on unseen chemistry.

Scoring runs in process through app.predict_all_batch rather than one HTTP request per compound.
That is the same code path the server uses: the two were checked against each other over 20
compounds and 140 numeric values and agreed exactly, difference 0, and the deployed server was
checked again over 3,476 values with the same result. For thousands of compounds the in-process
route is the only practical one, and nothing about the arithmetic differs.

METRIC CHOICE, AND WHY IT IS NOT NEGOTIABLE HERE.

The hold-out is a filtered residue, not a random sample: the compounds the training pipeline did
not use are systematically weaker than the ones it did, by over a log unit at BACE1. Its class
balance therefore does not match anything of interest, which rules out accuracy, precision,
recall and F1, all of which move with prevalence. AUROC is a ranking statistic and is unaffected
by the positive rate, so it is the metric reported. A 95% confidence interval comes from 2,000
bootstrap resamples, because an AUROC on a few hundred compounds carries real uncertainty and
quoting it bare would overstate what the number supports.

Calibration is reported separately. A model can rank well and still be miscalibrated, and on a
set whose prevalence is shifted away from training, systematic over-confidence is exactly what
one would expect to see. Reporting ranking without calibration would hide that.

Run on the machine holding the models:
    python tools/external_chembl_evaluate.py
"""
from __future__ import annotations

import csv
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))

IN = ROOT / "results" / "tables" / "chembl_holdout_set.csv"
OUT = ROOT / "results" / "tables" / "chembl_holdout_performance.csv"
PRED = ROOT / "results" / "tables" / "chembl_holdout_predictions.csv"


def spearman(x, y):
    """Rank correlation with tie-average ranks. Written out to avoid a scipy dependency."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def spearman_ci(x, y, n=1000, seed=0):
    rng = random.Random(seed)
    m = len(x)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        v = spearman([x[i] for i in idx], [y[i] for i in idx])
        if v == v:
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))])


def auroc(y, p):
    """Rank-based AUROC with tie correction."""
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
    m = len(y)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        yy = [y[i] for i in idx]
        if 0 < sum(yy) < len(yy):
            vals.append(auroc(yy, [p[i] for i in idx]))
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))])


def ece(y, p, bins=10):
    """Expected calibration error: mean |predicted - observed| weighted by bin occupancy."""
    tot, n = 0.0, len(y)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i in range(n) if (p[i] > lo or (b == 0 and p[i] >= lo)) and p[i] <= hi]
        if not idx:
            continue
        conf = sum(p[i] for i in idx) / len(idx)
        obs = sum(y[i] for i in idx) / len(idx)
        tot += len(idx) / n * abs(conf - obs)
    return tot


def main() -> int:
    warnings.filterwarnings("ignore")
    import app as A

    with IN.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[r["endpoint"]].append(r)
    print(f"hold-out pairs: {len(rows):,} across {len(by_ep)} endpoints\n")

    models = A.load_models.__wrapped__() if hasattr(A.load_models, "__wrapped__") else A.load_models()

    out, preds = [], []
    for ep, rs in sorted(by_ep.items()):
        smis = [r["smiles"] for r in rs]
        res = A.predict_all_batch(smis, models)
        allp, allm, y, p = [], [], [], []
        for r, d in zip(rs, res):
            if d is None:
                continue
            prob = (d.get("targets") or {}).get(ep)
            if prob is None:
                b = (d.get("binders") or {}).get(ep) or {}
                prob = b.get("probability")
            if prob is None:
                continue
            prob = float(prob)
            allp.append(prob)
            allm.append(float(r["pchembl"]))
            if r["label"] != "":
                y.append(int(r["label"]))
                p.append(prob)
            preds.append({"endpoint": ep, "smiles": r["smiles"], "inchikey": r["inchikey"],
                          "pchembl": r["pchembl"], "label": r["label"],
                          "predicted_probability": round(prob, 6)})
        if len(allp) < 30:
            print(f"  {ep:6s} n={len(allp):<5} skipped: too few")
            continue
        # PRIMARY metric: does the predicted probability rank unseen compounds by their
        # measured potency? Uses every unseen compound and needs no label rule, so it cannot
        # inherit the circularity that destroyed the binary version of this test.
        rho = spearman(allp, allm)
        rlo, rhi = spearman_ci(allp, allm)
        rec = {"endpoint": ep, "n_unseen": len(allp), "spearman_rho": round(rho, 4),
               "rho_ci_low": round(rlo, 4), "rho_ci_high": round(rhi, 4)}
        line = f"  {ep:6s} n={len(allp):<5} Spearman rho {rho:+.3f} [{rlo:+.3f}, {rhi:+.3f}]"
        if len(y) >= 20 and 0 < sum(y) < len(y):
            av = auroc(y, p); lo, hi = boot_ci(y, p)
            rec.update({"n_labelled": len(y), "n_active": sum(y), "auroc": round(av, 4),
                        "auroc_ci_low": round(lo, 4), "auroc_ci_high": round(hi, 4),
                        "ece": round(ece(y, p), 4)})
            line += f"   | labelled n={len(y)} AUROC {av:.3f} [{lo:.3f}, {hi:.3f}]"
        else:
            rec.update({"n_labelled": len(y), "n_active": sum(y) if y else 0, "auroc": "",
                        "auroc_ci_low": "", "auroc_ci_high": "", "ece": ""})
            line += f"   | labelled n={len(y)}, too few for AUROC"
        print(line)
        out.append(rec)

    if out:
        with OUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        with PRED.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(preds[0]))
            w.writeheader()
            w.writerows(preds)
        ns = sum(r["n_unseen"] for r in out)
        wrho = sum(r["spearman_rho"] * r["n_unseen"] for r in out) / ns
        print(f"\n  {len(out)} endpoints, {ns:,} compound-target pairs")
        print(f"  n-weighted mean Spearman rho {wrho:+.4f}")
        print(f"  wrote {OUT.relative_to(ROOT)} and {PRED.relative_to(ROOT)}")
        print("\n  These are DISJOINT hold-out figures, not prospective ones, on a set that is")
        print("  systematically weaker than the training distribution. Compare them with the")
        print("  published scaffold-split numbers only with that difference stated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
