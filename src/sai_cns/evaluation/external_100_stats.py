"""Statistical analysis of the prospective external validation.

Reported quantities and why each is chosen:

  * Top-1 and top-3 accuracy with Wilson score intervals. Wilson rather than the normal
    approximation because the sample is modest and several strata have proportions near 0 or 1,
    where the normal interval misbehaves and can leave the unit interval.

  * A permutation test against chance. With fourteen candidate diseases the naive chance rate is
    1/14, but the tool does not choose uniformly, so the null is built by permuting the expected
    labels across compounds and recomputing accuracy, which preserves both the label distribution
    and the tool's own prediction distribution.

  * Specificity on negative controls, measured separately, because a tool that fires on everything
    would score well on sensitivity alone.

  * Stratification by applicability domain. This is the pre-registered test of the central claim
    that predictive power is retained inside the domain and lost outside it. A difference in
    proportions is tested with a two-sided Fisher exact test.

  * Cohen's kappa, which corrects agreement for chance given the marginal distributions.

Output: results/tables/external_100_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
TAB = ROOT / "results" / "tables"
rng = np.random.default_rng(20260802)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def permutation_p(hits, expected, predicted, n_perm=20000):
    obs = hits.mean()
    exp = np.array(expected, dtype=object)
    pred = np.array(predicted, dtype=object)
    count = 0
    for _ in range(n_perm):
        if (rng.permutation(exp) == pred).mean() >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def main():
    df = pd.read_csv(TAB / "external_100_predictions.csv")
    cns = df[df.expected_class != "NONE"].copy()
    neg = df[df.expected_class == "NONE"].copy()
    rows = []

    def add(metric, k, n, extra=""):
        lo, hi = wilson(k, n)
        rows.append({"metric": metric, "k": int(k), "n": int(n),
                     "estimate": round(k / n, 4) if n else np.nan,
                     "ci95_low": round(lo, 4), "ci95_high": round(hi, 4), "note": extra})

    add("Top-1 accuracy (CNS compounds)", cns.hit_top1.sum(), len(cns))
    add("Top-3 accuracy (CNS compounds)", cns.hit_top3.sum(), len(cns))
    add("Specificity (negative controls silent)", neg.hit_top3.sum(), len(neg),
        "a control is correct when the tool reports no disease above threshold")

    # chance baselines
    n_dis = cns.expected_class.nunique()
    p_perm = permutation_p(cns.hit_top1.to_numpy(), cns.expected_class.tolist(),
                           cns.predicted_top1.tolist())
    rows.append({"metric": "Permutation p (top-1 vs shuffled labels)", "k": "", "n": len(cns),
                 "estimate": p_perm, "ci95_low": "", "ci95_high": "",
                 "note": f"{n_dis} distinct expected classes present"})

    # Cohen's kappa over CNS compounds
    labs = sorted(set(cns.expected_class) | set(cns.predicted_top1))
    idx = {l: i for i, l in enumerate(labs)}
    cm = np.zeros((len(labs), len(labs)))
    for a, b in zip(cns.expected_class, cns.predicted_top1):
        cm[idx[a], idx[b]] += 1
    po = np.trace(cm) / cm.sum()
    pe = (cm.sum(0) * cm.sum(1)).sum() / cm.sum() ** 2
    rows.append({"metric": "Cohen's kappa (top-1)", "k": "", "n": len(cns),
                 "estimate": round((po - pe) / (1 - pe), 4), "ci95_low": "", "ci95_high": "",
                 "note": "chance-corrected agreement"})

    # applicability-domain stratification
    ind = cns[cns.ad_max_tanimoto >= 0.5]
    near = cns[(cns.ad_max_tanimoto >= 0.3) & (cns.ad_max_tanimoto < 0.5)]
    outd = cns[cns.ad_max_tanimoto < 0.3]
    for nm, sub in [("in domain (T>=0.5)", ind), ("near domain (0.3-0.5)", near),
                    ("out of domain (T<0.3)", outd)]:
        if len(sub):
            add(f"Top-1 accuracy, {nm}", sub.hit_top1.sum(), len(sub))
    if len(ind) and len(outd):
        table = [[ind.hit_top1.sum(), len(ind) - ind.hit_top1.sum()],
                 [outd.hit_top1.sum(), len(outd) - outd.hit_top1.sum()]]
        _, pf = stats.fisher_exact(table)
        rows.append({"metric": "Fisher exact p (in domain vs out of domain)", "k": "",
                     "n": len(ind) + len(outd), "estimate": round(pf, 5),
                     "ci95_low": "", "ci95_high": "", "note": "two-sided"})

    # per-class recall
    for cl, sub in cns.groupby("expected_class"):
        add(f"Recall: {cl}", sub.hit_top3.sum(), len(sub), "top-3")

    out = pd.DataFrame(rows)
    out.to_csv(TAB / "external_100_summary.csv", index=False)
    pd.set_option("display.width", 170)
    print(out.to_string(index=False))
    print()
    print("Confusion of misses (expected -> predicted), CNS compounds only:")
    miss = cns[cns.hit_top1 == 0]
    for _, r in miss.iterrows():
        print(f"   {r.compound:24} {r.expected_class[:26]:28} -> {str(r.predicted_top1)[:26]:28} "
              f"AD {r.ad_max_tanimoto}")
    print("\nwrote", TAB / "external_100_summary.csv")


if __name__ == "__main__":
    main()
