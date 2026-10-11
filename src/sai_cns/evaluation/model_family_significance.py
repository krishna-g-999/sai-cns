"""Paired significance test on the model-family comparison.

`results/tables/model_comparison.csv` reports five families over thirteen endpoints under two split
regimes, and every document that cites it quotes point deltas alone. The decisions log makes exactly
this criticism of an earlier comparison ("point deltas alone do not establish significance") and
answers it with DeLong tests, but those were run on the superseded ensemble and were never re-run for
the deployed random forests. This script closes that gap for the current panel.

What it does and does not measure, stated plainly because the two are easy to confuse:

  It pairs the families by endpoint and asks, over the thirteen endpoints, whether the forest sits
  above an alternative more often than chance would give. The statistic is Wilcoxon signed-rank on
  the per-endpoint means. Because the panel mixes AUROC and R-squared, which are not the same
  quantity, the test is rank-based over "is the forest higher" and the median delta it reports is a
  summary of that ordering rather than an effect size with units.

  It is NOT a DeLong test. DeLong compares two ROC curves on identical compounds and accounts for
  their correlation, which is the stronger instrument where it applies. It applies per endpoint, on
  classification only, and would need the fold-level predictions rather than the summary table. That
  remains the better test and is not done here.

Run:  brainsafe_env/Scripts/python.exe src/sai_cns/evaluation/model_family_significance.py
Out:  results/tables/model_family_significance.csv
"""
from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "results" / "tables" / "model_comparison.csv"
OUT = ROOT / "results" / "tables" / "model_family_significance.csv"

REFERENCE = "RandomForest"
CLASSIFIERS = {"BBB", "AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG"}


def load() -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    with open(SRC, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["split"]][r["endpoint"]][r["model"]] = float(r["mean"])
    return out


def min_attainable_p(n: int) -> float:
    """The smallest two-sided p the signed-rank test can return at this many pairs.

    It is attained when every difference has the same sign, which puts all the rank mass in one tail.
    Below n = 6 that floor sits above 0.05, so the test cannot reject at its own stated level however
    large or however unanimous the effect. At n = 5 the floor is 0.0625.
    """
    if n < 1:
        return 1.0
    return float(wilcoxon(list(range(1, n + 1))).pvalue)


def compare(scores: dict[str, dict[str, float]], endpoints: list[str], alt: str,
            alpha: float = 0.05) -> dict:
    ref = [scores[e][REFERENCE] for e in endpoints]
    other = [scores[e][alt] for e in endpoints]
    delta = [a - b for a, b in zip(ref, other)]
    # wilcoxon is undefined when every difference is zero; report it rather than crashing
    if all(d == 0 for d in delta):
        p = 1.0
    else:
        p = float(wilcoxon(ref, other).pvalue)

    # Whether the test COULD have rejected, reported beside whether it did.
    #
    # The five regression endpoints cannot be tested at 0.05: the floor is 0.0625, and seven of the
    # eight regression rows sat exactly on it, flagged "not distinguishable". Two of those were
    # unanimous, XGBoost above the forest on 5 of 5 and the forest above logistic regression on 5 of
    # 5. Reporting a unanimous result as a null is a statement about the sample size, not about the
    # models, and a reader has no way to tell the two apart from a p-value alone. The distinction is
    # therefore made in the file rather than left to the reader.
    floor = min_attainable_p(len(endpoints))
    underpowered = floor > alpha
    if underpowered:
        verdict = "underpowered: no result at this n can reach alpha"
    elif p < alpha:
        verdict = "distinguishable"
    else:
        verdict = "not distinguishable"

    return {
        "alternative": alt,
        "n_endpoints": len(endpoints),
        "forest_higher_on": sum(1 for d in delta if d > 0),
        "median_delta": round(st.median(delta), 4),
        "mean_delta": round(st.mean(delta), 4),
        "min_delta": round(min(delta), 4),
        "max_delta": round(max(delta), 4),
        "wilcoxon_p": round(p, 5),
        "min_attainable_p": round(floor, 5),
        "underpowered_at_0.05": underpowered,
        "verdict": verdict,
        "distinguishable_at_0.05": (p < alpha) and not underpowered,
    }


def main() -> None:
    data = load()
    rows = []
    for split in ("scaffold", "random"):
        scores = data[split]
        subsets = {
            "all 13 endpoints (AUROC and R-squared pooled by rank)": sorted(scores),
            "8 classification endpoints (AUROC only)": sorted(e for e in scores if e in CLASSIFIERS),
            "5 regression endpoints (R-squared only)": sorted(e for e in scores if e not in CLASSIFIERS),
        }
        for label, eps in subsets.items():
            alts = sorted({m for e in eps for m in scores[e]} - {REFERENCE})
            for alt in alts:
                rows.append({"split": split, "subset": label, **compare(scores, eps, alt)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {OUT.relative_to(ROOT).as_posix()}  ({len(rows)} comparisons)")
    print()
    for label in ("all 13", "8 classification", "5 regression"):
        print(f"scaffold split, {label} endpoints:")
        for r in rows:
            if r["split"] == "scaffold" and r["subset"].startswith(label):
                print(f"  forest vs {r['alternative']:<22} higher on {r['forest_higher_on']:>2}/"
                      f"{r['n_endpoints']}  median {r['median_delta']:+.4f}  "
                      f"p={r['wilcoxon_p']:.4f}  (floor {r['min_attainable_p']:.4f})  "
                      f"{r['verdict']}")
        print()
    n_under = sum(1 for r in rows if r["underpowered_at_0.05"])
    print(f"{n_under} of {len(rows)} comparisons cannot reach alpha=0.05 at their sample size; "
          f"their p-values say nothing about the models.")


if __name__ == "__main__":
    main()
