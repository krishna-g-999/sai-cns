"""Figure 7: prospective (temporal) performance is governed by the applicability domain.

(a) Per-endpoint temporal AUROC split by how similar each future compound is to the training set.
(b) The same for the regression endpoints, using Spearman rank correlation, the decision-relevant
    metric for triage.
(c) Summary: mean performance by domain stratum, showing that predictive power is retained inside
    the domain and lost outside it.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TAB = ROOT / "results" / "tables"
OUT = ROOT / "manuscript" / "figures"
NAVY, GOLD, GREEN, GREY, RED = "#0D2137", "#F0A500", "#1B6B45", "#94A3B8", "#9B2335"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": "#3A4A5F",
                     "axes.linewidth": 0.8, "figure.dpi": 300})
S_IN, S_NEAR, S_OUT = "in_domain (T>=0.5)", "near_domain (0.3-0.5)", "out_domain (T<0.3)"
COLS = {S_IN: GREEN, S_NEAR: GOLD, S_OUT: RED}
LBL = {S_IN: "in domain (T $\\geq$ 0.5)", S_NEAR: "near domain (0.3-0.5)", S_OUT: "out of domain (T < 0.3)"}


def main():
    d = pd.read_csv(TAB / "temporal_by_domain.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), gridspec_kw={"width_ratios": [1.25, 1.1, 0.85]})

    # (a) classifiers
    c = d[d.task == "classification"]
    eps = list(dict.fromkeys(c.endpoint))
    x = np.arange(len(eps)); w = 0.26
    ax = axes[0]
    for i, s in enumerate([S_IN, S_NEAR, S_OUT]):
        v = [c[(c.endpoint == e) & (c.stratum == s)].auroc.mean() for e in eps]
        ax.bar(x + (i - 1) * w, v, w, color=COLS[s], edgecolor="white", label=LBL[s])
    ax.axhline(0.5, color="#3A4A5F", ls=":", lw=1)
    ax.text(len(eps) - 0.5, 0.512, "chance", fontsize=7.4, color="#3A4A5F", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(eps, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylim(0.3, 1.0); ax.set_ylabel("Temporal AUROC")
    ax.set_title("(a) Classifiers: future compounds", loc="left", fontweight="bold", color=NAVY)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=1)
    ax.spines[["top", "right"]].set_visible(False)

    # (b) regressions, Spearman
    r = d[d.task == "regression"]
    eps = list(dict.fromkeys(r.endpoint))
    x = np.arange(len(eps))
    ax = axes[1]
    for i, s in enumerate([S_IN, S_NEAR, S_OUT]):
        v = [r[(r.endpoint == e) & (r.stratum == s)].spearman.mean() for e in eps]
        ax.bar(x + (i - 1) * w, v, w, color=COLS[s], edgecolor="white")
    ax.axhline(0, color="#3A4A5F", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([e.replace("_DPPH", "") for e in eps], rotation=30,
                                         ha="right", fontsize=8.5)
    ax.set_ylim(-0.45, 0.85); ax.set_ylabel("Temporal Spearman $\\rho$")
    ax.set_title("(b) Potency models: rank correlation", loc="left", fontweight="bold", color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)

    # (c) summary
    ax = axes[2]
    strata = [S_IN, S_NEAR, S_OUT]
    cm = [c[c.stratum == s].auroc.mean() for s in strata]
    rm = [r[r.stratum == s].spearman.mean() for s in strata]
    xs = np.arange(3)
    ax.plot(xs, cm, "-o", color=NAVY, lw=2, ms=8, label="classifiers (AUROC)")
    ax.plot(xs, rm, "-s", color=GOLD, lw=2, ms=8, label="potency ($\\rho$)")
    for i, (a, b) in enumerate(zip(cm, rm)):
        ax.annotate(f"{a:.2f}", (i, a), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.6, color=NAVY, fontweight="bold")
        ax.annotate(f"{b:.2f}", (i, b), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8.6, color="#8A5E00", fontweight="bold")
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(["in", "near", "out"], fontsize=9.5)
    ax.set_xlabel("Applicability domain of the future compound")
    ax.set_ylim(-0.1, 1.0)
    ax.set_title("(c) Predictive power is retained\ninside the domain, lost outside",
                 loc="left", fontweight="bold", color=NAVY, fontsize=9.8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "Figure7_temporal_by_domain.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote Figure7_temporal_by_domain.png")
    print("  classifiers by domain:", [round(v, 3) for v in cm])
    print("  potency rho by domain:", [round(v, 3) for v in rm])


if __name__ == "__main__":
    main()
