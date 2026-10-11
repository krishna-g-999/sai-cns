"""Manuscript figures for the powered validation. Every value is read from a results table.

No number in these figures is typed in by hand: each panel loads the CSV written by the analysis
that produced it, so a figure cannot drift away from the result it depicts. If a required table is
missing the figure is skipped with a message rather than drawn from remembered values.

  Figure8_scaffold_holdout.png   prospective recall per target with Wilson intervals and the pooled
                                 estimate, from scaffold_holdout_results.csv
  Figure9_specificity.png        the 1000-compound non-CNS specificity test, score distribution and
                                 where the false positives land
  Figure10_performance.png       sensitivity against specificity with intervals, and the binder
                                 panel validated on held-out measured inactives
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
OUT.mkdir(parents=True, exist_ok=True)
NAVY, GOLD, GREEN, RED, GREY = "#0D2137", "#F0A500", "#1B6B45", "#9B2335", "#94A3B8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#3A4A5F",
                     "axes.linewidth": 0.8, "figure.dpi": 300})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fig_scaffold():
    f = TAB / "scaffold_holdout_results.csv"
    if not f.exists():
        print("skip Figure8: missing scaffold_holdout_results.csv"); return
    d = pd.read_csv(f)
    good = d[~d.threshold_collapsed].sort_values("holdout_recall")
    n_tot = int(good.holdout_actives.sum())
    k_tot = int((good.holdout_recall * good.holdout_actives).round().sum())
    pooled = k_tot / n_tot
    plo, phi = wilson(k_tot, n_tot)

    fig, ax = plt.subplots(1, 2, figsize=(13.6, 6.6), gridspec_kw={"width_ratios": [2.1, 1]})
    y = np.arange(len(good))
    err = np.vstack([good.holdout_recall - good.recall_ci95_low,
                     good.recall_ci95_high - good.holdout_recall])
    cols = [GREEN if r >= 0.80 else (GOLD if r >= 0.50 else RED) for r in good.holdout_recall]
    ax[0].barh(y, good.holdout_recall, color=cols, edgecolor="white", height=0.72)
    ax[0].errorbar(good.holdout_recall, y, xerr=err, fmt="none",
                   ecolor="#3A4A5F", elinewidth=0.9, capsize=2)
    ax[0].axvline(pooled, color=NAVY, ls="--", lw=1.4)
    ax[0].text(pooled + 0.012, 0.6, f"pooled {pooled:.3f}", color=NAVY,
               fontsize=8.6, fontweight="bold")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([f"{t}  (n={n})" for t, n in
                           zip(good.target, good.holdout_actives)], fontsize=7.4)
    ax[0].set_xlim(0, 1.02)
    ax[0].set_xlabel("Recall on held-out scaffolds (95% Wilson interval)")
    ax[0].set_title("(a) Prospective recall per target, entire scaffolds withheld",
                    loc="left", fontweight="bold", color=NAVY)
    ax[0].spines[["top", "right"]].set_visible(False)

    ax[1].hist(good.holdout_recall, bins=np.arange(0, 1.05, 0.1), color=NAVY,
               edgecolor="white", alpha=0.9)
    ax[1].axvline(pooled, color=GOLD, lw=2)
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Targets")
    ax[1].set_title("(b) Distribution across targets", loc="left", fontweight="bold", color=NAVY)
    ax[1].spines[["top", "right"]].set_visible(False)
    txt = (f"{len(good)} targets\n{n_tot:,} held-out compounds\n"
           f"{int(d.holdout_scaffolds.sum()):,} withheld scaffolds\n\n"
           f"pooled recall {pooled:.3f}\n95% CI [{plo:.3f}, {phi:.3f}]")
    ax[1].text(0.02, 0.97, txt, transform=ax[1].transAxes, va="top", fontsize=8.2,
               bbox=dict(boxstyle="round,pad=0.5", fc="#F4F7FC", ec="#D8E0EC"))
    fig.tight_layout()
    fig.savefig(OUT / "Figure8_scaffold_holdout.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote Figure8_scaffold_holdout.png  (pooled {pooled:.3f}, n={n_tot})")


def fig_specificity():
    f = TAB / "noncns_specificity_predictions.csv"
    if not f.exists():
        print("skip Figure9: missing noncns_specificity_predictions.csv"); return
    d = pd.read_csv(f)
    k, n = int(d.fired.sum()), len(d)
    lo, hi = wilson(n - k, n)

    fig, ax = plt.subplots(1, 3, figsize=(14.4, 4.5), gridspec_kw={"width_ratios": [1.1, 1, 1.1]})
    ax[0].hist(d.top_score, bins=40, color=NAVY, edgecolor="white")
    ax[0].axvline(0.30, color=RED, ls="--", lw=1.6)
    ax[0].text(0.32, ax[0].get_ylim()[1] * 0.84, "actionable\nthreshold 0.30",
               color=RED, fontsize=8.2, fontweight="bold")
    ax[0].set_xlabel("Highest disease score"); ax[0].set_ylabel("Compounds")
    ax[0].set_title("(a) Score distribution, 1000 non-CNS compounds",
                    loc="left", fontweight="bold", color=NAVY, fontsize=9.6)
    ax[0].spines[["top", "right"]].set_visible(False)

    ax[1].bar(["silent", "fired"], [n - k, k], color=[GREEN, RED], edgecolor="white", width=0.6)
    for i, v in enumerate([n - k, k]):
        ax[1].text(i, v + 8, f"{v}\n{v/n:.1%}", ha="center", fontsize=9, fontweight="bold")
    ax[1].set_ylim(0, n * 1.1); ax[1].set_ylabel("Compounds")
    ax[1].set_title(f"(b) Specificity {1-k/n:.1%}\n95% CI [{lo:.1%}, {hi:.1%}]",
                    loc="left", fontweight="bold", color=NAVY, fontsize=9.6)
    ax[1].spines[["top", "right"]].set_visible(False)

    fp = d[d.fired == 1].top_disease.value_counts().head(8)[::-1]
    ax[2].barh(range(len(fp)), fp.values, color=GOLD, edgecolor="white")
    ax[2].set_yticks(range(len(fp)))
    ax[2].set_yticklabels([s[:26] for s in fp.index], fontsize=8)
    ax[2].set_xlabel("False positives")
    ax[2].set_title("(c) Where false positives land", loc="left",
                    fontweight="bold", color=NAVY, fontsize=9.6)
    ax[2].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "Figure9_specificity.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote Figure9_specificity.png  (specificity {1-k/n:.3f}, n={n})")


def fig_performance():
    fs, fp = TAB / "scaffold_holdout_results.csv", TAB / "noncns_specificity_predictions.csv"
    fb = TAB / "manuscript_Table2_binder_validation.csv"
    if not (fs.exists() and fp.exists()):
        print("skip Figure10: missing inputs"); return
    sh = pd.read_csv(fs); sp = pd.read_csv(fp)
    good = sh[~sh.threshold_collapsed]
    n_s = int(good.holdout_actives.sum())
    k_s = int((good.holdout_recall * good.holdout_actives).round().sum())
    sens = k_s / n_s; slo, shi = wilson(k_s, n_s)
    k = int(sp.fired.sum()); n_p = len(sp); spec = 1 - k / n_p
    plo, phi = wilson(n_p - k, n_p)

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.9))
    ax[0].errorbar([spec], [sens], xerr=[[spec - plo], [phi - spec]],
                   yerr=[[sens - slo], [shi - sens]], fmt="o", ms=13, color=NAVY,
                   ecolor=GOLD, elinewidth=2.4, capsize=5, zorder=3)
    ax[0].annotate(f"sensitivity {sens:.3f}\nspecificity {spec:.3f}\n"
                   f"balanced {(sens+spec)/2:.3f}", (spec, sens),
                   textcoords="offset points", xytext=(-124, -52), fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.45", fc="#F4F7FC", ec="#D8E0EC"))
    ax[0].plot([0, 1], [1, 0], color=GREY, ls=":", lw=1)
    ax[0].set_xlim(0.5, 1.0); ax[0].set_ylim(0.5, 1.0)
    ax[0].set_xlabel("Specificity (1000 non-CNS compounds)")
    ax[0].set_ylabel("Sensitivity (scaffold-held-out)")
    ax[0].set_title("(a) Operating point with 95% intervals", loc="left",
                    fontweight="bold", color=NAVY)
    ax[0].grid(alpha=0.25); ax[0].spines[["top", "right"]].set_visible(False)

    if fb.exists():
        b = pd.read_csv(fb)
        col = [c for c in b.columns if "AUROC" in c][0]
        ax[1].hist(b[col], bins=np.arange(0.80, 1.005, 0.02), color=GREEN,
                   edgecolor="white", alpha=0.9)
        ax[1].axvline(b[col].mean(), color=NAVY, lw=2)
        ax[1].text(b[col].mean() - 0.004, ax[1].get_ylim()[1] * 0.88,
                   f"mean {b[col].mean():.3f}", ha="right", color=NAVY,
                   fontsize=8.8, fontweight="bold")
        ax[1].set_xlabel("AUROC against held-out measured inactives")
        ax[1].set_ylabel("Targets")
        ax[1].set_title(f"(b) Binder panel, {len(b)} targets", loc="left",
                        fontweight="bold", color=NAVY)
        ax[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "Figure10_performance.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote Figure10_performance.png  (sens {sens:.3f}, spec {spec:.3f})")


if __name__ == "__main__":
    fig_scaffold()
    fig_specificity()
    fig_performance()
