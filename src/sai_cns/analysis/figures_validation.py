"""Manuscript figures explaining the cross-validation design and the error bars.

Figure 5  (a) schematic of the 10-fold random vs scaffold-grouped design;
          (b) decomposition of the between-fold error bar into sampling noise and genuine
              chemotype heterogeneity, per endpoint and split.
Figure 6  Complete per-endpoint performance across all three model layers.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
TAB = ROOT / "results" / "tables"
OUT = ROOT / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NAVY, GOLD, GREEN, GREY, RED = "#0D2137", "#F0A500", "#1B6B45", "#94A3B8", "#9B2335"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": "#3A4A5F",
                     "axes.linewidth": 0.8, "figure.dpi": 300})


def fig_cv_and_errorbars():
    T4 = pd.read_csv(TAB / "manuscript_T4_variance_decomposition.csv")
    fig = plt.figure(figsize=(12.4, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.34, wspace=0.22)

    cols = [NAVY, GOLD, GREEN, RED]
    rng = np.random.default_rng(3)

    def _grid(ax, series_of_cell, title, sub):
        for c in range(10):
            for rw in range(4):
                ax.add_patch(Rectangle((c, -rw), 0.86, 0.86,
                                       fc=cols[series_of_cell(c, rw)], ec="white", lw=1))
            ax.add_patch(Rectangle((c - 0.07, -3.07), 1.0, 4.0, fill=False,
                                   ec="#3A4A5F", lw=0.7, ls=":"))
        ax.text(5, 1.62, title, ha="center", fontweight="bold", fontsize=11, color=NAVY)
        ax.text(5, 1.02, sub, ha="center", fontsize=8.6, color="#3A4A5F")
        ax.annotate("", xy=(10.0, -3.55), xytext=(0.0, -3.55),
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.9))
        ax.text(5, -4.0, "fold 1  ...  fold 10", ha="center", fontsize=8, color=GREY)
        ax.set_xlim(-0.8, 10.4); ax.set_ylim(-4.5, 2.3); ax.axis("off")

    # (a) random: every fold is a mixture of all series
    axa = fig.add_subplot(gs[0, 0])
    rand_cells = rng.integers(0, 4, (10, 4))
    _grid(axa, lambda c, rw: int(rand_cells[c, rw]), "Random 10-fold",
          "every fold contains all chemical series\n(test compounds have close training analogues)")

    # (b) scaffold: each fold is one intact chemical series
    axb = fig.add_subplot(gs[0, 1])
    _grid(axb, lambda c, rw: c % 4, "Scaffold-grouped 10-fold",
          "each fold holds out whole chemical series\n(test compounds are unseen chemotypes)")
    axb.legend(handles=[plt.Line2D([], [], marker="s", ls="", ms=8, mfc=c, mec="white",
                                   label=f"series {i+1}") for i, c in enumerate(cols)],
               loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))

    # ---- (c) variance decomposition ----
    axc = fig.add_subplot(gs[1, :])
    eps = [e for e in ["BBB", "AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG",
                       "D2", "A2A", "HT2A", "SERT", "antioxidant_DPPH"] if e in set(T4.endpoint)]
    x = np.arange(len(eps)); w = 0.38
    for k, (split, off) in enumerate([("random", -w / 2), ("scaffold", w / 2)]):
        d = T4[T4.split == split].set_index("endpoint").loc[eps]
        samp = d["sd_sampling"].values
        het = d["sd_heterogeneity"].values
        hatch = "" if split == "random" else "//"
        axc.bar(x + off, samp, w, color=GREY, edgecolor="white",
                label="sampling noise" if k == 0 else None)
        axc.bar(x + off, het, w, bottom=samp, color=GOLD if split == "random" else NAVY,
                edgecolor="white", hatch=hatch,
                label=("chemotype heterogeneity (random)" if split == "random"
                       else "chemotype heterogeneity (scaffold)"))
    ymax = float(T4[T4.split == "scaffold"]["sd_observed"].max())
    for i, e in enumerate(eps):
        s = T4[(T4.endpoint == e) & (T4.split == "scaffold")].iloc[0]
        axc.text(i + w / 2, float(s.sd_observed) + ymax * 0.02,
                 f"{s.pct_variance_heterogeneity:.0f}%", ha="center", va="bottom",
                 fontsize=8.2, color=NAVY, fontweight="bold", clip_on=False, zorder=6)
    axc.set_ylim(0, ymax * 1.22)
    axc.set_xticks(x); axc.set_xticklabels(eps, rotation=35, ha="right", fontsize=8.5)
    axc.set_ylabel("Between-fold standard deviation")
    axc.set_title("(c) The scaffold error bar is dominated by genuine chemotype heterogeneity, "
                  "not statistical noise\n(left bar of each pair, random split; right bar, scaffold split; "
                  "% = heterogeneity share of scaffold variance)",
                  loc="left", fontsize=9.6, fontweight="bold", color=NAVY)
    axc.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    axc.spines[["top", "right"]].set_visible(False)
    axa.set_title("(a)", loc="left", fontweight="bold", color=NAVY)
    axb.set_title("(b)", loc="left", fontweight="bold", color=NAVY)
    fig.savefig(OUT / "Figure5_cv_design_and_errorbars.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote Figure5_cv_design_and_errorbars.png")


def fig_all_endpoints():
    T1 = pd.read_csv(TAB / "manuscript_T1_endpoints.csv")
    adme = pd.read_csv(TAB / "adme_cv_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.4), gridspec_kw={"width_ratios": [1, 1.25, 1]})

    # (a) classifiers
    clf = ["BBB", "AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG"]
    ax = axes[0]
    r = T1[(T1.split == "random") & (T1.endpoint.isin(clf))].set_index("endpoint").loc[clf]
    s = T1[(T1.split == "scaffold") & (T1.endpoint.isin(clf))].set_index("endpoint").loc[clf]
    y = np.arange(len(clf))[::-1]
    ax.barh(y + 0.19, r.roc_auc_mean, 0.36, xerr=r.roc_auc_sd, color=GREY, edgecolor="white",
            error_kw=dict(ecolor="#3A4A5F", lw=0.9, capsize=2), label="random 10-fold")
    ax.barh(y - 0.19, s.roc_auc_mean, 0.36, xerr=s.roc_auc_sd, color=NAVY, edgecolor="white",
            error_kw=dict(ecolor=GOLD, lw=0.9, capsize=2), label="scaffold 10-fold")
    ax.set_yticks(y); ax.set_yticklabels(clf, fontsize=9)
    ax.set_xlim(0.6, 1.0); ax.set_xlabel("AUROC")
    ax.set_title("(a) Target classifiers", loc="left", fontweight="bold", color=NAVY)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.13))
    ax.spines[["top", "right"]].set_visible(False)

    # (b) binder classifiers
    ax = axes[1]
    order = ["D2", "A2A", "HT2A", "SERT", "HT1A", "HT6", "HT7", "H3", "DAT", "NET",
             "Sigma1", "CB1", "OPRK1", "OPRM1", "D3", "A1", "a7nAChR", "LRRK2"]
    vals, labs = [], []
    for ep in order:
        f = ROOT / "models_rf" / f"{ep}_binder_meta.json"
        if f.exists():
            m = json.loads(f.read_text())
            if m.get("auroc_hard_decoys"):
                vals.append(m["auroc_hard_decoys"]); labs.append(ep)
    y = np.arange(len(labs))[::-1]
    ax.barh(y, vals, 0.66, color=GREEN, edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8.5)
    ax.set_xlim(0.6, 1.0); ax.set_xlabel("AUROC (near-miss decoys)")
    ax.set_title("(b) Decoy-aware binder classifiers", loc="left", fontweight="bold", color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)

    # (c) ADME
    ax = axes[2]
    aorder = ["solubility", "lipophilicity", "caco2_permeability", "logbb", "kpuu",
              "plasma_protein_binding", "clearance_hepatocyte"]
    ar = adme[(adme.split == "scaffold") & (adme.endpoint.isin(aorder))].set_index("endpoint").loc[aorder]
    y = np.arange(len(aorder))[::-1]
    ax.barh(y, ar.r2_mean, 0.6, xerr=ar.r2_sd, color=GOLD, edgecolor="white",
            error_kw=dict(ecolor=NAVY, lw=0.9, capsize=2))
    ax.set_yticks(y); ax.set_yticklabels([a.replace("_", " ") for a in aorder], fontsize=8.5)
    ax.set_xlim(0, 1.0); ax.set_xlabel("R2 (scaffold 10-fold)")
    ax.set_title("(c) ADME regression endpoints", loc="left", fontweight="bold", color=NAVY)
    ax.axvline(0.3, color=RED, ls=":", lw=1)
    ax.text(0.30, -0.92, "R2 = 0.3", color=RED, fontsize=7.4, ha="center")
    ax.spines[["top", "right"]].set_visible(False)

    # P-gp classifiers noted separately (below the axis label)
    pg = adme[(adme.split == "scaffold") & (adme.endpoint.isin(["pgp_inhibition", "pgp_substrate"]))]
    txt = ", ".join(f"{r.endpoint.replace('_',' ')} AUROC {r.roc_auc_mean:.2f}" for _, r in pg.iterrows())
    ax.annotate(f"P-gp classifiers (scaffold): {txt}", xy=(0.5, -0.20), xycoords="axes fraction",
                ha="center", fontsize=7.6, color="#3A4A5F")

    fig.tight_layout()
    fig.savefig(OUT / "Figure6_all_endpoints.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote Figure6_all_endpoints.png")


if __name__ == "__main__":
    fig_cv_and_errorbars()
    fig_all_endpoints()
