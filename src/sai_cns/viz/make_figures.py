"""Render the result figures for the random-forest models and the feature analysis.

Every figure is built from a table under results/tables/, so the figures can never drift from the
numbers they display. Missing tables are skipped with a note rather than faked. Colours use the
Okabe-Ito colour-blind-safe palette.

Outputs (results/figures/):
  fig_rf_classification_auroc.png   AUROC per classifier, random vs scaffold 10-fold
  fig_rf_regression_r2.png          R-squared per regressor, random vs scaffold 10-fold
  fig_compound_counts.png           compounds per endpoint (by data source if available)
  fig_feature_block_ablation.png    fingerprint-only vs descriptors-only vs combined
  fig_descriptor_importance.png     permutation importance of the twelve descriptors per endpoint

Run:  python src/sai_cns/viz/make_figures.py
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
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermilion": "#D55E00",
         "skyblue": "#56B4E9", "yellow": "#F0E442", "purple": "#CC79A7", "grey": "#666666"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/figures/{name}")


def fig_classification():
    f = TAB / "rf_cv_summary.csv"
    if not f.exists():
        print("  skip classification: rf_cv_summary.csv missing")
        return
    s = pd.read_csv(f)
    s = s[s.task == "classification"]
    order = ["BBB", "AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG"]
    ep = [e for e in order if e in set(s.endpoint)]
    rnd = s[s.split == "random"].set_index("endpoint").reindex(ep)
    scf = s[s.split == "scaffold"].set_index("endpoint").reindex(ep)
    x = np.arange(len(ep)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - w / 2, rnd.roc_auc_mean, w, yerr=rnd.roc_auc_sd, capsize=3,
           color=OKABE["skyblue"], label="Random 10-fold")
    ax.bar(x + w / 2, scf.roc_auc_mean, w, yerr=scf.roc_auc_sd, capsize=3,
           color=OKABE["blue"], label="Scaffold 10-fold")
    ax.set_xticks(x); ax.set_xticklabels(ep, rotation=30, ha="right")
    ax.set_ylabel("AUROC"); ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color=OKABE["grey"], lw=0.8, ls="--")
    ax.set_title("Random-forest classifiers: ten-fold cross-validation")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "fig_rf_classification_auroc.png")


def fig_regression():
    f = TAB / "rf_cv_summary.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    s = s[s.task == "regression"]
    order = ["D2", "A2A", "HT2A", "SERT", "antioxidant_DPPH"]
    ep = [e for e in order if e in set(s.endpoint)]
    rnd = s[s.split == "random"].set_index("endpoint").reindex(ep)
    scf = s[s.split == "scaffold"].set_index("endpoint").reindex(ep)
    x = np.arange(len(ep)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(x - w / 2, rnd.r2_mean, w, yerr=rnd.r2_sd, capsize=3,
           color=OKABE["orange"], label="Random 10-fold")
    ax.bar(x + w / 2, scf.r2_mean, w, yerr=scf.r2_sd, capsize=3,
           color=OKABE["vermilion"], label="Scaffold 10-fold")
    ax.set_xticks(x); ax.set_xticklabels([e.replace("_DPPH", "") for e in ep], rotation=30, ha="right")
    ax.set_ylabel("R-squared"); ax.set_ylim(0, 1.0)
    ax.set_title("Random-forest regressors: ten-fold cross-validation")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, "fig_rf_regression_r2.png")


def fig_counts():
    f = TAB / "rf_cv_summary.csv"
    if not f.exists():
        return
    s = pd.read_csv(f).drop_duplicates("endpoint")[["endpoint", "task", "n"]]
    s = s.sort_values("n", ascending=True)
    colours = [OKABE["blue"] if t == "classification" else OKABE["orange"] for t in s.task]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(s.endpoint, s.n, color=colours)
    for y, n in enumerate(s.n):
        ax.text(n + max(s.n) * 0.01, y, f"{n:,}", va="center", fontsize=9)
    ax.set_xlabel("Compounds used (train + test, 10-fold)")
    ax.set_title("Measured compounds per endpoint")
    handles = [plt.Rectangle((0, 0), 1, 1, color=OKABE["blue"]),
               plt.Rectangle((0, 0), 1, 1, color=OKABE["orange"])]
    ax.legend(handles, ["classification", "regression"], frameon=False, loc="lower right")
    _save(fig, "fig_compound_counts.png")


def fig_ablation():
    f = TAB / "feature_block_ablation.csv"
    if not f.exists():
        print("  skip ablation: feature_block_ablation.csv missing")
        return
    a = pd.read_csv(f)
    ep = list(dict.fromkeys(a.endpoint))
    blocks = ["fingerprint_only", "descriptors_only", "combined"]
    cols = {"fingerprint_only": OKABE["skyblue"], "descriptors_only": OKABE["yellow"],
            "combined": OKABE["green"]}
    x = np.arange(len(ep)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, b in enumerate(blocks):
        d = a[a.block == b].set_index("endpoint").reindex(ep)
        ax.bar(x + (i - 1) * w, d["mean"], w, yerr=d["sd"], capsize=2,
               color=cols[b], label=b.replace("_", " "))
    ax.set_xticks(x); ax.set_xticklabels(ep, rotation=30, ha="right")
    ax.set_ylabel("AUROC (classifiers) / R-squared (regressors)")
    ax.set_title("Feature-block ablation: both blocks are retained on measured evidence")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    _save(fig, "fig_feature_block_ablation.png")


def fig_importance():
    f = TAB / "feature_descriptor_importance.csv"
    if not f.exists():
        print("  skip importance: feature_descriptor_importance.csv missing")
        return
    imp = pd.read_csv(f)
    piv = imp.pivot(index="descriptor", columns="endpoint", values="importance_mean")
    order = ["mw", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds", "aromatic_rings",
             "fraction_csp3", "ring_count", "heavy_atoms", "formal_charge", "qed"]
    piv = piv.reindex([d for d in order if d in piv.index])
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_title("Descriptor permutation importance (drop in performance when shuffled)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean importance")
    _save(fig, "fig_descriptor_importance.png")


def fig_calibration():
    f = TAB / "calibration.csv"
    if not f.exists():
        print("  skip calibration: calibration.csv missing")
        return
    c = pd.read_csv(f)
    x = np.arange(len(c)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - w / 2, c.ece_raw, w, color=OKABE["vermilion"], label="raw RF")
    ax.bar(x + w / 2, c.ece_calibrated, w, color=OKABE["green"], label="isotonic-calibrated")
    ax.set_xticks(x); ax.set_xticklabels(c.endpoint, rotation=30, ha="right")
    ax.set_ylabel("Expected calibration error (lower better)")
    ax.set_title("Probability calibration: isotonic recalibration per endpoint")
    ax.legend(frameon=False)
    _save(fig, "fig_calibration.png")


def fig_applicability():
    f = TAB / "applicability_coverage.csv"
    if not f.exists():
        print("  skip applicability: applicability_coverage.csv missing")
        return
    a = pd.read_csv(f).sort_values("in_domain_frac")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(a.endpoint, a.in_domain_frac, color=OKABE["blue"])
    for y, v in enumerate(a.in_domain_frac):
        ax.text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("Fraction of DrugBank inside the applicability domain (T >= 0.30)")
    ax.set_title("Applicability domain coverage of DrugBank per endpoint")
    _save(fig, "fig_applicability_coverage.png")


def fig_adme():
    f = TAB / "adme_cv_summary.csv"
    if not f.exists():
        print("  skip adme: adme_cv_summary.csv missing")
        return
    s = pd.read_csv(f)
    s = s[s.split == "scaffold"].copy()
    order = ["pgp_inhibition", "solubility", "pgp_substrate", "caco2_permeability",
             "lipophilicity", "logbb", "plasma_protein_binding", "kpuu", "clearance_hepatocyte"]
    s = s.set_index("endpoint").reindex([e for e in order if e in set(s.endpoint)]).reset_index()
    s["score"] = s.apply(lambda r: r.roc_auc_mean if r.task == "classification" else r.r2_mean, axis=1)
    s["label"] = s.apply(lambda r: f"{r.endpoint}\n(AUROC)" if r.task == "classification"
                         else f"{r.endpoint}\n(R2)", axis=1)
    colours = [OKABE["blue"] if t == "classification" else OKABE["green"] for t in s.task]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(s)), s.score, color=colours)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.label, fontsize=8)
    ax.invert_yaxis()
    for i, v in enumerate(s.score):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("Scaffold 10-fold performance (AUROC or R-squared)")
    ax.set_title("ADME / exposure models")
    handles = [plt.Rectangle((0, 0), 1, 1, color=OKABE["blue"]),
               plt.Rectangle((0, 0), 1, 1, color=OKABE["green"])]
    ax.legend(handles, ["classification (AUROC)", "regression (R-squared)"], frameon=False,
              loc="lower right")
    _save(fig, "fig_adme_performance.png")


def fig_kpuu_exposure():
    f = TAB / "cns_exposure_demo.csv"
    if not f.exists():
        print("  skip kpuu: cns_exposure_demo.csv missing")
        return
    d = pd.read_csv(f)
    names = [c.split(" (")[0] for c in d["compound"]]
    colour = [OKABE["green"] if v >= 0.3 else (OKABE["orange"] if v >= 0.1 else OKABE["vermilion"])
              for v in d["kpuu_pred"]]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(range(len(d)), d["kpuu_pred"], color=colour)
    ax.axhline(0.3, color=OKABE["grey"], ls="--", lw=1)
    ax.text(len(d) - 0.5, 0.32, "Kp,uu = 0.3 (meaningful free exposure)", ha="right", fontsize=8,
            color=OKABE["grey"])
    ax.set_xticks(range(len(d))); ax.set_xticklabels(names, rotation=20, ha="right")
    for i, v in enumerate(d["kpuu_pred"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("Predicted Kp,uu (unbound brain/plasma)")
    ax.set_title("Predicted free brain exposure on known drugs")
    _save(fig, "fig_kpuu_exposure.png")


def main():
    print("rendering figures:")
    for fn in (fig_classification, fig_regression, fig_counts, fig_ablation, fig_importance,
               fig_calibration, fig_applicability, fig_adme, fig_kpuu_exposure):
        fn()


if __name__ == "__main__":
    main()
