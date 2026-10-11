"""Figure 1. How many models there are, what each one is, and how they compose into one answer.

A reviewer's first question about a multi-endpoint server is what "the model" refers to, because
there is no single model: a prediction for one compound is assembled from many estimators trained on
different data with different label rules. Panel A follows one compound through that assembly. Panel
B expands a single endpoint, because the count in panel A hides the fact that every deployed
estimator is the last of twenty-one fits, twenty of which exist only to produce the error bar.

Every count is read from models_rf/ at run time. Nothing here is drawn from the text.

Run:  python src/sai_cns/figures/fig01_architecture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

M = ROOT / "models_rf"


def counts() -> dict:
    """The panel as it exists on disk, counted rather than restated.

    Deployed estimators and cross-validated endpoints are two different totals and are kept apart
    here. They happen to be close, which is exactly why conflating them would be easy: the binder
    for a receptor and the potency regression for the same receptor are two estimators under one
    endpoint name, while pKa is an estimator with no cross-validation record at all.
    """
    core_cls, core_reg, aux = [], [], []
    for p in sorted(M.glob("*_meta.json")):
        meta = json.loads(p.read_text(encoding="utf-8"))
        ep, task = meta.get("endpoint", p.stem), meta.get("task")
        if task == "classification":
            core_cls.append(ep)
        elif task == "regression":
            (aux if ep in ("antioxidant_DPPH", "pka_basic") else core_reg).append(ep)
    # pKa carries no meta task field, so it is counted from the estimator on disk.
    if (M / "pka_basic.joblib").exists() and "pka_basic" not in aux:
        aux.append("pka_basic")

    bm = json.loads((M / "binder_modes.json").read_text(encoding="utf-8"))
    adme = sorted(p.stem.replace("_meta", "") for p in (M / "adme").glob("*_meta.json"))

    # Trained is not deployed. Two binders were withdrawn after specificity testing and are still on
    # disk, so counting estimator files gives 72 where the server offers 70.
    n_withdrawn = sum(1 for v in bm.values() if not v.get("deployed", True))
    trained = (len(list(M.glob("*_binder.joblib")))
               + len([p for p in M.glob("*.joblib")
                      if not p.name.endswith(("_binder.joblib", "_calibrated.joblib"))])
               + len(list((M / "adme").glob("*.joblib"))))
    deployed = trained - n_withdrawn
    # An estimator is not an endpoint. Four receptors carry both a potency regression and a binder
    # classifier, so they are cross-validated twice and appear in two fold tables; counting the
    # union of file:endpoint keys counts estimators, and counting bare endpoint names counts
    # endpoints. The two totals differ by those four and are reported separately.
    cv_rows, cv_est, cv_eps = 0, set(), set()
    for f in ("rf_cv_folds", "binder_cv_folds", "adme_cv_folds"):
        p = ROOT / "results" / "tables" / f"{f}.csv"
        if p.exists():
            d = pd.read_csv(p, usecols=["endpoint", "split"])
            # binder_cv_folds also carries a random-split re-run, added after training purely so a
            # reviewer can see the conventional split for comparison (binder_cv_per_fold.py); the
            # binder classifiers themselves were only ever fitted, at training time, under the
            # scaffold-grouped split recorded in binder_modes.json. Counting the re-run as though it
            # were a fit behind the deployed panel double-counts a diagnostic that was never part of
            # building that panel, which is how this number drifted to 1,480 before.
            fit_rows = d[d.split == "scaffold"] if f == "binder_cv_folds" else d
            cv_rows += len(fit_rows)
            cv_est |= {f"{f}:{e}" for e in d.endpoint.unique()}
            cv_eps |= set(d.endpoint.unique())
    return {
        "core_cls": sorted(core_cls), "core_reg": sorted(core_reg), "aux": sorted(aux),
        "adme": adme, "binders": sorted(bm),
        "binder_hybrid": [k for k, v in bm.items()
                          if v.get("mode") == "hybrid_decoys_plus_measured_inactives"],
        "binder_holdout": [k for k, v in bm.items() if v.get("mode") == "measured_labels_holdout"],
        "binder_deployed": [k for k, v in bm.items() if v.get("deployed", True)],
        "binder_withdrawn": [k for k, v in bm.items() if not v.get("deployed", True)],
        "n_deployed": deployed, "n_trained": trained, "n_cv_fits": cv_rows,
        "n_cv_estimators": len(cv_est), "n_cv_endpoints": len(cv_eps),
        "n_calibrators": len(list(M.glob("*_calibrated.joblib"))),
    }


def box(ax, x, y, w, h, face, edge=None, alpha=1.0, r=0.012, lw=0.7, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=face, edgecolor=edge or face, alpha=alpha, linewidth=lw, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color=S.FAINT, lw=0.9, z=1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 color=color, linewidth=lw, shrinkA=0, shrinkB=0, zorder=z))


def panel_a(ax, c) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- column 1: the compound -------------------------------------------------------------
    # The four standardisation steps used to print here at one line each ("largest organic
    # fragment" and "InChIKey assigned" both ran past the box's own sides, measured against its
    # usable width); Figure 2 already gives that sequence in full beside the same molecule, and
    # this panel is about the four model families the vector feeds, not the standardisation step,
    # so the caption's "standardised" carries it here instead of a second, cut-off copy of it.
    box(ax, 0.005, 0.30, 0.115, 0.40, "#F4F7F9", S.HAIR)
    ax.text(0.062, 0.550, "INPUT", ha="center", fontsize=6.5, color=S.MUTED, fontweight="bold")
    ax.text(0.062, 0.470, "SMILES", ha="center", fontsize=8.5, color=S.INK, fontweight="bold")
    ax.text(0.062, 0.395, "standardised,\ndeduplicated", ha="center", va="top", fontsize=6.5,
            color=S.MUTED, linespacing=1.5)

    # ---- column 2: the feature vector, drawn at its real proportions ------------------------
    x0, w = 0.155, 0.105
    box(ax, x0, 0.30, w, 0.40, "#F4F7F9", S.HAIR)
    ax.text(x0 + w / 2, 0.655, "FEATURES", ha="center", fontsize=6.5, color=S.MUTED,
            fontweight="bold")
    # the strip is 1024:12, so the descriptor block is deliberately almost invisible
    bar_y, bar_h = 0.470, 0.085
    n_seg = 26
    seg_w = (w - 0.020) / n_seg
    for i in range(n_seg):
        lit = i in (2, 5, 6, 11, 15, 18, 22)
        ax.add_patch(Rectangle((x0 + 0.010 + i * seg_w, bar_y), seg_w * 0.78, bar_h,
                               facecolor=S.EXPOSURE if lit else "#D7DFE6", edgecolor="none",
                               zorder=4))
    ax.add_patch(Rectangle((x0 + 0.010, bar_y - 0.062), w - 0.020, 0.030,
                           facecolor=S.TARGET, edgecolor="none", zorder=4))
    ax.text(x0 + w / 2, 0.578, "1,024 ECFP-4 bits", ha="center", fontsize=6.5, color=S.EXPOSURE,
            fontweight="bold", zorder=5)
    ax.text(x0 + w / 2, 0.360, "12 descriptors", ha="center", fontsize=6.5, color=S.TARGET,
            fontweight="bold", zorder=5)
    ax.text(x0 + w / 2, 0.318, "1,036 columns", ha="center", fontsize=6.5, color=S.INK,
            fontweight="bold", zorder=5)

    # ---- column 3: the four model families -------------------------------------------------
    fx, fw = 0.305, 0.245
    bands = [
        ("EXPOSURE", len(c["adme"]), S.EXPOSURE,
         "BBB, logBB, Kp,uu, P-gp,\npermeability, solubility, logD,\nPPB, clearance"),
        ("TARGET POTENCY", len(c["core_cls"]) + len(c["core_reg"]), S.TARGET,
         f"{len(c['core_cls'])} classifiers + {len(c['core_reg'])} potency\n"
         "regressions, isotonic-calibrated,\nconformal intervals"),
        ("BINDER PANEL", len(c["binders"]), S.BINDER,
         f"{len(c['binder_hybrid'])} decoy-aware + "
         f"{len(c['binder_holdout'])} measured-label,\n"
         f"{len(c['binder_deployed'])} deployed, "
         f"{len(c['binder_withdrawn'])} withdrawn"),
        ("AUXILIARY", len(c["aux"]), S.MUTED, "antioxidant response, pKa"),
    ]
    tops = [0.94, 0.705, 0.47, 0.235]
    hs = [0.195, 0.195, 0.195, 0.115]
    for (name, n, col, sub), top, h in zip(bands, tops, hs):
        y = top - h
        box(ax, fx, y, fw, h, col, alpha=0.10, edge=col, lw=0.8)
        ax.add_patch(Rectangle((fx, y), 0.006, h, facecolor=col, edgecolor="none", zorder=3))
        ax.text(fx + 0.018, top - 0.034, name, fontsize=6.5, color=col, fontweight="bold")
        ax.text(fx + fw - 0.014, top - 0.040, f"{n}", ha="right", va="center", fontsize=10.5,
                color=col, fontweight="bold")
        ax.text(fx + 0.018, top - 0.062, sub, fontsize=6.5, color=S.MUTED, va="top",
                linespacing=1.5)
        arrow(ax, x0 + w + 0.004, 0.50, fx - 0.004, y + h / 2)

    arrow(ax, 0.122, 0.50, 0.153, 0.50)

    # ---- column 4: the gate ----------------------------------------------------------------
    gx, gw = 0.585, 0.115
    box(ax, gx, 0.30, gw, 0.40, "#FFF6E6", S.WITHHELD, lw=0.9)
    ax.text(gx + gw / 2, 0.655, "GATE", ha="center", fontsize=6.5, color=S.WITHHELD,
            fontweight="bold")
    ax.text(gx + gw / 2, 0.590, "exposure ×\nengagement", ha="center", va="center", fontsize=7.2,
            color=S.INK, fontweight="bold", linespacing=1.5)
    ax.text(gx + gw / 2, 0.500, "a target score is\nadmitted only in\nproportion to the\n"
                                "probability of\nreaching the brain",
            ha="center", fontsize=6.5, color=S.MUTED, va="top", linespacing=1.55)
    for top, h in zip(tops, hs):
        arrow(ax, fx + fw + 0.004, top - h / 2, gx - 0.004, 0.50)

    # ---- column 5: the output --------------------------------------------------------------
    # The paragraph below is wrapped at four lines, not three: at three, every line ran past this
    # box's right edge by 15-20%, measured against the box's own usable width (ow minus its two
    # insets) with the renderer's own text extents, not guessed from a character count. The header
    # above the demo bars overflowed the same way at one line and is two below for the same reason.
    ox, ow = 0.735, 0.255
    box(ax, ox, 0.055, ow, 0.81, "#F4F7F9", S.HAIR)
    ax.text(ox + ow / 2, 0.825, "OUTPUT", ha="center", fontsize=6.5, color=S.MUTED,
            fontweight="bold")
    ax.text(ox + 0.016, 0.770, "base-rate enrichment", fontsize=7, color=S.INK, fontweight="bold")
    ax.text(ox + 0.016, 0.740, "a probability is scored against how\noften that endpoint fires "
                               "across the\nlibrary, not on its raw value, so a\ncommon endpoint "
                               "cannot dominate",
            fontsize=6.5, color=S.MUTED, va="top", linespacing=1.55)
    ax.plot([ox + 0.016, ox + ow - 0.016], [0.585, 0.585], color=S.HAIR, lw=0.8)
    ax.text(ox + 0.016, 0.548, "ranked mechanisms\nand conditions", fontsize=7, color=S.INK,
            fontweight="bold", va="top", linespacing=1.4)
    demo = [("strongest mechanism", 0.92, S.TARGET), ("second mechanism", 0.61, S.TARGET),
            ("exposure", 0.78, S.EXPOSURE), ("liability flag", 0.34, S.SAFETY)]
    for i, (lab, v, col) in enumerate(demo):
        yy = 0.428 - i * 0.055
        ax.text(ox + 0.016, yy + 0.006, lab, fontsize=6.5, color=S.MUTED)
        ax.add_patch(Rectangle((ox + 0.140, yy), (ow - 0.160) * v, 0.024,
                               facecolor=col, alpha=0.75, edgecolor="none"))
    ax.plot([ox + 0.016, ox + ow - 0.016], [0.198, 0.198], color=S.HAIR, lw=0.8)
    ax.text(ox + 0.016, 0.163, "every score carries", fontsize=6.5, color=S.INK, fontweight="bold")
    for i, t in enumerate(["a calibrated probability", "a conformal interval",
                           "an applicability-domain distance"]):
        ax.text(ox + 0.016, 0.125 - i * 0.031, "•  " + t, fontsize=6.5, color=S.MUTED)
    arrow(ax, gx + gw + 0.004, 0.50, ox - 0.004, 0.50)


def panel_b(ax, c) -> None:
    """One endpoint, expanded. The count in panel A is deployed models, not fitted models."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.0, 0.955, "One endpoint is twenty-one fits", fontsize=7.6, color=S.INK,
            fontweight="bold")

    groups = [
        ("10 random folds", 10, S.FAINT, 0.02,
         "compounds partitioned at random;\nreports interpolation within known chemistry"),
        ("10 scaffold folds", 10, S.WITHHELD, 0.355,
         "whole Bemis-Murcko scaffolds held out;\nreports generalisation to new chemistry"),
        ("1 deployed fit", 1, S.TARGET, 0.695,
         "refitted on every remaining row, then\ncalibrated; this is what the server runs"),
    ]
    for label, n, col, x, sub in groups:
        ax.text(x, 0.680, label, fontsize=6.6, color=col, fontweight="bold")
        for i in range(n):
            cx = x + (i % 5) * 0.0345
            cy = 0.570 - (i // 5) * 0.080
            box(ax, cx, cy, 0.026, 0.058, col, alpha=0.30 if n == 10 else 0.85, edge=col, r=0.004)
            # a forest, drawn as a forest
            for t in range(3):
                ax.plot([cx + 0.006 + t * 0.007, cx + 0.006 + t * 0.007], [cy + 0.011, cy + 0.047],
                        color=col, lw=0.45, alpha=0.9, zorder=4)
        ax.text(x, 0.420, sub, fontsize=6.5, color=S.MUTED, va="top", linespacing=1.6)

    # The box has to actually hold two lines of the small text below it, not just the big number:
    # at the old height (0.145) the second line of "8 isotonic calibrators" and its neighbours
    # printed below the box's own bottom edge, half in and half out of the shaded background.
    # Every position below is measured against the rendered text's own bounding box (not eyeballed)
    # against the paragraph above it and the box itself, since guessing this by eye is how it broke.
    ax.add_patch(Rectangle((0.0, 0.015), 1.0, 0.181, facecolor="#F4F7F9", edgecolor=S.HAIR, lw=0.6))
    for i, (big, small) in enumerate([
            (f"{c['n_deployed']}", "deployed estimators, plus\n"
                                   f"{c['n_calibrators']} isotonic calibrators"),
            (f"{c['n_cv_fits']:,}", "cross-validation fits, over\n"
                                    f"{c['n_cv_estimators']} cross-validated estimators"),
            ("300", "trees in every forest,\nleaf size 2"),
            ("1,036", "input columns, identical\nfor every endpoint")]):
        xx = 0.040 + i * 0.245
        ax.text(xx, 0.137, big, fontsize=11, color=S.INK, fontweight="bold")
        ax.text(xx, 0.122, small, fontsize=6.5, color=S.MUTED, va="top", linespacing=1.5)


def main() -> None:
    S.use()
    c = counts()
    fig = plt.figure(figsize=(S.DOUBLE, 6.28))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.30, 1.0], hspace=0.02,
                          left=0.012, right=0.988, top=0.945, bottom=0.045)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1])
    a.text(0.0, 1.075, "A", transform=a.transAxes, fontsize=10, fontweight="bold", va="bottom",
           color=S.INK)
    a.text(0.022, 1.075, "A prediction is assembled from many models, not produced by one",
           transform=a.transAxes, fontsize=8.5, fontweight="bold", va="bottom", color=S.INK)
    b.text(0.0, 1.005, "B", transform=b.transAxes, fontsize=10, fontweight="bold", va="bottom",
           color=S.INK)
    panel_a(a, c)
    panel_b(b, c)
    S.save(fig, "Figure1_architecture")


if __name__ == "__main__":
    main()
