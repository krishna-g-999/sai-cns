"""Figure 2 of the manuscript. Every deployed estimator, what it predicts, and what stands behind it.

A multi-endpoint server is usually described by a panel mean, which is the one number that cannot be
checked. This figure shows the panel itself: one mark per estimator, positioned by the evidence
behind it and the performance it achieves, so a reader can see at once that the panel is not uniform
and can find any individual target rather than taking an average on trust.

Three things are encoded, and no more, because a figure that encodes five variables communicates
none:

  horizontal   the size of the training set, on a log axis, because it spans two orders of magnitude
               (37 to 15,831 rows as this docstring was last checked against the panel; read the axis
               of the actual figure, which is regenerated from MODEL_INVENTORY.csv on every build, for
               the current span) and a linear axis would collapse nine tenths of the panel. Binder
               training sets include property-matched decoys and the others do not, so the axis says
               so rather than implying that every row is a measurement
  vertical     the performance actually claimed for that estimator, on the split that is claimed
  colour       the model family, using the same four colours as every other figure in the set

Withdrawn estimators are drawn, in the colour used for a failure throughout, because a panel that
shows only what survived is a selection rather than an inventory.

The lower panel is the same population as a distribution, which answers the question the scatter
raises: how much of the panel sits where. Reading the two together is the point.

Everything is read from results/tables/MODEL_INVENTORY.csv, which is built directly from the
estimators on disk and carries the date it was built.

Run:  python src/sai_cns/figures/fig09_model_atlas.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

INV = ROOT / "results" / "tables" / "MODEL_INVENTORY.csv"

FAMILY_COLOUR = {"binder": S.BINDER, "target": S.TARGET,
                 "exposure": S.EXPOSURE, "safety": S.SAFETY}
FAMILY_LABEL = {"binder": "binder classifier", "target": "target potency and activity",
                "exposure": "exposure and ADME", "safety": "safety liability"}


def load() -> pd.DataFrame:
    if not INV.exists():
        raise SystemExit("run src/sai_cns/analysis/build_model_inventory.py first")
    d = pd.read_csv(INV)
    # the performance a reader would quote: the scaffold figure where there is one, since that is
    # the number the manuscript claims, and it is the harder of the two
    d["score"] = d.scaffold_split.where(d.scaffold_split.notna(), d.random_split)
    d["n"] = pd.to_numeric(d.n_train, errors="coerce")
    return d.dropna(subset=["score", "n"])


def panel_a(ax, d) -> None:
    """One mark per estimator.

    Marker shape carries the metric. AUROC and R2 both run to 1.0 and are not the same quantity: 0.5
    is chance for one and a respectable fit for the other. Drawing them on a shared axis without
    distinguishing them would invite a comparison that is not valid, so the shape says which is
    which and the axis label says they are not comparable.
    """
    for fam, g in d.groupby("family"):
        col = FAMILY_COLOUR.get(fam, S.MUTED)
        for task, marker in (("classification", "o"), ("regression", "s")):
            h = g[(g.task == task) & g.deployed]
            if len(h):
                ax.scatter(h.n, h.score, s=26 if task == "classification" else 22, marker=marker,
                           facecolor=col, edgecolor="white", linewidth=0.6, alpha=0.9, zorder=3)
        gone = g[~g.deployed]
        if len(gone):
            ax.scatter(gone.n, gone.score, s=48, marker="o", facecolor="none", edgecolor=S.WARN,
                       linewidth=1.3, zorder=4)

    # name every withdrawal, the weakest few, and the anchors a reader will look for. A fixed
    # vertical step only ever avoided the previously-placed LABELS, not the many unlabelled markers
    # in the dense high-performance cluster, which is exactly where BACE1 and hERG sit; adjustText
    # repels every label from every marker (not just the ones it names) and draws a leader line to
    # its own point whenever it has to move one to do it.
    show = set(d[~d.deployed].model) | {"BBB", "hERG", "BACE1"} | set(d.nsmallest(3, "score").model)
    rows = d[d.model.isin(show)]
    # A halo behind each label, as fig07_binder_panel.py uses for the same adjustText leader lines:
    # without it, a line that ends up passing almost underneath its own text (plasma_protein_binding,
    # clearance_hepatocyte) draws through the letters rather than stopping at them.
    # zorder must be set on the bbox dict itself: a Text's bbox patch defaults to zorder 1
    # regardless of the text's own zorder, which sits below the axisbelow grid (2) — the reason the
    # first version of this halo still let the y=0.4 gridline show straight through two labels.
    # pad=0.05 was not enough to cover BBB's own marker, which sits almost exactly under the label's
    # first letter on this log-x axis (1.05x is a small absolute shift wherever n itself is small).
    halo = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=1.0, zorder=5)
    texts = [ax.text(r["n"] * 1.08, r["score"],
                     r["model"].replace("_binder", "").replace("adme_", ""),
                     fontsize=S.pt(6.5), color=S.WARN if not r["deployed"] else S.INK, bbox=halo)
             for _, r in rows.iterrows()]
    adjust_text(texts, x=d.n.to_numpy(), y=d.score.to_numpy(), ax=ax,
                expand_text=(1.08, 1.25), expand_points=(1.8, 2.0), force_points=0.7,
                arrowprops=dict(arrowstyle="-", color=S.HAIR, lw=0.7, shrinkA=14, shrinkB=18,
                                zorder=1))
    # BBB's connector still crossed its own label after the halo and shrink fix above: adjustText's
    # fallback annotate (used whenever the axes transform doesn't support FancyArrowPatch, per the
    # warning this prints) draws its arrow last regardless of the zorder passed to arrowprops, so it
    # can still land on top of the halo. Redrawing every text and its halo after the arrows exist,
    # rather than relying on zorder to hold across that fallback, puts letters back on top for good.
    for t in texts:
        t.set_zorder(6)
        if t.get_bbox_patch() is not None:
            t.get_bbox_patch().set_zorder(6)

    ax.set_xscale("log")
    ax.set_xlabel("compounds in the training set\n"
                  "(binder sets include property-matched decoys; all others are measured only)")
    ax.set_ylabel("AUROC (circles) or R² (squares)\nnot comparable to one another",
                  linespacing=1.7)
    ax.set_ylim(0, 1.06)
    S.strip(ax, x=True, y=True)
    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=FAMILY_COLOUR[f],
                      markeredgecolor="white", markersize=5.5) for f in FAMILY_COLOUR]
    labels = [FAMILY_LABEL[f] for f in FAMILY_COLOUR]
    handles += [Line2D([], [], marker="s", linestyle="none", markerfacecolor=S.FAINT,
                       markeredgecolor="white", markersize=5),
                Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                       markeredgecolor=S.WARN, markeredgewidth=1.3, markersize=6.5)]
    labels += ["regression, scored by R²", "withdrawn after specificity testing"]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False,
              handletextpad=0.4, labelspacing=0.55, fontsize=S.pt(6.5))
    ax.text(0.0, 1.02, "each mark is one estimator; the panel spans two orders of magnitude in "
                       "training-set size",
            transform=ax.transAxes, fontsize=S.pt(6.5), color=S.MUTED, va="bottom")


def panel_b(ax, d) -> None:
    """The same population as a distribution, by family."""
    fams = ["binder", "target", "exposure", "safety"]
    fams = [f for f in fams if (d.family == f).any()]
    for i, fam in enumerate(fams):
        g = d[d.family == fam]
        col = FAMILY_COLOUR.get(fam, S.MUTED)
        jitter = (np.random.default_rng(42).random(len(g)) - 0.5) * 0.34
        ax.scatter(g.score, np.full(len(g), i) + jitter, s=16, facecolor=col, alpha=0.75,
                   edgecolor="none", zorder=3)
        med = float(g.score.median())
        ax.plot([med, med], [i - 0.30, i + 0.30], color=S.INK, lw=1.4, zorder=4)
        ax.text(1.015, i, f"n={len(g)}   median {med:.3f}", fontsize=S.pt(6.5), va="center",
                color=S.MUTED)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([FAMILY_LABEL[f] for f in fams], fontsize=S.pt(6.5))
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, len(fams) - 0.4)
    ax.set_xlabel("performance on the split that is claimed")
    S.strip(ax, x=True, y=False)
    ax.text(0.0, 1.06, "the vertical rule is the family median; the spread is the panel, "
                       "and it is what a single mean hides",
            transform=ax.transAxes, fontsize=S.pt(6.5), color=S.MUTED, va="bottom")


def main() -> None:
    S.use()
    d = load()
    date = str(d.inventory_date.iloc[0])
    fig = plt.figure(figsize=(S.DOUBLE, 5.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.58,
                          left=0.100, right=0.735, top=0.905, bottom=0.098)
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    S.panel(a, "A", "the whole panel, one mark per estimator", dx=-0.095, dy=1.075, gap=0.030)
    S.panel(b, "B", "and how it is distributed", dx=-0.095, dy=1.115, gap=0.030)
    panel_a(a, d); panel_b(b, d)
    # pKa has no cross-validation record, so it has no score to plot and is absent here
    total = len(pd.read_csv(INV))
    missing = total - len(d)
    note = f"{len(d)} of {total} estimators, inventory {date}"
    if missing:
        note += f" ({missing} without a cross-validation record, not plotted)"
    fig.text(0.735, 0.002, note, ha="right", fontsize=S.pt(6.5), color=S.FAINT)
    S.save(fig, "Figure9_model_atlas")


if __name__ == "__main__":
    main()
