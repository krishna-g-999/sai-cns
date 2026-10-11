"""Figure 3. Two cross-validation schemes, and the gap between them.

A single cross-validated AUROC is the number most easily overstated in this field, because the
partition decides what it means. Splitting compounds at random puts close analogues of a test
compound in the training set, so the score reports interpolation inside chemistry the model has
already seen. Splitting on Bemis-Murcko scaffolds withholds a whole structural class, so the score
reports what happens to a compound the model has no near neighbour for.

Both are reported here for every endpoint, and the distance between them is the honest statement of
how far the model travels. Panel A shows why they differ. Panel B pairs them per endpoint. Panel C
shows all ten folds behind each mean, because a mean over ten folds with a wide spread and a mean
over ten tight folds are not the same claim.

Reads results/tables/rf_cv_summary.csv and rf_cv_folds.csv.

Run:  python src/sai_cns/figures/fig03_cv_design.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

TAB = ROOT / "results" / "tables"


def panel_a(ax) -> None:
    """The same compounds, partitioned two ways."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    rng = np.random.default_rng(11)

    # Four scaffold classes, drawn as spatial clusters, because a scaffold class is a region of
    # chemical space and that is exactly what the split withholds.
    centres = [(0.20, 0.70), (0.44, 0.76), (0.30, 0.40), (0.55, 0.48)]
    pts, cls = [], []
    for k, (cx, cy) in enumerate(centres):
        n = [9, 8, 8, 7][k]
        p = rng.normal([cx, cy], 0.052, size=(n, 2))
        pts.append(p); cls += [k] * n
    pts = np.vstack(pts); cls = np.array(cls)

    for j, (title, mode) in enumerate([("random 10-fold", "random"),
                                       ("scaffold 10-fold", "scaffold")]):
        x0 = 0.02 + j * 0.51
        ax.add_patch(Rectangle((x0, 0.18), 0.45, 0.66, facecolor="#FBFCFD",
                               edgecolor=S.HAIR, lw=0.7))
        ax.text(x0 + 0.225, 0.875, title, ha="center", fontsize=7, fontweight="bold",
                color=S.FAINT if mode == "random" else S.WITHHELD)
        if mode == "random":
            held = rng.permutation(len(pts))[: len(pts) // 10 + 2]
            held_mask = np.zeros(len(pts), bool); held_mask[held] = True
        else:
            held_mask = cls == 2                      # one whole scaffold class withheld

        # The leftmost cluster (cx=0.20) mapped to X = x0 + 0.06, so its disc (radius 0.088)
        # reached 0.028 past the box's own left edge at x0, and the rightmost (cx=0.55) reached
        # past the right edge by about as much; both were visibly clipped by the box border. The
        # offset and scale below keep every disc's edge inside [x0, x0+0.45], not just its centre.
        CX_OFF, CX_SCALE = 0.10, 0.74
        for k, (cx, cy) in enumerate(centres):
            ax.add_patch(Circle((x0 + CX_OFF + (cx - 0.20) * CX_SCALE, cy - 0.06), 0.088,
                                facecolor=S.HAIR, edgecolor="none", alpha=0.55, zorder=1))
        for i, (px, py) in enumerate(pts):
            X = x0 + CX_OFF + (px - 0.20) * CX_SCALE
            Y = py - 0.06
            if held_mask[i]:
                ax.plot(X, Y, "o", ms=3.6, mfc=S.WITHHELD, mec="white", mew=0.5, zorder=3)
            else:
                ax.plot(X, Y, "o", ms=3.0, mfc=S.EXPOSURE, mec="white", mew=0.4, alpha=0.65,
                        zorder=2)
        # Short enough to sit inside its own half: at 6.5 pt the previous wording ran past the
        # midline and the two captions met.
        ax.text(x0 + 0.225, 0.145,
                "held out: scattered\nthrough every class"
                if mode == "random" else
                "held out: one whole\nclass, absent entirely",
                ha="center", va="top", fontsize=6.5, color=S.MUTED, linespacing=1.7)

    ax.plot([], [], "o", ms=3.2, mfc=S.EXPOSURE, mec="white", label="in training this fold")
    ax.plot([], [], "o", ms=3.6, mfc=S.WITHHELD, mec="white", label="held out this fold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.055), ncol=2, handletextpad=0.3,
              columnspacing=1.4)
    # Left-aligned at the same (0.0, 1.015) as B's and C's subheading, not centred at a height of
    # its own: it was the one panel whose subheading did not line up with the other two.
    ax.text(0.0, 1.015, "grey discs are Bemis-Murcko scaffold classes,\n"
                        "computed on the desalted parent",
            transform=ax.transAxes, fontsize=6.5, color=S.MUTED, va="bottom", linespacing=1.6)


def panel_b(ax, cv) -> None:
    """Every endpoint, random score paired with its scaffold score."""
    c = cv[cv.task == "classification"]
    piv = c.pivot_table(index="endpoint", values="roc_auc_mean", columns="split")
    piv = piv.dropna().sort_values("scaffold")
    ys = np.arange(len(piv))

    for y, (ep, row) in zip(ys, piv.iterrows()):
        ax.plot([row["scaffold"], row["random"]], [y, y], color=S.HAIR, lw=2.6,
                solid_capstyle="round", zorder=1)
        ax.plot(row["random"], y, "o", ms=4.2, mfc=S.FAINT, mec="white", mew=0.6, zorder=3)
        ax.plot(row["scaffold"], y, "o", ms=4.6, mfc=S.WITHHELD, mec="white", mew=0.6, zorder=3)

    ax.set_yticks(ys); ax.set_yticklabels(piv.index, fontsize=6.5)
    ax.set_xlabel("AUROC, mean over 10 folds")
    ax.set_xlim(0.82, 1.0)
    ax.set_ylim(-0.8, len(piv) - 0.2)
    S.strip(ax, x=True, y=False)
    gap = (piv["random"] - piv["scaffold"])
    ax.plot([], [], "o", ms=4.2, mfc=S.FAINT, mec="white", label="random")
    ax.plot([], [], "o", ms=4.6, mfc=S.WITHHELD, mec="white", label="scaffold")
    ax.legend(loc="lower right", handletextpad=0.3, borderpad=0.2)
    ax.text(0.0, 1.015, f"median cost {gap.median():.3f} AUROC\n"
                        f"(range {gap.min():.3f} to {gap.max():.3f})",
            transform=ax.transAxes, fontsize=6.5, color=S.MUTED, va="bottom", linespacing=1.6)


def panel_c(ax, folds) -> None:
    """All ten folds behind each mean."""
    f = folds[(folds.task == "classification") & folds.roc_auc.notna()]
    order = (f[f.split == "scaffold"].groupby("endpoint").roc_auc.mean()
             .sort_values().index.tolist())
    for i, ep in enumerate(order):
        for j, (split, col, off) in enumerate([("random", S.FAINT, -0.17),
                                               ("scaffold", S.WITHHELD, 0.17)]):
            v = f[(f.endpoint == ep) & (f.split == split)].roc_auc.values
            if not len(v):
                continue
            ax.plot(v, np.full(len(v), i + off), "o", ms=2.4, mfc=col, mec="none", alpha=0.75,
                    zorder=2)
            ax.plot([v.mean()], [i + off], "|", ms=9, color=S.INK, mew=1.1, zorder=3)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6.5)
    ax.set_xlabel("AUROC, individual folds")
    ax.set_ylim(-0.7, len(order) - 0.3)
    S.strip(ax, x=True, y=False)
    ax.text(0.0, 1.015, "each point is one fold;\nthe rule is their mean",
            transform=ax.transAxes, fontsize=6.5, color=S.MUTED, va="bottom", linespacing=1.6)


def main() -> None:
    S.use()
    cv = pd.read_csv(TAB / "rf_cv_summary.csv")
    folds = pd.read_csv(TAB / "rf_cv_folds.csv")

    fig = plt.figure(figsize=(S.DOUBLE, 4.09))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 0.92, 0.92], wspace=0.42,
                          left=0.045, right=0.99, top=0.815, bottom=0.135)
    a, b, c = (fig.add_subplot(gs[i]) for i in range(3))
    S.panel(a, "A", "what each split withholds", dx=-0.02, dy=1.145, gap=0.065)
    S.panel(b, "B", "the cost of withholding it", dx=-0.30, dy=1.145, gap=0.075)
    S.panel(c, "C", "all folds, not only means", dx=-0.30, dy=1.145, gap=0.075)
    panel_a(a); panel_b(b, cv); panel_c(c, folds)
    S.save(fig, "Figure3_cv_design")


if __name__ == "__main__":
    main()
