"""Figure 5. Recovering the measured negative class, and what it changed.

Public bioactivity databases record what bound. A compound tested and found inactive is often stored
only as a censored bound, "IC50 > 10 uM", and the standard query that filters on a pChEMBL value
discards exactly those rows. The panel was therefore trained on a positive class drawn from
measurement and a negative class drawn largely from property-matched decoys, and 35 of the 60
endpoints then in the panel sat above 90 per cent active (panel B computes this figure directly from
expansion_inactives.csv rather than restating it, so it cannot go stale the way this sentence did).
That is a property of the query, not of the chemistry.

This figure shows the correction. Panel A: a bound settles the label whenever the whole interval
falls on one side of the cut, so ">10 uM" is a measured non-binder and only bounds that straddle the
cut are undecidable. Panel B: what that recovered, per endpoint. Panel C: the effect on the models.

The scaffold split gains most, which is the direction that makes sense. A model taught its negative
class by decoys learns what a decoy looks like; one taught by compounds that were assayed and did
not bind learns where activity stops, and that is what transfers to unfamiliar chemistry.

Reads results/tables/expansion_inactives.csv, rf_cv_summary.csv, rf_cv_summary_pre_inactives.csv.

Run:  python src/sai_cns/figures/fig05_negative_class.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

TAB = ROOT / "results" / "tables"
CUT = 5.0                # pChEMBL at or below which a compound is inactive under the label rule


def panel_a(ax) -> None:
    """When a bound settles a label and when it does not."""
    ax.set_xlim(3.4, 9.6); ax.set_ylim(0, 1)
    ax.set_yticks([]); ax.set_xlabel("pChEMBL  (higher is more potent)")
    for side in ("left", "right"):
        ax.spines[side].set_visible(False)

    ax.axvspan(3.4, CUT, color=S.WARN, alpha=0.055, zorder=0)
    ax.axvspan(6.0, 9.6, color=S.TARGET, alpha=0.055, zorder=0)
    ax.axvspan(CUT, 6.0, color=S.FAINT, alpha=0.10, zorder=0)
    ax.axvline(CUT, color=S.MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.axvline(6.0, color=S.MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(4.2, 0.955, "inactive", ha="center", fontsize=6.5, color=S.WARN, fontweight="bold")
    ax.text(5.5, 0.955, "undecided", ha="center", fontsize=6.5, color=S.MUTED, fontweight="bold")
    ax.text(7.8, 0.955, "active", ha="center", fontsize=6.5, color=S.TARGET, fontweight="bold")

    # Both notes below, measured against the room actually open beside them (pChEMBL 3.60 to the
    # axis's own right edge at 9.6), ran about a sixth wider than that: with nothing to clip them,
    # they printed straight across the vertical seam into panel B's own y-axis label, the two
    # panels' text interleaved.
    rows = [
        ("exact value, pChEMBL 8.1", 8.1, None, S.TARGET, "an active, as it always was"),
        ("\"IC50 > 10 uM\"", 5.0, "lt", S.WARN,
         "whole interval below the cut: a measured non-binder,\nand the class that was missing"),
        ("\"IC50 > 100 nM\"", 7.0, "lt", S.MUTED,
         "interval spans both classes: undecidable,\nand discarded rather than guessed"),
    ]
    # The "exact value" row sits close enough to the undecided/active boundary at pChEMBL 6 that a
    # dash of that vertical line landed inside a letter of the label, reading as a typo ("exaot")
    # rather than a line crossing behind text. A halo, the same fix used for a leader line crossing
    # its own label elsewhere in this figure set, hides whatever passes behind any of these three.
    # zorder inside the bbox dict, not just on the text: a Text's bbox patch defaults to zorder 1
    # regardless of the text's own zorder, which sits below axisbelow-grid (2) and an explicit
    # axvline (also 2 here); without it the halo is painted, then drawn over by the line anyway.
    halo = dict(boxstyle="round,pad=0.05", facecolor=S.PAPER, edgecolor="none", alpha=0.92, zorder=5)
    for i, (label, x, arrow, col, note) in enumerate(rows):
        y = 0.755 - i * 0.245
        if arrow == "lt":
            ax.add_patch(FancyArrowPatch((x, y), (3.60, y), arrowstyle="-|>", mutation_scale=7,
                                         color=col, lw=1.2, shrinkA=0, shrinkB=0, zorder=3))
            ax.plot([x], [y], "|", ms=8, color=col, mew=1.4, zorder=4)
            ax.text(x + 0.14, y, label, fontsize=6.5, color=col, fontweight="bold", va="center",
                    zorder=5, bbox=halo)
        else:
            ax.plot([x], [y], "o", ms=5.0, mfc=col, mec="white", mew=0.7, zorder=4)
            ax.text(x - 0.16, y, label, fontsize=6.5, color=col, fontweight="bold", va="center",
                    ha="right", zorder=5, bbox=halo)
        ax.text(3.60, y - 0.088, note, fontsize=6.5, color=S.MUTED, va="center", linespacing=1.6)

    S.strip(ax, x=True, y=False)
    ax.text(0.0, 1.055, "a bound settles the label whenever the whole interval falls on one side "
                        "of the cut",
            transform=ax.transAxes, fontsize=6.5, color=S.MUTED, va="bottom")


def panel_b(ax, exp) -> None:
    """Per-endpoint class balance, before and after."""
    d = exp[(exp.status == "merged") & exp.pct_active_before.notna()].copy()
    d = d.sort_values("pct_active_before", ascending=False).reset_index(drop=True)
    ys = np.arange(len(d))

    for y, r in zip(ys, d.itertuples()):
        ax.plot([r.pct_active_after, r.pct_active_before], [y, y], color=S.HAIR, lw=1.5,
                solid_capstyle="round", zorder=1)
    ax.plot(d.pct_active_before, ys, "o", ms=2.5, mfc=S.FAINT, mec="none", zorder=3,
            label="before recovery")
    ax.plot(d.pct_active_after, ys, "o", ms=2.5, mfc=S.TARGET, mec="none", zorder=3,
            label="after recovery")
    ax.axvline(90, color=S.WARN, lw=0.8, ls=(0, (3, 2)), zorder=2)

    n_before = int((d.pct_active_before > 90).sum())
    n_after = int((d.pct_active_after > 90).sum())
    ax.set_yticks([]); ax.set_xlabel("per cent of rows labelled active")
    ax.set_ylabel(f"{len(d)} endpoints extended,\nordered by prior imbalance", linespacing=1.6)
    ax.set_xlim(30, 102); ax.set_ylim(-1, len(d))
    S.strip(ax, x=True, y=False)
    ax.legend(loc="upper left", handletextpad=0.3, borderpad=0.2, bbox_to_anchor=(0.0, 0.40))
    ax.text(0.02, 0.185, f"above 90 per cent active:\n{n_before} endpoints before, "
                         f"{n_after} after\n{int(d.added.sum()):,} measured non-binders added",
            transform=ax.transAxes, fontsize=6.5, color=S.INK, ha="left", va="top",
            linespacing=1.8)


def panel_c(ax, now, before) -> None:
    """The effect on cross-validated performance, per endpoint, in both directions.

    The comparison is against the panel as it stood immediately before this merge, recovered from
    version control so that the only difference between the two states is the contents of the
    tables. Aggregate bars are not used here: they hid that classification moves down and regression
    moves up, and a mean over four quantities of different sign says less than the endpoints do.
    """
    m = before.merge(now, on=["endpoint", "task", "split"], suffixes=("_b", "_a"))
    rows = []
    for _, r in m.iterrows():
        col = "roc_auc_mean" if r["task"] == "classification" else "r2_mean"
        b, a = r[f"{col}_b"], r[f"{col}_a"]
        if pd.notna(b) and pd.notna(a):
            rows.append((r["endpoint"], r["task"], r["split"], a - b))
    d = pd.DataFrame(rows, columns=["endpoint", "task", "split", "delta"])

    order = (d[d.split == "scaffold"].set_index("endpoint").delta.sort_values().index.tolist())
    order += [e for e in d.endpoint.unique() if e not in order]
    xs = np.arange(len(order))
    for split, off, mk in (("random", -0.16, "o"), ("scaffold", 0.16, "o")):
        sub = d[d.split == split].set_index("endpoint").reindex(order)
        cols = [S.FAINT if split == "random" else S.WITHHELD] * len(order)
        ax.bar(xs + off, sub.delta.fillna(0), 0.30, color=cols, edgecolor="none",
               label=f"{split} 10-fold")

    ax.axhline(0, color=S.MUTED, lw=0.8)
    ax.set_xticks(xs)
    # antioxidant_DPPH is wider than its bar slot on one line and was bleeding into its neighbours;
    # breaking on the underscore keeps every label within its own column regardless of name length.
    ax.set_xticklabels([f"{e.replace('_', chr(10))}\n"
                        f"{'R2' if e in set(d[d.task=='regression'].endpoint) else 'AUROC'}"
                        for e in order], fontsize=6.5, linespacing=1.6)
    ax.set_ylabel("change after recovery\n(after minus before)", linespacing=1.6)
    S.strip(ax)
    ax.legend(loc="upper left", handletextpad=0.4, borderpad=0.2, ncol=2)

    cls = d[d.task == "classification"].delta
    reg = d[d.task == "regression"].delta
    ax.text(0.995, 0.95, f"classification  median {cls.median():+.4f}\n"
                         f"regression      median {reg.median():+.4f}",
            transform=ax.transAxes, fontsize=6.5, color=S.INK, ha="right", va="top",
            family="monospace", linespacing=1.8)


def main() -> None:
    S.use()
    exp = pd.read_csv(TAB / "expansion_inactives.csv")
    now = pd.read_csv(TAB / "rf_cv_summary.csv")
    before = pd.read_csv(TAB / "rf_cv_summary_pre_expansion.csv")

    fig = plt.figure(figsize=(S.DOUBLE, 6.16))
    # hspace=0.32 left only 0.02 of figure height between A's x-axis label and C's header, measured
    # against their actual rendered extents; the two visibly ran into each other.
    gs = fig.add_gridspec(2, 2, height_ratios=[0.86, 1.0], hspace=0.45, wspace=0.24,
                          left=0.065, right=0.985, top=0.905, bottom=0.085)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, :])
    S.panel(a, "A", "what a censored measurement can settle", dx=-0.10, dy=1.13, gap=0.045)
    S.panel(b, "B", "class balance, before and after", dx=-0.12, dy=1.13, gap=0.045)
    S.panel(c, "C", "and what it did to the models", dx=-0.055, dy=1.10, gap=0.022)
    panel_a(a); panel_b(b, exp); panel_c(c, now, before)
    S.save(fig, "Figure5_negative_class")


if __name__ == "__main__":
    main()
