"""Figure 11. External validation: an apparent temporal decay that is a composition effect.

The panel's headline scores come from partitions of one retrospective dataset. Partitioning by date
instead puts the test compounds beyond the reach of the training data in the way a user's compound
is, and the aggregate result looks like decay over time. It is not, and the four panels are ordered
to show why.

A. The aggregate gap. Per endpoint, AUROC under a size-matched random split against AUROC under the
   time split. The size match matters: a time split trains on less data as well as none of the
   future, so without a control at the same n a drop cannot be attributed to either. Points fall
   below the diagonal, which reads as prospective decay.
B. The same picture for sensitivity at the frozen operating point, where the gap is far larger.
C. Why. A random split of medicinal-chemistry data holds out mostly close analogues of its own
   training set, because the published record is series. A time split does not. The two splits are
   not testing comparable populations.
D. The resolution. Recall against distance from training chemistry, for three test sets built by
   unrelated rules: withheld by date, withheld at random, withheld by curator. They trace one curve,
   so recall is a function of chemical distance and not of how the set was held out.

Reads external_prospective.csv and external_novelty_strata.csv.

Run:  python src/sai_cns/figures/fig11_external_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

TAB = ROOT / "results" / "tables"
SHORT = {"below 0.40 (different chemotype)": "< 0.40\ndifferent\nchemotype",
         "0.40 to 0.55 (related series)": "0.40-0.55\nrelated\nseries",
         "0.55 to 0.70 (same series)": "0.55-0.70\nsame\nseries",
         "0.70 and above (close analogue)": "> 0.70\nclose\nanalogue"}
ORDER = list(SHORT)


def _paired(ax, d, xcol, ycol, label, lo):
    x = pd.to_numeric(d[xcol], errors="coerce")
    y = pd.to_numeric(d[ycol], errors="coerce")
    m = x.notna() & y.notna()
    x, y, names = x[m], y[m], d.endpoint[m]
    ax.plot([lo, 1.0], [lo, 1.0], color=S.FAINT, lw=0.8, ls="--", zorder=1)
    ax.scatter(x, y, s=17, c=S.TARGET, edgecolor="white", linewidth=0.5, zorder=3)
    # A fixed offset (3, -7) points put GluN2B's label on the axis spine, since its own y sits
    # close enough to the axis's bottom (lo) that moving down 7 points landed at or past it; simply
    # flipping labels near that edge upward then ran two of them into each other in panel A, where
    # a7nAChR and Nav1_5 are only ~0.07 apart in y. adjustText, already used for this same problem
    # elsewhere in the figure set, repels each label from the others and from every point, not just
    # the one it names.
    halo = dict(boxstyle="round,pad=0.05", facecolor="white", edgecolor="none", alpha=1.0, zorder=5)
    idx = (x - y).sort_values(ascending=False).head(3).index
    texts = [ax.text(x[i] + 0.01, y[i] - 0.01, names[i], fontsize=S.MIN_PT - 0.5, color=S.MUTED,
                     bbox=halo) for i in idx]
    adjust_text(texts, x=x.to_numpy(), y=y.to_numpy(), ax=ax,
                expand_text=(1.1, 1.3), expand_points=(1.5, 1.8), force_points=0.6,
                arrowprops=dict(arrowstyle="-", color=S.HAIR, lw=0.6, shrinkA=2, shrinkB=3,
                                zorder=1))
    # adjustText's fallback connector (used whenever the axes transform doesn't support
    # FancyArrowPatch) can draw on top of the label regardless of the zorder passed to
    # arrowprops; forcing the label and its halo above that after the fact holds regardless.
    for t in texts:
        t.set_zorder(6)
        if t.get_bbox_patch() is not None:
            t.get_bbox_patch().set_zorder(6)
    ax.set_xlim(lo, 1.0); ax.set_ylim(lo, 1.0); ax.set_aspect("equal")
    ax.set_xlabel(f"{label}, size-matched random split")
    ax.set_ylabel(f"{label}, time split")
    ax.text(0.03, 0.97, f"median gap {float(np.median(x - y)):+.3f}\nn = {len(x)} endpoints",
            transform=ax.transAxes, va="top", fontsize=S.MIN_PT, color=S.INK)
    S.strip(ax, x=True, y=True)


def panel_c(ax, s):
    """Composition of the two test sets: the reason the aggregate gap exists."""
    t = s[(s.source == "prospective") & (s.split == "time")].set_index("novelty_band")
    r = s[(s.source == "prospective") & (s.split == "random")].set_index("novelty_band")
    bands = [b for b in ORDER if b in t.index and b in r.index]
    tv = np.array([t.loc[b, "n_actives"] for b in bands], float)
    rv = np.array([r.loc[b, "n_actives"] for b in bands], float)
    tv, rv = 100 * tv / tv.sum(), 100 * rv / rv.sum()
    xs = np.arange(len(bands)); w = 0.38
    ax.bar(xs - w / 2, tv, w, color=S.TARGET, edgecolor="white", linewidth=0.6,
           label="withheld by date", zorder=2)
    ax.bar(xs + w / 2, rv, w, color=S.FAINT, edgecolor="white", linewidth=0.6,
           label="withheld at random", zorder=2)
    for x, v in zip(xs - w / 2, tv):
        ax.text(x, v + 1.6, f"{v:.0f}", ha="center", fontsize=S.MIN_PT - 0.5, color=S.INK)
    for x, v in zip(xs + w / 2, rv):
        ax.text(x, v + 1.6, f"{v:.0f}", ha="center", fontsize=S.MIN_PT - 0.5, color=S.MUTED)
    ax.set_xticks(xs); ax.set_xticklabels([SHORT[b] for b in bands], fontsize=S.MIN_PT - 0.5)
    ax.set_xlabel("maximum Tanimoto to the training actives")
    ax.set_ylabel("per cent of the test actives")
    ax.set_ylim(0, max(rv.max(), tv.max()) * 1.18)
    ax.legend(loc="upper center", handletextpad=0.35, borderpad=0.25)
    S.strip(ax, x=False, y=True)


def panel_d(ax, s):
    """One curve from three unrelated hold-out rules."""
    series = [("prospective", "time", "withheld by date", S.TARGET, "o", "-"),
              ("prospective", "random", "withheld at random", S.FAINT, "s", "--"),
              ("cross-provenance", "cross_source", "withheld by curator", S.SAFETY, "^", "-.")]
    for src, split, lab, col, mk, ls in series:
        d = s[(s.source == src) & (s.split == split)].set_index("novelty_band")
        bands = [b for b in ORDER if b in d.index and pd.notna(d.loc[b, "recall_at_threshold"])]
        if not bands:
            continue
        ax.plot(np.arange(len(ORDER))[[ORDER.index(b) for b in bands]],
                [d.loc[b, "recall_at_threshold"] for b in bands],
                ls, marker=mk, color=col, ms=4.6, lw=1.5, mec="white", mew=0.5, label=lab, zorder=3)
    ax.set_xticks(np.arange(len(ORDER)))
    ax.set_xticklabels([SHORT[b] for b in ORDER], fontsize=S.MIN_PT - 0.5)
    ax.set_xlabel("maximum Tanimoto to the training actives")
    ax.set_ylabel("recall at the frozen threshold")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", handletextpad=0.35, borderpad=0.25)
    S.strip(ax, x=False, y=True)


def main() -> None:
    S.use()
    d = pd.read_csv(TAB / "external_prospective.csv")
    d = d[d.status == "ok"].reset_index(drop=True)
    s = pd.read_csv(TAB / "external_novelty_strata.csv")

    fig = plt.figure(figsize=(S.DOUBLE, 6.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.30, wspace=0.42,
                          left=0.085, right=0.985, top=0.915, bottom=0.10)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0]); e = fig.add_subplot(gs[1, 1])
    S.panel(a, "A", "an apparent cost of prospectivity", dx=-0.20, dy=1.10, gap=0.055)
    S.panel(b, "B", "larger still at the operating point", dx=-0.20, dy=1.10, gap=0.055)
    S.panel(c, "C", "but the two splits test different chemistry", dx=-0.20, dy=1.11, gap=0.055)
    S.panel(e, "D", "and all three hold-outs trace one curve", dx=-0.20, dy=1.11, gap=0.055)
    _paired(a, d, "random_auroc_vs_measured_inactives", "time_auroc_vs_measured_inactives",
            "AUROC", 0.4)
    _paired(b, d, "random_sensitivity", "time_sensitivity", "sensitivity", 0.0)
    panel_c(c, s); panel_d(e, s)
    S.save(fig, "Figure11_external_validation")


if __name__ == "__main__":
    main()
