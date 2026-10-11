"""Figure 10: the evidence behind which targets became endpoints and which did not.

The question this answers is the one asked of any panel: why these targets and not others. The
honest answer has two halves, and only one of them is about biology.

The biological half is stated in the manuscript: each axis is present because CNS attrition happens
on it. That is a literature argument and no figure can settle it.

The half a figure *can* settle is the second: a target only becomes an endpoint if the measured data
can support a model that survives a scaffold split, and if the fitted model can then be given a
threshold that finds actives without firing on unrelated chemistry. Those are measurable conditions,
they were applied uniformly, and they rejected targets that were wanted on mechanism. This figure
shows where those conditions bite.

  A  data volume against what the model achieves, over the whole panel, so the reader can see the
     relationship rather than take the bar on trust
  B  the condition that actually decides deployment, which is not AUROC but whether a usable
     threshold exists, and the endpoints that failed it despite ranking well
  C  the same bar applied to 1,688 surveyed candidate targets, showing how few clear it

Reads: results/tables/MODEL_INVENTORY.csv, binder_cv_summary.csv, np_target_survey.csv,
       models_rf/binder_modes.json, data/endpoints/*.csv

Run:  python src/sai_cns/figures/fig10_endpoint_selection.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

TAB = ROOT / "results" / "tables"

# The reliability gate, imported rather than copied. Panel B draws it, so a copy that drifted would
# mislabel the figure rather than merely misconfigure it, which is what happened when this file held
# its own 0.60.
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
import panel  # noqa: E402

SENS_FLOOR, AUROC_FLOOR = panel.MIN_SENSITIVITY, panel.MIN_AUROC


def load():
    modes = json.loads((ROOT / "models_rf" / "binder_modes.json").read_text(encoding="utf-8"))
    counts = {Path(f).stem: len(pd.read_csv(f, usecols=["smiles"]))
              for f in glob.glob(str(ROOT / "data" / "endpoints" / "*.csv"))}
    cv = pd.read_csv(TAB / "binder_cv_summary.csv")
    cv = cv[cv.split == "scaffold"].set_index("endpoint")
    rows = []
    for ep, rec in modes.items():
        rows.append({
            "endpoint": ep,
            "compounds": counts.get(ep, np.nan),
            "cv_auroc": cv.roc_auc_mean.get(ep, np.nan),
            "auroc_inactives": rec.get("auroc_vs_measured_inactives"),
            "sensitivity": rec.get("sensitivity_at_threshold"),
            "deployed": bool(rec.get("deployed", True)),
        })
    return pd.DataFrame(rows).dropna(subset=["compounds"])


def panel_a(ax, d):
    """Data volume against cross-validated discrimination, over the whole panel."""
    for dep, col, lab in ((True, S.BINDER, "deployed"), (False, S.WARN, "withdrawn")):
        g = d[(d.deployed == dep) & d.cv_auroc.notna()]
        ax.scatter(g.compounds, g.cv_auroc, s=26, c=col, alpha=.85,
                   edgecolors="white", linewidths=.5, label=lab, zorder=3)
    g = d.dropna(subset=["cv_auroc"])
    # A log-linear trend, drawn only to make the direction visible, not as a model of anything.
    x = np.log10(g.compounds.astype(float))
    b, a = np.polyfit(x, g.cv_auroc, 1)
    xx = np.linspace(x.min(), x.max(), 80)
    # Clipped at 1.0: the fit is a straight line in log-space and would otherwise be drawn above
    # the maximum an AUROC can take, which reads as a claim rather than as a direction.
    ax.plot(10 ** xx, np.minimum(a + b * xx, 1.0), color=S.MUTED, lw=1.1, ls="--", zorder=2,
            label=f"trend: {b:+.3f} AUROC per 10x data")
    ax.set_xscale("log")
    ax.set_xlabel("measured compounds in the endpoint's own training table")
    ax.set_ylabel("scaffold-split AUROC\n(cross-validated)", linespacing=1.5)
    ax.set_ylim(0.35, 1.02)
    ax.axhline(0.5, color=S.HAIR, lw=1, zorder=1)
    ax.text(ax.get_xlim()[0] * 1.15, 0.515, "chance", fontsize=S.pt(6.5), color=S.FAINT)
    S.strip(ax, x=True, y=True)
    ax.legend(loc="lower right", fontsize=S.pt(6.5), handletextpad=.3, borderpad=.35)


def panel_b(ax, d):
    """What actually decides deployment: whether a usable threshold exists."""
    g = d.dropna(subset=["sensitivity", "auroc_inactives"])
    for dep, col, lab in ((True, S.BINDER, "deployed"), (False, S.WARN, "withdrawn")):
        h = g[g.deployed == dep]
        ax.scatter(h.auroc_inactives, h.sensitivity, s=26, c=col, alpha=.85,
                   edgecolors="white", linewidths=.5, label=lab, zorder=3)
    # The gate is two-sided, so it is drawn as the corner it is rather than as one line. This panel
    # previously drew a single line at 0.60, the floor used by the earlier training stages, which put
    # three deployed endpoints on the wrong side of a line labelled as the gate.
    ax.axhline(SENS_FLOOR, color=S.WITHHELD, lw=1.2, ls="--", zorder=2)
    ax.axvline(AUROC_FLOOR, color=S.WITHHELD, lw=1.2, ls="--", zorder=2)
    ax.text(0.365, SENS_FLOOR + 0.015, f"sensitivity floor {SENS_FLOOR:.2f}", fontsize=S.pt(6.5),
            color=S.WITHHELD)
    ax.text(AUROC_FLOOR + 0.008, -0.005, f"AUROC floor {AUROC_FLOOR:.2f}", fontsize=S.pt(6.5),
            color=S.WITHHELD)
    ax.text(0.365, 0.85, "a call is reliable only above both", fontsize=S.pt(6.5), color=S.FAINT)
    # Name the endpoints that discriminate well and are still not deployable: they are the argument.
    for r in g[(~g.deployed) & (g.auroc_inactives >= 0.75)].itertuples():
        ax.annotate(r.endpoint, (r.auroc_inactives, r.sensitivity), fontsize=S.pt(6.5),
                    color=S.WARN, textcoords="offset points", xytext=(6, -2))
    ax.set_xlabel("AUROC against compounds measured and found inactive")
    ax.set_ylabel("sensitivity at the deployed threshold,\nheld-out actives by scaffold",
                  linespacing=1.5)
    ax.set_xlim(0.35, 1.02); ax.set_ylim(-0.04, 1.04)
    S.strip(ax, x=True, y=True)
    ax.legend(loc="upper left", fontsize=S.pt(6.5), handletextpad=.3, borderpad=.35)


def panel_c(ax):
    """The same bar applied to every surveyed candidate target."""
    p = TAB / "np_target_survey.csv"
    if not p.exists():
        ax.axis("off")
        return None
    s = pd.read_csv(p)
    n_total, n_train = len(s), int(s.trainable.sum())
    bins = np.logspace(0, np.log10(max(s.compounds.max(), 10)), 34)
    ax.hist(s.loc[~s.trainable, "compounds"], bins=bins, color=S.HAIR, edgecolor="white",
            linewidth=.4, label=f"below the bar ({n_total - n_train:,})")
    ax.hist(s.loc[s.trainable, "compounds"], bins=bins, color=S.TARGET, edgecolor="white",
            linewidth=.4, label=f"trainable ({n_train})")
    ax.axvline(60, color=S.WARN, lw=1.3, ls="--", zorder=4)
    ax.text(66, ax.get_ylim()[1] * .72, "60 compounds,\n15 per class", fontsize=S.pt(6.5),
            color=S.WARN, linespacing=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("measured compounds available for the target")
    ax.set_ylabel("candidate targets", linespacing=1.5)
    S.strip(ax, x=True, y=True)
    ax.legend(loc="upper right", fontsize=S.pt(6.5), handletextpad=.3, borderpad=.35)
    return n_total, n_train


def main() -> None:
    S.use()
    d = load()
    fig = plt.figure(figsize=(11.2, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=.34, wspace=.26)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    panel_a(ax_a, d)
    panel_b(ax_b, d)
    surveyed = panel_c(ax_c)

    S.panel(ax_a, "A", "more data buys discrimination, with diminishing returns")
    S.panel(ax_b, "B", "but discrimination is not what decides deployment")
    S.panel(ax_c, "C", "the same bar, applied to every candidate target surveyed", dx=-0.045)

    out = S.save(fig, "Figure10_endpoint_selection")
    print(f"  wrote {out.relative_to(ROOT)}")
    if surveyed:
        print(f"  surveyed {surveyed[0]:,} candidate targets, {surveyed[1]} clear the data bar")
    print(f"  panel: {len(d)} trained, {int(d.deployed.sum())} deployed")


if __name__ == "__main__":
    main()
