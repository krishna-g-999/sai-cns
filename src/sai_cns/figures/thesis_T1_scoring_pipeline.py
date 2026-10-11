"""Figure T1. What the system computes, from a structure to a reported condition.

The eleven manuscript figures describe the parts. None of them shows the transform that turns a
structure into the number a user reads, and that transform is the thesis's central object: a
calibrated probability is converted to a signed enrichment over the endpoint's own base rate, routed
through a curated graph, aggregated by a maximum rather than a sum, multiplied by an exposure gate,
and suppressed below a reporting threshold. Five operations, each of which changes what the number
means.

A. The chain, with the operation at each step and where it is argued in the thesis.
B. One compound through step two. Donepezil's eight core probabilities beside the base rate of the
   endpoint that produced each, because the same probability means opposite things at different base
   rates and the enrichment step exists for that reason.
C. Steps three and four. The targets the graph connects to Alzheimer's disease, the weighted signal
   of each, and the maximum that becomes the condition's score.
D. Step five. All sixteen conditions after gating, against the reporting threshold.

Donepezil is a training compound of the AChE endpoint, with a maximum Tanimoto of 1.000 to the
applicability reference. Its AChE probability is therefore recall and not prediction, and the figure
says so rather than presenting it as a forecast. That is the correct example for explaining what the
pipeline computes and the wrong one for judging how well it computes it; Figure 11 is the second
question, and the expected recall this compound's own distance implies is printed in panel D.

Every number is computed by importing app and scoring the structure, so this figure cannot disagree
with the server.

Run:  brainsafe_env/Scripts/python.exe src/sai_cns/figures/thesis_T1_scoring_pipeline.py
Out:  thesis/figures/FigureT1_scoring_pipeline.png and .pdf
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import style as S  # noqa: E402

OUTDIR = ROOT / "thesis" / "figures"
DONEPEZIL = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"
CORE = ["AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG", "BBB"]
# Registry identifiers are not names a reader should meet. NEURO in particular is not a protein:
# it is the neuroprotection axis, a percentile against measured chemistry rather than a binding
# model, and printing its internal token in a published figure invites a reader to look for a gene.
LABEL = {"AChE": "AChE", "BChE": "BChE", "BACE1": "BACE1", "GSK3B": "GSK-3β",
         "MAO_A": "MAO-A", "MAO_B": "MAO-B", "hERG": "hERG", "BBB": "BBB",
         "NLRP3": "NLRP3", "a7nAChR": "α7 nAChR", "NEURO": "neuroprotection axis"}
SHORT = {"Neuroprotection / oxidative stress": "Neuroprotection / ox. stress",
         "Amyotrophic lateral sclerosis": "ALS"}


def scored() -> dict:
    import app
    m = app.load_models()
    r = app.predict_all(DONEPEZIL, m)
    bbb, neuro, rows = app.disease_scores(r)
    dom = app.assess_domain(DONEPEZIL)
    core = [(ep, float(r["targets"][ep]), app.base_rate(ep), app.enrichment(ep, r["targets"][ep]))
            for ep in CORE if r["targets"].get(ep) is not None]
    ad = [(t, w, app.target_signal(r, neuro, t))
          for t, w in app.DISEASE_CONTRIB.get("Alzheimer's disease", [])]
    return {"bbb": bbb, "rows": rows, "core": core, "ad": ad, "dom": dom, "floor": 0.30}


def panel_a(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [
        ("structure", "SMILES,\ndesalted parent", S.MUTED, ""),
        ("features", "1,024 ECFP-4 bits\n+ 12 descriptors", S.MUTED, "Fig 2"),
        ("probability", "one calibrated\nforest per endpoint", S.TARGET, "Ch 3, 4"),
        ("enrichment", "signed distance\nabove base rate", S.BINDER, "Ch 7.2"),
        ("condition", "edge weight, then\nthe MAXIMUM", S.BINDER, "Ch 7.3, 7.4"),
        ("exposure", "$\\times$ barrier\nprobability", S.EXPOSURE, "Ch 7.5"),
        ("report", "suppress\nbelow 0.30", S.WITHHELD, "Ch 7.8"),
    ]
    n = len(steps)
    w, gap = 0.1235, 0.0193
    x = 0.0
    for i, (name, body, col, ref) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x, 0.50), w, 0.46,
                                    boxstyle="round,pad=0.005,rounding_size=0.010",
                                    linewidth=0.8, edgecolor=col, facecolor=col + "14", zorder=2))
        ax.text(x + w / 2, 0.865, name, ha="center", va="center",
                fontsize=S.pt(7.4), fontweight="bold", color=col)
        ax.text(x + w / 2, 0.675, body, ha="center", va="center",
                fontsize=S.pt(6.5), color=S.INK, linespacing=1.4)
        if ref:
            ax.text(x + w / 2, 0.455, ref, ha="center", va="top",
                    fontsize=S.pt(6.5), color=S.FAINT, style="italic")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.003, 0.73), (x + w + gap - 0.003, 0.73),
                                         arrowstyle="-|>", mutation_scale=7,
                                         linewidth=0.8, color=S.FAINT, zorder=1))
        x += w + gap
    ax.text(0.5, 0.10,
            r"$\tilde{S}_d(x)\;=\;\gamma_d(x)\cdot\max_{(t,w)\in G(d)}\;w\cdot"
            r"\max\!\left(0,\,E_t(\hat{q}_t(x))\right)$",
            ha="center", va="bottom", fontsize=S.pt(9), color=S.INK)


def panel_b(ax, F) -> None:
    core = F["core"]
    y = list(range(len(core)))[::-1]
    for i, (ep, p, br, e) in zip(y, core):
        col = S.GOOD if e > 0 else S.MUTED
        ax.plot([br, p], [i, i], color=col, linewidth=1.4, solid_capstyle="round", zorder=2)
        ax.plot([br], [i], marker="|", markersize=6, color=S.FAINT, zorder=3)
        ax.plot([p], [i], marker="o", markersize=4.2, color=col,
                markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        ax.text(1.06, i, f"{e:+.2f}", va="center", ha="left", fontsize=S.pt(6.8),
                color=col, fontweight="bold" if e > 0 else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[ep] for ep, *_ in core], fontsize=S.pt(7))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, len(core) - 0.3)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("calibrated probability", fontsize=S.pt(7))
    ax.text(1.06, len(core) - 0.35, "$E_t$", fontsize=S.pt(7.5), color=S.INK,
            fontweight="bold", ha="left", va="center")
    S.strip(ax, x=True, y=False)
    ax.text(0.0, -0.17, "tick = endpoint base rate,  dot = this compound",
            transform=ax.transAxes, fontsize=S.pt(6.5), color=S.MUTED, ha="left", va="top")


def panel_c(ax, F) -> None:
    ad = sorted(F["ad"], key=lambda t: t[1] * t[2])
    y = list(range(len(ad)))
    prod = [w * s for _, w, s in ad]
    best = max(prod) if prod else 0.0
    cols = [S.BINDER if p == best and p > 0 else S.HAIR for p in prod]
    ax.barh(y, prod, height=0.62, color=cols, edgecolor="none", zorder=2)
    for i, ((t, w, s), p) in enumerate(zip(ad, prod)):
        if p > 0:
            ax.text(p - 0.03, i, f"{w:.2f} × {s:.2f}", ha="right", va="center",
                    fontsize=S.pt(6.5), color="white", fontweight="bold")
        else:
            ax.plot([0.006], [i], marker="|", markersize=6.5, color=S.FAINT,
                    markeredgewidth=1.2, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL.get(t, t) for t, _, _ in ad], fontsize=S.pt(6.8))
    for lab, p in zip(ax.get_yticklabels(), prod):
        lab.set_color(S.INK if p > 0 else S.FAINT)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.55, len(ad) - 0.35)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("edge weight × engagement", fontsize=S.pt(7))
    ax.axvline(best, color=S.BINDER, linewidth=0.8, linestyle=(0, (2.5, 1.6)), zorder=3)
    ax.text(0.96, 2.4, "the maximum becomes\nthe condition score", ha="right", va="center",
            fontsize=S.pt(6.5), color=S.BINDER, fontweight="bold", linespacing=1.35)
    ax.text(0.0, -0.17, "tick at zero = engaged signal is exactly zero",
            transform=ax.transAxes, fontsize=S.pt(6.5), color=S.MUTED, ha="left", va="top")
    S.strip(ax, x=True, y=False)


def panel_d(ax, F) -> None:
    order = sorted(F["rows"], key=lambda d: d["gated"])
    y = list(range(len(order)))
    floor = F["floor"]
    for i, d in enumerate(order):
        g = d["gated"]
        above = g >= floor
        ax.barh(i, g, height=0.6, color=S.EXPOSURE if above else S.HAIR,
                edgecolor="none", zorder=2)
        if above:
            ax.text(g + 0.015, i, f"{g:.2f}", va="center", ha="left",
                    fontsize=S.pt(6.5), color=S.INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT.get(d["disease"], d["disease"]) for d in order],
                       fontsize=S.pt(6.5))
    for lab, d in zip(ax.get_yticklabels(), order):
        lab.set_color(S.INK if d["gated"] >= floor else S.FAINT)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.55, len(order) - 0.35)
    ax.set_xticks([0, 0.5, 1.0])
    ax.axvline(floor, color=S.WITHHELD, linewidth=0.9, zorder=4)
    ax.text(0.0, -0.17, "orange line = the 0.30 reporting threshold",
            transform=ax.transAxes, fontsize=S.pt(6.5), color=S.WITHHELD, ha="left", va="top")
    ax.set_xlabel("gated condition score", fontsize=S.pt(7))
    S.strip(ax, x=True, y=False)
    dom = F["dom"]
    er = dom.get("expected_recall") or {}
    ax.text(0.97, 5.6,
            f"barrier gate  {F['bbb']:.3f}\n"
            f"max Tanimoto  {dom.get('max_sim', float('nan')):.3f}\n"
            f"expected recall  {er.get('recall', float('nan')):.3f}",
            ha="right", va="center", fontsize=S.pt(6.5), color=S.MUTED, linespacing=1.6)


def main() -> None:
    S.use()
    F = scored()
    fig = plt.figure(figsize=(S.DOUBLE, 6.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.40, 1.0], width_ratios=[1.0, 0.80, 1.18],
                          hspace=0.20, wspace=0.72,
                          left=0.078, right=0.955, top=0.945, bottom=0.145)
    a = fig.add_subplot(gs[0, :])
    b = fig.add_subplot(gs[1, 0])
    c = fig.add_subplot(gs[1, 1])
    d = fig.add_subplot(gs[1, 2])

    panel_a(a)
    panel_b(b, F)
    panel_c(c, F)
    panel_d(d, F)

    S.panel(a, "A", "five operations, each changing what the number means",
            dx=-0.046, dy=1.02, gap=0.024)
    S.panel(b, "B", "probability against base rate", dx=-0.38, dy=1.05, gap=0.125)
    S.panel(c, "C", "one condition's targets", dx=-0.42, dy=1.05, gap=0.155)
    S.panel(d, "D", "what is reported", dx=-0.56, dy=1.05, gap=0.098)

    S.note(fig, "Worked on donepezil. It is a training compound of the AChE endpoint at a maximum "
                "Tanimoto of 1.000, so its AChE probability is recall rather than prediction. This "
                "figure explains what the pipeline computes; how well it computes it on novel "
                "chemistry is Figure 11.", y=0.028)
    S.save(fig, "FigureT1_scoring_pipeline", outdir=OUTDIR)


if __name__ == "__main__":
    main()
