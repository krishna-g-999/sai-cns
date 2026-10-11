"""Figure T4. The falsification suite: ten hypotheses stated so that they could fail.

Every other result in this project was produced by someone who wanted the tool to work. This suite
assumes each central claim is false and asks what evidence would prove it, pairing every hypothesis
with a null model capable of producing the same apparent success by accident. Four of the ten were
refuted and two weakened, and the refutations are the most valuable output: they cost the project a
claim about the knowledge graph, a claim about the exposure gate, a change to the interface, and four
withdrawn endpoints.

A. The ten verdicts, with the hypothesis stated as it was posed rather than as it turned out.
B. The leakage null. With labels permuted, the same pipeline on the same Bemis-Murcko folds returns
   chance on all sixteen endpoint-split combinations, against true AUROCs a third of a unit higher.
   The scaffold null matters most: whole scaffold classes could in principle carry enough
   class-frequency information for a label-free model to exploit the grouping, and they do not.
C. H10, the hypothesis the suite lacked until a thesis audit named the gap. The barrier model is the
   component the architecture is named for. Against descriptor rules and descriptor-only models on
   241 approved drugs it has never seen, the paired bootstrap interval for its margin crosses zero,
   which is why the verdict reads WEAKENED and not SUPPORTED.

Reads VERDICTS.csv, permutation_null.csv, rf_cv_summary.csv and H10_barrier_necessity.csv.

Run:  brainsafe_env/Scripts/python.exe src/sai_cns/figures/thesis_T4_falsification.py
Out:  thesis/figures/FigureT4_falsification.png and .pdf
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

OUTDIR = ROOT / "thesis" / "figures"
TAB = ROOT / "results" / "tables"
INV = ROOT / "inversion" / "results"
LABEL = {"AChE": "AChE", "BChE": "BChE", "BACE1": "BACE1", "GSK3B": "GSK-3β",
         "MAO_A": "MAO-A", "MAO_B": "MAO-B", "hERG": "hERG", "BBB": "BBB"}
SHORT = {
    "H1": "the disease score is informative",
    "H2": "the curated edge weights add value",
    "H3": "BBB gating discriminates between diseases",
    "H4": "specificity transfers to novel chemistry",
    "H5": "read-across beats a frequency baseline",
    "H6": "the disease scores match clinical indications",
    "H7": "some panel targets cannot rank at all",
    "H8": "engaged targets are independent observations",
    "H9": "the disease layer discriminates between compounds",
    "H10": "the barrier model earns its place over a rule",
}
COST = {
    "H2": "the graph is described as structure, not tuned parameters",
    "H3": "by construction. The gate is a filter, not a discriminator",
    "H7": "four endpoints withdrawn for firing on glucose and urea",
    "H8": "the interface now groups homologues and quotes the correlation",
}
H10_SHOW = ["descriptor forest, 12 features", "descriptor logistic regression",
            "tpsa alone", "hbd alone", "CNS heuristic: TPSA <= 90 and MW <= 400"]
# Short enough to sit in panel C's own margin. The first draft used the full method names, which
# were wide enough to run back across the gutter and print on top of panel B's data points.
H10_NICE = {"descriptor forest, 12 features": "forest, 12 desc.",
            "descriptor logistic regression": "logistic, 12 desc.",
            "tpsa alone": "TPSA alone",
            "hbd alone": "H-bond donors",
            "CNS heuristic: TPSA <= 90 and MW <= 400": "CNS rule"}


def rows(p: Path) -> list[dict]:
    with open(p, encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def facts() -> dict:
    ver = []
    for r in rows(INV / "VERDICTS.csv"):
        key = r["hypothesis"].split()[0].strip('"')
        ver.append((key, r["verdict"]))
    pn = rows(TAB / "permutation_null.csv")
    cv = {(r["endpoint"], r["split"]): float(r["roc_auc_mean"])
          for r in rows(TAB / "rf_cv_summary.csv") if r["task"] == "classification"}
    h10 = rows(INV / "H10_barrier_necessity.csv")
    return {"ver": ver, "pn": pn, "cv": cv, "h10": h10,
            "verdict10": h10[0]["verdict"]}


def panel_a(ax, F) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    col = {"SUPPORTED": S.TARGET, "REFUTED": S.WARN, "WEAKENED": S.WITHHELD}
    n = len(F["ver"])
    top, row = 0.955, 0.092
    ax.text(0.008, top + 0.045, "hypothesis, as posed", fontsize=S.pt(6.5), color=S.MUTED,
            fontweight="bold", va="center")
    ax.text(0.487, top + 0.045, "verdict", fontsize=S.pt(6.5), color=S.MUTED,
            fontweight="bold", va="center")
    ax.text(0.625, top + 0.045, "what the refutation cost", fontsize=S.pt(6.5), color=S.MUTED,
            fontweight="bold", va="center")
    for i, (key, verdict) in enumerate(F["ver"]):
        y = top - i * row
        base = verdict.split()[0]
        c = col.get(base, S.MUTED)
        if base == "REFUTED":
            ax.add_patch(plt.Rectangle((0.0, y - row / 2 + 0.006), 1.0, row - 0.012,
                                       facecolor=S.HAIR + "", alpha=0.45, edgecolor="none",
                                       zorder=1))
        ax.text(0.008, y, key, fontsize=S.pt(7), fontweight="bold", color=c, va="center")
        ax.text(0.058, y, SHORT.get(key, ""), fontsize=S.pt(6.8), color=S.INK, va="center")
        ax.text(0.487, y, base, fontsize=S.pt(6.8), fontweight="bold", color=c, va="center")
        if key in COST:
            ax.text(0.625, y, COST[key], fontsize=S.pt(6.5), color=S.MUTED, va="center")
    ax.text(0.008, top - n * row - 0.010,
            "H10 was added after a thesis audit found that the component the architecture is named "
            "for had no hypothesis.",
            fontsize=S.pt(6.5), color=S.BINDER, va="center", style="italic")


def panel_b(ax, F) -> None:
    pn = F["pn"]
    eps = [r["endpoint"] for r in pn if r["split"] == "random"]
    y = list(range(len(eps)))[::-1]
    for i, ep in zip(y, eps):
        for split, dy, mark in (("random", 0.16, "o"), ("scaffold", -0.16, "s")):
            null = float(next(r["permuted_roc_auc_mean"] for r in pn
                              if r["endpoint"] == ep and r["split"] == split))
            true = F["cv"][(ep, split)]
            ax.plot([null, true], [i + dy, i + dy], color=S.HAIR, linewidth=1.8,
                    solid_capstyle="round", zorder=1)
            ax.plot([null], [i + dy], marker=mark, markersize=3.6, color=S.WARN, zorder=3)
            ax.plot([true], [i + dy], marker=mark, markersize=3.6, color=S.TARGET,
                    markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    ax.axvline(0.5, color=S.INK, linewidth=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[e] for e in eps], fontsize=S.pt(7))
    ax.set_xlim(0.44, 1.0)
    ax.set_ylim(-0.55, len(eps) - 0.3)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("AUROC", fontsize=S.pt(7))
    nulls = [float(r["permuted_roc_auc_mean"]) for r in pn]
    mr = st.mean(float(r["permuted_roc_auc_mean"]) for r in pn if r["split"] == "random")
    ms = st.mean(float(r["permuted_roc_auc_mean"]) for r in pn if r["split"] == "scaffold")
    ax.text(0.505, len(eps) - 0.45, "chance", fontsize=S.pt(6.5), color=S.INK, ha="left",
            va="center")
    ax.text(0.74, len(eps) - 0.45, "permuted labels", fontsize=S.pt(6.8), color=S.WARN,
            ha="right", va="center", fontweight="bold")
    ax.text(0.76, len(eps) - 0.45, "→", fontsize=S.pt(6.8), color=S.FAINT, ha="center",
            va="center")
    ax.text(0.78, len(eps) - 0.45, "true labels", fontsize=S.pt(6.8), color=S.TARGET,
            ha="left", va="center", fontweight="bold")
    ax.text(0.0, -0.20,
            f"red = permuted labels, teal = true;  circle = random split, square = scaffold.\n"
            f"Permuted means {mr:.4f} and {ms:.4f}, """
            f"all sixteen within {max(abs(v - 0.5) for v in nulls):.4f} of chance",
            transform=ax.transAxes, fontsize=S.pt(6.5), color=S.MUTED, ha="left", va="top",
            linespacing=1.5)
    S.strip(ax, x=True, y=False)


def panel_c(ax, F) -> None:
    ext = {r["method"]: r for r in F["h10"] if r["population"] == "external approved"}
    dep = ext["deployed forest, 1,036 features"]
    shown = [m for m in H10_SHOW if m in ext]
    y = list(range(len(shown)))[::-1]
    for i, m in zip(y, shown):
        r = ext[m]
        d = float(r["delta_vs_deployed"])
        lo, hi = float(r["delta_ci95_low"]), float(r["delta_ci95_high"])
        crosses = hi >= 0
        c = S.WITHHELD if crosses else S.TARGET
        ax.plot([lo, hi], [i, i], color=c, linewidth=1.6, solid_capstyle="round", zorder=2)
        ax.plot([d], [i], marker="o", markersize=4.6, color=c, markeredgecolor="white",
                markeredgewidth=0.5, zorder=3)

    ax.axvline(0, color=S.INK, linewidth=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{H10_NICE[m]}  {float(ext[m]['auroc']):.4f}" for m in shown],
                       fontsize=S.pt(6.8))
    ax.set_ylim(-0.55, len(shown) - 0.35)
    ax.set_xlabel("AUROC margin against the deployed forest, 95% paired bootstrap",
                  fontsize=S.pt(7))
    ax.text(-0.004, len(shown) - 0.45, f"deployed forest  {float(dep['auroc']):.4f}",
            fontsize=S.pt(6.5), color=S.INK, ha="right", va="center", fontweight="bold")
    worst = ext["descriptor forest, 12 features"]
    ax.text(0.0, -0.20,
            f"on {int(float(dep['n']))} approved drugs the model has never seen, the strongest "
            f"null sits {abs(float(worst['delta_vs_deployed'])):.4f} behind\nwith an interval "
            f"reaching {float(worst['delta_ci95_high']):+.4f}, so the verdict is {F['verdict10']}",
            transform=ax.transAxes, fontsize=S.pt(6.5), color=S.WITHHELD, ha="left", va="top",
            linespacing=1.5)
    S.strip(ax, x=True, y=False)


def main() -> None:
    S.use()
    F = facts()
    fig = plt.figure(figsize=(S.DOUBLE, 6.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.98, 1.02], width_ratios=[1.0, 1.0],
                          hspace=0.24, wspace=0.46,
                          left=0.075, right=0.975, top=0.935, bottom=0.145)
    a = fig.add_subplot(gs[0, :])
    b = fig.add_subplot(gs[1, 0])
    c = fig.add_subplot(gs[1, 1])

    panel_a(a, F)
    panel_b(b, F)
    panel_c(c, F)

    S.panel(a, "A", "ten hypotheses, four refuted and two weakened",
            dx=-0.048, dy=1.035, gap=0.022)
    S.panel(b, "B", "the leakage null: labels permuted", dx=-0.19, dy=1.045, gap=0.062)
    S.panel(c, "C", "H10, the hypothesis the suite lacked", dx=-0.46, dy=1.045, gap=0.078)

    S.note(fig, "The suite is read-only: no model, dataset or threshold was changed to obtain any "
                "number in it, and acting on a finding is a separate commit, so a wording change "
                "cannot be mistaken for evidence.", y=0.022)
    S.save(fig, "FigureT4_falsification", outdir=OUTDIR)


if __name__ == "__main__":
    main()
