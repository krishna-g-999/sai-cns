"""Figure T3. The uncertainty stack, and the endpoints it does not reach.

Three layers sit on top of the point prediction: isotonic calibration of the probability, a Mondrian
conformal set at a stated error rate, and an applicability band derived from distance to training
chemistry. The stack is the reason a probability from this system is meant to be usable rather than
merely rankable, and Chapter 4 argues each layer. What no document states is that the stack is
complete for eight estimators and largely absent for the thirty-eight a user is most likely to query.

A. Calibration works on the core classifiers. Expected calibration error before and after isotonic
   regression on out-of-fold predictions, per endpoint. The mean falls from 0.0801 to 0.0147.
B. And the worst of the eight is the barrier model, at nearly three times that mean. This has not
   been stated anywhere in the project, and it matters more than the rank suggests, because the
   barrier probability multiplies every disease score rather than being reported on its own.
C. Conformal coverage holds where it is measured. Empirical coverage against the 0.90 target, with
   the average set size beside it: a set size near 1 means the interval is informative rather than
   returning both labels to be safe.
D. Where each layer stops, counted per layer rather than asserted. The layers do not all stop at
   the same place: nine of the 47 deployed binder endpoints carry no measured calibration error and
   eight carry no per-endpoint applicability reference, so a yes/no matrix would overstate two cells
   and understate none. Counts are read from the artefacts and the registry at build time.

An earlier draft of this figure headed the middle column "38 binder endpoints". That is the number
with a measured calibration error, not the number of endpoints: the eight core classifiers are not
in binder_modes.json at all, so the panel is 47 deployed binder endpoints of which 38 are measured.

Reads calibration.csv, rf_conformal.csv, integrity_calibration_per_target.csv, binder_modes.json
and the per-endpoint applicability reference.

Run:  brainsafe_env/Scripts/python.exe src/sai_cns/figures/thesis_T3_uncertainty_stack.py
Out:  thesis/figures/FigureT3_uncertainty_stack.png and .pdf
"""
from __future__ import annotations

import csv
import json
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
LABEL = {"AChE": "AChE", "BChE": "BChE", "BACE1": "BACE1", "GSK3B": "GSK-3β",
         "MAO_A": "MAO-A", "MAO_B": "MAO-B", "hERG": "hERG", "BBB": "BBB"}
# Registry identifiers are not protein names. Anything a reader sees is spelled the way a
# pharmacologist writes it, and identically in every panel of every figure.
NICE = {"GABA_A": "GABA-A", "GluN2B": "GluN2B", "Nav1_5": "Nav1.5", "Nav1_8": "Nav1.8",
        "a3b4nAChR": "α3β4 nAChR", "a4b2nAChR": "α4β2 nAChR", "a7nAChR": "α7 nAChR",
        "GSK3B": "GSK-3β", "MAO_A": "MAO-A", "MAO_B": "MAO-B", "mGluR5": "mGluR5",
        "NEURO": "neuroprotection", "Cav3_2": "Cav3.2", "Nav1_6": "Nav1.6", "Nav1_1": "Nav1.1",
        "Nav1_7": "Nav1.7"}


def rows(p: Path) -> list[dict]:
    with open(p, encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def facts() -> dict:
    cal = rows(TAB / "calibration.csv")
    conf = {r["endpoint"]: r for r in rows(TAB / "rf_conformal.csv")}
    ece_rows = rows(TAB / "integrity_calibration_per_target.csv")
    binder = [float(r["ece"]) for r in ece_rows]
    reg = json.loads((ROOT / "models_rf" / "binder_modes.json").read_text(encoding="utf-8"))
    dep = {k for k, v in reg.items() if v.get("deployed")}
    core = {r["endpoint"] for r in cal}

    # Per-endpoint applicability references, counted from the file the server reads rather than
    # assumed to cover everything.
    ad_path = ROOT / "models_rf" / "ad_per_endpoint.json"
    if ad_path.exists():
        ad = set(json.loads(ad_path.read_text(encoding="utf-8")))
    else:
        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        import app as _app
        ad = set(_app.load_ad_per_endpoint())

    n_adme = len(list((ROOT / "models_rf" / "adme").glob("*.joblib")))
    aux = [q for q in (ROOT / "models_rf").glob("*.joblib")
           if q.stem in ("antioxidant_DPPH", "pka_basic")]
    n_aux = n_adme + len(aux)
    ad_aux = sum(1 for name in ("antioxidant", "antioxidant_DPPH", "pka_basic") if name in ad)

    return {
        "cal": [(r["endpoint"], float(r["ece_raw"]), float(r["ece_calibrated"])) for r in cal],
        "conf": conf,
        "binder": binder,
        "n_core": len(core),
        "n_dep": len(dep),
        "n_ece": len([r for r in ece_rows if r["endpoint"] in dep]),
        "n_ad_dep": len(dep & ad),
        "n_aux": n_aux,
        "n_ad_aux": ad_aux,
        "no_ece": sorted(dep - {r["endpoint"] for r in ece_rows}),
    }


def panel_a(ax, F) -> None:
    cal = sorted(F["cal"], key=lambda t: t[2])
    y = list(range(len(cal)))[::-1]
    for i, (ep, raw, fin) in zip(y, cal):
        ax.plot([fin, raw], [i, i], color=S.HAIR, linewidth=2.2, solid_capstyle="round", zorder=1)
        ax.plot([raw], [i], marker="o", markersize=4.2, markerfacecolor="white",
                markeredgecolor=S.MUTED, markeredgewidth=0.9, zorder=3)
        worst = fin == max(c[2] for c in cal)
        ax.plot([fin], [i], marker="o", markersize=4.6,
                color=S.WARN if worst else S.TARGET, markeredgecolor="white",
                markeredgewidth=0.5, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[ep] for ep, _, _ in cal], fontsize=S.pt(7))
    for lab, (ep, _, fin) in zip(ax.get_yticklabels(), cal):
        if fin == max(c[2] for c in cal):
            lab.set_color(S.WARN)
            lab.set_fontweight("bold")
    ax.set_xlim(0, 0.105)
    ax.set_ylim(-0.9, len(cal) - 0.35)
    ax.set_xlabel("expected calibration error", fontsize=S.pt(7))
    mean_raw = st.mean(c[1] for c in cal)
    mean_fin = st.mean(c[2] for c in cal)
    ax.axvline(mean_fin, color=S.TARGET, linewidth=0.8, linestyle=(0, (2.5, 1.6)), zorder=2)
    ax.text(0.104, -0.78, f"open = raw,  filled = isotonic.  mean {mean_raw:.4f} → {mean_fin:.4f}",
            ha="right", va="center", fontsize=S.pt(6.5), color=S.MUTED)
    S.strip(ax, x=True, y=False)


def panel_b(ax, F) -> None:
    cal = F["cal"]
    fin = [c[2] for c in cal]
    worst = max(fin)
    ep_worst = [c[0] for c in cal if c[2] == worst][0]
    mean_fin = st.mean(fin)
    bars = [("best of the eight\n(BACE1)", min(fin), S.TARGET),
            ("mean of the eight", mean_fin, S.MUTED),
            (f"worst of the eight\n({LABEL[ep_worst]}, the gate)", worst, S.WARN)]
    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], width=0.52, color=[b[2] for b in bars],
           edgecolor="none", zorder=2)
    for i, b in enumerate(bars):
        ax.text(i, b[1] + 0.0016, f"{b[1]:.4f}", ha="center", va="bottom",
                fontsize=S.pt(7), fontweight="bold", color=b[2])
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=S.pt(6.5))
    ax.set_ylim(0, worst * 1.30)
    ax.set_ylabel("expected calibration error", fontsize=S.pt(7))
    S.strip(ax)
    ax.text(0.02, 0.97,
            "not reported on its own: it multiplies every one of\nthe fourteen gated conditions, "
            "so its error propagates",
            transform=ax.transAxes, fontsize=S.pt(6.8), color=S.WARN, linespacing=1.45,
            va="top", ha="left")


def panel_c(ax, F) -> None:
    conf = F["conf"]
    order = sorted(conf.values(), key=lambda r: float(r["empirical_coverage"]))
    y = list(range(len(order)))[::-1]
    target = float(order[0]["target_coverage"])
    for i, r in zip(y, order):
        cov = float(r["empirical_coverage"])
        ax.plot([target, cov], [i, i], color=S.HAIR, linewidth=2.0,
                solid_capstyle="round", zorder=1)
        ax.plot([cov], [i], marker="o", markersize=4.4,
                color=S.TARGET if cov >= target else S.WITHHELD,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.text(0.968, i, f"{float(r['avg_set_size']):.3f}", ha="right", va="center",
                fontsize=S.pt(6.5), color=S.MUTED)
    ax.axvline(target, color=S.INK, linewidth=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL.get(r["endpoint"], r["endpoint"]) for r in order],
                       fontsize=S.pt(7))
    ax.set_xlim(0.878, 0.972)
    ax.set_ylim(-0.9, len(order) - 0.35)
    ax.set_xticks([0.88, 0.90, 0.92, 0.94, 0.96])
    ax.set_xlabel("empirical coverage", fontsize=S.pt(7))
    ax.text(target + 0.0015, len(order) - 0.45, f"target {target:.2f}", fontsize=S.pt(6.5),
            color=S.INK, ha="left", va="center")
    ax.text(0.968, len(order) - 0.45, "mean set size", ha="right", va="center",
            fontsize=S.pt(6.5), color=S.MUTED, style="italic")
    ax.text(0.879, -0.78, "range 0.889 to 0.921;  a set size near 1 commits to one label",
            ha="left", va="center", fontsize=S.pt(6.5), color=S.MUTED)
    S.strip(ax, x=True, y=False)


def panel_d(ax, F) -> None:
    """Coverage counted, not asserted.

    A yes/no matrix would say the binder panel has an applicability band and a calibration error.
    It has both for most of its endpoints and neither for some, and the difference is the whole
    point of the panel, so each cell carries the count and is shaded by the fraction it covers.
    """
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    b = F["binder"]
    core_mean = st.mean(c[2] for c in F["cal"])
    nc, nd, na = F["n_core"], F["n_dep"], F["n_aux"]
    layers = ["isotonic\ncalibration", "conformal\nset", "applicability\nband",
              "per-endpoint\ncalibration error"]
    groups = [
        (f"{nc} core\nclassifiers", nc, [nc, nc, nc, nc]),
        (f"{nd} binder\nendpoints", nd, [0, 0, F["n_ad_dep"], F["n_ece"]]),
        (f"{na} ADME,\nauxiliary", na, [0, 0, F["n_ad_aux"], 0]),
    ]
    x0, y0, cw, ch = 0.315, 0.315, 0.222, 0.132
    for j, lay in enumerate(layers):
        ax.text(x0 - 0.022, y0 + (len(layers) - 1 - j) * ch + ch / 2, lay,
                ha="right", va="center", fontsize=S.pt(6.5), color=S.INK, linespacing=1.3)
    for i, (g, total, counts) in enumerate(groups):
        ax.text(x0 + i * cw + cw / 2, y0 + len(layers) * ch + 0.030, g,
                ha="center", va="bottom", fontsize=S.pt(6.5), color=S.INK,
                fontweight="bold", linespacing=1.3)
        for j, k in enumerate(counts):
            yy = y0 + (len(layers) - 1 - j) * ch
            frac = k / total if total else 0.0
            if frac >= 0.999:
                face, ink, txt = S.TARGET, "white", f"{k} of {total}"
            elif frac > 0:
                face, ink, txt = "#BFD9D5", S.INK, f"{k} of {total}"
            else:
                face, ink, txt = S.HAIR, S.MUTED, "none"
            ax.add_patch(plt.Rectangle((x0 + i * cw + 0.006, yy + 0.011),
                                       cw - 0.012, ch - 0.022,
                                       facecolor=face, edgecolor="none", zorder=2))
            ax.text(x0 + i * cw + cw / 2, yy + ch / 2, txt, ha="center", va="center",
                    fontsize=S.pt(6.5), color=ink, fontweight="bold")
    ax.text(0.5, 0.215,
            f"Binder calibration error, over the {len(b)} that carry one: mean {st.mean(b):.4f}, "
            f"median {st.median(b):.4f},\nrange {min(b):.4f} to {max(b):.4f}, against "
            f"{core_mean:.4f} for the core. The remaining {nd - len(b)} carry none:\n"
            + ", ".join(NICE.get(e, e) for e in F["no_ece"]),
            ha="center", va="top", fontsize=S.pt(6.5), color=S.WARN, linespacing=1.55)


def main() -> None:
    S.use()
    F = facts()
    fig = plt.figure(figsize=(S.DOUBLE, 5.85))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.94], width_ratios=[1.0, 0.92],
                          hspace=0.50, wspace=0.34,
                          left=0.088, right=0.972, top=0.935, bottom=0.125)
    a = fig.add_subplot(gs[0, 0])
    c = fig.add_subplot(gs[0, 1])
    b = fig.add_subplot(gs[1, 0])
    d = fig.add_subplot(gs[1, 1])

    panel_a(a, F)
    panel_b(b, F)
    panel_c(c, F)
    panel_d(d, F)

    S.panel(a, "A", "calibration works on the core", dx=-0.185, dy=1.045, gap=0.065)
    S.panel(c, "C", "conformal coverage, two endpoints short", dx=-0.185, dy=1.045, gap=0.062)
    S.panel(b, "B", "and the gate is the worst of them", dx=-0.185, dy=1.055, gap=0.070)
    S.panel(d, "D", "where each layer stops", dx=-0.075, dy=1.055, gap=0.062)

    S.note(fig, "The stack is fullest for the eight endpoints a user rarely queries and thinnest "
                "for the forty-seven they query constantly. Extending it to the panel needs no new "
                "data and is the largest single improvement available.", y=0.030)
    S.save(fig, "FigureT3_uncertainty_stack", outdir=OUTDIR)


if __name__ == "__main__":
    main()
