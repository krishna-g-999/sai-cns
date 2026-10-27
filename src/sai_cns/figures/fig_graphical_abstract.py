"""The graphical abstract NAR requires as a separate submission file.

NAR's own specification (author guidelines, "Graphical Abstract") asks for a single image in a 4:3
landscape aspect ratio, at least 100 x 75 mm, with text large enough to survive being shrunk for a
Table-of-Contents thumbnail or a social-media card. The previous version of this figure was drawn at
7.2 x 4.55 in, a 16:10 aspect ratio that does not meet that specification, and it illustrated the
biology (a molecule crossing a membrane and engaging a receptor) without showing the computation that
this paper is actually about: nothing in it depicted a feature vector, an ensemble of trees, or a
calibration step. A reader could not tell this apart from an illustration of pharmacology in general.

This version is drawn at 8 x 6 in (exactly 4:3) and extends the same mechanism diagram one step
further back, so the pipeline actually drawn is the one Figure 1 and Section 2 of Methods describe:

    SMILES -> 1,036-column feature vector -> 5-fold calibrated random-forest ensemble
           -> [barrier penetration, target engagement] -> exposure gate -> ranked disease call

The forest-in-a-box glyph and the GATE box reuse the exact icon and wording Figure 1 already
established, rather than inventing a second visual vocabulary for the same idea; a reader who has
seen Figure 1 recognises the shorthand immediately. Every value drawn (the bit pattern, the exposure
score, the target score, the gated score, the disease name, every count in the footer) is read live
from the deployed pipeline and the same result tables the manuscript cites, for the same worked
example (donepezil / Alzheimer's disease) used throughout the paper.

Output: manuscript/figures/GraphicalAbstract.png (and .pdf)

Run:  python src/sai_cns/figures/fig_graphical_abstract.py
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

logging.getLogger("streamlit").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
import style as S  # noqa: E402
from features.featurize import MORGAN_BITS, featurize_one  # noqa: E402

DONEPEZIL = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"


def molecule_image(smiles: str, size=(560, 420)):
    """The literal submitted structure, drawn rather than named."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    mol = Chem.MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = d.drawOptions()
    opts.bondLineWidth = 2
    opts.clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return plt.imread(io.BytesIO(d.GetDrawingText()), format="png")


def box(ax, x, y, w, h, face, edge=None, alpha=1.0, r=0.012, lw=0.8, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=face, edgecolor=edge or face, alpha=alpha, linewidth=lw,
                                transform=ax.transAxes, zorder=z))


def bit_grid(ax, x, y, bits, w=0.115, color=S.EXPOSURE):
    """Donepezil's complete, real 1,024-bit fingerprint, shrunk to an icon rather than sampled.

    A 5x5 crop of 25 bits was tried first and looked broken: at this fingerprint's true 4.6% bit
    density, any small contiguous window shows at most one or two set bits, which reads as an
    empty grid rather than a fingerprint. Keeping every one of the 1,024 bits, the way Figure 2
    draws it in full, means the icon shows the same genuinely sparse, speckled pattern rather than
    a cherry-picked or padded one.
    """
    from matplotlib.colors import ListedColormap
    side = int(np.sqrt(len(bits)))
    grid = np.asarray(bits[: side * side]).reshape(side, side)
    ax.imshow(grid, extent=(x - w / 2, x + w / 2, y - w / 2, y + w / 2), zorder=2,
              cmap=ListedColormap(["#EDF1F4", color]), vmin=0, vmax=1, interpolation="nearest",
              aspect="auto", transform=ax.transAxes)


def forest_box(ax, x, y, w, h, color, alpha=0.85):
    """One calibration fold's forest: the same box-plus-three-trunks glyph Figure 1 uses."""
    box(ax, x, y, w, h, color, alpha=alpha, edge=color, r=0.004, lw=0.0)
    for t in range(3):
        xt = x + w * 0.28 + t * w * 0.22
        ax.plot([xt, xt], [y + h * 0.16, y + h * 0.84], color="white", lw=0.9, alpha=0.95,
                zorder=4, transform=ax.transAxes)


def membrane(ax, x, y0, y1, n=6):
    """Two parallel lines with a phospholipid-bilayer hatch: the barrier, drawn schematically."""
    for side in (-1, 1):
        xline = x + side * 0.006
        ax.plot([xline, xline], [y0, y1], color=S.MUTED, lw=1.1, zorder=2, transform=ax.transAxes)
    for yy in np.linspace(y0 + 0.008, y1 - 0.008, n):
        for side in (-1, 1):
            ax.plot([x + side * 0.006, x + side * 0.020], [yy, yy], color=S.HAIR, lw=2.0,
                    solid_capstyle="round", zorder=1, transform=ax.transAxes)


def receptor(ax, x, y, color, r=0.024):
    """A binding pocket, drawn as a notched circle rather than borrowed clip-art."""
    ax.add_patch(Wedge((x, y), r, 200, 160, width=r * 0.55, facecolor=color, edgecolor="none",
                       transform=ax.transAxes, zorder=2, alpha=0.9))
    ax.add_patch(Circle((x, y), r * 0.42, facecolor=color, edgecolor="white", linewidth=1.0,
                        transform=ax.transAxes, zorder=3))


def wrap_to_width(fig, ax, text, fontsize, max_width, **kwargs):
    """Break `text` into lines that measure under `max_width` (axes-fraction) when actually
    rendered at `fontsize`, rather than trusting matplotlib's own `wrap=True`.

    `wrap=True` only prevents a line from crossing the edge of the whole *figure*, not this
    figure's own margin: it let one footer line run to within 3 pt of the physical page edge here.
    Measuring each candidate line's real rendered width against the margin this figure actually
    wants is the only way to guarantee the margin holds regardless of font or DPI.
    """
    words = text.split(" ")
    probe = ax.text(0, 0, "", fontsize=fontsize, transform=ax.transAxes, **kwargs)
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        probe.set_text(trial)
        fig.canvas.draw()
        width = probe.get_window_extent(renderer=fig.canvas.get_renderer()) \
                     .transformed(ax.transAxes.inverted()).width
        if width > max_width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    probe.remove()
    return "\n".join(lines)


def arrow(ax, xy0, xy1, color, rad=0.0, lw=1.3):
    ax.add_patch(FancyArrowPatch(xy0, xy1, transform=ax.transAxes, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=8.5, linewidth=lw, color=color,
                                 zorder=2))


def main() -> None:
    S.use()
    import app

    models = app.load_models()
    r = app.predict_all(DONEPEZIL, models)
    bbb, _neuro, dz = app.disease_scores(r)
    ache = r["targets"]["AChE"]
    top = dz[0]
    vec = np.asarray(featurize_one(DONEPEZIL), dtype=float)
    bits = (vec[:MORGAN_BITS] > 0).astype(int)

    facts = app.panel_facts()
    shape = app.panel_shape()

    # Exactly 4:3, per NAR's graphical-abstract specification (author guidelines, section on
    # Graphical Abstracts: landscape, 4:3, minimum 100 x 75 mm).
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Saving the literal 8 x 6 in canvas (below) rather than letting savefig's tight-bbox add its
    # usual 0.02 in pad means this figure has to draw its own margin: x=0.0 is the physical edge of
    # the page, not a safe left indent, and previous drafts of this figure had left-aligned text
    # starting flush against it (and, on the right, "Alzheimer's disease" running past x=1.0
    # outright). MX/MY below are that margin, held everywhere rather than eyeballed per element.
    MX, MY = 0.030, 0.028

    # Named vertical levels, top to bottom, so no two elements are tuned in isolation onto the
    # same row of the page.
    Y_TITLE, Y_SUBTITLE, Y_RULE1 = 1.0 - MY - 0.022, 0.912, 0.882
    DIAG_TOP, DIAG_BOT = 0.840, 0.300
    Y_BRANCH_TOP, Y_BRANCH_BOT = 0.730, 0.410
    DIAG_MID = (Y_BRANCH_TOP + Y_BRANCH_BOT) / 2          # 0.57: molecule / vector / model trunk
    Y_HEAD_OFF, Y_VAL_OFF = 0.088, 0.082                   # caption above / value below an icon
    Y_NOTE = 0.222
    Y_RULE2 = 0.190
    Y_STATS1, Y_STATS2, Y_URL = 0.148, 0.096, MY + 0.017

    ax.text(MX, Y_TITLE, "SAI-CNS", fontsize=16, fontweight="bold", ha="left", color=S.INK,
            transform=ax.transAxes)
    ax.text(MX, Y_SUBTITLE, "does a molecule reach the brain, and what does it do once it is there?",
            fontsize=9.3, ha="left", color=S.MUTED, style="italic", transform=ax.transAxes)
    ax.plot([MX, 1 - MX], [Y_RULE1, Y_RULE1], color=S.HAIR, lw=1.0, transform=ax.transAxes)

    # ---- the pipeline, drawn left to right, in the same order Figure 1 states it -------------
    x_mol, x_vec, x_model, x_branch, x_gate, x_out = 0.125, 0.320, 0.478, 0.648, 0.788, 0.905

    mol_w = 0.19
    mol_ax = fig.add_axes([x_mol - mol_w / 2, DIAG_MID - mol_w * 0.375, mol_w, mol_w * 0.75])
    mol_ax.imshow(molecule_image(DONEPEZIL))
    mol_ax.axis("off")
    ax.text(x_mol, DIAG_MID - Y_VAL_OFF - 0.03, "submitted structure", ha="center", va="center",
            fontsize=7.6, color=S.MUTED, transform=ax.transAxes)

    arrow(ax, (x_mol + 0.105, DIAG_MID), (x_vec - 0.062, DIAG_MID), S.MUTED)

    bit_grid(ax, x_vec, DIAG_MID, bits, color=S.EXPOSURE)
    ax.text(x_vec, DIAG_MID + Y_HEAD_OFF, "molecular features", ha="center", fontsize=7.8,
            color=S.MUTED, transform=ax.transAxes)
    ax.text(x_vec, DIAG_MID - Y_VAL_OFF, "1,036 columns:\nECFP4 + 12 descriptors", ha="center",
            va="top", fontsize=7.2, color=S.EXPOSURE, fontweight="bold", linespacing=1.5,
            transform=ax.transAxes)

    arrow(ax, (x_vec + 0.062, DIAG_MID), (x_model - 0.058, DIAG_MID), S.MUTED)

    fw, fh, fgap = 0.020, 0.052, 0.006
    fx0 = x_model - (5 * fw + 4 * fgap) / 2
    for i in range(5):
        forest_box(ax, fx0 + i * (fw + fgap), DIAG_MID - fh / 2, fw, fh, S.INK, alpha=0.80)
    ax.text(x_model, DIAG_MID + Y_HEAD_OFF, "deployed model", ha="center", fontsize=7.8,
            color=S.MUTED, transform=ax.transAxes)
    ax.text(x_model, DIAG_MID - Y_VAL_OFF, "5-fold calibrated ensemble,\n300 trees per fold",
            ha="center", va="top", fontsize=7.2, color=S.INK, fontweight="bold", linespacing=1.5,
            transform=ax.transAxes)

    # ---- the branch: one model, two predictions, gated together ------------------------------
    arrow(ax, (x_model + 0.058, DIAG_MID + 0.02), (x_branch - 0.026, Y_BRANCH_TOP), S.EXPOSURE,
          rad=-0.28)
    arrow(ax, (x_model + 0.058, DIAG_MID - 0.02), (x_branch - 0.026, Y_BRANCH_BOT), S.TARGET,
          rad=0.28)

    membrane(ax, x_branch, Y_BRANCH_TOP - 0.052, Y_BRANCH_TOP + 0.052)
    ax.text(x_branch, Y_BRANCH_TOP + Y_HEAD_OFF, "barrier penetration", ha="center", fontsize=7.6,
            color=S.MUTED, transform=ax.transAxes)
    ax.text(x_branch, Y_BRANCH_TOP - Y_VAL_OFF, f"γ = {bbb:.2f}", ha="center", va="center",
            fontsize=9.5, color=S.EXPOSURE, fontweight="bold", transform=ax.transAxes)

    receptor(ax, x_branch, Y_BRANCH_BOT, S.TARGET)
    ax.text(x_branch, Y_BRANCH_BOT + Y_HEAD_OFF, "target engagement", ha="center", fontsize=7.6,
            color=S.MUTED, transform=ax.transAxes)
    ax.text(x_branch, Y_BRANCH_BOT - Y_VAL_OFF, f"AChE = {ache:.2f}", ha="center", va="center",
            fontsize=9.5, color=S.TARGET, fontweight="bold", transform=ax.transAxes)

    arrow(ax, (x_branch + 0.026, Y_BRANCH_TOP), (x_gate - 0.040, DIAG_MID + 0.018), S.EXPOSURE,
          rad=0.22)
    arrow(ax, (x_branch + 0.026, Y_BRANCH_BOT), (x_gate - 0.040, DIAG_MID - 0.018), S.TARGET,
          rad=-0.22)

    gw, gh = 0.088, 0.135
    box(ax, x_gate - gw / 2, DIAG_MID - gh / 2, gw, gh, "#FFF6E6", edge=S.WITHHELD, lw=1.0)
    ax.text(x_gate, DIAG_MID + 0.032, "GATE", ha="center", va="center", fontsize=7.6,
            color=S.WITHHELD, fontweight="bold", transform=ax.transAxes)
    ax.text(x_gate, DIAG_MID - 0.014, "exposure\n× engagement", ha="center", va="center",
            fontsize=6.9, color=S.INK, fontweight="bold", linespacing=1.4, transform=ax.transAxes)

    arrow(ax, (x_gate + gw / 2, DIAG_MID), (x_out - 0.035, DIAG_MID), S.BINDER)

    # However long the reported disease name turns out to be, it must not be able to run past the
    # page edge the way "Alzheimer's disease" did on one line at the old, wider column position:
    # broken on its own first space rather than left to matplotlib's figure-wide auto-wrap, which
    # would not have wrapped a two-word name at all.
    disease_label = top["disease"]
    if len(disease_label) > 13 and " " in disease_label:
        head, _, tail = disease_label.partition(" ")
        disease_label = f"{head}\n{tail}"
    ax.text(x_out, DIAG_MID + Y_HEAD_OFF, "disease relevance", ha="center", fontsize=7.4,
            color=S.MUTED, transform=ax.transAxes)
    ax.text(x_out, DIAG_MID + 0.010, disease_label, ha="center", va="center", fontsize=8.4,
            color=S.INK, fontweight="bold", transform=ax.transAxes, linespacing=1.35)
    ax.text(x_out, DIAG_MID - Y_VAL_OFF - 0.01, f"{top['gated']:.2f}", ha="center", va="center",
            fontsize=12, color=S.BINDER, fontweight="bold", transform=ax.transAxes)

    ax.text(0.5, Y_NOTE,
            "a target score is admitted only in proportion to predicted exposure, so potency\n"
            "the compound cannot reach contributes nothing",
            fontsize=7.8, ha="center", va="center", color=S.MUTED, style="italic",
            transform=ax.transAxes, linespacing=1.5)

    ax.plot([MX, 1 - MX], [Y_RULE2, Y_RULE2], color=S.HAIR, lw=1.0, transform=ax.transAxes)

    # ---- the evidence, set as a caption rather than as headline numbers ----------------------
    max_w = 1 - 2 * MX
    stats1 = (f"{facts['n_records']:,} measured compound-endpoint records"
              f"  ·  {shape['targets']} molecular targets plus exposure, ADME and safety"
              f"  ·  {shape['deployed']} of {shape['trained']} trained estimators deployed"
              f"  ·  scaffold-held-out, calibrated and conformal")
    ax.text(MX, Y_STATS1, wrap_to_width(fig, ax, stats1, 7.6, max_w), fontsize=7.6, ha="left",
            va="top", color=S.INK, transform=ax.transAxes, linespacing=1.5)
    stats2 = ("negative class recovered from measurement, not decoys  ·  thresholds set on a "
              "disjoint pool  ·  every validation reported whichever way it falls")
    ax.text(MX, Y_STATS2, wrap_to_width(fig, ax, stats2, 7.0, max_w), fontsize=7.0, ha="left",
            va="top", color=S.MUTED, transform=ax.transAxes, linespacing=1.5)

    ax.text(MX, Y_URL,
            "freely available, no registration · github.com/krishna-g-999/sai-cns",
            fontsize=7.8, ha="left", va="center", color=S.EXPOSURE, fontweight="bold",
            transform=ax.transAxes)

    # Not S.save(): its house style sets bbox_inches="tight", which crops the canvas to content
    # and pulls the aspect ratio off 4:3 by a fraction of a percent. NAR's own specification names
    # 4:3 explicitly, so this file is saved at the literal 8.0 x 6.0 in canvas the figure was built
    # on, uncropped.
    png = S.FIGDIR / "GraphicalAbstract.png"
    pdf = S.FIGDIR / "GraphicalAbstract.pdf"
    S.FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=400, bbox_inches=None, pad_inches=0)
    fig.savefig(pdf, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"  wrote {png.relative_to(ROOT).as_posix()} and .pdf, "
          f"{fig.get_size_inches()[0]:.2f} x {fig.get_size_inches()[1]:.2f} in "
          f"({fig.get_size_inches()[0] / fig.get_size_inches()[1]:.4f} aspect)")


if __name__ == "__main__":
    main()
