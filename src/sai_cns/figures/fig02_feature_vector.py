"""Figure 2. What the estimator is actually given, computed for a real molecule.

Reviewers asked, reasonably, what the input to these models is. The answer is a fixed-length numeric
vector, and this figure shows it for donepezil rather than describing it: the structure, the
substructure environments that set bits, the 1,024-bit fingerprint drawn as a grid with the set bits
marked, and the twelve descriptors with their computed values.

The figure also states the two properties of this representation that a reader should hold on to,
because both bound what the models can do. Folding means a bit is not a unique substructure. Leaving
chirality out means two enantiomers are one row, which is why the deduplication step exists at all.

Everything is computed here by the same featuriser the models use, imported rather than reimplemented.

Run:  python src/sai_cns/figures/fig02_feature_vector.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
import style as S  # noqa: E402
from features.featurize import (MORGAN_BITS, MORGAN_RADIUS, _DESCRIPTORS,  # noqa: E402
                                featurize_one)

# Donepezil: an approved AChE inhibitor, in the training data, and a molecule a CNS reader knows.
SMILES = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"
NAME = "donepezil"

DESC_LABEL = {
    "mw": "molecular weight", "clogp": "Crippen logP", "tpsa": "TPSA (A^2)",
    "hbd": "H-bond donors", "hba": "H-bond acceptors", "rotatable_bonds": "rotatable bonds",
    "aromatic_rings": "aromatic rings", "fraction_csp3": "fraction sp3 C",
    "ring_count": "rings", "heavy_atoms": "heavy atoms", "formal_charge": "formal charge",
    "qed": "QED drug-likeness",
}


def structure_image():
    """Draw the molecule with RDKit, or report that it could not be drawn."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(SMILES)
        d = rdMolDraw2D.MolDraw2DCairo(760, 470)
        opts = d.drawOptions()
        opts.bondLineWidth = 2
        opts.clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        import io
        return plt.imread(io.BytesIO(d.GetDrawingText()), format="png")
    except Exception as exc:                                  # pragma: no cover - drawing only
        print(f"  structure not drawn: {exc}")
        return None


def main() -> None:
    S.use()
    vec = featurize_one(SMILES)
    if vec is None:
        raise SystemExit(f"{NAME} did not parse; the figure would be describing nothing")
    vec = np.asarray(vec, dtype=float)
    bits, desc = vec[:MORGAN_BITS], vec[MORGAN_BITS:]
    on = np.flatnonzero(bits)

    fig = plt.figure(figsize=(S.DOUBLE, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.30, 0.92], wspace=0.20,
                          left=0.035, right=0.985, top=0.855, bottom=0.075)

    # ---- A: the molecule -------------------------------------------------------------------
    a = fig.add_subplot(gs[0]); a.axis("off")
    # imshow below forces an equal aspect ratio, which shrinks this axis's box around its centre
    # and moves its top edge, so a fraction of ITS OWN height no longer lands beside B and C's
    # letters, whatever fraction is chosen: B's box shrinks by a different amount than A's, and
    # C never shrinks at all. Anchoring to the top stops the box drifting downward; the panel
    # letter itself is placed afterwards, in panel_fig(), from each box's actual final position.
    a.set_anchor("N")
    img = structure_image()
    if img is not None:
        a.imshow(img)
    a.text(0.5, -0.045, SMILES, transform=a.transAxes, ha="center", fontsize=6.5, color=S.MUTED,
           family="monospace", wrap=True)
    a.text(0.5, -0.135, "standardised first: largest organic fragment, salts stripped,\n"
                        "sanitised, then keyed by the InChIKey of that parent",
           transform=a.transAxes, ha="center", va="top", fontsize=6.5, color=S.MUTED,
           linespacing=1.6)

    # ---- B: the fingerprint, all 1,024 bits ------------------------------------------------
    b = fig.add_subplot(gs[1]); b.axis("off")
    b.set_anchor("N")
    side = int(np.sqrt(MORGAN_BITS))                       # 32 x 32
    grid = bits.reshape(side, side)
    b.imshow(grid, cmap=plt.matplotlib.colors.ListedColormap(["#EDF1F4", S.EXPOSURE]),
             vmin=0, vmax=1, interpolation="nearest", aspect="equal", extent=(0, side, side, 0))
    for k in range(0, side + 1, 8):
        b.axhline(k, color="white", lw=0.7); b.axvline(k, color="white", lw=0.7)
    b.set_xlim(0, side); b.set_ylim(side, 0)
    b.text(0.0, 1.002, "each cell is one bit, read left to right, top to bottom",
           transform=b.transAxes, fontsize=6.5, color=S.MUTED, va="bottom")
    # One line measured almost a fifth wider than B's own axes, running into C's column; the
    # break below keeps both lines inside B, measured the same way.
    b.text(0.0, -0.045, f"Morgan / ECFP-4, radius {MORGAN_RADIUS}, folded to {MORGAN_BITS:,} "
                        "bits,\nchirality NOT included.",
           transform=b.transAxes, fontsize=6.5, color=S.MUTED, va="top", linespacing=1.6)

    # ---- C: the descriptors, with the values this molecule has -----------------------------
    c = fig.add_subplot(gs[2]); c.axis("off")
    names = list(_DESCRIPTORS)          # dict, ordered as the featuriser concatenates them
    # Row spacing was a fixed 0.0645 regardless of how many descriptors there are; with 12 rows that
    # stopped at axes-fraction 0.235, leaving a quarter of the panel's own height empty above the
    # summary box below. Spreading the fixed number of rows over the space actually available (down
    # to just above the box) fills that height instead of stranding it as dead space.
    y_top, y_bottom = 0.945, 0.075
    step = (y_top - y_bottom) / (len(names) - 1)
    for i, (nm, val) in enumerate(zip(names, desc)):
        y = y_top - i * step
        c.add_patch(Rectangle((0.0, y - step / 2), 1.0, step * 0.8, transform=c.transAxes,
                              facecolor="#F4F7F9" if i % 2 == 0 else "white", edgecolor="none"))
        c.text(0.03, y, DESC_LABEL.get(nm, nm), transform=c.transAxes, fontsize=6.5,
               color=S.MUTED, va="center")
        c.text(0.97, y, f"{val:,.2f}" if abs(val) < 1e4 else f"{val:,.0f}",
               transform=c.transAxes, fontsize=6.5, color=S.TARGET, va="center", ha="right",
               fontweight="bold")
    # One line, even at the print floor MIN_PT, measured wider than the box (and the axes column
    # holding it); two lines fits with room to spare.
    c.add_patch(FancyBboxPatch((0.0, -0.078), 1.0, 0.108, transform=c.transAxes,
                               boxstyle="round,pad=0,rounding_size=0.02", clip_on=False,
                               facecolor="#F4F7F9", edgecolor=S.HAIR, lw=0.6))
    c.text(0.5, -0.010, f"{MORGAN_BITS:,} + {len(names)} = {MORGAN_BITS + len(names):,} columns,",
           transform=c.transAxes, ha="center", va="top", fontsize=S.MIN_PT, color=S.INK,
           fontweight="bold")
    c.text(0.5, -0.045, "every endpoint", transform=c.transAxes, ha="center", va="top",
           fontsize=S.MIN_PT, color=S.INK, fontweight="bold")

    # Panel letters last, from each axes' actual final box: A and B were shrunk to a square by
    # imshow's forced aspect, by different amounts, and C never was, so a fraction of any one
    # panel's own height cannot place all three level. draw() first so get_position() reflects the
    # aspect adjustment rather than the pre-draw box add_subplot handed back.
    fig.canvas.draw()
    # A, B and C share one top edge (all three are anchored "N" against the same gridspec top), so
    # giving B alone a taller top clearance to clear its own "each cell is one bit" caption, sitting
    # just above B's box, put its letter visibly higher than A's and C's. All three now share the
    # clearance B needs; A and C do not carry a caption that close to their box, so it only adds a
    # little air above two boxes that already had room, rather than misaligning the row to save it.
    top = 0.028
    S.panel_fig(fig, a, "A", f"the molecule: {NAME}", top=top)
    S.panel_fig(fig, b, "B", f"the fingerprint: {MORGAN_BITS} bits, {int(bits.sum())} set", top=top)
    S.panel_fig(fig, c, "C", f"the descriptors: {len(_DESCRIPTORS)} values", top=top)

    S.save(fig, "Figure2_feature_vector")


if __name__ == "__main__":
    main()
