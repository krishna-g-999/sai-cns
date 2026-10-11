"""Shared visual language for the manuscript figures.

One palette and one set of typographic rules, so eleven figures read as one document rather than as
eleven separate plots. The constraints are the journal's and the reader's, not decorative:

  - nothing below MIN_PT at final size. Journal figures are reproduced at the width given here and
    not reduced further, so the point sizes in these scripts are the point sizes a reader sees. An
    earlier version used 4.9 to 5.9 pt in fifty-four places, which is legible on screen at 400 dpi
    and is not legible in print; the floor is enforced by pt() rather than left to each script
  - the palette is distinguishable in the three common forms of colour blindness and in greyscale,
    so category is never carried by hue alone; position, shape or a label carries it too
  - no chartjunk that encodes nothing: no gradients behind data, no 3-D, no drop shadows on marks
  - axes start at a meaningful baseline, and any truncated axis says so

Colour meaning is fixed across the whole figure set, so a reader learns it once:

  EXPOSURE   the blood-brain barrier and ADME layer
  TARGET     the core target panel, potency and activity
  BINDER     the decoy-aware binder classifiers
  SAFETY     liabilities, hERG above all
  WITHHELD   anything held out: scaffolds, actives, the evaluation pool
  WARN       a failure, a withdrawal, or a limitation being stated
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
FIGDIR = ROOT / "manuscript" / "figures"

# Column widths in inches, from the NAR author guidelines: single 89 mm, double 183 mm.
SINGLE, DOUBLE = 3.50, 7.20

# The smallest type allowed anywhere in the figure set, in points at final size.
MIN_PT = 6.5


def pt(size: float) -> float:
    """Clamp a requested point size to the print floor.

    Scripts ask for the size they want and get the size a reader can read. Keeping the clamp here
    rather than editing every call site means the floor can be raised once, for the whole set.
    """
    return max(float(size), MIN_PT)

INK = "#12181F"          # near-black; pure black reads as heavy in print
MUTED = "#5A6672"
FAINT = "#98A4AE"
HAIR = "#DFE4E8"         # grid and separator lines
PAPER = "#FFFFFF"

# Okabe & Ito's colour-universal-design palette (2008), the standard reference set for figures that
# must stay distinguishable under the common forms of colour vision deficiency: protanopia and
# deuteranopia (red-green, ~8% of men) and, less severely, tritanopia (blue-yellow). It was adopted
# here after the previous set paired a teal "GOOD" against a red "WARN" as the panel's one good/bad
# signal, which is exactly the confusion protanopic and deuteranopic vision cannot resolve. Every hue
# below is one of the eight Okabe-Ito colours; none was picked to merely resemble one.
EXPOSURE = "#0072B2"     # blue
TARGET = "#009E73"       # bluish green
BINDER = "#CC79A7"       # reddish purple
SAFETY = "#D55E00"       # vermillion
WITHHELD = "#E69F00"     # orange
WARN = "#D55E00"
GOOD = "#009E73"

FAMILY = {"exposure": EXPOSURE, "target": TARGET, "binder": BINDER,
          "safety": SAFETY, "auxiliary": MUTED}

# Ordered ramp for quantities, dark = more. Sequential, perceptually even enough for print.
RAMP = ["#E8EEF3", "#C3D3E0", "#8FADC6", "#5A83A8", "#2E5C86", "#123A5E"]


def use() -> None:
    """Apply the house style. Call once at the top of every figure script."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        # "tight" crops to the rendered content's own bounding box, and the panel letter is often
        # the topmost thing in that box (it sits above the axes, not inside them). At 0.02 in this
        # left the letter within a point or two of the image's physical edge, printing as though the
        # headline runs into the trim rather than sitting in a margin.
        "savefig.pad_inches": 0.06,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 6,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": HAIR,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def strip(ax, x=False, y=True) -> None:
    """A grid behind the data on one axis only, so it guides without competing."""
    ax.set_axisbelow(True)
    if y:
        ax.yaxis.grid(True)
    if x:
        ax.xaxis.grid(True)


def panel(ax, letter: str, title: str = "", dx: float = -0.085, dy: float = 1.045,
          gap: float = 0.030) -> None:
    """Panel letter in the margin, title set beside it on the same baseline.

    The title is drawn as text rather than through set_title so that it starts after the letter
    instead of underneath it; a bold 10 pt letter and a left-aligned title otherwise land on the
    same point.

    dx/dy/gap are fractions of THIS AXES' own box, not of the figure. That is exactly right when
    every panel in a figure is a normal data axes of comparable shape, and wrong whenever one panel
    calls imshow or otherwise forces an equal aspect ratio: matplotlib then shrinks that axes' box
    around its centre to keep pixels square, so its top edge moves and a fixed dy no longer lines up
    with a neighbouring panel's. panel_fig() below is immune to this because it reads the box after
    any such adjustment and works in figure fractions throughout. Reach for it whenever a figure
    mixes an image panel with plotted panels; it is not the default only because most figures here
    do not mix the two and the existing dx/dy tuning for them should not be disturbed.
    """
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left", color=INK)
    if title:
        ax.text(dx + gap, dy, title, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                va="bottom", ha="left", color=INK)


def panel_fig(fig, ax, letter: str, title: str = "", left: float = 0.010, top: float = 0.012,
              gap: float = 0.022) -> None:
    """Panel letter aligned by its axes' actual position on the canvas, not by axes-fraction.

    Reads ax.get_position() (figure fractions, current as of the last draw) and places the letter
    `left` to the left of the axes' left edge and `top` above its top edge, both in figure fractions,
    so two panels line up whenever a reader would expect them to: because their axes start at the
    same place on the page, not because someone tuned a fraction of each one's own, possibly
    aspect-shrunk, box to match by eye.
    """
    box = ax.get_position()
    x, y = box.x0 - left, box.y1 + top
    fig.text(x, y, letter, fontsize=10, fontweight="bold", va="bottom", ha="left", color=INK)
    if title:
        fig.text(x + gap, y, title, fontsize=8.5, fontweight="bold", va="bottom", ha="left",
                 color=INK)


def note(fig, text: str, y: float = -0.01) -> None:
    """A caption line inside the image, for the thing a reader would otherwise have to be told."""
    fig.text(0.5, y, text, ha="center", va="top", fontsize=6.5, color=MUTED, wrap=True)


def save(fig, name: str, outdir: Path | None = None) -> Path:
    """Write PNG for review and PDF for submission; vector where the journal wants vector.

    outdir lets the thesis figures share this style while living beside the chapters that cite them.
    They are the same visual system and must stay so: a reader meeting Figure 3 in the manuscript and
    Figure T1 in the thesis should not have to relearn what a colour means.
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    d = Path(outdir) if outdir else FIGDIR
    d.mkdir(parents=True, exist_ok=True)
    png, pdf = d / f"{name}.png", d / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {png.relative_to(ROOT).as_posix()} and .pdf")
    return png
