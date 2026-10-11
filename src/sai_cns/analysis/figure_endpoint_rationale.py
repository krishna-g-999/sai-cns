"""Figure 1: why these endpoints, and how they compose the tool.

A single schematic showing the four functional layers a CNS candidate must satisfy
(exposure, target engagement, safety, developability), the endpoints modelled in each layer,
their measured-data size, and the brain conditions they inform. This is the scientific rationale
for the endpoint selection rather than a performance plot.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NAVY, GOLD, GREEN, RED, BLUE = "#0D2137", "#F0A500", "#1B6B45", "#9B2335", "#1D4ED8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "figure.dpi": 300})

LAYERS = [
    ("1. Can it reach the brain?", "EXPOSURE", BLUE,
     ["BBB penetration (7,805)", "Unbound brain exposure Kp,uu (566)", "logBB (1,058)",
      "P-glycoprotein efflux (1,371)", "Caco-2 permeability (897)"],
     "A candidate that reaches no free concentration in brain tissue\ncannot act centrally, however potent it is."),
    ("2. What does it engage?", "ENGAGEMENT", GREEN,
     ["Cholinesterases AChE, BChE (7,008)", "Amyloid BACE1 (8,501) and tau GSK-3b (4,958)",
      "Monoamine oxidase A and B (5,893)", "18 receptors and transporters (56,000)"],
     "Symptomatic and disease-modifying mechanisms spanning the major\nneurodegenerative and psychiatric axes."),
    ("3. Is it safe?", "SAFETY", RED,
     ["hERG cardiotoxicity liability (5,875)", "P-glycoprotein inhibition (1,212)"],
     "hERG blockade is a leading cause of late-stage\ncardiovascular attrition."),
    ("4. Is it developable?", "DEVELOPABILITY", GOLD,
     ["Aqueous solubility (9,573), logD (4,200)", "Plasma protein binding (1,797)",
      "Hepatocyte clearance (1,020)", "Antioxidant capacity (2,862)"],
     "Physicochemical and metabolic properties that decide whether\nan achievable dose sustains exposure."),
]

DISEASES = [
    ("Alzheimer's disease", "AChE, BChE, BACE1, GSK-3b, a7-nAChR, 5-HT6"),
    ("Parkinson's disease", "MAO-B, LRRK2, A2A, D3"),
    ("Depression and anxiety", "SERT, MAO-A, 5-HT1A, 5-HT7, NET"),
    ("Psychosis", "D2, 5-HT2A, D3"),
    ("Addiction and ADHD", "DAT, NET, mu-opioid, D3"),
    ("Chronic pain", "mu-opioid, kappa-opioid, CB1"),
    ("Sleep and wakefulness", "H3, 5-HT7"),
    ("Epilepsy and neuroprotection", "A1, sigma-1, antioxidant"),
]

HDR, ITEM, WHY, GAP = 0.62, 0.40, 0.78, 0.26


def main():
    left_h = sum(HDR + ITEM * len(it) + WHY for _, _, _, it, _ in LAYERS) + GAP * (len(LAYERS) - 1)
    right_h = 0.95 * len(DISEASES) + 1.15
    top = 1.30
    H = max(left_h, right_h) + top

    fig = plt.figure(figsize=(13.4, 0.86 * H))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.10)
    axl, axr = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for a in (axl, axr):
        a.set_xlim(0, 10); a.set_ylim(0, H); a.axis("off")

    axl.text(0, H - 0.42, "Four questions a CNS candidate must answer",
             fontsize=13, fontweight="bold", color=NAVY)
    axl.text(0, H - 0.88, "Endpoints modelled in each layer, with measured compounds in parentheses",
             fontsize=9.2, color="#3A4A5F")

    y = H - top
    for title, tag, col, items, why in LAYERS:
        h = HDR + ITEM * len(items) + WHY
        axl.add_patch(FancyBboxPatch((0.08, y - h), 9.75, h,
                                     boxstyle="round,pad=0.03,rounding_size=0.14",
                                     fc="#FBFCFE", ec=col, lw=1.7))
        axl.add_patch(FancyBboxPatch((0.30, y - 0.50), 2.45, 0.42,
                                     boxstyle="round,pad=0.02,rounding_size=0.08", fc=col, ec=col))
        axl.text(1.525, y - 0.29, tag, ha="center", va="center", fontsize=7.4,
                 color="white", fontweight="bold")
        axl.text(3.05, y - 0.29, title, va="center", fontsize=11.2, fontweight="bold", color=NAVY)
        yy = y - HDR - ITEM * 0.55
        for it in items:
            axl.text(0.55, yy, "•  " + it, fontsize=9.0, color="#22303F", va="center")
            yy -= ITEM
        axl.text(0.55, y - h + WHY * 0.62, why, fontsize=8.2, color="#5A6B82", va="center", style="italic")
        y -= h + GAP

    axr.text(0, H - 0.42, "Conditions the panel informs", fontsize=13, fontweight="bold", color=NAVY)
    axr.text(0, H - 0.88, "Each condition is scored from its strongest engaged target,\ngated by predicted BBB penetration",
             fontsize=9.2, color="#3A4A5F", va="top")
    yy = H - top
    for name, tg in DISEASES:
        axr.add_patch(FancyBboxPatch((0.05, yy - 0.80), 9.7, 0.80,
                                     boxstyle="round,pad=0.02,rounding_size=0.1",
                                     fc="#F4F7FC", ec="#DCE4F0", lw=1.1))
        axr.text(0.38, yy - 0.28, name, fontsize=10.0, fontweight="bold", color=NAVY, va="center")
        axr.text(0.38, yy - 0.59, tg, fontsize=8.2, color="#5A6B82", va="center")
        yy -= 0.95
    axr.add_patch(FancyBboxPatch((0.05, yy - 0.92), 9.7, 0.92,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 fc="#FFF8E6", ec=GOLD, lw=1.4))
    axr.text(0.38, yy - 0.30, "Not yet modelled", fontsize=9.6, fontweight="bold", color="#7A5B00", va="center")
    axr.text(0.38, yy - 0.65, "NMDA and GABA-A receptors, protein aggregation,\nneuroinflammation, epigenetic targets",
             fontsize=8.0, color="#7A5B00", va="center")

    fig.savefig(OUT / "Figure1_endpoint_rationale.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote Figure1_endpoint_rationale.png")


if __name__ == "__main__":
    main()
