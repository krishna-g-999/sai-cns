"""Figure 6. Four validations that a cross-validated score cannot substitute for.

Cross-validation says how the model behaves on the data it was built from. Each panel here asks a
question that partition cannot answer.

A. Calibration. A probability is only usable if 0.8 means eight times in ten. Isotonic regression is
   fitted on out-of-fold predictions, so the calibrator never sees a compound in its own fit.
B. Prospective sensitivity. Whole scaffold classes were withheld before the panel was trained, and
   the recall on them is measured with a Wilson interval, because a recall over twenty compounds
   and one over four hundred are not the same evidence.
C. Specificity on chemistry the server should stay quiet about: non-CNS compounds and, separately,
   an external set of approved drugs absent from the training source.
D. The adversarial suite, every check at the same size whatever its verdict.

Panel D is included deliberately. A validation figure that shows only the checks that passed is a
selection of results, not a validation. All six now pass, but one of them passes only because its
controls were corrected: most of the originals were measured compounds inside the flag's own
reference library, where in-domain is the truthful answer. The passing criterion did not move.

Reads calibration.csv, rf_conformal.csv, scaffold_holdout_results.csv,
noncns_specificity_summary.csv, external_bbb_validation.csv, inversion_validation.csv.

Run:  python src/sai_cns/figures/fig06_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

TAB = ROOT / "results" / "tables"


def panel_a(ax, cal, con) -> None:
    """Expected calibration error, before and after, per endpoint."""
    d = cal.sort_values("ece_raw", ascending=False).reset_index(drop=True)
    ys = np.arange(len(d))
    for y, r in zip(ys, d.itertuples()):
        ax.plot([r.ece_calibrated, r.ece_raw], [y, y], color=S.HAIR, lw=2.4,
                solid_capstyle="round", zorder=1)
    ax.plot(d.ece_raw, ys, "o", ms=3.6, mfc=S.FAINT, mec="white", mew=0.5, zorder=3,
            label="raw forest vote")
    ax.plot(d.ece_calibrated, ys, "o", ms=4.0, mfc=S.TARGET, mec="white", mew=0.5, zorder=3,
            label="after isotonic")
    ax.set_yticks(ys); ax.set_yticklabels(d.endpoint, fontsize=6.5)
    ax.set_xlabel("expected calibration error")
    ax.set_xlim(-0.004, max(d.ece_raw) * 1.10)
    ax.set_ylim(-0.9, len(d) - 0.1)
    S.strip(ax, x=True, y=False)
    ax.legend(loc="lower right", handletextpad=0.3, borderpad=0.2)
    cov = con.empirical_coverage
    ax.text(0.0, 1.02, f"mean ECE {d.ece_raw.mean():.4f} to {d.ece_calibrated.mean():.4f};  "
                       f"conformal coverage {cov.min():.3f} to {cov.max():.3f} "
                       f"against a {con.target_coverage.iloc[0]:.2f} target",
            transform=ax.transAxes, fontsize=6.5, color=S.MUTED, va="bottom")


def panel_b(ax, hold) -> None:
    """Recall on withheld scaffold classes, with intervals that show the evidence.

    Filtered on threshold_collapsed, NOT on the `usable` column. `usable` is defined in
    scaffold_holdout_report.py as `not collapsed and recall >= 0.50`, so it conditions on the very
    quantity being summarised: selecting on it drops the four weakest targets because they are weak
    and then reports the median of the survivors, which reads as 0.835 where the panel's median is
    0.815. Excluding a degenerate threshold is legitimate; excluding a poor result is not.
    """
    d = hold[~hold.threshold_collapsed.astype(bool)] if "threshold_collapsed" in hold.columns \
        else hold
    d = d.dropna(subset=["holdout_recall"]).sort_values("holdout_recall").reset_index(drop=True)
    ys = np.arange(len(d))
    ax.hlines(ys, d.recall_ci95_low, d.recall_ci95_high, color=S.HAIR, lw=1.6, zorder=1)
    sizes = 1.6 + 3.2 * np.sqrt(d.holdout_actives / d.holdout_actives.max())
    for y, r, s in zip(ys, d.itertuples(), sizes):
        ax.plot(r.holdout_recall, y, "o", ms=s, mfc=S.WITHHELD, mec="white", mew=0.4, zorder=3)
    ax.axvline(d.holdout_recall.median(), color=S.INK, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.set_yticks([]); ax.set_xlabel("recall on withheld scaffold classes")
    ax.set_ylabel(f"{len(d)} targets,\nordered by recall", linespacing=1.6)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-1, len(d))
    S.strip(ax, x=True, y=False)
    ax.text(0.03, 0.955, f"median {d.holdout_recall.median():.2f}\n"
                         "marker size is the number of\nwithheld actives; bars are\n"
                         "95 per cent Wilson intervals",
            transform=ax.transAxes, fontsize=6.5, color=S.MUTED, va="top", linespacing=1.7)


def panel_c(ax, spec, ext) -> None:
    """Specificity and external discrimination, as estimates with intervals."""
    def wrap(text, width=30):
        words, lines, cur = str(text).split(), [], ""
        for w in words:
            cur = f"{cur} {w}".strip() if len(f"{cur} {w}") <= width else (lines.append(cur), w)[1]
        lines.append(cur)
        return "\n".join(lines)

    rows = []
    s = spec[spec.metric.astype(str).str.startswith("Specificity")]
    for _, r in s.iterrows():
        rows.append((wrap(r["metric"]), float(r["estimate"]),
                     float(r.get("ci95_low", np.nan)), float(r.get("ci95_high", np.nan)),
                     int(r["n"]) if pd.notna(r.get("n")) else None, S.TARGET))
    for _, r in ext.iterrows():
        rows.append((wrap(r["set"]), float(r["auroc"]), np.nan, np.nan, int(r["n"]),
                     S.EXPOSURE))

    ys = np.arange(len(rows))[::-1]
    for y, (lab, est, lo, hi, n, col) in zip(ys, rows):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.hlines(y, lo, hi, color=S.HAIR, lw=2.0, zorder=1)
        ax.plot(est, y, "o", ms=4.2, mfc=col, mec="white", mew=0.5, zorder=3)
        ax.text(est, y + 0.30, f"{est:.3f}" + (f"  n={n:,}" if n else ""), fontsize=6.5,
                color=S.INK, ha="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.5, linespacing=1.5)
    ax.set_xlabel("estimate (specificity, or AUROC for the external sets)")
    ax.set_xlim(0.4, 1.02); ax.set_ylim(-0.8, len(rows) - 0.2)
    S.strip(ax, x=True, y=False)
    ax.axvline(0.5, color=S.FAINT, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.5, -0.255, "teal: specificity on chemistry the server should stay quiet about.\n"
                         "blue: external discrimination on approved drugs absent from the "
                         "training source.",
            transform=ax.transAxes, fontsize=6.5, color=S.MUTED, ha="center", va="top",
            linespacing=1.7)


def panel_d(ax, inv) -> None:
    """The adversarial suite, every check shown at the same size whatever its verdict."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    d = inv.reset_index(drop=True)
    n_pass = int((d.result.astype(str).str.upper() == "PASS").sum())
    ax.text(0.0, 0.95, f"{n_pass} of {len(d)} adversarial checks pass", fontsize=6.8,
            color=S.INK, fontweight="bold", va="top")
    for i, r in d.iterrows():
        y = 0.80 - i * 0.135
        ok = str(r["result"]).upper() == "PASS"
        col = S.GOOD if ok else S.WARN
        ax.text(0.0, y, "PASS" if ok else "FAIL", fontsize=6.5, color=col, fontweight="bold",
                family="monospace", va="center")
        text = str(r["check"])
        ax.text(0.085, y, text if len(text) <= 74 else text[:72] + "...", fontsize=6.5,
                color=S.INK if ok else S.WARN, va="center")
    if n_pass < len(d):
        note = ("The failing check is reported at the same size as the others and is not tuned "
                "until it passes.\nIt is a finding about the domain flag, and the Discussion "
                "treats it as one.")
    else:
        note = ("Every check is drawn at the same size whatever its verdict, so a check that later "
                "fails would be exactly this visible.\nThe domain-flag check above failed once, on "
                "a control set later found to include compounds inside its own reference library.")
    ax.text(0.0, 0.055, note, fontsize=6.5, color=S.MUTED, va="top", linespacing=1.7)


def main() -> None:
    S.use()
    cal = pd.read_csv(TAB / "calibration.csv")
    con = pd.read_csv(TAB / "rf_conformal.csv")
    hold = pd.read_csv(TAB / "scaffold_holdout_results.csv")
    spec = pd.read_csv(TAB / "noncns_specificity_summary.csv")
    ext = pd.read_csv(TAB / "external_bbb_validation.csv")
    inv = pd.read_csv(TAB / "inversion_validation.csv")

    fig = plt.figure(figsize=(S.DOUBLE, 6.53))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.94], hspace=0.55, wspace=0.40,
                          left=0.075, right=0.985, top=0.915, bottom=0.095)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0]); d = fig.add_subplot(gs[1, 1])
    S.panel(a, "A", "probabilities that mean what they say", dx=-0.24, dy=1.10, gap=0.055)
    S.panel(b, "B", "sensitivity on chemistry withheld", dx=-0.13, dy=1.10, gap=0.055)
    S.panel(c, "C", "and quiet where it should be", dx=-0.55, dy=1.09, gap=0.075)
    S.panel(d, "D", "and every adversarial check, passed or not", dx=-0.30, dy=1.09, gap=0.045)
    panel_a(a, cal, con); panel_b(b, hold); panel_c(c, spec, ext); panel_d(d, inv)
    S.save(fig, "Figure6_validation")


if __name__ == "__main__":
    main()
