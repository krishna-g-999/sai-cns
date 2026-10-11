"""How the deployed BBB model turns one molecule into one probability, hand-drawn rather than
auto-laid-out, because scikit-learn's own `plot_tree` renderer does not reserve space between
sibling leaves and left the previous version of this figure with overlapping text at depth 3.

The manuscript states that a random forest of 300 trees is fitted per endpoint, and that the eight
core classifiers are actually served through a five-fold `CalibratedClassifierCV`. Neither statement
answers what a reader actually wants to see: what one split looks like, and how five folds' worth of
those forests turn into the single number the server returns. This figure answers both, for one real
molecule, with every threshold, sample count and probability read directly from the deployed
`BBB_calibrated.joblib` rather than typed in.

Panel A is tree 0 of fold 0 of that object, to depth 3 (the full 1,247-node tree is exported as text
alongside this figure). Panel B is the five-fold calibration pipeline itself: each fold's 300-tree
forest vote, passed through that fold's own isotonic calibrator, averaged. The two panels are the
same computation at two different scales, one tree and one fold apart, and the donepezil path drawn
through Panel A lands on the same leaf that panel B's fold-0 forest vote (0.9170) is one of 300 trees
contributing to.

Reads models_rf/BBB_calibrated.joblib, models_rf/feature_names.json.
Writes manuscript/figures/FigureS_decision_tree_example.png (and .pdf) and
results/tables/decision_tree_example_full.txt (the complete tree, as text rules).

Run:  python src/sai_cns/figures/fig_decision_tree_example.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from sklearn.tree import export_text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
import style as S  # noqa: E402
from features.featurize import featurize_one  # noqa: E402

MODEL = "BBB"
MAX_DEPTH_SHOWN = 3
SMILES = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"
NAME = "donepezil"

FEAT_LABEL = {"mw": "MW", "qed": "QED", "hba": "H-bond acceptors", "tpsa": "TPSA",
              "clogp": "cLogP", "rotatable_bonds": "rotatable bonds",
              "formal_charge": "formal charge"}


def split_label(feature: str, threshold: float) -> str:
    if feature.startswith("ecfp4_"):
        return f"ECFP4 bit {feature.split('_')[1]}\nabsent (bit = 0)"
    label = FEAT_LABEL.get(feature, feature)
    prec = 1 if abs(threshold) >= 10 else 2
    return f"{label} ≤ {threshold:.{prec}f}"


def walk(t, i: int, depth: int, names: list[str]) -> dict:
    """Recurse the fitted tree's own arrays to `MAX_DEPTH_SHOWN`, exactly what plot_tree reads."""
    feat_idx = t.feature[i]
    val = t.value[i][0]
    n = int(t.n_node_samples[i])
    frac_pen = float(val[1] / (val[0] + val[1]))
    is_leaf = (t.children_left[i] == -1) or depth == MAX_DEPTH_SHOWN
    node = {"idx": int(i), "depth": depth, "n": n, "frac_pen": frac_pen, "is_leaf": is_leaf}
    if not is_leaf:
        node["feature"] = names[feat_idx]
        node["threshold"] = float(t.threshold[i])
        node["left"] = walk(t, int(t.children_left[i]), depth + 1, names)
        node["right"] = walk(t, int(t.children_right[i]), depth + 1, names)
    return node


def leaves_in_order(node: dict) -> list[dict]:
    if node["is_leaf"]:
        return [node]
    return leaves_in_order(node["left"]) + leaves_in_order(node["right"])


def assign_x(node: dict, xs: dict) -> float:
    """Post-order: a leaf's x comes from its slot; an internal node's x is its children's mean."""
    if node["is_leaf"]:
        return xs[node["idx"]]
    lx = assign_x(node["left"], xs)
    rx = assign_x(node["right"], xs)
    x = (lx + rx) / 2
    xs[node["idx"]] = x
    return x


def donepezil_path(node: dict, vec: dict) -> list[int]:
    """The real route this molecule takes, by evaluating each split against its own feature values."""
    path = [node["idx"]]
    while not node["is_leaf"]:
        val = vec[node["feature"]]
        goes_left = val <= node["threshold"]
        node = node["left"] if goes_left else node["right"]
        path.append(node["idx"])
    return path


def leaf_fill(frac_pen: float) -> tuple[str, float]:
    if frac_pen >= 0.5:
        return S.TARGET, 0.18 + 0.55 * (frac_pen - 0.5) * 2
    return S.SAFETY, 0.18 + 0.55 * (0.5 - frac_pen) * 2


def draw_box(ax, x, y, w, h, lines, fill, edge=S.HAIR, textcolor=S.INK, weight="normal",
             fontsize=7.2):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor=fill, edgecolor=edge, linewidth=0.8, zorder=3))
    ax.text(x, y, lines, ha="center", va="center", fontsize=fontsize, color=textcolor,
            fontweight=weight, linespacing=1.55, zorder=4)


def draw_tree_panel(ax, root: dict, path: set[int]) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    S.panel(ax, "A", "one tree's vote for donepezil "
                     f"(tree 0, fold 0 of the {MODEL} model, shown to depth {MAX_DEPTH_SHOWN} of "
                     f"{root['full_depth']})", dx=-0.01, dy=1.10, gap=0.235)

    leaves = leaves_in_order(root)
    n_leaves = len(leaves)
    margin, span = 0.05, 0.90
    slot = span / n_leaves
    xs = {lf["idx"]: margin + slot * (k + 0.5) for k, lf in enumerate(leaves)}
    assign_x(root, xs)

    y_by_depth = {0: 0.86, 1: 0.60, 2: 0.35, 3: 0.09}
    w_by_depth = {0: 0.20, 1: 0.185, 2: 0.155, 3: slot * 0.86}
    h_by_depth = {0: 0.115, 1: 0.115, 2: 0.115, 3: 0.135}

    # edges first, so boxes sit on top of the lines that terminate at their borders
    def draw_edges(node):
        if node["is_leaf"]:
            return
        x0, y0 = xs[node["idx"]], y_by_depth[node["depth"]]
        for child, side in ((node["left"], "left"), (node["right"], "right")):
            x1, y1 = xs[child["idx"]], y_by_depth[child["depth"]]
            on_path = node["idx"] in path and child["idx"] in path
            ax.plot([x0, x1], [y0 - h_by_depth[node["depth"]] / 2,
                               y1 + h_by_depth[child["depth"]] / 2],
                    color=S.EXPOSURE if on_path else S.HAIR,
                    linewidth=2.4 if on_path else 1.0,
                    zorder=2 if on_path else 1, solid_capstyle="round")
            if node["depth"] == 0:
                lab = "True" if side == "left" else "False"
                lx = x0 + (x1 - x0) * 0.32
                ly = y0 - h_by_depth[0] / 2 - 0.028
                ax.text(lx, ly, lab, fontsize=6.6, color=S.MUTED, style="italic", ha="center")
            draw_edges(child)

    draw_edges(root)

    def draw_nodes(node):
        x, y, d = xs[node["idx"]], y_by_depth[node["depth"]], node["depth"]
        w, h = w_by_depth[d], h_by_depth[d]
        on_path = node["idx"] in path
        if node["is_leaf"]:
            fill_c, alpha = leaf_fill(node["frac_pen"])
            import matplotlib.colors as mcolors
            fill = mcolors.to_rgba(fill_c, alpha)
            pen_pct = node["frac_pen"] * 100
            lean = f"{pen_pct:.0f}% penetrant" if node["frac_pen"] >= 0.5 \
                else f"{100 - pen_pct:.0f}% non-penetrant"
            draw_box(ax, x, y, w, h, f"n = {node['n']:,}\n{lean}", fill,
                     edge=S.EXPOSURE if on_path else S.MUTED,
                     textcolor=S.INK, fontsize=6.5)
        else:
            draw_box(ax, x, y, w, h,
                     split_label(node["feature"], node["threshold"]) + f"\nn = {node['n']:,}",
                     fill="white", edge=S.EXPOSURE if on_path else S.HAIR,
                     weight="bold" if d == 0 else "normal", fontsize=7.3 if d == 0 else 6.9)
            draw_nodes(node["left"]); draw_nodes(node["right"])
        if on_path:
            ax.scatter([x], [y + h / 2 + 0.006], marker="v", s=26, color=S.EXPOSURE, zorder=5)

    draw_nodes(root)

    traversed_leaf = next(lf for lf in leaves if lf["idx"] in path)
    tx = xs[traversed_leaf["idx"]]
    leaf_top = y_by_depth[3] + h_by_depth[3] / 2
    depth2_bottom = y_by_depth[2] - h_by_depth[2] / 2
    ax.annotate(f"{NAME} lands here (tree 0, fold 0 only)",
                xy=(tx, leaf_top + 0.006), xytext=(tx, (leaf_top + depth2_bottom) / 2),
                ha="center", va="center", fontsize=S.MIN_PT, color=S.EXPOSURE, style="italic",
                fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none"),
                arrowprops=dict(arrowstyle="-", color=S.EXPOSURE, lw=0.9, shrinkA=9, shrinkB=2))


def draw_ensemble_panel(ax, per_fold_raw, per_fold_cal, mean_val, live_check) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # The formula's summation sign is tall enough that, measured against its own rendered bbox, it
    # reached from axes-fraction 1.045 to 1.20; the title at dy=1.11 sat inside that span instead of
    # above it. The title now clears the formula's own measured top, not a guessed one.
    ax.text(0.5, 1.045,
            r"$\hat{p}(x) = \frac{1}{5}\sum_{k=1}^{5} g_k\left(\hat{p}_{forest,k}(x)\right)$",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=S.INK)
    S.panel(ax, "B", "the deployed probability: five folds' calibrated votes, averaged",
            dx=-0.01, dy=1.225, gap=0.28)

    n = len(per_fold_raw)
    centers = [0.10 + (0.80 / n) * (k + 0.5) for k in range(n)]
    w = (0.80 / n) * 0.80

    y_forest, y_g, y_cal = 0.80, 0.565, 0.35
    for k, (x, raw, cal) in enumerate(zip(centers, per_fold_raw, per_fold_cal)):
        draw_box(ax, x, y_forest, w, 0.20,
                 f"fold {k}\n300-tree forest\nraw vote\n{raw:.4f}",
                 fill="#F4F7F9", edge=S.HAIR, fontsize=6.7)
        ax.annotate("", xy=(x, y_cal + 0.10), xytext=(x, y_forest - 0.10),
                    arrowprops=dict(arrowstyle="-|>", color=S.MUTED, lw=1.0))
        ax.text(x, y_g, "isotonic\n$g_k$", ha="center", va="center", fontsize=S.MIN_PT,
                color=S.MUTED, style="italic", linespacing=1.3)
        import matplotlib.colors as mcolors
        fill = mcolors.to_rgba(S.TARGET, 0.18 + 0.60 * cal)
        draw_box(ax, x, y_cal, w, 0.145, f"calibrated\n{cal:.4f}", fill,
                 edge=S.TARGET, weight="bold", fontsize=7.3)

    # converge the five calibrated votes into one mean, drawn as a literal funnel
    apex_y = 0.215
    for x in centers:
        ax.plot([x, 0.5], [y_cal - 0.075, apex_y], color=S.MUTED, linewidth=0.9,
                 alpha=0.75, zorder=1)
    ax.annotate("", xy=(0.5, 0.135), xytext=(0.5, apex_y),
                arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=1.4))

    consistent = abs(mean_val - live_check) < 1e-9
    tick = "matches" if consistent else "DIFFERS FROM"
    draw_box(ax, 0.5, 0.075, 0.94, 0.13,
              f"deployed probability = mean of 5 folds = {mean_val:.4f}\n"
              f"{tick} app.py's own live output for {NAME} ({live_check:.4f})",
              fill="#F4F7F9", edge=S.INK, weight="bold", fontsize=7.6)


def main() -> None:
    warnings.filterwarnings("ignore")
    S.use()
    cal = joblib.load(ROOT / "models_rf" / f"{MODEL}_calibrated.joblib")
    names = json.load((ROOT / "models_rf" / "feature_names.json").open())
    fold0_forest = cal.calibrated_classifiers_[0].estimator
    tree = fold0_forest.estimators_[0]

    root = walk(tree.tree_, 0, 0, names)
    root["full_depth"] = int(tree.tree_.max_depth)

    vec = np.asarray(featurize_one(SMILES), dtype=float)
    vec_by_name = dict(zip(names, vec))
    path = set(donepezil_path(root, vec_by_name))

    per_fold_raw, per_fold_cal = [], []
    for cc in cal.calibrated_classifiers_:
        per_fold_raw.append(float(cc.estimator.predict_proba(vec.reshape(1, -1))[0, 1]))
        per_fold_cal.append(float(cc.predict_proba(vec.reshape(1, -1))[0, 1]))
    mean_val = float(np.mean(per_fold_cal))
    live_check = float(cal.predict_proba(vec.reshape(1, -1))[0, 1])

    fig = plt.figure(figsize=(13.5, 11.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.62, 1.0], hspace=0.20,
                          left=0.025, right=0.975, top=0.895, bottom=0.06)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    draw_tree_panel(ax_a, root, path)
    draw_ensemble_panel(ax_b, per_fold_raw, per_fold_cal, mean_val, live_check)

    fig.text(0.5, 0.985, f"BBB — {NAME} — "
                         f"{SMILES}", ha="center", va="top", fontsize=8.5, color=S.MUTED,
              family="monospace")

    out = S.save(fig, "FigureS_decision_tree_example")
    print(f"  wrote {out.relative_to(ROOT)}")
    print(f"  donepezil per-fold calibrated: {[round(v, 4) for v in per_fold_cal]}  "
          f"mean {mean_val:.6f}  live app.py-equivalent {live_check:.6f}")

    full_text = export_text(tree, feature_names=names, show_weights=True)
    out_txt = ROOT / "results" / "tables" / "decision_tree_example_full.txt"
    header = (f"Tree 0 of 300, from fold 0 of the 5-fold CalibratedClassifierCV ensemble that "
              f"serves the deployed {MODEL} model ({MODEL}_calibrated.joblib).\n"
              f"{tree.tree_.node_count:,} nodes, {tree.get_n_leaves():,} leaves, "
              f"max depth {root['full_depth']}.\n"
              f"Feature names from models_rf/feature_names.json; weights are [non-penetrant, "
              f"penetrant] sample counts reaching that node.\n"
              f"Generated by src/sai_cns/figures/fig_decision_tree_example.py\n"
              + "=" * 90 + "\n\n")
    out_txt.write_text(header + full_text, encoding="utf-8")
    print(f"  wrote {out_txt.relative_to(ROOT)} ({len(full_text):,} chars, "
          f"{tree.tree_.node_count:,} nodes)")


if __name__ == "__main__":
    main()
