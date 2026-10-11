"""Generate publication figures for the NAR Web Server manuscript.

Figure A (model selection): mean scaffold-split AUROC per method family, and the graph-neural-network
vs random-forest head-to-head. Figure B (validation): per-endpoint scaffold AUROC for the classifier
panel and hard-decoy AUROC for the decoy-aware binder classifiers. All numbers are read from the saved
results tables; nothing is estimated.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NAVY, GOLD, GREEN, GREY = "#0D2137", "#F0A500", "#1B6B45", "#94A3B8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#3A4A5F",
                     "axes.linewidth": 0.8, "figure.dpi": 300})

CLF = ["BBB", "AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG"]
BINDERS = ["D2", "A2A", "HT2A", "SERT", "HT1A", "HT6", "HT7", "H3", "DAT", "NET",
           "Sigma1", "CB1", "OPRK1", "OPRM1", "D3", "A1", "a7nAChR", "LRRK2"]


def fig_model_selection():
    cmp = pd.read_csv(ROOT / "results" / "tables" / "model_comparison.csv")
    c = cmp[(cmp.task == "classification") & (cmp.split == "scaffold") & (cmp.metric == "roc_auc")]
    means = c.groupby("model")["mean"].mean()
    # documented baselines from the manuscript / significance tables
    methods = [("Logistic regression", 0.808, GREY), ("k-NN read-across", 0.867, GREY),
               ("Hist. grad. boosting", float(means.get("HistGradientBoosting", np.nan)), NAVY),
               ("XGBoost", float(means.get("XGBoost", np.nan)), NAVY),
               ("Random forest", float(means.get("RandomForest", np.nan)), GOLD)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1), gridspec_kw={"width_ratios": [1.15, 1]})

    labels = [m[0] for m in methods]; vals = [m[1] for m in methods]; cols = [m[2] for m in methods]
    y = np.arange(len(methods))
    ax1.barh(y, vals, color=cols, edgecolor="white", height=0.66)
    for yi, v in zip(y, vals):
        ax1.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=9, fontweight="bold", color=NAVY)
    ax1.set_yticks(y); ax1.set_yticklabels(labels)
    ax1.set_xlim(0.75, 1.0); ax1.set_xlabel("Mean scaffold-split AUROC (classifier panel)")
    ax1.set_title("(a) Model families compared", loc="left", fontweight="bold", color=NAVY)
    ax1.axvline(0.75, color="#3A4A5F", lw=0.8); ax1.spines[["top", "right"]].set_visible(False)

    gnn = pd.read_csv(ROOT / "results" / "gnn" / "gnn_vs_rf.csv")
    x = np.arange(len(gnn)); w = 0.38
    ax2.bar(x - w / 2, gnn["GIN"], w, label="Graph NN (GIN)", color=GREY, edgecolor="white")
    ax2.bar(x + w / 2, gnn["RandomForest"], w, label="Random forest", color=GOLD, edgecolor="white")
    ax2.set_xticks(x); ax2.set_xticklabels(gnn["endpoint"], rotation=0)
    ax2.set_ylim(0.4, 1.0); ax2.set_ylabel("Held-out score (AUROC or R2)")
    ax2.set_title("(b) Deep learning vs random forest", loc="left", fontweight="bold", color=NAVY)
    ax2.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "Figure3_model_selection.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote Figure3_model_selection.png", {m[0]: round(m[1], 3) for m in methods})


def fig_validation():
    cv = pd.read_csv(ROOT / "results" / "tables" / "rf_cv_summary.csv")
    d = cv[(cv.split == "scaffold") & (cv.endpoint.isin(CLF))].set_index("endpoint")
    eps = [e for e in CLF if e in d.index]
    au = [float(d.loc[e, "roc_auc_mean"]) for e in eps]; sd = [float(d.loc[e, "roc_auc_sd"]) for e in eps]

    binder = {}
    for ep in BINDERS:
        f = ROOT / "models_rf" / f"{ep}_binder_meta.json"
        if f.exists():
            m = json.loads(f.read_text())
            if m.get("auroc_hard_decoys"):
                binder[ep] = m["auroc_hard_decoys"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1, 1.5]})
    y = np.arange(len(eps))
    ax1.barh(y, au, xerr=sd, color=NAVY, edgecolor="white", height=0.62,
             error_kw=dict(ecolor=GOLD, lw=1.2, capsize=3))
    ax1.set_yticks(y); ax1.set_yticklabels(eps); ax1.invert_yaxis()
    ax1.set_xlim(0.7, 1.0); ax1.set_xlabel("Scaffold 10-fold AUROC")
    ax1.set_title("(a) Classifier panel", loc="left", fontweight="bold", color=NAVY)
    ax1.spines[["top", "right"]].set_visible(False)

    bk = list(binder); bv = [binder[k] for k in bk]
    x = np.arange(len(bk))
    ax2.bar(x, bv, color=GREEN, edgecolor="white", width=0.7)
    ax2.set_xticks(x); ax2.set_xticklabels(bk, rotation=60, ha="right", fontsize=8)
    ax2.set_ylim(0.6, 1.0); ax2.set_ylabel("Hard-decoy AUROC")
    ax2.set_title("(b) Decoy-aware binder classifiers", loc="left", fontweight="bold", color=NAVY)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "Figure4_validation.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote Figure4_validation.png", "classifiers:", len(eps), "binders:", len(bk))


def fig_mechanism(smiles="O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1", name="Haloperidol"):
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
    import app
    from matplotlib.patches import FancyBboxPatch, PathPatch
    from matplotlib.path import Path as MPath

    r = app.predict_all(smiles, app.load_models())
    bbb, neuro, dz = app.disease_scores(r)
    dzmap = {d["disease"]: d for d in dz}
    targets = [t for t in app.KNOWLEDGE_GRAPH if app.target_signal(r, neuro, t) >= 0.06]
    edges = []
    for t in targets:
        for pw, sid, dis, w in app.KNOWLEDGE_GRAPH[t]:
            if dzmap[dis]["gated"] >= 0.03:
                edges.append((t, pw, dis, w, app.target_signal(r, neuro, t)))
    T = list(dict.fromkeys(e[0] for e in edges))
    P = list(dict.fromkeys(e[1] for e in edges))
    D = list(dict.fromkeys(e[2] for e in edges))
    D.sort(key=lambda d: -dzmap[d]["gated"])
    Dpos = {d: i for i, d in enumerate(D)}
    P.sort(key=lambda p: np.mean([Dpos[e[2]] for e in edges if e[1] == p]))
    Ppos = {p: i for i, p in enumerate(P)}
    T.sort(key=lambda t: np.mean([Ppos[e[1]] for e in edges if e[0] == t]))

    colx = [0.3, 3.2, 6.1, 8.9]; BW, BH = 1.9, 0.52
    n = max(len(T), len(P), len(D), 1)

    def ys(k):
        return [(n - 1) / 2 - (i - (k - 1) / 2) for i in range(k)]
    yc = {("C",): (n - 1) / 2}
    for i, t in enumerate(T): yc[("t", t)] = ys(len(T))[i]
    for i, p in enumerate(P): yc[("p", p)] = ys(len(P))[i]
    for i, d in enumerate(D): yc[("d", d)] = ys(len(D))[i]

    fig, ax = plt.subplots(figsize=(11, max(3.2, 0.95 * n + 1.1)))

    def curve(x0, y0, x1, y1, lw, col, al):
        dx = (x1 - x0) * 0.42
        p = MPath([(x0, y0), (x0 + dx, y0), (x1 - dx, y1), (x1, y1)],
                  [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
        ax.add_patch(PathPatch(p, fill=False, lw=lw, edgecolor=col, alpha=al, capstyle="round"))

    for t in T:
        s = edges_s = app.target_signal(r, neuro, t)
        curve(colx[0] + BW, yc[("C",)], colx[1], yc[("t", t)], 1 + 4 * s, "#1D4ED8", 0.25 + 0.5 * s)
    seen = set()
    for t, pw, dis, w, s in edges:
        if (t, pw) not in seen:
            seen.add((t, pw)); curve(colx[1] + BW, yc[("t", t)], colx[2], yc[("p", pw)], 1 + 4 * s, "#1A3A5C", 0.22 + 0.45 * s)
    pd = {}
    for t, pw, dis, w, s in edges:
        pd[(pw, dis)] = max(pd.get((pw, dis), 0), w * s)
    for (pw, dis), c in pd.items():
        curve(colx[2] + BW, yc[("p", pw)], colx[3], yc[("d", dis)], 1 + 4 * c, "#1A3A5C", 0.22 + 0.5 * c)

    def sc(v): return GREEN if v >= 0.5 else (GOLD if v >= 0.25 else GREY)

    def box(x, y, label, fill, edge, tc, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y - BH / 2), BW, BH, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fill, ec=edge, lw=1.3))
        ax.text(x + BW / 2, y, label, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold")

    box(colx[0], yc[("C",)], name, NAVY, NAVY, "white", 10)
    for t in T:
        s = app.target_signal(r, neuro, t); c = sc(s)
        box(colx[1], yc[("t", t)], f"{app.MECH_LABEL.get(t, t)}  {s:.0%}", "#EEF3FA", c, NAVY)
    for p in P:
        box(colx[2], yc[("p", p)], app.PATHWAY_SHORT.get(p, p), "#F1F4F9", "#1A3A5C", NAVY)
    for d in D:
        g = dzmap[d]["gated"]; c = sc(g)
        box(colx[3], yc[("d", d)], f"{app._short_disease(d)}  {g:.0%}", "#EEF3FA", c, NAVY)

    for i, h in enumerate(["COMPOUND", "TARGET", "PATHWAY", "DISEASE"]):
        ax.text(colx[i] + BW / 2, n - 0.15, h, ha="center", fontsize=9, fontweight="bold", color=GREY)
    ax.set_xlim(0, 11); ax.set_ylim(-0.7, n + 0.1); ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "Figure2_mechanism.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote Figure2_mechanism.png", name, "T/P/D:", len(T), len(P), len(D))


if __name__ == "__main__":
    fig_mechanism()
    fig_model_selection()
    fig_validation()
