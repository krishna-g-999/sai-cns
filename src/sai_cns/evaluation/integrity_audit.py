"""Integrity and reliability audit for a tool intended to displace wet-lab experiments.

Discrimination statistics such as AUROC answer whether a model ranks compounds correctly. They do
not answer the question a scientist actually faces, which is what happens if they act on a number.
Acting wrongly is not symmetric or free: a false positive spends animals, money and months on a
compound that was never going to work, and a false negative discards a real drug. This audit is
therefore organised around decisions rather than around rankings.

  A. Determinism. Identical input must give identical output, or nothing downstream is reproducible.

  B. Calibration. A stated probability is a promise about long-run frequency. If the tool says 80%
     it must be right about 80% of the time, otherwise the number cannot be used to weigh an
     experiment. Reported as expected calibration error and Brier score against held-out measured
     inactives, which is the only negative set with experimental authority.

  C. Decision analysis. Precision at the deployed threshold under realistic screening prevalence,
     with the number of wasted experiments per true hit made explicit. AUROC is prevalence-free;
     a laboratory is not.

  D. Leakage. Verify directly that the scaffold hold-out contained no training scaffold, rather
     than trusting that the splitting code did what it claimed.

Output: results/tables/integrity_audit.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize, featurize_one  # noqa: E402
from models.train_rf import _scaffold_groups  # noqa: E402

M = ROOT / "models_rf"
TAB = ROOT / "results" / "tables"
rows = []


def add(section, metric, value, note=""):
    rows.append({"section": section, "metric": metric, "value": value, "note": note})
    print(f"  {metric:52} {value}   {note}", flush=True)


def ece(y, p, bins=10):
    """Expected calibration error: mean gap between confidence and accuracy, weighted by bin size."""
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(e)


def main():
    import app

    print("=" * 92)
    print("A. DETERMINISM")
    print("=" * 92)
    models = app.load_models()
    smi = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"
    r1 = app.predict_all(smi, models)
    r2 = app.predict_all(smi, models)
    same = all(abs(r1["targets"][k] - r2["targets"][k]) < 1e-12 for k in r1["targets"]) and \
        all(abs(r1["adme"][k] - r2["adme"][k]) < 1e-12 for k in r1["adme"])
    add("determinism", "repeated prediction is bit-identical", "PASS" if same else "FAIL")
    v1 = app.disease_scores(r1)[2][0]["gated"]
    v2 = app.disease_scores(r2)[2][0]["gated"]
    add("determinism", "disease score reproducible", "PASS" if abs(v1 - v2) < 1e-12 else "FAIL")

    print()
    print("=" * 92)
    print("B. CALIBRATION against held-out actives and held-out measured inactives")
    print("=" * 92)
    modes = json.loads((M / "binder_modes.json").read_text())
    eces, briers, per = [], [], []
    skipped_no_holdout = []
    for ep, v in list(modes.items()):
        f = ROOT / "data" / "endpoints" / f"{ep}.csv"
        mp = M / f"{ep}_binder.joblib"
        if not (f.exists() and mp.exists()):
            continue
        # Read what training withheld. This block used to take every active at pChEMBL >= 7 and
        # every measured inactive straight from the endpoint table, which is exactly the data the
        # model was fitted to, and reported the result under a heading saying "held-out".
        hp = M / "holdout" / f"{ep}_binder_holdout.json"
        if not hp.exists():
            skipped_no_holdout.append(ep)
            continue
        h = json.loads(hp.read_text())
        act = list(h.get("active_holdout") or [])
        ina = list(h.get("measured_inactive_holdout") or [])
        if len(act) < 40 or len(ina) < 40:
            continue
        mdl = joblib.load(mp)
        Xa, _ = featurize(act)
        Xi, _ = featurize(ina)
        p = np.r_[mdl.predict_proba(Xa)[:, 1], mdl.predict_proba(Xi)[:, 1]]
        y = np.r_[np.ones(len(act)), np.zeros(len(ina))]
        e, b = ece(y, p), brier_score_loss(y, p)
        eces.append(e); briers.append(b)
        per.append({"endpoint": ep, "ece": round(e, 3), "brier": round(b, 3)})
    if skipped_no_holdout:
        print(f"  {len(skipped_no_holdout)} endpoint(s) skipped for want of a holdout record: "
              f"{', '.join(skipped_no_holdout[:8])}"
              + (" ..." if len(skipped_no_holdout) > 8 else ""))
    if not eces:
        add("calibration", "mean expected calibration error", None,
            "no endpoint had a holdout record; retrain the panel first")
        return
    add("calibration", "mean expected calibration error", round(float(np.mean(eces)), 3),
        f"across {len(eces)} targets, held-out compounds only, 0 is perfect")
    add("calibration", "worst expected calibration error", round(float(np.max(eces)), 3),
        max(per, key=lambda x: x["ece"])["endpoint"])
    add("calibration", "mean Brier score", round(float(np.mean(briers)), 3),
        "lower is better; 0.25 is uninformative")
    pd.DataFrame(per).to_csv(TAB / "integrity_calibration_per_target.csv", index=False)

    print()
    print("=" * 92)
    print("C. DECISION ANALYSIS: what acting on a call actually costs")
    print("=" * 92)
    sens = float(np.mean([v["sensitivity_at_threshold"] for v in modes.values()
                          if v.get("sensitivity_at_threshold")]))
    spec_tbl = TAB / "noncns_specificity_predictions.csv"
    fpr = float(pd.read_csv(spec_tbl).fired.mean()) if spec_tbl.exists() else np.nan
    add("decision", "mean sensitivity at deployed threshold", round(sens, 3))
    add("decision", "false-positive rate on non-CNS compounds", round(fpr, 3))
    for prev in (0.50, 0.10, 0.01):
        ppv = (sens * prev) / (sens * prev + fpr * (1 - prev))
        npv = ((1 - fpr) * (1 - prev)) / ((1 - fpr) * (1 - prev) + (1 - sens) * prev)
        waste = (1 - ppv) / ppv if ppv > 0 else np.inf
        add("decision", f"precision (PPV) at {prev:.0%} prevalence", round(ppv, 3),
            f"NPV {npv:.3f}; {waste:.1f} false leads per true hit")

    print()
    print("=" * 92)
    print("D. LEAKAGE: was the scaffold hold-out genuinely disjoint?")
    print("=" * 92)
    # The per-endpoint records written by the training scripts, not heldout_actives.json, which
    # scaffold_holdout_panel.py writes on its own schedule and which goes stale the moment the panel
    # is retrained. Checking a current panel against a previous run's holdout list reports overlaps
    # that are an artefact of the mismatch rather than a property of either.
    files = sorted((M / "holdout").glob("*_binder_holdout.json"))
    if files:
        held = {}
        for f in files:
            d = json.loads(f.read_text())
            if d.get("active_holdout"):
                held[d["endpoint"]] = d["active_holdout"]
        checked, bad = 0, 0
        # Every target, not the first eight. The truncation was undocumented, and json
        # preserves insertion order, so the same eight were checked every run and
        # thirty-eight were never checked at all while the output implied a panel sweep.
        for ep, smis in held.items():
            f = ROOT / "data" / "endpoints" / f"{ep}.csv"
            if not f.exists() or not smis:
                continue
            df = pd.read_csv(f).dropna(subset=["smiles"])
            pv = pd.to_numeric(df.get("pchembl"), errors="coerce")
            act = df.loc[pv >= 7, "smiles"].astype(str).tolist()
            train = [s for s in act if s not in set(smis)]
            if not train:
                continue
            g_all = _scaffold_groups(train + list(smis))
            g_tr = set(g_all[:len(train)])
            g_ho = set(g_all[len(train):])
            overlap = len(g_tr & g_ho)
            checked += 1
            bad += int(overlap > 0)
            if overlap:
                print(f"     {ep}: {overlap} shared scaffolds", flush=True)
        add("leakage", "targets checked for scaffold overlap", checked)
        add("leakage", "targets with any shared scaffold", bad,
            "PASS" if bad == 0 else "FAIL: hold-out was not disjoint")
    else:
        add("leakage", "hold-out record present", "MISSING")

    out = pd.DataFrame(rows)
    out.to_csv(TAB / "integrity_audit.csv", index=False)
    print()
    print("wrote", TAB / "integrity_audit.csv")


if __name__ == "__main__":
    main()
