"""The same prospective test for the twelve core endpoints, with the control the old one lacked.

rf_conformal_temporal.py already trains the core models on compounds up to a cutoff year and tests on
later ones, and reports AUROC 0.72 to 0.91 for the classifiers and R-squared 0.01 to 0.23 for the
regressors. Those numbers have been quoted for a year without the one comparison that makes them
readable.

A time split withholds the most recent quarter of the data, so a model fitted to it has both less
training data and none of the future. When its score is lower than the deployed one, that alone
cannot say which of the two caused the drop, and the difference matters: less data is fixed by
waiting, while a genuine temporal shift is a property of the field and is not fixed by anything. The
regression R-squared values are the acute case. Read on their own they suggest the receptor models
barely work; read against a control they may say something quite different.

Each endpoint is therefore fitted twice, on training sets of identical size: once cut by date, once
cut at random. Everything else is the deployed configuration from train_rf.py, so the gap between
the two is the cost of not knowing the future and nothing else.

Classifiers are scored by AUROC and regressors by R-squared and Spearman, matching what
rf_conformal_temporal.py reports, so the two tables can be read side by side. No decision threshold
is invented for the core models: the deployed panel calibrates them separately and reports intervals
rather than a cut, so a sensitivity figure here would describe an operating point that does not
exist.

Output: results/tables/external_prospective_core.csv
        appends to results/tables/external_prospective_compounds.csv is NOT done; the core
        compound-level rows go to their own file so the two panels stay separable.

Run:  python src/sai_cns/evaluation/external_prospective_core.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
sys.path.insert(0, str(ROOT / "src" / "sai_cns" / "evaluation"))
from features.featurize import featurize  # noqa: E402
from models.train_rf import CLASSIFICATION, REGRESSION, RF_COMMON, SEED, _load  # noqa: E402
from external_prospective import CUT_PERCENTILE, _fp, _max_sim  # noqa: E402

TAB = ROOT / "results" / "tables"
MIN_TEST = 30
MIN_TRAIN = 200


def _prepare(ep: str, target: str | None):
    df = _load(ep).dropna(subset=["smiles"]).reset_index(drop=True)
    if "year" not in df.columns:
        return None
    col = "label" if target is None else target
    if col not in df.columns:
        return None
    df = df.dropna(subset=[col]).reset_index(drop=True)
    X, mask = featurize(df["smiles"].astype(str).tolist())
    df = df.loc[mask].reset_index(drop=True)
    y = pd.to_numeric(df[col], errors="coerce").to_numpy()
    ok = ~np.isnan(y)
    return X[ok], y[ok], df.loc[ok].reset_index(drop=True)


def _score(task, y_true, pred):
    if task == "classification":
        if len(np.unique(y_true)) < 2:
            return {"metric": "auroc", "score": None}
        return {"metric": "auroc", "score": round(float(roc_auc_score(y_true, pred)), 4)}
    return {"metric": "r2", "score": round(float(r2_score(y_true, pred)), 4),
            "spearman": round(float(spearmanr(y_true, pred).statistic), 4)}


def _fit_predict(task, Xtr, ytr, Xte):
    if task == "classification":
        m = RandomForestClassifier(**RF_COMMON).fit(Xtr, ytr.astype(int))
        return m.predict_proba(Xte)[:, 1]
    return RandomForestRegressor(**RF_COMMON).fit(Xtr, ytr).predict(Xte)


def _one(ep: str, task: str, target: str | None):
    prep = _prepare(ep, target)
    if prep is None:
        return None, []
    X, y, df = prep
    yr = pd.to_numeric(df["year"], errors="coerce").to_numpy()
    if np.isfinite(yr).sum() < 100:
        return {"endpoint": ep, "task": task, "status": "too few dated compounds"}, []
    cut = int(np.nanpercentile(yr[np.isfinite(yr)], CUT_PERCENTILE))
    # Undated rows are treated as already known, never as future, so the test set can only ever be
    # too small and never contaminated with chemistry of unknown date.
    post = np.where(np.isfinite(yr), yr > cut, False)
    pre = ~post
    if pre.sum() < MIN_TRAIN or post.sum() < MIN_TEST:
        return {"endpoint": ep, "task": task, "cutoff_year": cut,
                "status": f"{int(pre.sum())} train / {int(post.sum())} test is too few"}, []
    if task == "classification" and len(np.unique(y[post])) < 2:
        return {"endpoint": ep, "task": task, "cutoff_year": cut,
                "status": "post-cutoff set has one class only"}, []

    p_time = _fit_predict(task, X[pre], y[pre], X[post])
    s_time = _score(task, y[post], p_time)

    # Size-matched random control: identical counts, dates ignored.
    g = np.random.default_rng(SEED)
    idx = g.permutation(len(y))
    r_tr, r_te = idx[:int(pre.sum())], idx[int(pre.sum()):int(pre.sum()) + int(post.sum())]
    if task == "classification" and len(np.unique(y[r_te])) < 2:
        s_rand = {"metric": s_time["metric"], "score": None}
    else:
        s_rand = _score(task, y[r_te], _fit_predict(task, X[r_tr], y[r_tr], X[r_te]))

    tr_fps = [f for f in (_fp(s) for s in df.loc[pre, "smiles"].astype(str)) if f is not None]
    te_smi = df.loc[post, "smiles"].astype(str).tolist()
    nov = _max_sim([_fp(s) for s in te_smi], tr_fps)

    row = {"endpoint": ep, "task": task, "status": "ok", "cutoff_year": cut,
           "n_train": int(pre.sum()), "n_test": int(post.sum()),
           "metric": s_time["metric"],
           "time_split_score": s_time["score"], "random_control_score": s_rand["score"],
           "median_test_novelty": round(float(np.median(nov)), 4),
           "test_below_tanimoto_0.4": int((nov < 0.40).sum())}
    if s_time["score"] is not None and s_rand["score"] is not None:
        row["cost_of_prospectivity"] = round(s_rand["score"] - s_time["score"], 4)
    if task == "regression":
        row["time_split_spearman"] = s_time.get("spearman")
        row["random_control_spearman"] = s_rand.get("spearman")

    comps = [{"endpoint": ep, "split": "time", "smiles": s, "measured": float(v),
              "probability": round(float(pp), 5),
              "max_tanimoto_to_training": round(float(nv), 4)}
             for s, v, pp, nv in zip(te_smi, y[post], p_time, nov)]
    return row, comps


def main() -> None:
    want = sys.argv[1:]
    jobs = ([(e, "classification", None) for e in CLASSIFICATION] +
            [(e, "regression", t) for e, t in REGRESSION.items()])
    if want:
        jobs = [j for j in jobs if j[0] in want]
    print(f"prospective validation of {len(jobs)} core endpoints\n", flush=True)

    rows, comps = [], []
    for ep, task, target in jobs:
        try:
            row, cs = _one(ep, task, target)
        except Exception as exc:
            row, cs = {"endpoint": ep, "task": task, "status": f"error: {exc}"}, []
        if row is None:
            print(f"  {ep:18s} skipped: no year column", flush=True)
            continue
        rows.append(row)
        comps.extend(cs)
        if row.get("status") == "ok":
            t, r = row["time_split_score"], row["random_control_score"]
            print(f"  {ep:18s} cut {row['cutoff_year']}  {row['n_train']:>5} -> "
                  f"{row['n_test']:>4}   {row['metric']} time "
                  f"{t if t is not None else float('nan'):.3f} / random "
                  f"{r if r is not None else float('nan'):.3f}", flush=True)
        else:
            print(f"  {ep:18s} skipped: {row['status']}", flush=True)
        TAB.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(TAB / "external_prospective_core.csv", index=False)
        pd.DataFrame(comps).to_csv(TAB / "external_prospective_core_compounds.csv", index=False)

    r = pd.DataFrame(rows)
    ok = r[r.status == "ok"] if len(r) else r
    print()
    for task in ("classification", "regression"):
        s = ok[ok.task == task] if len(ok) else ok
        if not len(s):
            continue
        a = pd.to_numeric(s.time_split_score, errors="coerce")
        b = pd.to_numeric(s.random_control_score, errors="coerce")
        m = s.metric.iloc[0]
        print(f"  {task:15s} n={len(s):>2}  mean {m} time {a.mean():.4f} / "
              f"random {b.mean():.4f}   cost {(b - a).mean():+.4f}")
    print(f"\nwrote {(TAB / 'external_prospective_core.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
