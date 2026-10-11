"""Why is temporal performance low, and what can be done about it?

The headline temporal numbers (classifier AUROC 0.752, regression R2 0.134) are alarming at face
value. Before accepting them we test four specific hypotheses, because each implies a different
remedy:

  H1  Range restriction. R2 = 1 - SSE/SST is normalised by the variance of the TEST set. Medicinal
      chemistry converges over time on a narrower potency window, so SST shrinks and R2 collapses
      even when absolute error is unchanged. Diagnostic: compare SD(y) train vs test, and report
      RMSE, MAE and Spearman rank correlation alongside R2. For a triage tool, ranking is what
      matters, so Spearman is the decision-relevant metric.

  H2  Training handicap. The temporal model is fit on only the pre-cutoff 75% of the data, whereas
      the deployed model is fit on everything. Diagnostic: compare the temporal model against a
      random-split model trained on the same reduced sample size.

  H3  Applicability domain. Failure may be concentrated in future compounds that are structurally
      far from the training set. Diagnostic: stratify the temporal test set by maximum Tanimoto
      similarity to the training set and recompute metrics per stratum. If in-domain performance
      holds up, the domain flag is an actionable safeguard rather than a disclaimer.

  H4  Recency weighting. Weighting recent training compounds more heavily may track drift.
      Diagnostic: refit with exponential recency weights and compare.

Output: results/tables/temporal_diagnosis.csv, results/tables/temporal_by_domain.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize  # noqa: E402

TAB = ROOT / "results" / "tables"
RF = dict(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
CLASSIFIERS = ["AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG"]
REGRESSORS = ["D2", "A2A", "HT2A", "SERT", "antioxidant_DPPH"]


def _load(ep):
    p = ROOT / "data" / "endpoints" / f"{ep}.csv"
    if not p.exists():
        p = ROOT / "data" / "endpoints_reg" / "antioxidant_dpph.csv"
    return pd.read_csv(p)


def _fps(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        out.append(_GEN.GetFingerprint(m) if m else None)
    return out


def max_sim(train_smiles, test_smiles):
    ref = [f for f in _fps(train_smiles) if f is not None]
    res = []
    for f in _fps(test_smiles):
        res.append(max(DataStructs.BulkTanimotoSimilarity(f, ref)) if f is not None else np.nan)
    return np.array(res)


def run_endpoint(ep, task):
    target = "label" if task == "classification" else ("y" if ep == "antioxidant_DPPH" else "pchembl")
    df = _load(ep)
    if "year" not in df.columns:
        return None, None
    df = df.dropna(subset=["smiles", target, "year"]).reset_index(drop=True)
    if len(df) < 200:
        return None, None
    cutoff = np.percentile(df["year"], 75)
    X, mask = featurize(df["smiles"].tolist())
    df = df.loc[mask].reset_index(drop=True)
    y = df[target].to_numpy()
    tr = (df["year"] <= cutoff).to_numpy()
    te = (df["year"] > cutoff).to_numpy()
    if te.sum() < 30:
        return None, None

    row = {"endpoint": ep, "task": task, "cutoff": int(cutoff),
           "n_train": int(tr.sum()), "n_test": int(te.sum())}

    if task == "regression":
        y = y.astype(float)
        # H1: range restriction
        row["sd_train"] = round(float(np.std(y[tr])), 3)
        row["sd_test"] = round(float(np.std(y[te])), 3)
        row["variance_ratio_test_train"] = round(float(np.var(y[te]) / np.var(y[tr])), 3)
        m = RandomForestRegressor(**RF).fit(X[tr], y[tr])
        p = m.predict(X[te])
        row["r2"] = round(float(r2_score(y[te], p)), 3)
        row["rmse"] = round(float(np.sqrt(mean_squared_error(y[te], p))), 3)
        row["mae"] = round(float(mean_absolute_error(y[te], p)), 3)
        row["spearman"] = round(float(spearmanr(y[te], p).statistic), 3)
        # in-sample-era reference: random split on the SAME reduced sample size (H2)
        rng = np.random.default_rng(0)
        idx = rng.permutation(np.where(tr)[0])
        k = min(int(te.sum()), len(idx) // 4)
        hold, keep = idx[:k], idx[k:]
        m2 = RandomForestRegressor(**RF).fit(X[keep], y[keep])
        p2 = m2.predict(X[hold])
        row["r2_random_same_n"] = round(float(r2_score(y[hold], p2)), 3)
        row["spearman_random_same_n"] = round(float(spearmanr(y[hold], p2).statistic), 3)
        row["rmse_random_same_n"] = round(float(np.sqrt(mean_squared_error(y[hold], p2))), 3)
        # H4 recency weighting
        yrs = df["year"].to_numpy()[tr]
        w = np.exp((yrs - yrs.max()) / 5.0)
        m3 = RandomForestRegressor(**RF).fit(X[tr], y[tr], sample_weight=w)
        p3 = m3.predict(X[te])
        row["r2_recency_weighted"] = round(float(r2_score(y[te], p3)), 3)
        row["spearman_recency_weighted"] = round(float(spearmanr(y[te], p3).statistic), 3)
        pred, truth = p, y[te]
    else:
        y = y.astype(int)
        if len(set(y[te])) < 2:
            return None, None
        row["base_rate_train"] = round(float(y[tr].mean()), 3)
        row["base_rate_test"] = round(float(y[te].mean()), 3)
        m = RandomForestClassifier(class_weight="balanced", **RF).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        row["auroc"] = round(float(roc_auc_score(y[te], p)), 3)
        rng = np.random.default_rng(0)
        idx = rng.permutation(np.where(tr)[0])
        k = min(int(te.sum()), len(idx) // 4)
        hold, keep = idx[:k], idx[k:]
        if len(set(y[hold])) > 1:
            m2 = RandomForestClassifier(class_weight="balanced", **RF).fit(X[keep], y[keep])
            row["auroc_random_same_n"] = round(float(roc_auc_score(y[hold], m2.predict_proba(X[hold])[:, 1])), 3)
        yrs = df["year"].to_numpy()[tr]
        w = np.exp((yrs - yrs.max()) / 5.0)
        m3 = RandomForestClassifier(class_weight="balanced", **RF).fit(X[tr], y[tr], sample_weight=w)
        row["auroc_recency_weighted"] = round(float(roc_auc_score(y[te], m3.predict_proba(X[te])[:, 1])), 3)
        pred, truth = p, y[te]

    # H3: stratify by applicability domain
    sim = max_sim(df["smiles"][tr].tolist(), df["smiles"][te].tolist())
    strata = []
    for name, sel in [("in_domain (T>=0.5)", sim >= 0.5),
                      ("near_domain (0.3-0.5)", (sim >= 0.3) & (sim < 0.5)),
                      ("out_domain (T<0.3)", sim < 0.3)]:
        if sel.sum() < 20:
            continue
        t, q = truth[sel], pred[sel]
        rec = {"endpoint": ep, "task": task, "stratum": name, "n": int(sel.sum()),
               "mean_max_tanimoto": round(float(np.nanmean(sim[sel])), 3)}
        if task == "classification":
            if len(set(t)) > 1:
                rec["auroc"] = round(float(roc_auc_score(t, q)), 3)
        else:
            rec["r2"] = round(float(r2_score(t, q)), 3)
            rec["spearman"] = round(float(spearmanr(t, q).statistic), 3)
            rec["rmse"] = round(float(np.sqrt(mean_squared_error(t, q))), 3)
        strata.append(rec)
    row["frac_test_in_domain"] = round(float(np.mean(sim >= 0.5)), 3)
    row["frac_test_out_domain"] = round(float(np.mean(sim < 0.3)), 3)
    return row, strata


def main():
    rows, strata = [], []
    for ep in CLASSIFIERS:
        r, s = run_endpoint(ep, "classification")
        if r:
            rows.append(r); strata.extend(s)
            print(f"[{ep}] AUROC temporal {r['auroc']} | same-n random {r.get('auroc_random_same_n')} "
                  f"| recency {r['auroc_recency_weighted']} | base rate {r['base_rate_train']}->{r['base_rate_test']}",
                  flush=True)
    for ep in REGRESSORS:
        r, s = run_endpoint(ep, "regression")
        if r:
            rows.append(r); strata.extend(s)
            print(f"[{ep}] R2 {r['r2']} rho {r['spearman']} RMSE {r['rmse']} | "
                  f"SD train {r['sd_train']} -> test {r['sd_test']} (var ratio {r['variance_ratio_test_train']}) | "
                  f"same-n random R2 {r['r2_random_same_n']} rho {r['spearman_random_same_n']}", flush=True)
    pd.DataFrame(rows).to_csv(TAB / "temporal_diagnosis.csv", index=False)
    pd.DataFrame(strata).to_csv(TAB / "temporal_by_domain.csv", index=False)
    print("\nwrote temporal_diagnosis.csv and temporal_by_domain.csv")


if __name__ == "__main__":
    main()
