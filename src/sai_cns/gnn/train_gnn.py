"""Train the GIN graph network and compare it, honestly, to the random forest on the same split.

For each selected endpoint a single scaffold hold-out is used (20% of scaffolds as test, a slice of the
rest as validation for early stopping). The GIN and a random forest are trained on the identical
training compounds and scored on the identical test compounds, so the only thing that differs is the
model. This is a fair, like-for-like test of whether a learned graph representation beats a fixed
fingerprint at this data scale.

CPU is fine for this demonstration. For the full 13-endpoint, 10-fold run, move to the GPU cluster
(see src/sai_cns/gnn/README.md).

Output: results/gnn/gnn_vs_rf.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.featurize import featurize_one  # noqa: E402  (ECFP for the RF baseline)
from models.train_rf import CLASSIFICATION, REGRESSION, SEED, _load, _scaffold_groups  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_features import mol_to_graph, NODE_DIM  # noqa: E402
from gin_model import GIN, collate  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ENDPOINTS = ["BBB", "BACE1", "MAO_A", "A2A"]
BATCH = 128
MAX_EPOCHS = 120
PATIENCE = 18
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_aligned(ep):
    task = "classification" if ep in CLASSIFICATION else "regression"
    target = "label" if task == "classification" else REGRESSION[ep]
    df = _load(ep).dropna(subset=["smiles", target]).reset_index(drop=True)
    graphs, X_ecfp, y, smi = [], [], [], []
    for s, t in zip(df["smiles"], df[target]):
        g = mol_to_graph(s)
        v = featurize_one(s)
        if g is not None and v is not None:
            graphs.append(g); X_ecfp.append(v); y.append(t); smi.append(s)
    y = np.array(y, dtype=float)
    groups = _scaffold_groups(smi)
    return task, graphs, np.vstack(X_ecfp), y, groups


def scaffold_split(groups, test_frac=0.2, val_frac=0.1):
    rng = np.random.RandomState(SEED)
    uniq = np.unique(groups); rng.shuffle(uniq)
    n_test = int(len(uniq) * test_frac)
    n_val = int(len(uniq) * val_frac)
    test_s, val_s = set(uniq[:n_test]), set(uniq[n_test:n_test + n_val])
    test = np.isin(groups, list(test_s))
    val = np.isin(groups, list(val_s))
    train = ~(test | val)
    return np.where(train)[0], np.where(val)[0], np.where(test)[0]


def _minibatches(idx, rng):
    idx = idx.copy(); rng.shuffle(idx)
    for k in range(0, len(idx), BATCH):
        b = idx[k:k + BATCH]
        if len(b) >= 2:
            yield b


def _predict(model, graphs, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(idx), BATCH):
            b = idx[k:k + BATCH]
            X, EI, B, _ = collate([graphs[i] for i in b], [0] * len(b))
            out.append(model(X, EI, B, len(b)).cpu().numpy().reshape(-1))
    return np.concatenate(out)


def train_gin(task, graphs, y, tr, va, te):
    model = GIN(NODE_DIM, hidden=64, n_layers=3, out_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    rng = np.random.RandomState(SEED)
    if task == "classification":
        pos = max(y[tr].sum(), 1); neg = len(tr) - y[tr].sum()
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
        ymean = ystd = None
    else:
        ymean, ystd = y[tr].mean(), y[tr].std() + 1e-8
        lossf = nn.MSELoss()

    def target(i):
        return (y[i] - ymean) / ystd if task == "regression" else y[i]

    best, best_state, wait = (-np.inf, None, 0)
    for epoch in range(MAX_EPOCHS):
        model.train()
        for b in _minibatches(tr, rng):
            X, EI, B, yy = collate([graphs[i] for i in b], [target(i) for i in b])
            opt.zero_grad()
            loss = lossf(model(X, EI, B, len(b)), yy)
            loss.backward(); opt.step()
        pred = _predict(model, graphs, va)
        if task == "classification":
            score = roc_auc_score(y[va], pred) if len(set(y[va])) > 1 else 0.5
        else:
            score = r2_score(y[va], pred * ystd + ymean)
        if score > best:
            best, best_state, wait = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    pred = _predict(model, graphs, te)
    if task == "regression":
        pred = pred * ystd + ymean
    return pred


def main():
    rows = []
    for ep in ENDPOINTS:
        task, graphs, X_ecfp, y, groups = load_aligned(ep)
        tr, va, te = scaffold_split(groups)
        print(f"\n[{ep}] {task}: {len(y)} compounds; train {len(tr)}, val {len(va)}, test {len(te)}")

        # GIN
        gin_pred = train_gin(task, graphs, y, tr, va, te)
        # RF on the identical split (ECFP features)
        if task == "classification":
            rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                        n_jobs=-1, random_state=SEED).fit(X_ecfp[tr], y[tr].astype(int))
            rf_pred = rf.predict_proba(X_ecfp[te])[:, 1]
            gin_s = roc_auc_score(y[te], gin_pred); rf_s = roc_auc_score(y[te], rf_pred)
            metric = "roc_auc"
        else:
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                       random_state=SEED).fit(X_ecfp[tr], y[tr])
            rf_pred = rf.predict(X_ecfp[te])
            gin_s = r2_score(y[te], gin_pred); rf_s = r2_score(y[te], rf_pred)
            metric = "r2"
        rows.append({"endpoint": ep, "task": task, "metric": metric, "n_test": len(te),
                     "GIN": round(float(gin_s), 4), "RandomForest": round(float(rf_s), 4),
                     "winner": "GIN" if gin_s > rf_s else "RandomForest"})
        print(f"[{ep}] {metric}: GIN {gin_s:.3f} vs RF {rf_s:.3f} -> {rows[-1]['winner']}")

    out = pd.DataFrame(rows)
    (ROOT / "results" / "gnn").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "results" / "gnn" / "gnn_vs_rf.csv", index=False)
    print("\n=== GIN vs Random Forest (same scaffold hold-out) ===")
    print(out.to_string(index=False))
    print("wrote results/gnn/gnn_vs_rf.csv")


if __name__ == "__main__":
    main()
