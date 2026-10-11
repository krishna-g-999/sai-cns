"""What happens if we add more training data? A scaffold-honest learning curve.

For each selected endpoint a fixed scaffold-held-out test set is set aside, then the model is trained
on an increasing fraction of the remaining *scaffolds* (10 to 100%) and scored on that same test set.
Subsampling by scaffold, not by random compound, mimics how real new chemistry arrives. If the curve
is still rising at 100%, more data would help; if it has flattened, the endpoint is data-saturated and
the ceiling is the representation or the task, not the sample size.

Outputs:
  results/tables/learning_curve.csv
  results/figures/fig_learning_curve.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.featurize import featurize  # noqa: E402
from models.train_rf import CLASSIFICATION, REGRESSION, SEED, _load, _scaffold_groups  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ENDPOINTS = ["BBB", "MAO_A", "BACE1", "A2A"]   # large + small classifier + big enzyme + regressor
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
TEST_SCAFFOLD_FRAC = 0.2
RNG = np.random.RandomState(SEED)


def _rf(task):
    return (RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                   n_jobs=-1, random_state=SEED) if task == "classification"
            else RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                       random_state=SEED))


def curve_for(ep):
    task = "classification" if ep in CLASSIFICATION else "regression"
    target = "label" if task == "classification" else REGRESSION[ep]
    df = _load(ep).dropna(subset=["smiles", target]).reset_index(drop=True)
    X, mask = featurize(df["smiles"].tolist())
    df = df.loc[mask].reset_index(drop=True)
    y = df[target].to_numpy()
    y = y.astype(int) if task == "classification" else y.astype(float)
    groups = _scaffold_groups(df["smiles"].tolist())

    uniq = np.unique(groups)
    RNG.shuffle(uniq)
    n_test = int(len(uniq) * TEST_SCAFFOLD_FRAC)
    test_scaf, train_scaf = set(uniq[:n_test]), uniq[n_test:]
    test = np.isin(groups, list(test_scaf))
    Xte, yte = X[test], y[test]

    rows = []
    for frac in FRACTIONS:
        k = max(1, int(len(train_scaf) * frac))
        sel = set(train_scaf[:k])
        tr = np.isin(groups, list(sel))
        model = _rf(task).fit(X[tr], y[tr])
        if task == "classification":
            score = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
        else:
            score = r2_score(yte, model.predict(Xte))
        rows.append({"endpoint": ep, "task": task, "train_fraction": frac,
                     "n_train": int(tr.sum()), "n_test": int(test.sum()),
                     "metric": "roc_auc" if task == "classification" else "r2",
                     "score": round(float(score), 4)})
        print(f"  [{ep}] {int(frac*100):3d}% -> n_train={int(tr.sum()):5d}  score={score:.3f}")
    return rows


def main():
    all_rows = []
    for ep in ENDPOINTS:
        print(f"[{ep}]")
        all_rows.extend(curve_for(ep))
    out = pd.DataFrame(all_rows)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "results" / "tables" / "learning_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for ep, g in out.groupby("endpoint"):
        ax.plot(g.n_train, g.score, marker="o", label=f"{ep} ({g.metric.iloc[0]})")
    ax.set_xlabel("Training compounds (subsampled by scaffold)")
    ax.set_ylabel("Score on fixed scaffold-held-out test set")
    ax.set_title("Learning curves: does more data still help?")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "fig_learning_curve.png", dpi=150, bbox_inches="tight")
    print("wrote results/tables/learning_curve.csv and results/figures/fig_learning_curve.png")


if __name__ == "__main__":
    main()
