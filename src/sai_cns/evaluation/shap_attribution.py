"""Whether SHAP attribution actually recovers known physicochemistry, measured rather than asserted.

The manuscript claims specific Spearman correlations between a descriptor's value and its SHAP
contribution for the barrier model (TPSA, molecular weight and hydrogen-bond donor count pushing away
from penetration; drug-likeness pushing towards it) and for hERG (lipophilicity pushing towards
blockade). No script in the repository computed these, no results table held them, and no section of
the technical report mentions SHAP at all, which put this among the unbacked hardcoded-prose figures
the rest of this project's auditing has been closing one at a time. `shap` is installed in the
environment but is not a runtime dependency of the server; it is a one-off interpretability check, run
here and recorded so the manuscript's specific numbers are something a reviewer can reproduce.

Method. TreeExplainer on the deployed, uncalibrated random forest (models_rf/{BBB,hERG}.joblib) is
exact for a random forest, unlike a model-agnostic approximation, and calibration is a separate
monotonic remapping of the forest's own vote share, so explaining the forest explains the score the
calibrator maps from. A random sample of compounds from the endpoint's own training table is scored;
for each of the twelve interpretable descriptors, the Spearman correlation between the descriptor's
raw value and its SHAP contribution across the sample is the number the manuscript quotes. The 1,024
fingerprint bits are excluded from this correlation because a folded bit has no single physical
meaning to correlate against (see fingerprint_collisions.py); SHAP is computed over the full 1,036
columns so the fingerprint's share of the explanation is not hidden from the forest, only from this
particular readout.

Output: results/tables/shap_attribution.csv

Run:  python src/sai_cns/evaluation/shap_attribution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize, _DESCRIPTORS, MORGAN_BITS  # noqa: E402

SEED = 42
SAMPLE_N = 800
ENDPOINTS = ["BBB", "hERG"]


def main() -> None:
    import shap  # imported here: not a runtime dependency of the server

    rng = np.random.default_rng(SEED)
    rows = []
    for ep in ENDPOINTS:
        table = pd.read_csv(ROOT / "data" / "endpoints" / f"{ep}.csv").dropna(subset=["smiles", "label"])
        n = min(SAMPLE_N, len(table))
        sample = table.sample(n=n, random_state=SEED)
        X, mask = featurize(sample["smiles"].astype(str).tolist())
        model = joblib.load(ROOT / "models_rf" / f"{ep}.joblib")

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        # A binary RandomForestClassifier returns either a single (n, 1036) array of contributions
        # to the positive class, or a (n, 1036, 2) array with one slice per class; both forms occur
        # across shap versions, and the positive-class slice is the one the manuscript's "pushes
        # towards/away from" language describes.
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        for j, desc in enumerate(_DESCRIPTORS):
            col = MORGAN_BITS + j
            r, p = spearmanr(X[:, col], sv[:, col])
            rows.append({"endpoint": ep, "descriptor": desc, "n": n,
                         "spearman_r": round(float(r), 4), "p_value": p})

    d = pd.DataFrame(rows)
    out = ROOT / "results" / "tables" / "shap_attribution.csv"
    d.to_csv(out, index=False)

    print(d.to_string(index=False))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
