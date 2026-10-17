"""Does the conformal guarantee hold for unfamiliar chemistry, and does conditioning fix it?

WHAT ALREADY EXISTS. The panel carries a Mondrian conformal predictor on its eight core
classifiers, conditioned on class: the nonconformity threshold for each class is a quantile of the
calibration compounds of that class, and empirical coverage runs 0.876 to 0.933 against a 0.90
target. That is a genuine guarantee and it is a MARGINAL one. Averaged over all test compounds,
90 per cent of prediction sets contain the truth.

WHY THAT IS NOT ENOUGH HERE. Marginal coverage permits arbitrary variation underneath. A predictor
can cover 97 per cent of familiar chemistry and 60 per cent of unfamiliar chemistry and still
report 0.90 overall, and this server's central caution is precisely that unfamiliar chemistry
behaves differently: recall falls from 0.86 to 0.16 across the applicability-domain range. If the
same gradient exists in conformal coverage, then the interval shown to a user submitting a novel
scaffold means something weaker than the interval shown to a user submitting an analogue, and
nothing currently says so.

WHAT THIS MEASURES. Two things, in order.

  1. Coverage of the EXISTING class-conditional predictor, broken out by applicability-domain
     band. This asks whether the guarantee already holds conditionally, in which case no change
     is needed and that is worth knowing.

  2. Coverage of a predictor conditioned on class AND band, where each cell takes its threshold
     from its own calibration compounds. This is Mondrian with a finer taxonomy, and it restores
     conditional validity by construction, at the cost of smaller calibration sets and therefore
     wider sets.

Distance is measured to the TRAINING fold of each split, not to the deployed reference, because
the model being evaluated is the one fitted on that fold. Cells with fewer than 30 calibration
compounds are reported but flagged: a quantile of twenty points is not a guarantee.

Run:
    python tools/conformal_domain_conditional.py
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
sys.path.insert(0, str(ROOT / "src" / "sai_cns" / "evaluation"))

OUT = ROOT / "results" / "tables" / "conformal_domain_conditional.csv"
EPS = 0.10                      # 90 per cent target, as deployed
BANDS = [(0.0, 0.40), (0.40, 0.55), (0.55, 0.70), (0.70, 1.01)]
MIN_CELL = 30


def band_of(t):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= t < hi:
            return i
    return len(BANDS) - 1


def quantile_threshold(alphas, eps=EPS):
    """The conformal quantile: the ceil((n+1)(1-eps))-th smallest nonconformity."""
    if len(alphas) == 0:
        return 1.0
    k = int(np.ceil((len(alphas) + 1) * (1 - eps)))
    return float(np.sort(alphas)[min(k, len(alphas)) - 1])


def main() -> int:
    warnings.filterwarnings("ignore")
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator as fgen
    from sklearn.ensemble import RandomForestClassifier
    RDLogger.DisableLog("rdApp.*")
    import rf_conformal_temporal as C

    GEN = fgen.GetMorganGenerator(radius=2, fpSize=2048)
    rows = []

    for endpoint in ("BBB", "AChE", "BACE1", "GSK3B", "MAO_B", "hERG"):
        for split in ("random", "scaffold"):
            try:
                df = C._load(endpoint).dropna(subset=["smiles", "label"]).reset_index(drop=True)
                X, mask = C.featurize(df["smiles"].tolist())
                y = df.loc[mask, "label"].to_numpy().astype(int)
                smiles = [s for s, m in zip(df["smiles"].tolist(), mask) if m]
                groups = C._scaffold_groups(smiles)
                X, y, groups, smiles, _ = C._dedup_features(X, y, groups, smiles, "classification")
                y = np.asarray(y).astype(int)
            except Exception as e:
                print(f"  {endpoint}/{split}: load failed, {type(e).__name__}")
                continue

            n = len(y)
            rng = np.random.RandomState(C.SEED)
            if split == "random":
                idx = rng.permutation(n)
            else:
                order = rng.permutation(np.unique(groups))
                rank = {g: i for i, g in enumerate(order)}
                idx = np.argsort([rank[g] for g in groups], kind="stable")
            a, b = int(0.6 * n), int(0.8 * n)
            tr, cal, te = idx[:a], idx[a:b], idx[b:]

            model = RandomForestClassifier(class_weight="balanced", **C.RF_COMMON).fit(X[tr], y[tr])
            proba = model.predict_proba(X)
            classes = list(model.classes_)

            fps = [GEN.GetFingerprint(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
                   for s in smiles]
            tr_fps = [fps[i] for i in tr if fps[i] is not None]
            sim = np.zeros(n)
            for i in range(n):
                sim[i] = max(DataStructs.BulkTanimotoSimilarity(fps[i], tr_fps)) if fps[i] else 0.0
            bands = np.array([band_of(s) for s in sim])

            # (1) existing predictor: thresholds by class only
            thr_c = {}
            for c in classes:
                ci = classes.index(c)
                sel = cal[y[cal] == c]
                thr_c[c] = quantile_threshold(1.0 - proba[sel, ci])

            # (2) proposed: thresholds by class AND band
            thr_cb = {}
            for c in classes:
                ci = classes.index(c)
                for bnd in range(len(BANDS)):
                    sel = cal[(y[cal] == c) & (bands[cal] == bnd)]
                    thr_cb[(c, bnd)] = (quantile_threshold(1.0 - proba[sel, ci]), len(sel))

            for bnd in range(len(BANDS)):
                m = te[bands[te] == bnd]
                if len(m) < 20:
                    continue
                cov_c, cov_cb, size_c, size_cb = [], [], [], []
                thin = False
                for i in m:
                    ps1 = [c for c in classes
                           if (1.0 - proba[i, classes.index(c)]) <= thr_c[c]]
                    ps2 = []
                    for c in classes:
                        t, ncal = thr_cb[(c, bnd)]
                        if ncal < MIN_CELL:
                            thin = True
                        if (1.0 - proba[i, classes.index(c)]) <= t:
                            ps2.append(c)
                    cov_c.append(y[i] in ps1); size_c.append(len(ps1))
                    cov_cb.append(y[i] in ps2); size_cb.append(len(ps2))
                lo, hi = BANDS[bnd]
                rows.append({
                    "endpoint": endpoint, "split": split, "band": f"{lo:.2f}-{hi:.2f}",
                    "n_test": len(m),
                    "coverage_class_only": round(float(np.mean(cov_c)), 3),
                    "coverage_class_and_band": round(float(np.mean(cov_cb)), 3),
                    "set_size_class_only": round(float(np.mean(size_c)), 3),
                    "set_size_class_and_band": round(float(np.mean(size_cb)), 3),
                    "thin_calibration_cell": int(thin),
                })

    if not rows:
        print("  nothing computed")
        return 1
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"  target coverage {1-EPS:.2f}\n")
    print(f"  {'endpoint':8s} {'split':9s} {'band':11s} {'n':>5s} "
          f"{'class-only':>11s} {'+band':>7s} {'size':>6s} {'size+':>6s}")
    print("  " + "-" * 72)
    for r in rows:
        flag = " *" if r["thin_calibration_cell"] else ""
        print(f"  {r['endpoint']:8s} {r['split']:9s} {r['band']:11s} {r['n_test']:>5} "
              f"{r['coverage_class_only']:>11.3f} {r['coverage_class_and_band']:>7.3f} "
              f"{r['set_size_class_only']:>6.2f} {r['set_size_class_and_band']:>6.2f}{flag}")
    cls = [r["coverage_class_only"] for r in rows]
    cnd = [r["coverage_class_and_band"] for r in rows]
    print(f"\n  class-only coverage across bands: min {min(cls):.3f} max {max(cls):.3f}")
    print(f"  class+band coverage across bands : min {min(cnd):.3f} max {max(cnd):.3f}")
    print("  * at least one calibration cell held fewer than "
          f"{MIN_CELL} compounds, so its quantile is not a guarantee")
    print(f"\n  wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
