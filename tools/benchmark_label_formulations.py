"""Benchmark three ways of handling the 5 to 6 pChEMBL band, on the same compounds.

THE QUESTION. The deployed panel labels actives at pChEMBL >= 6.0 and inactives at <= 5.0 and
discards everything between, which is 23.5 per cent of records panel-wide and 43.0 per cent at
hERG. An order of magnitude in potency is removed from exactly where the decision boundary sits.
Two things follow that no document currently answers: whether including the band would improve
discrimination near that boundary, and whether the band carries orderable information at all.

THE DESIGN. Three formulations are fitted on identical scaffold-grouped folds and scored on
identical test compounds, which is the only way the comparison means anything:

  A  binary, band dropped from training        the deployed formulation
  B  three-class, band as its own class        score = P(active)
  C  regression on pChEMBL, all rows           score = predicted pChEMBL

All three produce a ranking, so all three are scored by AUROC and average precision on the
DECIDABLE part of each test fold, the compounds at <= 5.0 or >= 6.0. That subset is identical for
all three formulations and is the fair comparison. Formulation A is not penalised for never having
seen the band; it is simply asked whether dropping it cost anything on the compounds everyone can
be scored on.

The band compounds in each test fold are then used for the second question. A score that orders
potency inside the band should correlate with the true value there. If none of the three correlates,
the band is noise and discarding it is free. If B or C correlates and A does not, the exclusion is
discarding usable signal.

WHAT THIS DOES NOT CLAIM. The endpoints pool heterogeneous assays with different censoring
conventions, so the regression arm mixes measurement scales and its absolute error is not
interpretable as a potency prediction. It is used here only as a ranking, which is what the
comparison needs. Censored records are excluded throughout: only exact pChEMBL values are read, so
the three formulations differ in their treatment of the band and in nothing else.

Deduplication is on the feature vector with the median taken per group, the regression convention,
because the binary convention drops label-conflicting groups and that would remove different rows
from different formulations and break the comparison.

Reads data/_chembl_cache/<ep>_y.json. Writes two tables and no model files.

    python tools/benchmark_label_formulations.py [--endpoints AChE,hERG] [--folds 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CACHE = ROOT / "data" / "_chembl_cache"
OUT_FOLDS = ROOT / "results" / "tables" / "label_formulation_folds.csv"
OUT_SUMMARY = ROOT / "results" / "tables" / "label_formulation_benchmark.csv"

ACTIVE_CUT = 6.0
INACTIVE_CUT = 5.0
MIN_PER_CLASS = 25
MIN_BAND_FOR_RHO = 30


def auroc(y, s):
    """Mann-Whitney U with tie-averaged ranks."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(y, s):
    """Step-form average precision, matching sklearn's definition."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3:
        return np.nan

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        sv = v[order]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def load_endpoint(ep):
    recs = json.loads((CACHE / f"{ep}_y.json").read_text(encoding="utf-8"))
    smi = [r["smiles"] for r in recs if r.get("pchembl") is not None]
    y = np.array([float(r["pchembl"]) for r in recs if r.get("pchembl") is not None])
    return smi, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", default="")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import GroupKFold

    from sai_cns.features.featurize import featurize
    from sai_cns.models.train_rf import RF_COMMON, _dedup_features, _scaffold_groups

    eps = ([e.strip() for e in args.endpoints.split(",") if e.strip()]
           or sorted(p.name[:-7] for p in CACHE.glob("*_y.json")))
    print(f"endpoints: {', '.join(eps)}")
    print(f"folds: {args.folds}  (scaffold-grouped)\n")

    fold_rows = []
    for ep in eps:
        t0 = time.time()
        smi, y = load_endpoint(ep)
        X, mask = featurize(smi)
        y = y[mask]
        smi = [s for s, m in zip(smi, mask) if m]
        groups = _scaffold_groups(smi)
        X, y, groups, smi, rep = _dedup_features(X, y, groups, smi, "regression")

        band = (y > INACTIVE_CUT) & (y < ACTIVE_CUT)
        dec = ~band
        print(f"[{ep}] {rep['rows_in']:,} records -> {len(y):,} after dedup "
              f"({rep['duplicate_rows_removed']:,} duplicates); "
              f"band {band.sum():,} ({100 * band.mean():.1f}%), "
              f"active {(y >= ACTIVE_CUT).sum():,}, inactive {(y <= INACTIVE_CUT).sum():,}",
              flush=True)

        n_groups = len(np.unique(groups))
        folds = min(args.folds, n_groups)
        if folds < 2:
            print(f"  skipped: only {n_groups} scaffold group(s)")
            continue

        for k, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
            y_tr, y_te = y[tr], y[te]
            te_dec = te[dec[te]]
            te_band = te[band[te]]
            if len(te_dec) < 2 * MIN_PER_CLASS:
                continue
            truth = (y[te_dec] >= ACTIVE_CUT).astype(int)
            if truth.sum() < MIN_PER_CLASS or (1 - truth).sum() < MIN_PER_CLASS:
                continue

            scores = {}

            # A: binary, band dropped from training
            tr_dec = tr[dec[tr]]
            lab = (y[tr_dec] >= ACTIVE_CUT).astype(int)
            if len(np.unique(lab)) == 2:
                m = RandomForestClassifier(class_weight="balanced", **RF_COMMON)
                m.fit(X[tr_dec], lab)
                scores["A_binary_band_dropped"] = m.predict_proba(X[te])[:, 1]

            # B: three-class, band as its own class
            cls = np.where(y_tr >= ACTIVE_CUT, 2, np.where(y_tr <= INACTIVE_CUT, 0, 1))
            if len(np.unique(cls)) == 3:
                m = RandomForestClassifier(class_weight="balanced", **RF_COMMON)
                m.fit(X[tr], cls)
                p = m.predict_proba(X[te])
                scores["B_three_class"] = p[:, list(m.classes_).index(2)]

            # C: regression on pChEMBL, all rows
            m = RandomForestRegressor(**RF_COMMON)
            m.fit(X[tr], y_tr)
            scores["C_regression"] = m.predict(X[te])

            pos_in_te = {int(i): j for j, i in enumerate(te)}
            idx_dec = [pos_in_te[int(i)] for i in te_dec]
            idx_band = [pos_in_te[int(i)] for i in te_band]

            for name, sc in scores.items():
                row = {
                    "endpoint": ep, "fold": k, "formulation": name,
                    "n_test": len(te), "n_decidable": len(te_dec), "n_band": len(te_band),
                    "n_active": int(truth.sum()), "n_inactive": int((1 - truth).sum()),
                    "auroc_decidable": auroc(truth, sc[idx_dec]),
                    "ap_decidable": average_precision(truth, sc[idx_dec]),
                }
                if len(idx_band) >= MIN_BAND_FOR_RHO:
                    row["spearman_within_band"] = spearman(sc[idx_band], y[te_band])
                    lo = np.percentile(sc[idx_dec][truth == 0], 50)
                    hi = np.percentile(sc[idx_dec][truth == 1], 50)
                    row["band_median_score"] = float(np.median(sc[idx_band]))
                    row["inactive_median_score"] = float(lo)
                    row["active_median_score"] = float(hi)
                fold_rows.append(row)
        print(f"  {time.time() - t0:.0f}s", flush=True)

    if not fold_rows:
        print("no usable folds")
        return 1

    folds_df = pd.DataFrame(fold_rows)
    OUT_FOLDS.parent.mkdir(parents=True, exist_ok=True)
    folds_df.to_csv(OUT_FOLDS, index=False)

    agg = (folds_df.groupby(["endpoint", "formulation"])
           .agg(n_folds=("fold", "count"),
                auroc=("auroc_decidable", "mean"),
                ap=("ap_decidable", "mean"),
                rho_band=("spearman_within_band", "mean"),
                n_band=("n_band", "sum"))
           .reset_index())
    agg.to_csv(OUT_SUMMARY, index=False)

    print("\n=== AUROC on the decidable test compounds, by formulation ===")
    piv = agg.pivot(index="endpoint", columns="formulation", values="auroc")
    print(piv.round(4).to_string())
    print("\npanel mean:")
    print(piv.mean().round(4).to_string())

    print("\n=== average precision ===")
    pap = agg.pivot(index="endpoint", columns="formulation", values="ap")
    print(pap.round(4).to_string())
    print("\npanel mean:")
    print(pap.mean().round(4).to_string())

    print("\n=== Spearman of score against true pChEMBL WITHIN the discarded band ===")
    prho = agg.pivot(index="endpoint", columns="formulation", values="rho_band")
    print(prho.round(4).to_string())
    print("\npanel mean:")
    print(prho.mean().round(4).to_string())

    # Paired comparison across endpoints: same folds, same test compounds.
    print("\n=== paired differences across endpoints (Wilcoxon signed-rank, two-sided) ===")
    try:
        from scipy.stats import wilcoxon
        base = "A_binary_band_dropped"
        for other in [c for c in piv.columns if c != base]:
            both = piv[[base, other]].dropna()
            if len(both) < 5:
                print(f"  {other} vs {base}: n={len(both)}, too few to test")
                continue
            d = both[other] - both[base]
            st = wilcoxon(both[other], both[base])
            print(f"  {other} minus {base}: mean {d.mean():+.4f}  median {d.median():+.4f}  "
                  f"wins {int((d > 0).sum())}/{len(d)}  p={st.pvalue:.4f}  "
                  f"(floor for n={len(d)} is {2 ** -(len(d) - 1):.4f})")
    except ImportError:
        print("  scipy unavailable; paired test skipped")

    print(f"\nwrote {OUT_FOLDS.relative_to(ROOT).as_posix()}")
    print(f"wrote {OUT_SUMMARY.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
