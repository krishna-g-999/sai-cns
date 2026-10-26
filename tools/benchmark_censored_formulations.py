"""Benchmark the censored-likelihood formulation against the deployed labelling rule.

THE QUESTION THIS CLOSES. The label-formulation benchmark
(tools/benchmark_label_formulations.py) compared three ways of handling the 5 to 6 pChEMBL band and
found the deployed binary rule as good as a three-class model and better than a potency regression.
It read exact pChEMBL values only, deliberately, so that the three arms differed in their treatment
of the band and in nothing else. That design is what left one formulation untested, and it is the
formulation with the strongest theoretical claim on the data the project throws away: a censored
likelihood, which treats every measurement as an interval the true potency lies in rather than as a
point or a class.

The deployed rule keeps a censored record only when its bound settles a class on its own. "IC50 >
10 uM" bounds potency below pChEMBL 5.0 and is inactive whatever the true value is, so it is kept
as a negative; "IC50 > 1 uM" bounds potency below 6.0, is consistent with either class, and is
discarded. A censored likelihood needs no such rule: the second record still rules out the potent
region, so it is usable. It also uses the bound's magnitude, which the binary rule discards when it
collapses every settled bound to the single label 0: under that rule a compound bounded below
pChEMBL 1.3 and one bounded below 5.0 are the same observation.

THE DESIGN. Five arms on identical scaffold-grouped folds, scored on identical test compounds. The
arms are a two-by-two of learner against loss, plus the deployed rule, so that a win for the
censored likelihood cannot be a win for gradient boosting wearing its coat:

                      exact rows, binary label        all rows, censored likelihood
  random forest       A_rf_binary_exact               D_rf_censored_em
  gradient boosting   E_xgb_binary_exact              D_xgb_aft
  deployed rule       Aplus_rf_binary_deployed  (exact rows plus the bounds that settle a class)

A_rf_binary_exact is formulation A of the earlier benchmark: the band dropped and censored records
ignored. Aplus_rf_binary_deployed adds the censored records whose bound settles the class, which is
what the panel actually trains on, so the pair measures what censored recovery already buys. The two
D arms then measure what is left on the table by not modelling the bounds as bounds.

D_rf_censored_em keeps the learner fixed and changes only the loss, by iterated truncated
imputation: censored targets start at their bound, a forest is fitted, and each censored target is
replaced by the fitted value clipped back into its own censoring interval, for a few rounds. That is
a tree-ensemble analogue of Buckley-James, not a maximum-likelihood fit, and it is reported as what
it is. D_xgb_aft is the genuine article: XGBoost's accelerated-failure-time objective maximises an
interval-censored likelihood directly, with a log-normal error. E_xgb_binary_exact is its learner
control.

SCORING. Every arm produces a ranking, so all five are scored by AUROC and average precision on the
DECIDABLE exact test compounds, those at or below 5.0 or at or above 6.0. That subset is identical
for all five and contains no censored record, so no arm is scored on the extra data it was given.
Spearman correlation against true potency inside the band is reported on the exact band compounds,
which are likewise common to all arms.

WHAT THIS CANNOT SHOW. The AUROC values here are not directly comparable with the earlier
benchmark's, because adding the censored compounds changes the scaffold-group population and
therefore the fold boundaries. A_rf_binary_exact is refitted here for that reason; it is the within-
table baseline, and the earlier table remains the comparison between the band formulations.

Censored records are intersected per compound rather than averaged, since every measurement on one
compound has to hold at once, and a compound whose bounds contradict each other is dropped and
counted rather than reconciled.

Reads data/_chembl_cache/<ep>_y.json and <ep>_censored.json; the latter is written by
tools/fetch_censored_records.py. Writes two tables and no model files.

    python tools/benchmark_censored_formulations.py [--endpoints AChE,hERG] [--folds 5]
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
sys.path.insert(0, str(ROOT / "tools"))

CACHE = ROOT / "data" / "_chembl_cache"
OUT_FOLDS = ROOT / "results" / "tables" / "censored_formulation_folds.csv"
OUT_SUMMARY = ROOT / "results" / "tables" / "censored_formulation_benchmark.csv"

ACTIVE_CUT = 6.0
INACTIVE_CUT = 5.0
MIN_PER_CLASS = 25
MIN_BAND_FOR_RHO = 30
EM_ROUNDS = 4
POTENCY_FLOOR = 0.1          # AFT models a positive quantity; no pChEMBL sits at or below this
XGB_ROUNDS = 300
XGB_DEPTH = 6
XGB_ETA = 0.1


def load_exact(ep):
    recs = json.loads((CACHE / f"{ep}_y.json").read_text(encoding="utf-8"))
    smi = [r["smiles"] for r in recs if r.get("pchembl") is not None]
    val = np.array([float(r["pchembl"]) for r in recs if r.get("pchembl") is not None])
    return smi, val


def load_censored(ep):
    """Returns (smiles, lower, upper) with POTENCY_FLOOR and +inf for the open ends."""
    f = CACHE / f"{ep}_censored.json"
    if not f.exists():
        return [], np.array([]), np.array([])
    recs = json.loads(f.read_text(encoding="utf-8"))
    smi, lo, hi = [], [], []
    for r in recs:
        l, h = r.get("pchembl_lower"), r.get("pchembl_upper")
        if l is None and h is None:
            continue
        smi.append(r["smiles"])
        lo.append(POTENCY_FLOOR if l is None else float(l))
        hi.append(np.inf if h is None else float(h))
    return smi, np.array(lo), np.array(hi)


def dedup_intervals(X, lo, hi, groups, smiles):
    """Collapse byte-identical feature vectors, intersecting their intervals.

    Exact measurements dominate a censored bound on the same vector: a recorded potency is a
    tighter statement than a bound, and the fetchers draw the two from different queries, so a
    bound that contradicts an exact value is a provenance disagreement rather than new information.
    Those are counted and reported.
    """
    seen: dict[bytes, list[int]] = {}
    for i, vec in enumerate(X):
        seen.setdefault(vec.tobytes(), []).append(i)

    keep, LO, HI = [], [], []
    merged = contradictory = exact_vs_bound = 0
    for idxs in seen.values():
        if len(idxs) > 1:
            merged += len(idxs) - 1
        exact = [i for i in idxs if lo[i] == hi[i]]
        if exact:
            v = float(np.median([lo[i] for i in exact]))
            cens = [i for i in idxs if lo[i] != hi[i]]
            if cens:
                cl = max(lo[i] for i in cens)
                ch = min(hi[i] for i in cens)
                if not (cl <= v <= ch):
                    exact_vs_bound += 1
            keep.append(exact[0])
            LO.append(v)
            HI.append(v)
        else:
            cl = max(lo[i] for i in idxs)
            ch = min(hi[i] for i in idxs)
            if cl > ch:
                contradictory += 1
                continue
            keep.append(idxs[0])
            LO.append(float(cl))
            HI.append(float(ch))

    order = np.argsort(keep)
    k = np.asarray(keep)[order]
    report = {"rows_in": int(len(lo)), "duplicate_rows_removed": int(merged),
              "contradictory_bounds_dropped": int(contradictory),
              "exact_outside_its_own_bound": int(exact_vs_bound), "rows_out": int(len(k))}
    return (X[k], np.asarray(LO)[order], np.asarray(HI)[order], groups[k],
            [smiles[i] for i in k], report)


def fit_rf_censored_em(X, lo, hi, rf_common, rounds=EM_ROUNDS):
    """Random forest regression under iterated truncated imputation.

    Censored targets begin at the finite end of their interval, which is the naive guess the
    deployed rule effectively makes when it calls such a compound inactive. Each round refits and
    then clips every censored target back inside its own interval, so the fitted value is used
    where the interval permits it and the bound where it does not. The exact targets never move.
    """
    from sklearn.ensemble import RandomForestRegressor

    exact = lo == hi
    t = np.where(exact, lo, np.where(np.isfinite(hi), hi, lo)).astype(float)
    model = None
    for _ in range(rounds):
        model = RandomForestRegressor(**rf_common)
        model.fit(X, t)
        pred = model.predict(X)
        t = np.where(exact, t, np.clip(pred, lo, hi))
    return model


def aft_scale_from_data(lo, hi):
    """Plug-in estimate of the AFT error scale, from the spread of the response itself.

    THIS WAS A REAL DEFECT AND IT CHANGED THE RESULT. The scale was previously fixed at 1.0, the
    XGBoost default. With aft_loss_distribution "normal" the error is Normal on log(label), and on
    these endpoints sd(log(pChEMBL)) is about 0.15 to 0.21, so a scale of 1.0 makes the likelihood
    roughly five times wider than the entire marginal spread of the response. The censored terms
    then contribute almost no gradient, the fit barely resolves anything, and predictions leave the
    physical scale altogether. Measured at MAO_A, holding everything else fixed: 0.6972 AUROC at
    scale 1.0 against 0.7701 at 0.2, with the random forest at 0.8245. The arm was losing to its own
    nuisance parameter.

    That mattered more than a tuning detail, because this arm carries the comparison's strongest
    theoretical claim and every other arm sits at a default adequate for its objective. Leaving the
    one estimator with a free scale parameter unfitted is not a fair comparison, and a reviewer
    would be right to say so.

    The scale is therefore estimated per fit from the training rows, which is the natural plug-in
    for a log-normal AFT and introduces no tuning loop and no extra data. Exact rows are used where
    available, since a censored bound is not a draw from the response distribution; the finite end
    of each censored interval is used only when too few exact rows exist to estimate a spread.
    """
    exact = lo == hi
    src = lo[exact] if int(exact.sum()) >= 30 else np.where(np.isfinite(hi), hi, lo)
    src = src[src > 0]
    if len(src) < 2:
        return 1.0
    sd = float(np.std(np.log(src)))
    return float(np.clip(sd, 0.05, 1.0))


def fit_xgb_aft(X, lo, hi, seed, scale=None):
    """XGBoost's interval-censored accelerated-failure-time objective."""
    import xgboost as xgb

    if scale is None:
        scale = aft_scale_from_data(lo, hi)
    d = xgb.DMatrix(X)
    d.set_float_info("label_lower_bound", np.maximum(lo, POTENCY_FLOOR))
    d.set_float_info("label_upper_bound", hi)
    params = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
              "aft_loss_distribution": "normal", "aft_loss_distribution_scale": scale,
              "tree_method": "hist", "max_depth": XGB_DEPTH, "eta": XGB_ETA,
              "seed": seed, "verbosity": 0}
    return xgb.train(params, d, num_boost_round=XGB_ROUNDS), scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", default="")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    from benchmark_label_formulations import auroc, average_precision, spearman
    from sai_cns.features.featurize import featurize
    from sai_cns.models.train_rf import RF_COMMON, SEED, _scaffold_groups

    eps = ([e.strip() for e in args.endpoints.split(",") if e.strip()]
           or sorted(p.name[:-7] for p in CACHE.glob("*_y.json")
                     if (CACHE / f"{p.name[:-7]}_censored.json").exists()))
    print(f"endpoints: {', '.join(eps)}")
    print(f"folds: {args.folds} (scaffold-grouped over the union of exact and censored rows)\n")

    fold_rows, diag_rows = [], []
    for ep in eps:
        t0 = time.time()
        smi_e, val = load_exact(ep)
        smi_c, clo, chi = load_censored(ep)
        if not smi_c:
            print(f"[{ep}] no censored cache; skipped")
            continue

        smiles = list(smi_e) + list(smi_c)
        lo = np.concatenate([val, clo])
        hi = np.concatenate([val, chi])
        X, mask = featurize(smiles)
        lo, hi = lo[mask], hi[mask]
        smiles = [s for s, m in zip(smiles, mask) if m]
        groups = _scaffold_groups(smiles)
        X, lo, hi, groups, smiles, rep = dedup_intervals(X, lo, hi, groups, smiles)

        exact = lo == hi
        band = exact & (lo > INACTIVE_CUT) & (lo < ACTIVE_CUT)
        dec = exact & ~band
        settles = (~exact) & (((hi <= INACTIVE_CUT) & np.isfinite(hi))
                              | (lo >= ACTIVE_CUT))
        print(f"[{ep}] {rep['rows_in']:,} rows -> {rep['rows_out']:,} after dedup "
              f"({rep['duplicate_rows_removed']:,} duplicates, "
              f"{rep['contradictory_bounds_dropped']:,} contradictory, "
              f"{rep['exact_outside_its_own_bound']:,} exact-vs-bound disagreements)", flush=True)
        print(f"        exact {int(exact.sum()):,} (decidable {int(dec.sum()):,}, "
              f"band {int(band.sum()):,}); censored {int((~exact).sum()):,} "
              f"({int(settles.sum()):,} settle a class, "
              f"{int(((~exact) & ~settles).sum()):,} usable only by a censored model)", flush=True)
        diag_rows.append({"endpoint": ep, **rep, "n_exact": int(exact.sum()),
                          "n_decidable": int(dec.sum()), "n_band": int(band.sum()),
                          "n_censored": int((~exact).sum()),
                          "n_censored_settling": int(settles.sum()),
                          "n_censored_discarded_by_rule": int(((~exact) & ~settles).sum())})

        n_groups = len(np.unique(groups))
        folds = min(args.folds, n_groups)
        if folds < 2:
            print(f"  skipped: only {n_groups} scaffold group(s)")
            continue

        for k, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(X, lo, groups)):
            te_dec = te[dec[te]]
            te_band = te[band[te]]
            if len(te_dec) < 2 * MIN_PER_CLASS:
                continue
            truth = (lo[te_dec] >= ACTIVE_CUT).astype(int)
            if truth.sum() < MIN_PER_CLASS or (1 - truth).sum() < MIN_PER_CLASS:
                # Said out loud. HT2A has 104 measured inactives panel-wide, so every one
                # of its folds fails here and the endpoint vanished from both benchmark
                # tables while still appearing in the inventory. An endpoint dropped by a
                # data-dependent rule with no record of it is exactly what a referee finds.
                print(f"  [{ep}] fold {k} skipped: {int(truth.sum())} active / "
                      f"{int((1 - truth).sum())} inactive test compounds, "
                      f"below MIN_PER_CLASS={MIN_PER_CLASS}", flush=True)
                continue

            tr_dec = tr[dec[tr]]
            lab = (lo[tr_dec] >= ACTIVE_CUT).astype(int)
            if len(np.unique(lab)) < 2:
                continue
            tr_set = tr[settles[tr]]
            tr_dep = np.concatenate([tr_dec, tr_set])
            lab_dep = np.concatenate([lab, (lo[tr_set] >= ACTIVE_CUT).astype(int)])

            scores = {}

            m = RandomForestClassifier(class_weight="balanced", **RF_COMMON)
            m.fit(X[tr_dec], lab)
            scores["A_rf_binary_exact"] = m.predict_proba(X[te])[:, 1]

            if len(np.unique(lab_dep)) == 2:
                m = RandomForestClassifier(class_weight="balanced", **RF_COMMON)
                m.fit(X[tr_dep], lab_dep)
                scores["Aplus_rf_binary_deployed"] = m.predict_proba(X[te])[:, 1]

            m = fit_rf_censored_em(X[tr], lo[tr], hi[tr], RF_COMMON)
            scores["D_rf_censored_em"] = m.predict(X[te])

            pos = max(int((lab == 0).sum()), 1) / max(int((lab == 1).sum()), 1)
            xc = xgb.XGBClassifier(n_estimators=XGB_ROUNDS, max_depth=XGB_DEPTH,
                                   learning_rate=XGB_ETA, tree_method="hist",
                                   scale_pos_weight=pos, random_state=SEED,
                                   eval_metric="logloss", verbosity=0)
            xc.fit(X[tr_dec], lab)
            scores["E_xgb_binary_exact"] = xc.predict_proba(X[te])[:, 1]

            bst, aft_scale = fit_xgb_aft(X[tr], lo[tr], hi[tr], SEED)
            scores["D_xgb_aft"] = bst.predict(xgb.DMatrix(X[te]))

            pos_in_te = {int(i): j for j, i in enumerate(te)}
            idx_dec = [pos_in_te[int(i)] for i in te_dec]
            idx_band = [pos_in_te[int(i)] for i in te_band]

            for name, sc in scores.items():
                row = {"endpoint": ep, "fold": k, "formulation": name,
                       "n_test": len(te), "n_decidable": len(te_dec), "n_band": len(te_band),
                       "n_active": int(truth.sum()), "n_inactive": int((1 - truth).sum()),
                       "n_train_exact_decidable": int(len(tr_dec)),
                       "n_train_censored": int((~exact[tr]).sum()),
                       "aft_scale": round(aft_scale, 4) if name == "D_xgb_aft" else None,
                       "auroc_decidable": auroc(truth, np.asarray(sc)[idx_dec]),
                       "ap_decidable": average_precision(truth, np.asarray(sc)[idx_dec])}
                if len(idx_band) >= MIN_BAND_FOR_RHO:
                    row["spearman_within_band"] = spearman(np.asarray(sc)[idx_band], lo[te_band])
                fold_rows.append(row)
        print(f"  {time.time() - t0:.0f}s", flush=True)

    if not fold_rows:
        print("no usable folds")
        return 1

    folds_df = pd.DataFrame(fold_rows)
    OUT_FOLDS.parent.mkdir(parents=True, exist_ok=True)
    folds_df.to_csv(OUT_FOLDS, index=False)

    agg = (folds_df.groupby(["endpoint", "formulation"])
           .agg(n_folds=("fold", "count"), auroc=("auroc_decidable", "mean"),
                ap=("ap_decidable", "mean"), rho_band=("spearman_within_band", "mean"),
                n_censored_train=("n_train_censored", "max"))
           .reset_index())
    agg.to_csv(OUT_SUMMARY, index=False)
    pd.DataFrame(diag_rows).to_csv(
        ROOT / "results" / "tables" / "censored_data_inventory.csv", index=False)

    order = ["A_rf_binary_exact", "Aplus_rf_binary_deployed", "D_rf_censored_em",
             "E_xgb_binary_exact", "D_xgb_aft"]
    for label, col in (("AUROC on the decidable exact test compounds", "auroc"),
                       ("average precision", "ap"),
                       ("Spearman against true potency WITHIN the band", "rho_band")):
        piv = agg.pivot(index="endpoint", columns="formulation", values=col)
        piv = piv[[c for c in order if c in piv.columns]]
        print(f"\n=== {label} ===")
        print(piv.round(4).to_string())
        print("\npanel mean:")
        print(piv.mean().round(4).to_string())

    piv = agg.pivot(index="endpoint", columns="formulation", values="auroc")
    print("\n=== paired differences in AUROC across endpoints (Wilcoxon, two-sided) ===")
    try:
        from scipy.stats import wilcoxon
        pairs = [("Aplus_rf_binary_deployed", "A_rf_binary_exact"),
                 ("D_rf_censored_em", "A_rf_binary_exact"),
                 ("D_rf_censored_em", "Aplus_rf_binary_deployed"),
                 ("D_xgb_aft", "E_xgb_binary_exact"),
                 ("E_xgb_binary_exact", "A_rf_binary_exact")]
        for b, a in pairs:
            if b not in piv.columns or a not in piv.columns:
                continue
            both = piv[[a, b]].dropna()
            if len(both) < 5:
                print(f"  {b} vs {a}: n={len(both)}, too few to test")
                continue
            d = both[b] - both[a]
            st = wilcoxon(both[b], both[a])
            print(f"  {b}\n    minus {a}: mean {d.mean():+.4f}  median {d.median():+.4f}  "
                  f"wins {int((d > 0).sum())}/{len(d)}  p={st.pvalue:.4f}  "
                  f"(floor for n={len(d)} is {2 ** -(len(d) - 1):.4f})")
    except ImportError:
        print("  scipy unavailable; paired test skipped")

    print(f"\nwrote {OUT_FOLDS.relative_to(ROOT).as_posix()}")
    print(f"wrote {OUT_SUMMARY.relative_to(ROOT).as_posix()}")
    print("wrote results/tables/censored_data_inventory.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
