"""Prospective validation: what would this panel have said about chemistry that did not yet exist?

External validation is the thinnest part of this project's evidence, and the reason is structural
rather than neglect. An external test set must be measured chemistry absent from training, and for
most of these targets the public measured chemistry *is* the training set. Buying a genuinely
independent set would mean running new assays.

There is one axis along which an independent set can still be constructed from what exists, and it
is the axis that matters most for a server people will submit new compounds to: time. Every measured
row carries the year of the document it came from. Freezing the data at a year, fitting the panel on
what was known then, and testing on compounds first published afterwards reproduces the situation a
user is actually in. It is the accepted simulator of prospective performance (Sheridan, J Chem Inf
Model 2013), and it is stronger evidence than a random split because a compound published after the
cutoff was not available to be memorised, whatever its scaffold.

What makes this a real test rather than a comfortable one:

  The models are refitted. Scoring the deployed models on post-cutoff compounds would measure
  nothing, because the deployed models were fitted on them. Every endpoint here is trained again
  from its pre-cutoff rows alone.

  The operating point is also frozen. A threshold tuned on the full data leaks the future into the
  decision even when the model does not. Thresholds here are quantiles of pre-cutoff held-out
  measured inactives, so the reported sensitivity is what a user in the cutoff year would have got.

  Decoys are matched to pre-cutoff actives only, and every post-cutoff compound is barred from decoy
  eligibility, so a test compound can never appear as a training negative.

  Nothing else changes. Same featuriser, same forest, same calibration, same target false-positive
  rate as the deployed panel, so the difference between these numbers and the deployed ones is the
  cost of not knowing the future and nothing else.

The headline number is not the point. A time split flatters a field that publishes analogue series:
if the compounds appearing after the cutoff are close relatives of those before it, high performance
demonstrates interpolation and not much else. Every test compound is therefore scored for its
maximum Tanimoto similarity to the pre-cutoff training actives, and the companion analysis
(external_novelty_strata.py) reports performance as a function of that distance. The question worth
answering is not whether the panel works on later chemistry but whether it works on later chemistry
that does not resemble what it was trained on.

Outputs: results/tables/external_prospective.csv         one row per endpoint
         results/tables/external_prospective_compounds.csv  one row per test compound

Run:  python src/sai_cns/evaluation/external_prospective.py            (all deployed endpoints)
      python src/sai_cns/evaluation/external_prospective.py D2 SERT    (named endpoints)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize  # noqa: E402
from models.pools import background_pools  # noqa: E402
import panel  # noqa: E402

TAB = ROOT / "results" / "tables"
CACHE = ROOT / "data" / "_prospective_cache"

# Every constant below is the deployed one, imported in spirit rather than by name so that this file
# reads as a specification of what was done. They must not drift from train_binders_hybrid.py.
ACTIVE_P, DECOY_RATIO, TAN_MAX = 7.0, 3, 0.35
MEASURED_ACTIVE_P, MEASURED_INACTIVE_P = 6.0, 5.0
TARGET_FPR = 0.10
RF = dict(n_estimators=300, min_samples_leaf=4, n_jobs=-1, random_state=42,
          class_weight="balanced")
SEED = 42

CUT_PERCENTILE = 75      # the convention already used by rf_conformal_temporal.py
MIN_TRAIN_ACTIVES = 100  # the deployed trainer's own floor for fitting an endpoint at all
MIN_TEST_ACTIVES = 30    # below this a prospective AUROC is an anecdote
MIN_TEST_NEGATIVES = 30
N_BACKGROUND = 6000      # evaluation-pool sample used as the presumed-inactive comparator

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
rng = np.random.default_rng(SEED)


def _fp(smi):
    m = Chem.MolFromSmiles(str(smi))
    return None if m is None else _GEN.GetFingerprint(m)


def _max_sim(query_fps, ref_fps) -> np.ndarray:
    """Maximum Tanimoto of each query against the reference set, the novelty axis."""
    out = np.zeros(len(query_fps))
    for i, fp in enumerate(query_fps):
        out[i] = 0.0 if fp is None or not ref_fps else max(
            DataStructs.BulkTanimotoSimilarity(fp, ref_fps))
    return out


def _split_by_year(df: pd.DataFrame, smiles: list[str], cut: int) -> tuple[list, list]:
    """Earliest recorded year per structure decides its side of the wall.

    A compound measured in 2012 and again in 2021 belongs before the cutoff: it was available to be
    known. Taking the latest year instead would move known chemistry into the test set and inflate
    every number here.
    """
    y = pd.to_numeric(df["year"], errors="coerce")
    first = df.assign(_y=y).dropna(subset=["_y"]).groupby("smiles")["_y"].min()
    pre, post = [], []
    for s in smiles:
        yr = first.get(s)
        if yr is None or np.isnan(yr):
            pre.append(s)          # undated chemistry is treated as already known, never as future
        else:
            (post if yr > cut else pre).append(s)
    return pre, post


def _load_pools():
    pools = background_pools(with_fingerprints=True)
    dec_smiles, dec_fps = pools["decoy"]
    eval_smiles, _ = pools["evaluation"]
    CACHE.mkdir(parents=True, exist_ok=True)
    dec_cache, ev_cache = CACHE / "decoy.npz", CACHE / "evaluation.npz"

    if dec_cache.exists():
        z = np.load(dec_cache, allow_pickle=True)
        bgX, bg_ok, bg_keep = z["X"], z["smiles"], z["keep"]
    else:
        print(f"featurising {len(dec_smiles):,} decoy-pool compounds (cached after this run)",
              flush=True)
        bgX, bgm = featurize(list(dec_smiles))
        bg_ok = np.array(dec_smiles)[bgm]
        bg_keep = np.asarray(bgm)
        np.savez_compressed(dec_cache, X=bgX, smiles=bg_ok, keep=bg_keep)
    bg_fp_ok = [f for f, k in zip(dec_fps, bg_keep) if k]

    if ev_cache.exists():
        z = np.load(ev_cache, allow_pickle=True)
        evX = z["X"]
    else:
        print(f"featurising {len(eval_smiles):,} evaluation-pool compounds", flush=True)
        evX, evm = featurize(list(eval_smiles))
        np.savez_compressed(ev_cache, X=evX)
    if len(evX) > N_BACKGROUND:
        evX = evX[np.random.default_rng(SEED).choice(len(evX), N_BACKGROUND, replace=False)]
    return bgX, bg_ok, bg_fp_ok, evX


def _fit(X, y):
    """The deployed fit exactly: forest, then Platt calibration on a stratified fifth."""
    Xt, Xc, yt, yc = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    forest = RandomForestClassifier(**RF).fit(Xt, yt)
    return CalibratedClassifierCV(FrozenEstimator(forest), method="sigmoid").fit(Xc, yc)


def _run_split(split, ep, mode, act_train, act_test, ina_train, ina_thr, ina_test,
               barred, bgX, bg_ok, bg_fp_ok, evX):
    """Fit and score one split. Identical code for both, so the two are comparable by construction.

    `barred` is every measured compound at this endpoint, whichever side of the split it fell on, so
    a test compound can never be drawn as a training decoy.
    """
    aX, am = featurize(act_train)
    act_train = [s for s, k in zip(act_train, am) if k]
    a_fps = [f for f in (_fp(s) for s in act_train) if f is not None]

    dec = []
    if mode != panel.MEASURED_LABEL:
        # Property-matched to the training actives of this split only. Matching to all actives would
        # let the composition of the test set choose the training negatives.
        bg_mw, bg_lp = bgX[:, -12], bgX[:, -11]
        mw_lo, mw_hi = np.quantile(aX[:, -12], .02), np.quantile(aX[:, -12], .98)
        lp_lo, lp_hi = np.quantile(aX[:, -11], .02), np.quantile(aX[:, -11], .98)
        cand = (bg_mw >= mw_lo) & (bg_mw <= mw_hi) & (bg_lp >= lp_lo) & (bg_lp <= lp_hi)
        idx = [i for i in np.where(cand)[0] if bg_ok[i] not in barred]
        np.random.default_rng(SEED).shuffle(idx)
        need = max(DECOY_RATIO * len(act_train) - len(ina_train), len(act_train))
        for i in idx:
            fp = bg_fp_ok[i]
            if fp is None:
                continue
            if max(DataStructs.BulkTanimotoSimilarity(fp, a_fps)) < TAN_MAX:
                dec.append(str(bg_ok[i]))
            if len(dec) >= need:
                break

    smiles = act_train + dec + ina_train
    y = np.array([1] * len(act_train) + [0] * (len(dec) + len(ina_train)))
    X, mask = featurize(smiles)
    y = y[mask]
    if len(np.unique(y)) < 2 or (y == 1).sum() < MIN_TRAIN_ACTIVES:
        return None, []
    cal = _fit(X, y)

    def score(smis):
        if not smis:
            return np.array([]), []
        Xq, m = featurize(list(smis))
        return cal.predict_proba(Xq)[:, 1], [s for s, k in zip(smis, m) if k]

    p_thr, _ = score(ina_thr)
    if len(p_thr) >= 15:
        thr = float(np.clip(np.quantile(p_thr, 1.0 - TARGET_FPR), 0.05, 0.999))
        basis = "held-out measured inactives from the training side of the split"
    else:
        thr = 0.40
        basis = "global fallback (too few held-out inactives on the training side)"

    p_act, act_kept = score(act_test)
    p_ina, ina_kept = score(ina_test)
    p_bg = cal.predict_proba(evX)[:, 1]
    if not len(p_act):
        return None, []

    auroc_meas = (round(float(roc_auc_score(np.r_[np.ones(len(p_act)), np.zeros(len(p_ina))],
                                            np.r_[p_act, p_ina])), 4)
                  if len(p_ina) >= MIN_TEST_NEGATIVES else None)
    auroc_bg = round(float(roc_auc_score(np.r_[np.ones(len(p_act)), np.zeros(len(p_bg))],
                                         np.r_[p_act, p_bg])), 4)
    nov_a = _max_sim([_fp(s) for s in act_kept], a_fps)
    nov_i = _max_sim([_fp(s) for s in ina_kept], a_fps) if ina_kept else np.array([])

    metrics = {
        "n_train_actives": len(act_train), "n_train_decoys": len(dec),
        "n_train_measured_inactives": len(ina_train),
        "n_test_actives": len(p_act), "n_test_measured_inactives": len(p_ina),
        "auroc_vs_measured_inactives": auroc_meas, "auroc_vs_background": auroc_bg,
        "threshold": round(thr, 4), "threshold_basis": basis,
        "sensitivity": round(float((p_act >= thr).mean()), 4),
        "fpr_measured_inactives": (round(float((p_ina >= thr).mean()), 4) if len(p_ina) else None),
        "fpr_background": round(float((p_bg >= thr).mean()), 4),
        "median_test_novelty": round(float(np.median(nov_a)), 4),
        "test_actives_below_tanimoto_0.4": int((nov_a < 0.40).sum()),
    }
    # Both classes are recorded so novelty can be stratified on a matched comparison. Keeping only
    # the actives would allow a stratum whose actives are novel and whose negatives are not, and the
    # AUROC of that stratum would measure the mismatch rather than the model.
    compounds = ([{"endpoint": ep, "split": split, "smiles": s, "measured": 1,
                   "probability": round(float(pp), 5), "called": bool(pp >= thr),
                   "max_tanimoto_to_training": round(float(nv), 4)}
                  for s, pp, nv in zip(act_kept, p_act, nov_a)] +
                 [{"endpoint": ep, "split": split, "smiles": s, "measured": 0,
                   "probability": round(float(pp), 5), "called": bool(pp >= thr),
                   "max_tanimoto_to_training": round(float(nv), 4)}
                  for s, pp, nv in zip(ina_kept, p_ina, nov_i)])
    return metrics, compounds


def _endpoint(ep, mode, bgX, bg_ok, bg_fp_ok, evX, deployed_rec):
    """Two splits of the same endpoint, matched in size, differing only in how they were cut.

    The size match is the whole point. A time split trains on the earlier three quarters, so it has
    less data than the deployed model as well as none of the future. Comparing it against the
    deployed figure therefore measures two things at once and cannot say which caused a drop. The
    random control trains on the same number of actives drawn without regard to date and tests on
    the same number, so the difference between the two isolates the cost of not knowing the future.
    """
    f = ROOT / "data" / "endpoints" / f"{ep}.csv"
    if not f.exists():
        return None, []
    df = pd.read_csv(f).dropna(subset=["smiles"])
    df["smiles"] = df["smiles"].astype(str)
    if "year" not in df.columns:
        return {"endpoint": ep, "status": "no year column"}, []
    p = pd.to_numeric(df.get("pchembl"), errors="coerce")

    if mode == panel.MEASURED_LABEL:
        act = sorted(set(df.loc[p >= MEASURED_ACTIVE_P, "smiles"]))
        ina_all = sorted(set(df.loc[(p < MEASURED_INACTIVE_P) | (df["label"] == 0), "smiles"])
                         - set(act))
    else:
        act = sorted(set(df.loc[p >= ACTIVE_P, "smiles"]))
        ina_all = sorted(set(df.loc[df["label"] == 0, "smiles"]) - set(act))

    yr_act = pd.to_numeric(df.loc[df["smiles"].isin(act), "year"], errors="coerce").dropna()
    if len(yr_act) < 50:
        return {"endpoint": ep, "status": "too few dated actives"}, []
    cut = int(np.percentile(yr_act, CUT_PERCENTILE))

    act_pre, act_post = _split_by_year(df, act, cut)
    ina_pre, ina_post = _split_by_year(df, ina_all, cut)
    if len(act_pre) < MIN_TRAIN_ACTIVES:
        return {"endpoint": ep, "status": "only %d pre-cutoff actives" % len(act_pre),
                "cutoff_year": cut}, []
    if len(act_post) < MIN_TEST_ACTIVES:
        return {"endpoint": ep, "status": "only %d post-cutoff actives" % len(act_post),
                "cutoff_year": cut}, []

    g = np.random.default_rng(SEED)
    ina_pre_p = list(g.permutation(ina_pre))
    half = len(ina_pre_p) // 2
    barred = set(act) | set(ina_all)

    base = {"endpoint": ep, "mode": mode, "status": "ok", "cutoff_year": cut,
            "n_actives_total": len(act), "n_measured_inactives_total": len(ina_all)}
    row, comps = dict(base), []

    m_t, c_t = _run_split("time", ep, mode, act_pre, act_post,
                          ina_pre_p[:half], ina_pre_p[half:], ina_post,
                          barred, bgX, bg_ok, bg_fp_ok, evX)
    if m_t is None:
        return {**base, "status": "degenerate training set"}, []
    row.update({"time_" + k: v for k, v in m_t.items()})
    comps += c_t

    # The size-matched random control. Same counts, dates ignored.
    perm_a = list(g.permutation(act))
    r_train = perm_a[:len(act_pre)]
    r_test = perm_a[len(act_pre):len(act_pre) + len(act_post)]
    perm_i = list(g.permutation(ina_all))
    n_tr, n_th, n_te = len(ina_pre_p[:half]), len(ina_pre_p[half:]), len(ina_post)
    r_ina_train = perm_i[:n_tr]
    r_ina_thr = perm_i[n_tr:n_tr + n_th]
    r_ina_test = perm_i[n_tr + n_th:][:n_te]
    m_r, c_r = _run_split("random", ep, mode, r_train, r_test,
                          r_ina_train, r_ina_thr, r_ina_test,
                          barred, bgX, bg_ok, bg_fp_ok, evX)
    if m_r is not None:
        row.update({"random_" + k: v for k, v in m_r.items()})
        comps += c_r
        if m_t["auroc_vs_measured_inactives"] and m_r["auroc_vs_measured_inactives"]:
            row["auroc_cost_of_prospectivity"] = round(
                m_r["auroc_vs_measured_inactives"] - m_t["auroc_vs_measured_inactives"], 4)
        row["sensitivity_cost_of_prospectivity"] = round(
            m_r["sensitivity"] - m_t["sensitivity"], 4)

    row.update({
        "deployed_auroc_vs_measured_inactives": deployed_rec.get("auroc_vs_measured_inactives"),
        "deployed_sensitivity": deployed_rec.get("sensitivity_at_threshold"),
        "deployed_threshold": deployed_rec.get("threshold")})
    return row, comps


def main() -> None:
    want = sys.argv[1:]
    modes = json.loads((ROOT / "models_rf" / "binder_modes.json").read_text())
    eps = [(e.name, e.mode) for e in panel.binders(deployed=True)]
    if want:
        eps = [(n, m) for n, m in eps if n in want]
    print(f"prospective validation of {len(eps)} deployed endpoints", flush=True)

    bgX, bg_ok, bg_fp_ok, evX = _load_pools()
    print(f"background: {len(bg_ok):,} decoy-eligible, {len(evX):,} evaluation compounds\n",
          flush=True)

    rows, comps = [], []
    for i, (ep, mode) in enumerate(eps, 1):
        try:
            row, cs = _endpoint(ep, mode, bgX, bg_ok, bg_fp_ok, evX, modes.get(ep, {}))
        except Exception as exc:                      # one bad endpoint must not lose 40 retrains
            row, cs = {"endpoint": ep, "mode": mode, "status": f"error: {exc}"}, []
        if row is None:
            continue
        rows.append(row)
        comps.extend(cs)
        if row.get("status") == "ok":
            a = row.get("time_auroc_vs_measured_inactives")
            r = row.get("random_auroc_vs_measured_inactives")
            fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "  -  "
            print(f"[{i:>2}/{len(eps)}] {ep:12s} cut {row['cutoff_year']}  "
                  f"train {row['time_n_train_actives']:>5} -> test "
                  f"{row['time_n_test_actives']:>4}  "
                  f"AUROC time {fmt(a)} / random {fmt(r)}  "
                  f"sens {row['time_sensitivity']:.3f} vs "
                  f"{row.get('random_sensitivity', float('nan')):.3f} "
                  f"(deployed {row['deployed_sensitivity']})", flush=True)
        else:
            print(f"[{i:>2}/{len(eps)}] {ep:12s} skipped: {row['status']}", flush=True)

        TAB.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(TAB / "external_prospective.csv", index=False)
        pd.DataFrame(comps).to_csv(TAB / "external_prospective_compounds.csv", index=False)

    ok = pd.DataFrame(rows)
    ok = ok[ok.status == "ok"] if len(ok) else ok
    print()
    if len(ok):
        num = lambda c: pd.to_numeric(ok.get(c), errors="coerce")
        at, ar = num("time_auroc_vs_measured_inactives"), num("random_auroc_vs_measured_inactives")
        both = at.notna() & ar.notna()
        print(f"  endpoints validated prospectively   : {len(ok)}")
        print(f"  mean AUROC, time split              : {at.mean():.4f}  (n={int(at.notna().sum())})")
        print(f"  mean AUROC, size-matched random     : {ar.mean():.4f}")
        print(f"  mean cost of prospectivity, AUROC   : "
              f"{(ar[both] - at[both]).mean():+.4f}  (n={int(both.sum())} paired)")
        print(f"  mean sensitivity, time split        : {num('time_sensitivity').mean():.4f}")
        print(f"  mean sensitivity, random control    : {num('random_sensitivity').mean():.4f}")
        print(f"  mean sensitivity, deployed panel    : {num('deployed_sensitivity').mean():.4f}")
        print(f"  mean background FPR, time split     : {num('time_fpr_background').mean():.4f}")
        print(f"  test compounds scored               : {len(comps):,}")
    print(f"\nwrote {(TAB / 'external_prospective.csv').relative_to(ROOT)} and "
          f"{(TAB / 'external_prospective_compounds.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
