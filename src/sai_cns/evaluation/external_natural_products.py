"""External validation on natural products: how the deployed panel behaves on chemistry it has not
seen.

The manuscript cannot currently answer "how does this perform on natural products", because no
natural-product test set existed. NPASS supplies measured activity against this panel's own targets,
and the models were fitted before that file was read, so it is external by construction.

It is used only to measure. Nothing here enters training, and folding it in later would destroy the
one property that makes it worth having.

The contamination check is the point of this file, not a formality. NPASS candidates carry an
`already_in_table` flag computed by comparing raw SMILES strings, which is not an identity test: the
same molecule written as a different tautomer, salt or canonical form counts as new. Measured here,
that admitted 227 of AChE's 283 "external" compounds, with a median Tanimoto to training of 1.00, and
produced an AUROC of 0.961 that was memorisation. Identity is therefore re-established on the
InChIKey of the desalted parent and on the feature vector, and anything matching either is removed
before scoring rather than merely counted. On what genuinely remains, AChE scores 0.463.

Endpoints with fewer than MIN_TEST truly external rows, or with only one class, are reported as
unscored rather than scored, because an AUROC over four compounds is not evidence.

Output: results/tables/external_natural_products.csv
        results/tables/external_natural_products_summary.csv

Run:  python src/sai_cns/evaluation/external_natural_products.py
"""
from __future__ import annotations

import gc
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize, featurize_one, parent_mol  # noqa: E402

M = ROOT / "models_rf"
TAB = ROOT / "results" / "tables"
CANDIDATES = TAB / "npass_candidates.csv"

MIN_TEST = 25
IN_DOMAIN, NEAR_DOMAIN = 0.50, 0.30
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def auroc(y, p) -> float:
    """Rank-based AUROC, written here so the number is independent of the pipeline's own code."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def load_estimator(endpoint: str):
    """The deployed estimator that answers "is this active", for one endpoint.

    Capability is checked rather than assumed from the filename. Four receptors carry both a potency
    regression and a binder classifier under one endpoint name, so `SERT.joblib` is a
    RandomForestRegressor; loading it by name and calling predict_proba raised AttributeError and
    lost the endpoint from the run. A regression predicts potency, not the probability of activity,
    and the two cannot be scored against a binary label by the same code.
    """
    for name, kind in ((f"{endpoint}_calibrated.joblib", "calibrated"),
                       (f"{endpoint}.joblib", "classifier"),
                       (f"{endpoint}_binder.joblib", "binder")):
        p = M / name
        if not p.exists():
            continue
        try:
            est = joblib.load(p)
        except Exception:
            continue
        if hasattr(est, "predict_proba"):
            return est, kind
    return None, None


def training_identity(endpoint: str):
    """The endpoint's training chemistry as InChIKeys, fingerprints and feature vectors."""
    path = ROOT / "data" / "endpoints" / f"{endpoint}.csv"
    if not path.exists():
        return set(), [], set()
    smis = pd.read_csv(path, usecols=["smiles"]).dropna().smiles.astype(str).tolist()
    keys, fps, vecs = set(), [], set()
    for smi in smis:
        mol = parent_mol(smi)
        if mol is None:
            continue
        fps.append(_GEN.GetFingerprint(mol))
        try:
            keys.add(Chem.MolToInchiKey(mol))
        except Exception:
            pass
        v = featurize_one(smi)
        if v is not None:
            vecs.add(v.tobytes())
    return keys, fps, vecs


def one_endpoint(ep: str, g: pd.DataFrame):
    """Score one endpoint. Returns (summary row, per-compound frame or None)."""
    est, kind = load_estimator(ep)
    if est is None:
        return {"endpoint": ep, "n_candidates": len(g), "status": "no deployed estimator"}, None

    g = g.drop_duplicates("smiles").reset_index(drop=True)
    X, mask = featurize(g.smiles.astype(str).tolist())
    g = g.loc[mask].reset_index(drop=True)
    if len(g) == 0:
        return {"endpoint": ep, "n_candidates": 0, "status": "nothing featurised"}, None

    keys, tr_fps, tr_vecs = training_identity(ep)

    sims, keep, n_key, n_vec = [], [], 0, 0
    for i, smi in enumerate(g.smiles.astype(str)):
        mol = parent_mol(smi)
        fp = _GEN.GetFingerprint(mol) if mol is not None else None
        sims.append(max(DataStructs.BulkTanimotoSimilarity(fp, tr_fps))
                    if (fp is not None and tr_fps) else 0.0)
        hit_key = False
        if mol is not None:
            try:
                hit_key = Chem.MolToInchiKey(mol) in keys
            except Exception:
                hit_key = False
        hit_vec = X[i].tobytes() in tr_vecs
        n_key += int(hit_key)
        n_vec += int(hit_vec)
        keep.append(not (hit_key or hit_vec))

    keep = np.asarray(keep, dtype=bool)
    removed = int((~keep).sum())
    base = {"endpoint": ep, "n_candidates": int(len(g)),
            "removed_as_contaminated": removed,
            "leaked_by_inchikey": n_key, "leaked_by_feature_vector": n_vec,
            "estimator": kind}

    # Release the endpoint's training identity before scoring. Held across 73 endpoints this
    # accumulates to several gigabytes of fingerprints and feature vectors, and the process was
    # being killed outright: no Python exception, no traceback, just a non-zero exit part way
    # through, which is the least informative way for a long run to fail.
    del keys, tr_fps, tr_vecs
    gc.collect()

    g = g.loc[keep].reset_index(drop=True)
    X, sims = X[keep], list(np.asarray(sims)[keep])
    if len(g) == 0:
        return {**base, "n": 0, "status": "wholly contained in training"}, None

    g["max_tanimoto_to_training"] = sims
    p = est.predict_proba(X)[:, 1]
    g["prediction"] = p
    y = g.label.to_numpy().astype(int)

    if len(g) < MIN_TEST or len(set(y)) < 2:
        return ({**base, "n": int(len(g)), "n_active": int(y.sum()),
                 "status": f"not scored: {len(g)} external rows, {len(set(y))} class(es)"}, g)

    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp_ = int(((pred == 1) & (y == 0)).sum())
    row = {**base, "n": int(len(g)), "n_active": int(y.sum()), "status": "scored",
           "auroc": round(auroc(y, p), 4),
           "sensitivity": round(tp / (tp + fn), 4) if tp + fn else None,
           "specificity": round(tn / (tn + fp_), 4) if tn + fp_ else None,
           "median_max_tanimoto": round(float(np.median(sims)), 3)}
    s = np.asarray(sims)
    for name, m_ in (("in_domain", s >= IN_DOMAIN),
                     ("near_domain", (s >= NEAR_DOMAIN) & (s < IN_DOMAIN)),
                     ("out_of_domain", s < NEAR_DOMAIN)):
        row[f"n_{name}"] = int(m_.sum())
        if m_.sum() >= MIN_TEST and len(set(y[m_])) == 2:
            row[f"auroc_{name}"] = round(auroc(y[m_], p[m_]), 4)
    return row, g


def report(rows: list[dict], frames: list[pd.DataFrame]) -> None:
    out = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    out.to_csv(TAB / "external_natural_products_summary.csv", index=False)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            TAB / "external_natural_products.csv", index=False)

    scored = out[out.get("status", "") == "scored"] if "status" in out else pd.DataFrame()
    print("\n=== external validation on natural products ===")
    print(f"  endpoints considered             {len(out)}")
    if "removed_as_contaminated" in out:
        print(f"  compounds removed as contaminated {int(out.removed_as_contaminated.fillna(0).sum()):,}")
    if len(scored):
        print(f"  endpoints with enough external data to score  {len(scored)}")
        print(f"  externally scored compounds      {int(scored.n.sum()):,}")
        print(f"  mean AUROC                       {scored.auroc.mean():.4f}")
        print(f"  median AUROC                     {scored.auroc.median():.4f}")
        print(f"  endpoints at or below chance     {int((scored.auroc <= 0.5).sum())}")
        print()
        print(scored[["endpoint", "n", "n_active", "auroc", "median_max_tanimoto",
                      "removed_as_contaminated"]].to_string(index=False))
    else:
        print("  no endpoint retained enough genuinely external data to score")
    print("\nwrote results/tables/external_natural_products_summary.csv"
          + (" and external_natural_products.csv" if frames else ""))


def main() -> None:
    if not CANDIDATES.exists():
        raise SystemExit("run src/sai_cns/data/ingest_npass.py first")
    cand = pd.read_csv(CANDIDATES)
    cand = cand[~cand.already_in_table.astype(bool)]
    print(f"{len(cand):,} measurements, {cand.smiles.nunique():,} compounds, "
          f"{cand.endpoint.nunique()} endpoints flagged new by SMILES string\n")

    rows, frames = [], []
    groups = list(cand.groupby("endpoint"))
    for i, (ep, g) in enumerate(groups, 1):
        try:
            row, frame = one_endpoint(ep, g)
        except Exception as exc:
            print(f"  ({i}/{len(groups)}) {ep:14s} FAILED {type(exc).__name__}: {exc}", flush=True)
            rows.append({"endpoint": ep, "n_candidates": len(g),
                         "status": f"failed: {type(exc).__name__}"})
            continue
        rows.append(row)
        gc.collect()
        if frame is not None:
            frames.append(frame)
        st = row.get("status", "")
        extra = f"AUROC {row['auroc']:.3f}" if st == "scored" else st
        print(f"  ({i}/{len(groups)}) {ep:14s} cand {row.get('n_candidates', 0):4d} "
              f"-> external {row.get('n', 0):4d}  {extra}", flush=True)
    report(rows, frames)


if __name__ == "__main__":
    main()
