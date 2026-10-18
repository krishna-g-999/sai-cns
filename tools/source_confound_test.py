"""Does provenance leak into the headline AUROC? Measured, with a size-matched control.

THE CONCERN. Rows in the endpoint tables carry a source tag, and for AChE that tag is not
independent of the label: every BindingDB row and every BindingDB-and-ChEMBL row is an active,
every ChEMBL_inactive row is an inactive, and together those account for 45 per cent of the table.
Provenance is not a feature and cannot leak directly. But the sources curate different chemistry,
so a model can in principle learn "looks like a BindingDB compound" as a proxy for "active", and
the reported AUROC would then be measuring the proxy rather than the pharmacology.

THE OBVIOUS TEST, AND WHY IT IS NOT ENOUGH. Refit on the ChEMBL-only subset, where provenance
carries no information about the label, and compare. That comparison is confounded in turn: it
also cuts the sample roughly in half, and a smaller training set scores worse for reasons that
have nothing to do with the confound. Reporting that difference as leakage would be an error, and
a tempting one, because it runs in the direction a sceptical reader expects.

So this script runs the control as well. Scaffold-grouped subsamples of the FULL table, matched to
the size of the ChEMBL-only subset, give the distribution the ChEMBL-only score should be read
against. If the ChEMBL-only score sits inside that distribution, the apparent inflation is a
sample-size effect and the confound has no detectable influence on the headline. The spread of the
subsamples also sets the detection floor, which is reported rather than left implicit: a test that
cannot see an inflation smaller than its own noise should say so.

Every fit is scaffold-grouped (GroupKFold on Bemis-Murcko scaffolds), because a random split of
medicinal-chemistry data mostly measures recall of the series a model was trained on.

Run:
    python tools/source_confound_test.py
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

OUT = ROOT / "results" / "tables" / "source_confound_test.csv"
ENDPOINT = "AChE"
N_SUBSAMPLES = 5
RF = dict(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
          n_jobs=-1, random_state=0)


def auroc(y, p):
    pr = sorted(zip(p, y))
    rk, i = [0.0] * len(pr), 0
    while i < len(pr):
        j = i
        while j + 1 < len(pr) and pr[j + 1][0] == pr[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rk[k] = r
        i = j + 1
    pos = sum(1 for _, v in pr if v == 1)
    neg = len(pr) - pos
    if not pos or not neg:
        return float("nan")
    s = sum(r for r, (_, v) in zip(rk, pr) if v == 1)
    return (s - pos * (pos + 1) / 2.0) / (pos * neg)


def main() -> int:
    warnings.filterwarnings("ignore")
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    RDLogger.DisableLog("rdApp.*")
    import app as A

    rows = list(csv.DictReader((ROOT / "data" / "endpoints" / f"{ENDPOINT}.csv")
                               .open(encoding="utf-8")))
    X, y, g, src = [], [], [], []
    for r in rows:
        f = A.featurize_one(r["smiles"])
        if f is None:
            continue
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None:
            continue
        try:
            sk = MurckoScaffold.MurckoScaffoldSmiles(mol=m) or ""
        except Exception:
            continue
        X.append(f); y.append(int(r["label"])); g.append(sk); src.append(r["source"])
    X = np.vstack(X); y = np.asarray(y); g = np.asarray(g); src = np.asarray(src)

    det = sum(1 for s, l in zip(src, y)
              if (s in ("BindingDB", "BindingDB+ChEMBL") and l == 1)
              or (s == "ChEMBL_inactive" and l == 0))
    print(f"  {ENDPOINT}: {len(y):,} rows, {det:,} source-determined "
          f"({det / len(y) * 100:.1f} per cent)")

    def cv(Xs, ys, gs):
        oof = np.zeros(len(ys))
        for tr, te in GroupKFold(n_splits=5).split(Xs, ys, gs):
            oof[te] = RandomForestClassifier(**RF).fit(Xs[tr], ys[tr]).predict_proba(Xs[te])[:, 1]
        return auroc(ys.tolist(), oof.tolist())

    is_bdb = (np.char.find(src.astype(str), "BindingDB") >= 0).astype(int)
    a_src = cv(X, is_bdb, g)
    a_all = cv(X, y, g)
    m = src == "ChEMBL"
    a_ch = cv(X[m], y[m], g[m])
    n_ch = int(m.sum())
    print(f"  source predictable from structure alone   AUROC {a_src:.4f}")
    print(f"  label, all rows            n={len(y):>6,}   AUROC {a_all:.4f}")
    print(f"  label, ChEMBL-only         n={n_ch:>6,}   AUROC {a_ch:.4f}")
    print(f"  apparent inflation {a_all - a_ch:+.4f}, which the control below tests\n")

    rng = np.random.RandomState(0)
    uniq = np.unique(g)
    vals = []
    for _ in range(N_SUBSAMPLES):
        order = rng.permutation(uniq)
        keep, n = set(), 0
        for s in order:
            idx = np.where(g == s)[0]
            if n + len(idx) > n_ch:
                continue
            keep.update(idx.tolist()); n += len(idx)
            if n >= n_ch * 0.98:
                break
        sel = np.array(sorted(keep))
        if len(np.unique(y[sel])) < 2:
            continue
        a = cv(X[sel], y[sel], g[sel]); vals.append(a)
        print(f"  size-matched subsample of full table  n={len(sel):>6,}   AUROC {a:.4f}")

    mu, sd = float(np.mean(vals)), float(np.std(vals))
    print(f"\n  size-matched full-table mean {mu:.4f} (sd {sd:.4f})")
    inside = abs(a_ch - mu) <= 2 * sd
    print(f"  ChEMBL-only {a_ch:.4f} is {'INSIDE' if inside else 'OUTSIDE'} "
          f"two standard deviations of the size-matched mean")
    print(f"  detection floor about {2 * sd:.3f} AUROC")
    print("\n  Inside the interval means the apparent inflation is a sample-size effect and the")
    print("  confound has no detectable influence on the headline. It does not mean none exists.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "n", "auroc"])
        w.writerow(["source_predictable_from_structure", len(y), round(a_src, 4)])
        w.writerow(["label_all_rows", len(y), round(a_all, 4)])
        w.writerow(["label_chembl_only", n_ch, round(a_ch, 4)])
        for i, a in enumerate(vals):
            w.writerow([f"size_matched_subsample_{i + 1}", n_ch, round(a, 4)])
        w.writerow(["size_matched_mean", n_ch, round(mu, 4)])
        w.writerow(["size_matched_sd", n_ch, round(sd, 4)])
        w.writerow(["rows_source_determined_fraction", len(y), round(det / len(y), 4)])
    print(f"\n  wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
