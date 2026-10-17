"""Would refitting a receptor regressor on the full potency range remove its optimism?

THE DECISION THIS INFORMS. The four receptor regressors over-predict by 1.04 log units on the
compounds a user actually sees a number for. The diagnosis is that they are fitted on binder-only
assays, so they answer "given that this binds, how potent is it" and have never seen a weak
compound. The obvious remedy is to refit on the full potency range.

That remedy is expensive in a way the others are not. Refitting changes the deployed estimators,
which invalidates every validation figure in both manuscripts: the scaffold cross-validation, the
binder panel AUROCs, the calibration error, the external hold-outs, the specificity, all of it
would have to be recomputed, and the published model archive republished. It is not a change to
make on a diagnosis alone.

So this script tests the diagnosis before anyone acts on it, without touching a deployed model.
For one receptor it fits two regressors on identical features and an identical test set:

    binder-only   compounds with pChEMBL >= 7, reproducing how the deployed models were trained
    full-range    every compound with a measured pChEMBL, weak ones included

Both are tested on the same held-out compounds, which are absent from the training index and from
the applicability-domain reference, and the quantity compared is the mean signed error. If
full-range training removes the optimism the diagnosis is confirmed and the refit is justified. If
it does not, the cause lies elsewhere and a refit would invalidate the validation for nothing.

Run:
    python tools/regressor_fullrange_experiment.py
"""
from __future__ import annotations

import collections
import csv
import json
import ssl
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))

OUT = ROOT / "results" / "tables" / "regressor_fullrange_experiment.csv"
HOLD = ROOT / "results" / "tables" / "chembl_holdout_set.csv"
API = "https://www.ebi.ac.uk/chembl/api/data"
TARGETS = {"D2": "CHEMBL217", "A2A": "CHEMBL251", "HT2A": "CHEMBL224", "SERT": "CHEMBL228"}
BINDER_CUT = 7.0          # the cut the deployed binder-only regressors were trained above

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def get(url, timeout=180):
    for a in range(4):
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=_CTX) as r:
                return json.load(r)
        except Exception:
            if a == 3:
                raise
            time.sleep(3 * (a + 1))


def activities(tid):
    """{smiles: median pChEMBL} over human binding assays."""
    vals = collections.defaultdict(list)
    offset, page = 0, 1000
    while True:
        d = get(f"{API}/activity.json?target_chembl_id={tid}&pchembl_value__isnull=false"
                f"&assay_type=B&target_organism=Homo%20sapiens&limit={page}&offset={offset}")
        acts = d.get("activities", [])
        for a in acts:
            s, p = a.get("canonical_smiles"), a.get("pchembl_value")
            if s and p is not None:
                try:
                    vals[s].append(float(p))
                except (TypeError, ValueError):
                    pass
        offset += page
        if offset >= d["page_meta"]["total_count"] or not acts:
            break
    return {s: float(np.median(v)) for s, v in vals.items()}


def main() -> int:
    warnings.filterwarnings("ignore")
    from rdkit import RDLogger
    from sklearn.ensemble import RandomForestRegressor
    RDLogger.DisableLog("rdApp.*")
    import app as A

    hold = collections.defaultdict(dict)
    for r in csv.DictReader(HOLD.open(encoding="utf-8")):
        if r["endpoint"] in TARGETS:
            hold[r["endpoint"]][r["smiles"]] = float(r["pchembl"])

    rows = []
    print(f"  {'target':6s} {'train binder-only':>18s} {'train full-range':>17s} {'test':>6s}"
          f" {'bias binder':>12s} {'bias full':>10s} {'MAE binder':>11s} {'MAE full':>9s}")
    print("  " + "-" * 94)
    for ep, tid in TARGETS.items():
        test = hold.get(ep, {})
        if len(test) < 60:
            print(f"  {ep:6s} only {len(test)} held-out compounds; skipped")
            continue
        try:
            acts = activities(tid)
        except Exception as e:
            print(f"  {ep:6s} fetch failed: {type(e).__name__}")
            continue
        # the test compounds must not appear in either training set
        train = {s: p for s, p in acts.items() if s not in test}

        def build(d):
            X, y = [], []
            for s, p in d.items():
                f = A.featurize_one(s)
                if f is None:
                    continue
                X.append(f)
                y.append(p)
            return (np.vstack(X), np.asarray(y)) if X else (None, None)

        Xb, yb = build({s: p for s, p in train.items() if p >= BINDER_CUT})
        Xf, yf = build(train)
        Xt, yt = build(test)
        if Xb is None or Xf is None or Xt is None or len(yb) < 100:
            print(f"  {ep:6s} insufficient training data")
            continue

        common = dict(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=0)
        pb = RandomForestRegressor(**common).fit(Xb, yb).predict(Xt)
        pf = RandomForestRegressor(**common).fit(Xf, yf).predict(Xt)
        bias_b, bias_f = float(np.mean(pb - yt)), float(np.mean(pf - yt))
        mae_b, mae_f = float(np.mean(np.abs(pb - yt))), float(np.mean(np.abs(pf - yt)))
        print(f"  {ep:6s} {len(yb):>18,} {len(yf):>17,} {len(yt):>6,}"
              f" {bias_b:>+12.3f} {bias_f:>+10.3f} {mae_b:>11.3f} {mae_f:>9.3f}")
        rows.append({"endpoint": ep, "n_train_binder_only": len(yb), "n_train_full": len(yf),
                     "n_test": len(yt), "bias_binder_only": round(bias_b, 4),
                     "bias_full_range": round(bias_f, 4), "mae_binder_only": round(mae_b, 4),
                     "mae_full_range": round(mae_f, 4)})

    if rows:
        with OUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        bb = np.mean([r["bias_binder_only"] for r in rows])
        bf = np.mean([r["bias_full_range"] for r in rows])
        print(f"\n  mean signed error: binder-only {bb:+.3f}, full-range {bf:+.3f} log units")
        print(f"  wrote {OUT.relative_to(ROOT)}")
        print("\n  A full-range bias near zero confirms the diagnosis and justifies the refit.")
        print("  A full-range bias still near +1 would mean the cause lies elsewhere.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
