"""Regenerate the headline validation table from the deployed state.

This table carries the numbers most likely to be quoted from the manuscript, and until now it was
maintained by hand. It had drifted to describe 39 targets when 46 have hold-out twins, which is
exactly the failure the provenance audit exists to catch and which the audit itself missed because
the file was not on its list. Both are fixed: the table is derived, and it is added to the audit.

Every figure traces to a file rather than to a recollection:

  prospective sensitivity   models_rf/holdout/binder_modes.json, the scaffold hold-out twins, where
                            a held-out compound shares no Bemis-Murcko scaffold with anything its
                            model saw
  specificity               results/tables/noncns_specificity_summary.csv, compounds with no measured
                            activity at any modelled target. They are presumed inactive rather than
                            proven so, which makes this a lower bound and it is labelled as one
  deployed specificity      results/deployed_specificity_audit.csv, the false-positive rate of the
                            served models on structures drawn at random from PubChem

Targets whose hold-out threshold collapsed to the permitted floor are excluded from the pooled
recall, because a threshold at the floor means no separation from background chemistry was achieved
and a recall computed against it is meaningless rather than merely poor.

Writes results/tables/MASTER_validation_summary.csv
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
TAB = ROOT / "results" / "tables"
FLOOR = 0.051   # a hold-out threshold at or below this is the permitted floor, not a real cut


def wilson(k, n, z=1.96):
    if not n:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    import app

    modes = json.loads((ROOT / "models_rf" / "binder_modes.json").read_text())
    hold = json.loads((ROOT / "models_rf" / "holdout" / "binder_modes.json").read_text())
    deployed = {t for t in app.TARGET_KIND if t != "NEURO"
                and modes.get(t, {}).get("deployed", True)}

    usable, excluded = {}, []
    for ep, v in hold.items():
        if ep not in deployed:
            continue
        if float(v.get("threshold", 0)) <= FLOOR:
            excluded.append(ep)
            continue
        usable[ep] = v
    k = sum(round(v["holdout_recall_at_threshold"] * v["n_holdout_actives"]) for v in usable.values())
    n = sum(v["n_holdout_actives"] for v in usable.values())
    lo, hi = wilson(k, n)
    rec = pd.Series([v["holdout_recall_at_threshold"] for v in usable.values()])

    rows = [{
        "test": "Prospective sensitivity (scaffold-held-out)", "statistic": "recall",
        "k": k, "n": n, "estimate": round(k / n, 4),
        "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
        "design": f"20% of Bemis-Murcko scaffolds withheld per target; {len(usable)} deployed targets "
                  f"with a usable hold-out threshold; held-out compounds share no scaffold with "
                  f"training. {len(excluded)} excluded for a threshold at the floor: "
                  f"{', '.join(sorted(excluded)) or 'none'}"}]

    spec = TAB / "noncns_specificity_summary.csv"
    if spec.exists():
        s = pd.read_csv(spec)
        # read the row by the metric it names, not by guessing a column. Guessing silently dropped
        # this row on the first run and produced a balanced accuracy computed from one term.
        r = s[s.metric.astype(str).str.startswith("Specificity")]
        if len(r):
            kk, nn = int(r.k.iloc[0]), int(r.n.iloc[0])
            slo, shi = wilson(kk, nn)
            rows.append({"test": "Specificity (non-CNS compounds)", "statistic": "true-negative rate",
                         "k": kk, "n": nn, "estimate": round(kk / nn, 4),
                         "ci95_low": round(slo, 4), "ci95_high": round(shi, 4),
                         "design": "library compounds with no measured activity at any modelled "
                                   "target; presumed inactive rather than proven so, therefore a "
                                   "lower bound"})

    dep = ROOT / "results" / "deployed_specificity_audit.csv"
    if dep.exists():
        d = pd.read_csv(dep)
        rows.append({"test": "Deployed specificity on random chemistry",
                     "statistic": "median false-positive rate per endpoint",
                     "k": "", "n": len(d), "estimate": round(float(d.random_fpr.median()), 4),
                     "ci95_low": "", "ci95_high": "",
                     "design": "every served model at its deployed threshold against 600 structures "
                               "drawn by random PubChem identifier, plus molecules no CNS target "
                               f"plausibly binds; {int((d.n_trivial_fired > 0).sum())} endpoints "
                               f"fire on a trivial molecule"})

    spec_row = next((r for r in rows if r["test"].startswith("Specificity")), None)
    if spec_row is not None:
        rows.append({"test": "Balanced accuracy", "statistic": "(sensitivity+specificity)/2",
                     "k": "", "n": n + spec_row["n"],
                     "estimate": round((k / n + spec_row["estimate"]) / 2, 4),
                     "ci95_low": "", "ci95_high": "", "design": "combines the two tests above"})
    else:
        print("WARNING: no specificity row found; balanced accuracy omitted rather than "
              "computed from one term", flush=True)

    rows += [
        {"test": "Targets reaching >=0.80 prospective recall", "statistic": "count",
         "k": int((rec >= 0.80).sum()), "n": len(rec),
         "estimate": round(float((rec >= 0.80).mean()), 4), "ci95_low": "", "ci95_high": "",
         "design": f"per-target, excluding {len(excluded)} whose hold-out threshold collapsed"},
        {"test": "Targets below 0.50 prospective recall", "statistic": "count",
         "k": int((rec < 0.50).sum()), "n": len(rec),
         "estimate": round(float((rec < 0.50).mean()), 4), "ci95_low": "", "ci95_high": "",
         "design": ", ".join(sorted(e for e, v in usable.items()
                                    if v["holdout_recall_at_threshold"] < 0.50)) or "none"},
        {"test": "Binder discrimination vs measured inactives", "statistic": "mean AUROC",
         "k": "", "n": sum(1 for t in deployed if "auroc_vs_measured_inactives" in modes.get(t, {})),
         "estimate": round(float(np.mean([modes[t]["auroc_vs_measured_inactives"] for t in deployed
                                          if "auroc_vs_measured_inactives" in modes.get(t, {})])), 4),
         "ci95_low": "", "ci95_high": "",
         "design": "compounds experimentally tested on the same target and found inactive, held out "
                   "from training"},
    ]

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "MASTER_validation_summary.csv", index=False)
    pd.set_option("display.width", 230)
    pd.set_option("display.max_colwidth", 58)
    print(df.drop(columns=["design"]).to_string(index=False))
    print(f"\npooled prospective recall {k:,}/{n:,} = {k/n:.4f} over {len(usable)} deployed targets")
    print("wrote", TAB / "MASTER_validation_summary.csv")


if __name__ == "__main__":
    main()
