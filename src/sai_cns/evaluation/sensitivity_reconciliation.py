"""Reconcile the four figures this project reports for deployed sensitivity.

Four artefacts report "sensitivity at the deployed threshold" for the binder panel and they do not
agree. A thesis audit traced the disagreement to the sets they are measured on, and one of them is
not the set its own label claims.

  results/tables/final_thresholds.csv        held-out actives only, from models_rf/holdout/*.json
  results/tables/background_specificity.csv  every active in the endpoint table, training included
  models_rf/binder_modes.json                whatever the last writer put there
  inversion/results/H7_*.csv                 held-out actives scored against random PubChem

The threshold sequence runs final_thresholds.py and then calibrate_background_specificity.py, and
both write `sensitivity_at_threshold` into the registry. The second wins. But the second selects its
actives with

    act = df.loc[p >= 7, "smiles"]

over the whole endpoint table, so the model is scored on the compounds it was fitted on, while the
registry still carries `sensitivity_basis: "held_out_actives_by_scaffold"` written by the first. The
published mean of 0.898 is therefore measured on a set that is mostly training data and labelled as
though it were not.

This script quantifies the gap per endpoint rather than asserting it: for each deployed binder it
reports the held-out figure, the published figure, and what fraction of the published figure's
scoring set the model was trained on.

Run:  brainsafe_env/Scripts/python.exe src/sai_cns/evaluation/sensitivity_reconciliation.py
Out:  results/tables/sensitivity_reconciliation.csv
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REG = ROOT / "models_rf" / "binder_modes.json"
HOLD = ROOT / "models_rf" / "holdout"
FT = ROOT / "results" / "tables" / "final_thresholds.csv"
H7 = ROOT / "inversion" / "results" / "H7_target_discrimination.csv"
EP = ROOT / "data" / "endpoints"
OUT = ROOT / "results" / "tables" / "sensitivity_reconciliation.csv"


def _rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    reg = json.loads(REG.read_text(encoding="utf-8"))
    deployed = {k: v for k, v in reg.items() if v.get("deployed")}
    held = {r["target"]: r for r in _rows(FT)}
    h7 = {r["target"]: r for r in _rows(H7)}

    rows = []
    for ep, v in sorted(deployed.items()):
        f = EP / f"{ep}.csv"
        n_scored = trained_on = n_holdout = None
        if f.exists():
            df = pd.read_csv(f).dropna(subset=["smiles"])
            p = pd.to_numeric(df.get("pchembl"), errors="coerce")
            n_scored = int((p >= 7).sum())
            if n_scored < 30:                      # the fallback the second script also uses
                n_scored = int((df["label"] == 1).sum())
        hf = HOLD / f"{ep}_binder_holdout.json"
        if hf.exists():
            hj = json.loads(hf.read_text(encoding="utf-8"))
            trained_on = len(hj.get("active_train") or [])
            n_holdout = len(hj.get("active_holdout") or [])

        s_held = float(held[ep]["sensitivity"]) if held.get(ep, {}).get("sensitivity") else None
        s_pub = v.get("sensitivity_at_threshold")
        s_h7 = float(h7[ep]["deployed_sensitivity"]) if ep in h7 else None
        rows.append({
            "target": ep,
            "threshold_registry": v.get("threshold"),
            "threshold_final_thresholds": float(held[ep]["threshold"]) if ep in held else None,
            "sensitivity_heldout": s_held,
            "sensitivity_published": s_pub,
            "sensitivity_h7_vs_random": s_h7,
            "published_minus_heldout": (round(s_pub - s_held, 4)
                                        if s_pub is not None and s_held is not None else None),
            "n_actives_scored_by_published": n_scored,
            "n_actives_model_trained_on": trained_on,
            "n_actives_held_out": n_holdout,
            "train_fraction_of_published_set": (round(trained_on / n_scored, 4)
                                                if trained_on and n_scored else None),
            "registry_claims_basis": v.get("sensitivity_basis"),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    hv = [r["sensitivity_heldout"] for r in rows if r["sensitivity_heldout"] is not None]
    pv = [r["sensitivity_published"] for r in rows if r["sensitivity_published"] is not None]
    tf = [r["train_fraction_of_published_set"] for r in rows
          if r["train_fraction_of_published_set"] is not None]
    dv = [r["published_minus_heldout"] for r in rows if r["published_minus_heldout"] is not None]

    print(f"wrote {OUT.relative_to(ROOT).as_posix()}  ({len(rows)} deployed binder endpoints)")
    print()
    print(f"  held out only   mean {st.mean(hv):.4f}  median {st.median(hv):.4f}  "
          f"below 0.50 on {sum(1 for x in hv if x < 0.5)} of {len(hv)}")
    print(f"  as published    mean {st.mean(pv):.4f}  median {st.median(pv):.4f}  "
          f"below 0.50 on {sum(1 for x in pv if x < 0.5)} of {len(pv)}")
    print(f"  published is higher on {sum(1 for x in dv if x > 0)} of {len(dv)}, "
          f"by a mean of {st.mean(dv):+.4f}")
    # Three endpoints report a ratio above 1, which is not possible for one population and is a
    # second inconsistency rather than an arithmetic slip: the binder pipeline selects its training
    # actives by the `label` column while calibrate_background_specificity.py selects by
    # pChEMBL >= 7, and at Nav1.5, SIRT1 and TAAR1 those differ sharply. Nav1.5 has 242 actives by
    # label and 40 at pChEMBL >= 7, so the published sensitivity there is measured on a set defined
    # by a different rule from the one the model was fitted with. The median is quoted as the
    # headline because it is unaffected by those three; the mean over the well-defined endpoints is
    # given beside it.
    clean = [x for x in tf if x <= 1.0]
    odd = [r["target"] for r in rows
           if r["train_fraction_of_published_set"] and r["train_fraction_of_published_set"] > 1.0]
    print(f"  the published set is {st.median(clean):.1%} training compounds at the median, "
          f"{st.mean(clean):.1%} at the mean over the {len(clean)} endpoints where the two scripts "
          f"agree on what an active is")
    if odd:
        print(f"  {len(odd)} endpoints define 'active' differently in the two scripts and are "
              f"excluded from that figure: {', '.join(odd)}")
    print()
    print("  largest gaps:")
    for r in sorted((r for r in rows if r["published_minus_heldout"] is not None),
                    key=lambda r: -r["published_minus_heldout"])[:5]:
        print(f"    {r['target']:<10} held out {r['sensitivity_heldout']:.3f}  "
              f"published {r['sensitivity_published']:.3f}  "
              f"gap {r['published_minus_heldout']:+.3f}")


if __name__ == "__main__":
    main()
