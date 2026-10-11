"""Cross-provenance validation: does the panel survive a change of curator?

The time split (external_prospective.py) holds out the future. It cannot hold out a laboratory, and
a model can be prospectively accurate while depending on the habits of one curation pipeline:
ChEMBL's assay selection, its pChEMBL derivation, its choice of which papers to abstract. A second
kind of independence is available in this data and is worth measuring separately.

Every measured row carries the database it came from. Most rows are ChEMBL; some are present in both
ChEMBL and BindingDB; and a few thousand were deposited in BindingDB and are absent from ChEMBL
entirely. Those last are curated by different people, from different papers, under a different
protocol for deciding what counts as a measurement. Training on the ChEMBL side and testing on the
BindingDB-only side therefore asks a question the time split cannot: is the signal a property of the
chemistry, or of one database's way of recording it?

What this test can and cannot say, stated before the numbers rather than after:

  It cannot report AUROC. Every BindingDB-only row at every endpoint is an active. BindingDB
  deposits affinities, so a compound absent from ChEMBL and present in BindingDB is a compound
  someone measured and found to bind. There are no independently curated negatives to rank against,
  and inventing them from the background pool would test the background pool.

  It therefore reports recall at the deployed operating point, paired with the false-positive rate
  that same threshold produces on the background evaluation pool. Recall alone is meaningless: a
  model answering "active" to everything scores 1.0. Recall at a measured false-positive rate is a
  real statement, and it is the same pair the deployed panel reports.

  Recall is also reported against how far each test compound sits from the training chemistry, for
  the same reason as in the time split. A BindingDB-only compound from the same paper series as a
  ChEMBL one is not independent evidence however different its provenance.

The models are refitted on ChEMBL-provenance rows alone. Rows marked as present in both databases
count as ChEMBL and train; only BindingDB-only rows are held out.

Output: results/tables/external_cross_source.csv
        results/tables/external_cross_source_compounds.csv

Run:  python src/sai_cns/evaluation/external_cross_source.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
sys.path.insert(0, str(ROOT / "src" / "sai_cns" / "evaluation"))
from features.featurize import featurize  # noqa: E402
import panel  # noqa: E402
from external_prospective import (  # noqa: E402
    ACTIVE_P, MEASURED_ACTIVE_P, MEASURED_INACTIVE_P, MIN_TRAIN_ACTIVES, SEED,
    _fp, _load_pools, _max_sim, _run_split)

TAB = ROOT / "results" / "tables"
BDB_ONLY = "BindingDB"       # deposited in BindingDB and absent from ChEMBL
MIN_TEST = 50                # below this a recall figure is an anecdote


def main() -> None:
    modes = json.loads((ROOT / "models_rf" / "binder_modes.json").read_text())
    deployed = {e.name: e.mode for e in panel.binders(deployed=True)}
    bgX, bg_ok, bg_fp_ok, evX = _load_pools()
    print(f"background: {len(bg_ok):,} decoy-eligible, {len(evX):,} evaluation compounds\n",
          flush=True)

    rows, comps = [], []
    for ep, mode in sorted(deployed.items()):
        f = ROOT / "data" / "endpoints" / f"{ep}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f).dropna(subset=["smiles"])
        df["smiles"] = df["smiles"].astype(str)
        if "source" not in df.columns:
            continue
        p = pd.to_numeric(df.get("pchembl"), errors="coerce")
        cut_a = MEASURED_ACTIVE_P if mode == panel.MEASURED_LABEL else ACTIVE_P
        is_act = p >= cut_a
        is_ina = ((p < MEASURED_INACTIVE_P) | (df["label"] == 0)
                  if mode == panel.MEASURED_LABEL else df["label"] == 0)

        ext = df["source"] == BDB_ONLY
        test_act = sorted(set(df.loc[ext & is_act, "smiles"]))
        if len(test_act) < MIN_TEST:
            continue

        # A compound recorded under both provenances belongs to the training side. Leaving it in the
        # test set would score the model on chemistry it was fitted to and call the result external.
        train_act = sorted(set(df.loc[~ext & is_act, "smiles"]) - set(test_act))
        train_ina = sorted(set(df.loc[~ext & is_ina, "smiles"]) - set(train_act) - set(test_act))
        if len(train_act) < MIN_TRAIN_ACTIVES:
            continue

        g = np.random.default_rng(SEED)
        ina = list(g.permutation(train_ina))
        half = len(ina) // 2
        barred = set(df["smiles"])

        m, cs = _run_split("cross_source", ep, mode, train_act, test_act,
                           ina[:half], ina[half:], [], barred, bgX, bg_ok, bg_fp_ok, evX)
        if m is None:
            continue
        # A BindingDB-only compound can still be feature-identical to a ChEMBL training compound:
        # different provenance, same molecule after standardisation and folding. Recalling those
        # measures memorisation and would be reported as external evidence. The distinguishable
        # subset is reported alongside the whole set, exactly as the barrier model's external table
        # separates the two.
        cd = pd.DataFrame(cs)
        act = cd[cd.measured == 1] if len(cd) else cd
        novel = act[act.max_tanimoto_to_training < 0.999] if len(act) else act
        dep = modes.get(ep, {})
        rows.append({
            "endpoint": ep, "mode": mode,
            "n_train_actives_chembl": m["n_train_actives"],
            "n_test_actives_bindingdb_only": m["n_test_actives"],
            "n_test_actives_distinguishable": int(len(novel)),
            "threshold": m["threshold"], "threshold_basis": m["threshold_basis"],
            "recall_on_external_actives": m["sensitivity"],
            "recall_on_distinguishable_only": (round(float(novel.called.mean()), 4)
                                               if len(novel) else None),
            "fpr_background_at_same_threshold": m["fpr_background"],
            "auroc_vs_background": m["auroc_vs_background"],
            "median_novelty_of_test_set": m["median_test_novelty"],
            "test_actives_below_tanimoto_0.4": m["test_actives_below_tanimoto_0.4"],
            "deployed_sensitivity": dep.get("sensitivity_at_threshold"),
            "deployed_threshold": dep.get("threshold")})
        comps.extend(cs)
        print(f"  {ep:10s} train {m['n_train_actives']:>5} ChEMBL -> test "
              f"{m['n_test_actives']:>4} BindingDB-only ({len(novel)} distinguishable)   "
              f"recall {m['sensitivity']:.3f} "
              f"at background FPR {m['fpr_background']:.4f}   "
              f"(deployed sens {dep.get('sensitivity_at_threshold')})", flush=True)

        TAB.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(TAB / "external_cross_source.csv", index=False)
        pd.DataFrame(comps).to_csv(TAB / "external_cross_source_compounds.csv", index=False)

    r = pd.DataFrame(rows)
    print()
    if len(r):
        print(f"  endpoints with an independently curated test set : {len(r)}")
        print(f"  external actives scored                          : "
              f"{int(r.n_test_actives_bindingdb_only.sum()):,}")
        print(f"  mean recall on BindingDB-only actives            : "
              f"{r.recall_on_external_actives.mean():.4f}")
        print(f"  of which distinguishable from training           : "
              f"{int(r.n_test_actives_distinguishable.sum()):,}")
        print(f"  mean recall on the distinguishable subset        : "
              f"{pd.to_numeric(r.recall_on_distinguishable_only, errors='coerce').mean():.4f}")
        print(f"  mean deployed sensitivity, same endpoints        : "
              f"{pd.to_numeric(r.deployed_sensitivity, errors='coerce').mean():.4f}")
        print(f"  mean background FPR at that threshold            : "
              f"{r.fpr_background_at_same_threshold.mean():.4f}")
    print(f"\nwrote {(TAB / 'external_cross_source.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
