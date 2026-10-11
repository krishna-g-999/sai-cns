"""Two master files: every training input record, and every compound tested outside training.

Reviewers asked for the inputs in one place, and for evidence on compounds other than the ones the
models were fitted to. Those are different questions and are answered by different files, so they are
kept separate rather than merged into something that answers neither cleanly.

  MASTER_training_inputs.csv    every measured record used for training, one row per endpoint per
                                compound, with the endpoint it belongs to and the provenance of the
                                measurement. This is the union of data/endpoints/*.csv and the ADME
                                and antioxidant tables, stacked with an endpoint column.

  MASTER_external_test_results.csv   every compound the deployed models were scored on that was NOT
                                fitted to, drawn from five independent evaluations, with the
                                prediction each one produced.

Two things about the external file are worth stating plainly.

First, membership is checked rather than assumed. Every external compound's canonical SMILES is
tested against the union of all training tables, and the result is written to `found_in_training`.
A compound that appears there is a genuine overlap and is labelled as one; it is not silently
dropped, because how much overlap exists is part of what a reviewer is entitled to see.

Second, decoys are not training inputs in the sense the first file means. They are background
compounds selected by property matching, not measurements, so they appear in neither file; their
counts per endpoint are in the workbook's compound-accounting sheet instead.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
EP = ROOT / "data" / "endpoints"
EPR = ROOT / "data" / "endpoints_reg"
TAB = ROOT / "results" / "tables"
INV = ROOT / "inversion" / "results"


def wilson(k, n, z=1.96):
    """Wilson score interval. Used rather than the normal approximation because several strata here
    have rates near 0 or 1, where the normal interval runs outside [0, 1]."""
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def canon(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None


def master_inputs():
    import app
    frames = []
    named = [("Core target/property classifier", list(app.TARGET_CLASSIFIERS)),
             ("Receptor potency regressor", list(app.RECEPTOR_REGRESSORS)),
             ("Binder classifier", list(app.BINDER_TARGETS))]
    for panel, names in named:
        for n in names:
            p = EP / f"{n}.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            d.insert(0, "endpoint", n)
            d.insert(1, "panel", panel)
            d.insert(2, "source_file", f"data/endpoints/{n}.csv")
            frames.append(d)
    for p in sorted(EPR.glob("*.csv")) if EPR.exists() else []:
        d = pd.read_csv(p)
        d.insert(0, "endpoint", p.stem)
        d.insert(1, "panel", "Regression endpoint")
        d.insert(2, "source_file", f"data/endpoints_reg/{p.name}")
        frames.append(d)
    m = pd.concat(frames, ignore_index=True, sort=False)
    front = ["endpoint", "panel", "source_file", "smiles", "label", "pchembl", "year", "source"]
    cols = [c for c in front if c in m.columns] + [c for c in m.columns if c not in front]
    return m[cols]


def external_results():
    """Every evaluation run on compounds outside the fitted sets, stacked with a test_set column."""
    out = []

    p = INV / "H6_clinical_indication_predictions.csv"
    if p.exists():
        d = pd.read_csv(p)
        out.append(pd.DataFrame({
            "test_set": "H6 clinical indication",
            "what_it_tests": "whether a drug's licensed indication is recovered in the top three "
                             "predicted diseases, against ChEMBL indication records",
            "compound_id": d.get("chembl_id"), "smiles": d["smiles"],
            "expected": d.get("true_indications"), "predicted": d.get("predicted_top3"),
            "outcome": d["hit"].map({1: "recovered", 0: "not recovered"}) if "hit" in d else None}))

    p = TAB / "external_100_predictions.csv"
    if p.exists():
        d = pd.read_csv(p)
        out.append(pd.DataFrame({
            "test_set": "External reference set",
            "what_it_tests": "whether a compound of known pharmacological class is assigned the "
                             "matching disease",
            "compound_id": d.get("compound"), "smiles": d["smiles"],
            "expected": d.get("expected_class"), "predicted": d.get("predicted_top3"),
            "outcome": d["hit_top1"].map({1: "top-1 correct", 0: "top-1 incorrect"})
            if "hit_top1" in d else None}))

    p = TAB / "noncns_specificity_predictions.csv"
    if p.exists():
        d = pd.read_csv(p)
        out.append(pd.DataFrame({
            "test_set": "Non-CNS specificity",
            "what_it_tests": "how often a drug with no central action is nonetheless reported as "
                             "engaging a brain mechanism; every firing here is a false positive",
            "compound_id": None, "smiles": d["smiles"],
            "expected": "no brain mechanism", "predicted": d.get("top_disease"),
            "outcome": d["fired"].map({1: "fired (false positive)", 0: "silent (correct)"})
            if "fired" in d else None}))

    p = INV / "H4_distant_predictions.csv"
    if p.exists():
        d = pd.read_csv(p)
        out.append(pd.DataFrame({
            "test_set": "H4 distant chemistry",
            "what_it_tests": "the false-positive rate on chemistry increasingly unlike anything "
                             "trained on, stratified by nearest-neighbour Tanimoto",
            "compound_id": None, "smiles": d["smiles"],
            "expected": "no confident call outside the applicability domain",
            "predicted": d.get("top_score"),
            "outcome": d["fired"].map({1: "fired", 0: "silent"}) if "fired" in d else None}))

    e = pd.concat(out, ignore_index=True, sort=False)

    train = set()
    for p in list(EP.glob("*.csv")) + (list(EPR.glob("*.csv")) if EPR.exists() else []):
        d = pd.read_csv(p, usecols=lambda c: c == "smiles")
        train |= {canon(s) for s in d["smiles"].astype(str)}
    train.discard(None)

    # object dtype, not bool, so a structure RDKit could not parse reads as "unparsed" rather than
    # being silently coerced to False and counted as a clean non-overlap.
    e["canonical_smiles"] = [canon(s) for s in e["smiles"]]
    e["found_in_training"] = [
        "unparsed" if c is None else ("yes" if c in train else "no")
        for c in e["canonical_smiles"]]
    return e, len(train)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reviewer_package" / "08.09.26"
    out_dir.mkdir(parents=True, exist_ok=True)

    mi = master_inputs()
    mi.to_csv(out_dir / "MASTER_training_inputs.csv", index=False)
    print(f"MASTER_training_inputs.csv     {len(mi):,} rows, "
          f"{mi.endpoint.nunique()} endpoints, "
          f"{mi.smiles.astype(str).nunique():,} distinct SMILES as written")

    ex, n_train = external_results()
    ex.to_csv(out_dir / "MASTER_external_test_results.csv", index=False)
    print(f"MASTER_external_test_results.csv {len(ex):,} rows across "
          f"{ex.test_set.nunique()} independent evaluations")
    print(f"  checked against {n_train:,} canonical training structures")
    for name, g in ex.groupby("test_set"):
        ov = int((g["found_in_training"] == "yes").sum())
        bad = int((g["found_in_training"] == "unparsed").sum())
        print(f"    {name:26} n={len(g):5,}  overlapping training chemistry: {ov} "
              f"({ov / len(g) * 100:.1f}%)" + (f", unparsed {bad}" if bad else ""))

    # The point of the external file is what happens on chemistry the models were not fitted to, so
    # every result is also reported split on that. Approved-drug sets overlap ChEMBL heavily by
    # construction, and a headline computed over the mixture would be carried by the seen half.
    good = {"recovered", "top-1 correct", "silent (correct)", "silent"}
    rows = []
    for (name, seen), g in ex.groupby(["test_set", "found_in_training"]):
        o = g["outcome"].dropna()
        if not len(o):
            continue
        k = int(o.isin(good).sum())
        lo, hi = wilson(k, len(o))
        rows.append({
            "test_set": name,
            "compound_seen_in_training": {"yes": "yes, overlaps training chemistry",
                                          "no": "no, held out from every training table",
                                          "unparsed": "structure could not be parsed"}[seen],
            "n": len(o), "n_correct_or_correctly_silent": k,
            "rate": round(k / len(o), 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "what_rate_means": {
                "H6 clinical indication": "licensed indication recovered in the predicted top three",
                "External reference set": "pharmacological class recovered as the top prediction",
                "Non-CNS specificity": "correctly silent; a drug with no central action was not "
                                       "reported as engaging a brain mechanism",
                "H4 distant chemistry": "correctly silent on chemistry outside the applicability "
                                        "domain",
            }.get(name),
        })
    s = pd.DataFrame(rows).sort_values(["test_set", "compound_seen_in_training"])
    s.to_csv(out_dir / "MASTER_external_test_summary.csv", index=False)
    print(f"\nMASTER_external_test_summary.csv  performance split on training overlap")
    for _, r in s.iterrows():
        print(f"    {r.test_set:26} {r.compound_seen_in_training[:34]:34} "
              f"n={r.n:5,}  {r.rate:.3f} [{r.ci95_low:.3f}, {r.ci95_high:.3f}]")


if __name__ == "__main__":
    main()
