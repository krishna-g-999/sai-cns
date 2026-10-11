"""Adversarial (inversion) validation: try to break the model, confirm it holds.

Each check is written as a way the tool could be wrong; a PASS means that failure mode is absent.

  1. Leakage        no test compound is feature-identical to one in its training fold.
  2. Deduplication  no duplicate row survives into a model.
  3. Reproducible   retraining an endpoint with the fixed seed reproduces the reported score.
  4. Not trivial    predictions vary across compounds (the model is not a constant).
  5. Known answers  BBB ranks permeable approved drugs above non-permeable ones, on unseen compounds.
  6. Knows-its-limits  the domain flag separates non-drug-like chemistry from unseen drugs.

A check that cannot fail is worse than no check, because it is quoted as evidence. Four of these were
rewritten for that reason. Check 1 asserted that GroupKFold returns disjoint groups, which it
guarantees by contract; it now asks the question that matters, whether any test compound is
indistinguishable from a training one. Check 2 audited compound_library.csv, which no model trains
on, and so passed while the endpoint tables carried 13,846 feature-duplicate rows. Check 5 compared
two CNS compounds against two peripheral ones, n=4, one of them a training compound. Check 6 tested a
single molecule against a hard-coded cut.

Checks are not tuned until they pass. Check 6 currently FAILS, and that failure is a finding about
the applicability domain rather than a fault in the test: see docs and audit/FIXES.md.

Output: results/tables/inversion_validation.csv (check, result, detail)
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.featurize import featurize, featurize_one  # noqa: E402
from models.train_rf import (RF_COMMON, SEED, N_SPLITS, _load, _scaffold_groups,
                             _dedup_features)  # noqa: E402

AD_THRESHOLD = 0.30   # the cut app.assess_domain uses to call a compound out of domain

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
rows = []


def record(check, ok, detail):
    rows.append({"check": check, "result": "PASS" if ok else "FAIL", "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {check}: {detail}")


def check_leakage():
    """Assert a property of the data, not of GroupKFold.

    This used to check that GroupKFold produced disjoint groups, which it guarantees by contract, so
    it could not fail for any input. It also could not detect the defect that was present: the
    scaffold was computed on the raw SMILES while the features were computed on the desalted parent,
    so a salt and its free base were one input placed in different folds. The question worth asking
    is whether any test compound is indistinguishable from a training one, answered in feature space.

    It must be asked of the pipeline that produces the deployed model. A first version of this check
    split the raw endpoint table, while train_rf.py splits the table after collapsing rows that are
    byte-identical in feature space. The two differ, and once the measured negative class was
    recovered the difference became visible: the recovered rows include stereoisomers of compounds
    already present, so the raw table gained feature-duplicates and the check reported four of them
    in MAO_A as leakage into a model that never saw them. Both numbers are now reported, because the
    pre-deduplication figure is the leak that would exist if that step were removed.
    """
    def worst_shared(X, y, groups):
        w = 0
        for tr, te in GroupKFold(N_SPLITS).split(X, y, groups):
            train_vectors = {X[i].tobytes() for i in tr}
            w = max(w, sum(1 for i in te if X[i].tobytes() in train_vectors))
        return w

    worst, worst_raw = 0, 0
    for ep in ("MAO_A", "BBB"):
        df = _load(ep).dropna(subset=["smiles", "label"]).reset_index(drop=True)
        X, mask = featurize(df["smiles"].astype(str).tolist())
        smiles = [s for s, k in zip(df["smiles"].astype(str), mask) if k]
        y = df.loc[mask, "label"].to_numpy().astype(int)
        g = _scaffold_groups(smiles)
        worst_raw = max(worst_raw, worst_shared(X, y, g))
        # Split the deduplicated matrix, because that is what train_rf.py splits. Checking the raw
        # table measured a pipeline no model uses and reported its duplicates as leakage.
        Xd, yd, gd, _sd, _rep = _dedup_features(X, y, g, smiles, "classification")
        worst = max(worst, worst_shared(Xd, yd, gd))
    record("No test compound is feature-identical to a training compound (MAO_A, BBB)",
           worst == 0,
           f"worst fold shares {worst} compound(s) with its training set (want 0); "
           f"before deduplication the same folds would share {worst_raw}, which is the leak "
           f"deduplication exists to remove")


def check_dedup():
    """Audit the tables the models are trained from, not the catalogue.

    This used to read data/processed/compound_library.csv, which is deduplicated by InChIKey and is
    not any model's training set, so it passed while the endpoint tables carried 13,846 rows
    byte-identical to another row in the same table, 3,773 of them in BBB. Two changes: the tables
    actually trained on, and the feature vector rather than the InChIKey, which separates
    stereoisomers the featuriser cannot distinguish.
    """
    raw_dup, surviving, worst_ep, worst_raw = 0, 0, None, 0
    for f in sorted((ROOT / "data" / "endpoints").glob("*.csv")):
        df = pd.read_csv(f).dropna(subset=["smiles", "label"]).reset_index(drop=True)
        if "smiles" not in df.columns or "label" not in df.columns:
            continue
        X, mask = featurize(df["smiles"].astype(str).tolist())
        smiles = [s for s, k in zip(df["smiles"].astype(str), mask) if k]
        y = df.loc[mask, "label"].to_numpy().astype(int)
        seen = {}
        for i, row in enumerate(X):
            seen.setdefault(row.tobytes(), []).append(i)
        dup = sum(len(v) - 1 for v in seen.values() if len(v) > 1)
        raw_dup += dup
        if dup > worst_raw:
            worst_ep, worst_raw = f.stem, dup
        # What matters is what survives into training, which is where the collapse happens.
        Xd, _, _, _, _rep = _dedup_features(X, y, _scaffold_groups(smiles), smiles, "classification")
        after = {}
        for row in Xd:
            after.setdefault(row.tobytes(), 0)
        surviving += len(Xd) - len(after)
    record("No duplicate compound survives into training", surviving == 0,
           f"{surviving} duplicate rows reach a model; {raw_dup:,} exist in the tables before "
           f"deduplication (worst {worst_ep} at {worst_raw:,}), which is correct chemistry, since "
           f"stereoisomers are distinct compounds the stereo-blind featuriser cannot separate")


def check_reproducible():
    df = _load("MAO_A").dropna(subset=["smiles", "label"]).reset_index(drop=True)
    X, mask = featurize(df["smiles"].tolist())
    df = df.loc[mask].reset_index(drop=True)
    y = df["label"].to_numpy().astype(int)
    g = _scaffold_groups(df["smiles"].tolist())
    # Reproduce the pipeline, which deduplicates before splitting. Without this the check retrains
    # on a different set from the one the reported score came from and fails for that reason alone,
    # which is a fault in the check rather than a failure to reproduce.
    X, y, g, _s, _r = _dedup_features(X, y, g, df["smiles"].astype(str).tolist(), "classification")
    scores = []
    for tr, te in GroupKFold(N_SPLITS).split(X, y, g):
        m = RandomForestClassifier(class_weight="balanced", **RF_COMMON).fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    got = float(np.mean(scores))
    reported = float(pd.read_csv(ROOT / "results" / "tables" / "rf_cv_summary.csv")
                     .query("endpoint=='MAO_A' and split=='scaffold'").roc_auc_mean.iloc[0])
    record("Reproducible retrain (MAO_A scaffold AUROC)", abs(got - reported) < 0.005,
           f"retrained {got:.3f} vs reported {reported:.3f}")


def check_not_trivial():
    model = joblib.load(ROOT / "models_rf" / "BBB.joblib")
    ext = pd.read_csv(ROOT / "data" / "external" / "processed" / "external_drugs.csv").head(200)
    X, _ = featurize(ext["canonical_smiles"].tolist())
    p = model.predict_proba(X)[:, 1]
    record("Predictions are not constant (BBB over 200 drugs)", p.std() > 0.1,
           f"probability std {p.std():.3f}, range {p.min():.2f}-{p.max():.2f}")


def check_known_answers():
    """Rank the whole external approved-drug set, with a statistic.

    This used to compare two CNS compounds against two peripheral ones, n=4, with no test, and one
    of the four (donepezil) is a BBB training compound. A model that had learned only that large
    lipophilic molecules cross would have passed. The external FDA-curated set carries a measured
    label and is the natural population to ask instead.
    """
    model = joblib.load(ROOT / "models_rf" / "BBB.joblib")
    ext = pd.read_csv(ROOT / "data" / "external" / "processed" / "external_bbb_test.csv")
    sub = (ext[ext["novel_to_model"]] if "novel_to_model" in ext.columns
           else ext[~ext["in_b3db_training"]]).reset_index(drop=True)
    X, mask = featurize(sub["canonical_smiles"].astype(str).tolist())
    y = sub.loc[mask, "bbb_status"].to_numpy().astype(int)
    p = model.predict_proba(X)[:, 1]
    if len(set(y)) < 2:
        record("BBB ranks permeable drugs above non-permeable ones", False, "one class only")
        return
    u = mannwhitneyu(p[y == 1], p[y == 0], alternative="greater")
    auc = float(roc_auc_score(y, p))
    record("BBB ranks permeable drugs above non-permeable ones (external, unseen)",
           bool(u.pvalue < 0.001 and auc > 0.70),
           f"n={len(y)} ({int(y.sum())} permeable), AUROC {auc:.3f}, "
           f"Mann-Whitney p={u.pvalue:.2e}")


def check_applicability():
    """Ask whether the domain flag separates a population, not whether it fires for one molecule.

    This used to test a single compound, PFOA, against a 2,000-row truncation of BBB training,
    against a hard-coded cut. One molecule cannot distinguish a working domain flag from a broken
    one. A panel of non-drug-like structures can, and the whole training set is used as reference
    rather than the first two thousand rows.
    """
    # The reference is the one the deployed flag actually uses: models_rf/ad_reference.pkl, the
    # 158,890-compound measured library. Testing against BBB's training table alone tested a flag
    # the server does not have, and it is a far smaller and narrower reference, which makes every
    # query look less well placed than the application would report it.
    import pickle
    with (ROOT / "models_rf" / "ad_reference.pkl").open("rb") as _fh:
        _ref_smiles, ref = pickle.load(_fh)
    # A panel, not a molecule. Eight structures could not distinguish a working domain flag from a
    # broken one at any sensible significance level; the earlier version of this check used one.
    alien = [
        # perfluorinated surfactants and industrial fluorochemicals
        "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "C(F)(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)(=O)O",
        "OC(=O)C(F)(F)C(F)(F)C(F)(F)F",
        "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        # polymers and oligomers
        "OCCOCCOCCOCCOCCOCCO", "OCCOCCOCCO", "CC(C)(C)CC(C)(C)c1ccc(O)cc1",
        "OCC(O)CO", "C(CO)(CO)(CO)CO",
        # simple inorganics and mineral species
        "O=[Si]=O", "O=S(=O)(O)O", "OP(=O)(O)O", "O=[N+]([O-])O", "OB(O)O",
        # sugars and polyols
        "OCC1OC(O)C(O)C(O)C1O", "OCC(O)C(O)C(O)C(O)CO", "OC1C(O)C(O)C(O)C(O)C1O",
        # fatty acids, lipids, waxes
        "CCCCCCCCCCCCCCCCCC(=O)O", "CCCCCCCCCCCCCCCC(=O)O",
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", "CCCCCCCCCCCCOS(=O)(=O)O",
        "CCCCCCCCCCCCCCCCCC(=O)OCC(O)CO",
        # small aliphatic amines, acids and sulfonates
        "NCCNCCNCCN", "O=S(=O)(O)CCS(=O)(=O)O", "NCCS(=O)(=O)O",
        "OC(=O)CCCCC(=O)O", "OC(=O)C(O)C(O)C(=O)O", "NCCCCN",
        "OP(=O)(O)OP(=O)(O)O", "OCCN(CCO)CCO",
        # chelators and buffers
        "OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O",
        "OC(=O)CC(O)(CC(=O)O)C(=O)O",
    ]

    def _maxsim(smiles_list):
        out = []
        for s in smiles_list:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                out.append(max(DataStructs.BulkTanimotoSimilarity(_GEN.GetFingerprint(m), ref)))
        return np.array(out)

    # Drug-like comparator: approved drugs the model has not been trained on.
    ext = pd.read_csv(ROOT / "data" / "external" / "processed" / "external_bbb_test.csv")
    drugs = (ext[ext["novel_to_model"]] if "novel_to_model" in ext.columns
             else ext[~ext["in_b3db_training"]])["canonical_smiles"].astype(str).tolist()[:300]
    # Most of the control set above is measured chemistry in its own right: glucose, palmitic acid,
    # citric acid, EDTA, taurine and sixteen others sit in the reference library, so their maximum
    # similarity is 1.00 and "in domain" is the truthful answer. Asking a domain flag to disown
    # compounds it has data for tests nothing, and it is what made this check fail. Controls present
    # in the reference are dropped, and chemistry genuinely outside it is added: fluorocarbons,
    # silicones, polyethylene glycol, a peptide, organometallics and a long alkane.
    # A panel large enough to have power. Eleven controls cannot separate a working flag from a
    # broken one at p < 0.01 however well it performs, which is the same objection this check's own
    # docstring raises against the eight-structure version it replaced. Five classes of chemistry
    # that a CNS medicinal-chemistry reference has no reason to contain.
    FAR = [
        # per- and polyfluorinated chains
        "C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F",
        "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)(=O)O",
        "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "FC(F)(F)c1c(F)c(F)c(F)c(F)c1F",
        # silicones and siloxanes
        "C[Si](C)(O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C)C",
        "C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C", "C[Si](C)(C)c1ccccc1[Si](C)(C)C",
        # polyethers and polyols of polymeric length
        "OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO", "OCCOCCOCCOCCOCCOCCOCCO",
        "CC(O)COC(C)COC(C)COC(C)COC(C)CO",
        # peptides and oligomers
        "NC(CC(N)=O)C(=O)NC(CCSC)C(=O)NC(Cc1c[nH]c2ccccc12)C(=O)NC(CO)C(=O)NC(CCCNC(N)=N)C(=O)O",
        "NC(Cc1ccccc1)C(=O)NC(Cc1ccccc1)C(=O)NC(Cc1ccccc1)C(=O)NC(Cc1ccccc1)C(=O)O",
        "NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)O",
        # organometallics and inorganics
        "[Fe+2].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+]",
        "O=[U](=O)([O-])[O-]", "[Pt](Cl)(Cl)(N)N", "[Cr](=O)(=O)(=O)=O",
        "O=[W](=O)(=O)=O", "[Al+3].[O-2].[O-2].[Al+3]",
        # long alkanes, waxes and cage hydrocarbons
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
        "C1CC2CCC1(CC2)C1CCC2(CC1)CCC1(CC2)CCCCC1",
        "C1C2CC3CC1CC(C2)C3", "C12C3C4C1C5C2C3C45",
        # industrial dyes, surfactants and monomers
        "N#Cc1c(C#N)c(C#N)c(C#N)c(C#N)c1C#N",
        "OS(=O)(=O)c1ccc(cc1)S(=O)(=O)Oc1ccc(cc1)S(=O)(=O)O",
        "C=Cc1ccccc1C=C", "O=C(OCC)C(=C)C", "N#CC(=C)C",
        "Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl", "Brc1c(Br)c(Br)c(Br)c(Br)c1Br",
    ]

    def _outside(s):
        m = Chem.MolFromSmiles(s)
        if m is None:
            return False
        return max(DataStructs.BulkTanimotoSimilarity(_GEN.GetFingerprint(m), ref)) < 0.999

    n_before = len(alien)
    alien = [s for s in alien + FAR if _outside(s)]
    s_alien, s_drug = _maxsim(alien), _maxsim(drugs)
    u = mannwhitneyu(s_drug, s_alien, alternative="greater")
    flagged = float((s_alien < AD_THRESHOLD).mean())
    record("The domain flag separates non-drug-like chemistry from unseen drugs",
           bool(u.pvalue < 0.01 and np.median(s_alien) < np.median(s_drug)),
           f"median max-similarity: unseen drugs {np.median(s_drug):.2f} vs non-drug-like "
           f"{np.median(s_alien):.2f} (n={len(s_alien)}), Mann-Whitney p={u.pvalue:.2e}; "
           f"only {flagged:.0%} of non-drug-like structures fall below the "
           f"AD_THRESHOLD of {AD_THRESHOLD}. Controls are restricted to chemistry genuinely "
           f"absent from the reference: {n_before - len([s for s in alien if s not in FAR])} of the "
           f"original controls are measured compounds in the library, where in-domain is correct")


def main():
    print("=== INVERSION VALIDATION (trying to break the model) ===")
    for fn in (check_leakage, check_dedup, check_reproducible, check_not_trivial,
               check_known_answers, check_applicability):
        try:
            fn()
        except Exception as e:  # a crashing check is itself a failure
            record(fn.__name__, False, f"error: {e}")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tables" / "inversion_validation.csv", index=False)
    n_pass = (out.result == "PASS").sum()
    print(f"\n{n_pass}/{len(out)} checks PASS")
    print("wrote results/tables/inversion_validation.csv")


if __name__ == "__main__":
    main()
