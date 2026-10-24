"""Which applicability-domain measure actually separates chemistry the panel cannot place?

The deployed measure is the maximum Tanimoto similarity to the endpoint's own measured chemistry,
and it fails the one adversarial check written to test it: shown sugars, fatty acids, buffers and
simple inorganic acids, it scores them a median 0.48 against 0.44 for approved drugs the model has
never seen. It rates non-drug-like chemistry as *better* placed than real drugs.

The reason is structural rather than a bug. A maximum over a folded fingerprint rewards being a
substructure of something in the training set. A hexadecanoic acid is a long alkyl chain with a
carboxylate, and both fragments appear in thousands of medicinal-chemistry compounds, so its single
best match is high while it resembles no training compound as a whole. Maximum similarity asks "is
any part of you familiar", when the question that matters is "do you sit among compounds we have
measured".

Four alternatives are measured here against the same adversarial set, all on the same fingerprints,
so the comparison is between definitions rather than between implementations:

  max              the deployed measure, as a baseline
  mean_top5        mean similarity to the five nearest training compounds. A compound genuinely
                   inside the domain has a neighbourhood, not one lucky match
  kth_5            similarity to the fifth nearest neighbour alone, the strictest form of the same
                   idea
  density_0.4      how many training compounds sit within Tanimoto 0.4, which measures support
                   directly rather than by proxy

A measure passes if non-drug-like chemistry scores below unseen drugs, one-sided Mann-Whitney
p < 0.01, and if it flags a useful share of the aliens at a threshold that does not also reject the
drugs. Both halves matter: a measure that flags everything separates nothing.

Output: results/tables/applicability_measures.csv

Run:  python src/sai_cns/evaluation/applicability_measures.py
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
OUT = ROOT / "results" / "tables" / "applicability_measures.csv"

# The applicability-domain fingerprint has one definition, and this file previously
# carried a fourteenth private copy of it. A reference built with one fingerprint and
# queried with another returns similarities that are wrong rather than missing.
import ad_fingerprint as _adfp  # noqa: E402

_GEN = _adfp.generator()

# The adversarial set from validate_inversion.py: chemistry no CNS panel should claim to place.
ALIEN = [
    "O=[Si]=O", "O=S(=O)(O)O", "OP(=O)(O)O", "O=[N+]([O-])O", "OB(O)O",
    "OCC1OC(O)C(O)C(O)C1O", "OCC(O)C(O)C(O)C(O)CO", "OC1C(O)C(O)C(O)C(O)C1O",
    "CCCCCCCCCCCCCCCCCC(=O)O", "CCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", "CCCCCCCCCCCCOS(=O)(=O)O",
    "CCCCCCCCCCCCCCCCCC(=O)OCC(O)CO",
    "NCCNCCNCCN", "O=S(=O)(O)CCS(=O)(=O)O", "NCCS(=O)(=O)O",
    "OC(=O)CCCCC(=O)O", "OC(=O)C(O)C(O)C(=O)O", "NCCCCN",
    "OP(=O)(O)OP(=O)(O)O", "OCCN(CCO)CCO",
    "OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O",
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",
]


# Chemistry that is certainly outside a CNS medicinal-chemistry reference: polymers, per-fluorinated
# chains, silicones, peptides and organometallics. The original control set could not test the flag,
# because a measured sugar or fatty acid is in the reference library and the flag is right to say so.
FAR_CHEMISTRY = [
    "C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F",
    "C[Si](C)(O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C)C",
    "OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO",
    "CC(C)(C)c1ccc(O)cc1.CC(C)(C)c1ccc(O)cc1.CC(C)(C)c1ccc(O)cc1",
    "NC(CC(N)=O)C(=O)NC(CCSC)C(=O)NC(Cc1c[nH]c2ccccc12)C(=O)NC(CO)C(=O)NC(CCCNC(N)=N)C(=O)O",
    "[Fe+2].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+]",
    "O=[U](=O)([O-])[O-]", "[Pt](Cl)(Cl)(N)N",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "OS(=O)(=O)c1ccc(cc1)S(=O)(=O)Oc1ccc(cc1)S(=O)(=O)O",
    "C1CC2CCC1(CC2)C1CCC2(CC1)CCC1(CC2)CCCCC1",
    "N#Cc1c(C#N)c(C#N)c(C#N)c(C#N)c1C#N",
]


def measures(fp, ref) -> dict:
    """Every candidate measure for one query, from one similarity vector."""
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, ref))
    top = np.sort(sims)[::-1]
    return {"max": float(top[0]),
            "mean_top5": float(top[:5].mean()),
            "kth_5": float(top[4]) if len(top) >= 5 else float(top[-1]),
            "density_0.4": float((sims >= 0.40).sum())}


def main() -> None:
    with (ROOT / "models_rf" / "ad_reference.pkl").open("rb") as fh:
        obj = pickle.load(fh)
    # A three-element reference carries the fingerprint spec it was built with; a two-element one
    # predates the spec field and was necessarily built stereochemistry-blind.
    _smiles, ref = obj[0], obj[1]
    spec = obj[2] if len(obj) == 3 else _adfp.LEGACY_SPEC
    if spec != _adfp.SPEC:
        raise SystemExit(f"the reference was built with '{spec}' but this script computes "
                         f"'{_adfp.SPEC}'; rebuild it with src/sai_cns/build_ad_reference.py")
    print(f"reference chemistry: {len(ref):,} measured compounds")

    ext = pd.read_csv(ROOT / "data" / "external" / "processed" / "external_bbb_test.csv")
    col = "novel_to_model" if "novel_to_model" in ext.columns else None
    drugs = (ext[ext[col]] if col else ext[~ext["in_b3db_training"]])
    drugs = drugs["canonical_smiles"].astype(str).tolist()[:300]
    print(f"unseen approved drugs: {len(drugs)}")

    def score(smis):
        rows = []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            rows.append(measures(_GEN.GetFingerprint(m), ref))
        return pd.DataFrame(rows)

    # The original adversarial set cannot test a domain flag, because the flag's reference library
    # contains most of it. Glucose, palmitic acid, citric acid, EDTA, taurine and six others are
    # measured compounds in their own right and sit in the 158,890-compound reference, so their
    # maximum similarity is 1.00 and "in domain" is the truthful answer rather than a failure. A
    # test that asks a flag to disown chemistry it has data for is testing the wrong thing. Only
    # controls genuinely absent from the reference are kept.
    from features.featurize import parent_mol
    canon = set()
    for s in _smiles:
        m = parent_mol(str(s))
        if m is not None:
            canon.add(Chem.MolToSmiles(m))
    # Absence has to be judged on the fingerprint the flag actually compares, not on a canonical
    # SMILES: two different structures can be fingerprint-identical after folding, and such a pair
    # scores 1.00 whatever the canonical strings say. A control is kept only if its maximum
    # similarity to the reference is below 1.0, which is the same test the flag applies.
    def truly_absent(s):
        m = parent_mol(s)
        if m is None:
            return False
        sims = DataStructs.BulkTanimotoSimilarity(_GEN.GetFingerprint(m), ref)
        return max(sims) < 0.999

    absent = [s for s in ALIEN + FAR_CHEMISTRY if truly_absent(s)]
    present = [s for s in ALIEN if s not in absent]
    print(f"non-drug-like controls: {len(present)} of the original {len(ALIEN)} are themselves in "
          f"the reference library, so the flag calling them in-domain is correct rather than a "
          f"failure; {len(absent)} controls are genuinely absent and are used")

    A, D = score(absent), score(drugs)

    out = []
    for name in ("max", "mean_top5", "kth_5", "density_0.4"):
        a, d = A[name].to_numpy(), D[name].to_numpy()
        u = mannwhitneyu(d, a, alternative="greater")
        # A threshold set to reject at most 10 per cent of real drugs; how many aliens does it catch?
        thr = float(np.quantile(d, 0.10))
        caught = float((a <= thr).mean())
        separates = bool(u.pvalue < 0.01 and np.median(a) < np.median(d))
        out.append({"measure": name,
                    "median_unseen_drugs": round(float(np.median(d)), 4),
                    "median_non_drug_like": round(float(np.median(a)), 4),
                    "mann_whitney_p": f"{u.pvalue:.2e}",
                    "separates": separates,
                    "threshold_at_10pct_drug_loss": round(thr, 4),
                    "non_drug_like_caught": round(caught, 3)})
    res = pd.DataFrame(out)
    print()
    print(res.to_string(index=False))

    winners = res[res.separates]
    print()
    if len(winners):
        best = winners.sort_values("non_drug_like_caught", ascending=False).iloc[0]
        print(f"  best separating measure: {best.measure}, catching "
              f"{best.non_drug_like_caught:.0%} of non-drug-like chemistry at a threshold that "
              f"rejects 10 per cent of real drugs")
    else:
        print("  no measure separates the two populations; the honest conclusion is that a "
              "similarity-based domain flag cannot do this job")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
