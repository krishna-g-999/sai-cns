"""The label-permutation null the technical report asserts but never wrote to a file.

Section 6.9 of the technical report states that with labels permuted the same pipeline on the same
folds returns a mean AUROC of 0.4938 random and 0.4921 scaffold, worst single endpoint 0.5174. A
thesis audit found that those three numbers are hard-coded in a literal prose block in
`build_technical_report.py`, that no artefact in the repository holds them, and that no script
computes them. The permutation null is the evidence that the cross-validated figures are not
inflated by leakage, so it is the one claim in the validation chapter a reader cannot check.

This script computes it. It imports train_rf rather than reimplementing it, so the featuriser, the
deduplication, the scaffold grouping, the fold objects and the forest hyper-parameters are the same
by construction and not by resemblance. The only change is that the label vector is permuted once
per endpoint before cross-validation begins.

What a permuted label destroys, and what it does not:

  destroyed    any association between a compound's features and its class
  preserved    the class balance, the fold sizes, the scaffold grouping, and the tendency of whole
               scaffold classes to be sampled together

The second column is the point. A scaffold-grouped fold could report an inflated AUROC without any
chemistry being learned, if scaffold classes differed enough in class frequency for a model to
exploit the grouping alone. Permuting the labels breaks the chemistry but keeps the grouping, so a
scaffold figure that stays at chance under permutation cannot be an artefact of the grouping.

One permutation is drawn per endpoint, with a seed derived from the endpoint name so the draw is
reproducible and independent across endpoints. That gives eight paired observations per split rather
than a null distribution for any one endpoint, which is what the report's claim is: a statement
about the panel, not a per-endpoint p-value. The spread across the eight is reported alongside the
mean, because a mean of 0.49 over a range of 0.44 to 0.55 and a mean of 0.49 over a range of 0.489
to 0.491 are different results.

Run:  brainsafe_env/Scripts/python.exe src/sai_cns/evaluation/permutation_null.py
Out:  results/tables/permutation_null.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
sys.path.insert(0, str(ROOT / "src" / "sai_cns" / "models"))

from models.train_rf import (  # noqa: E402
    CLASSIFICATION, _cv, _dedup_features, _load, _scaffold_groups,
)
from features.featurize import featurize  # noqa: E402

OUT = ROOT / "results" / "tables" / "permutation_null.csv"


def _seed(name: str) -> int:
    """A per-endpoint seed that does not depend on iteration order."""
    return int.from_bytes(name.encode("utf-8"), "little") % (2**31 - 1)


def main() -> None:
    rows = []
    for ep in CLASSIFICATION:
        t0 = time.time()
        df = _load(ep).dropna(subset=["smiles", "label"]).reset_index(drop=True)
        X, mask = featurize(df["smiles"].tolist())
        df = df.loc[mask].reset_index(drop=True)
        y = df["label"].to_numpy().astype(int)
        groups = _scaffold_groups(df["smiles"].tolist())
        smiles = df["smiles"].tolist()
        X, y, groups, smiles, _ = _dedup_features(X, y, groups, smiles, "classification")

        rng = np.random.default_rng(_seed(ep))
        y_perm = rng.permutation(y)
        assert y_perm.sum() == y.sum(), "permutation must preserve the class balance"

        fold_rows, _ = _cv(ep, "classification", X, y_perm, groups, smiles)
        fr = pd.DataFrame(fold_rows)
        for split, g in fr.groupby("split"):
            rows.append({
                "endpoint": ep, "split": split, "n": int(X.shape[0]),
                "n_positive": int(y.sum()), "n_scaffolds": int(len(set(groups))),
                "permuted_roc_auc_mean": round(float(g.roc_auc.mean()), 4),
                "permuted_roc_auc_sd": round(float(g.roc_auc.std(ddof=1)), 4),
                "permuted_roc_auc_min_fold": round(float(g.roc_auc.min()), 4),
                "permuted_roc_auc_max_fold": round(float(g.roc_auc.max()), 4),
            })
        print(f"[{ep:8}] n={X.shape[0]:6}  "
              + "  ".join(f"{r['split']} {r['permuted_roc_auc_mean']:.4f}"
                          for r in rows[-2:])
              + f"   ({time.time() - t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print()
    print(out.to_string(index=False))
    print()
    for split, g in out.groupby("split"):
        v = g.permuted_roc_auc_mean
        worst = g.loc[(v - 0.5).abs().idxmax()]
        print(f"{split:9} mean {v.mean():.4f}  range {v.min():.4f} to {v.max():.4f}  "
              f"furthest from chance: {worst.endpoint} at {worst.permuted_roc_auc_mean:.4f}")
    print(f"\nwrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
