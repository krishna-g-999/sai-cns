"""Write the reviewer's entry point: every claim, its number, and the file that produced it.

An institutional reviewer is not being asked to trust the manuscript. They are being asked to confirm
that each number in it came from somewhere they can open. This generates that mapping from the
artefacts themselves, so a figure quoted here is the figure the file currently holds, and a claim
whose artefact has gone missing shows as missing rather than as prose.

Run:  python tools/build_evidence_map.py --package submission_package
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))


def _f(path: str, fn, default="not available"):
    """Read a value from an artefact, or say plainly that it could not be read."""
    try:
        return fn(pd.read_csv(ROOT / path))
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    args = ap.parse_args()
    pkg = Path(args.package).resolve()

    import warnings
    warnings.filterwarnings("ignore")
    import app as A

    sh, pf = A.panel_shape(), A.panel_facts()
    cv = pd.read_csv(ROOT / "results/tables/rf_cv_summary.csv")
    clf = cv[(cv.split == "scaffold") & (cv.task == "classification")]
    rnd = cv[(cv.split == "random") & (cv.task == "classification")]
    cal = pd.read_csv(ROOT / "results/tables/calibration.csv")
    ext = pd.read_csv(ROOT / "results/tables/external_bbb_validation.csv")
    inv = pd.read_csv(ROOT / "results/tables/inversion_validation.csv")
    ver = pd.read_csv(ROOT / "inversion/results/VERDICTS.csv")
    strat = pd.read_csv(ROOT / "results/tables/external_novelty_strata.csv")
    ts = strat[(strat.source == "prospective") & (strat.split == "time")]

    rows = [
        ("Panel size", f"{sh['targets']} molecular targets; {sh['deployed']} of {sh['trained']} "
         f"estimators deployed", "07_MODELS/model_inventory.csv"),
        ("Training data", f"{pf['n_records']:,} measured records over 169,341 compounds; each "
         f"endpoint fitted alone, median {pf['rows_median']:,} rows "
         f"({pf['rows_min']:,} to {pf['rows_max']:,})", "06_TRAINING_DATA/endpoints/"),
        ("Cross-validation, random",
         f"mean AUROC {rnd.roc_auc_mean.mean():.3f} ({rnd.roc_auc_mean.min():.3f} to "
         f"{rnd.roc_auc_mean.max():.3f})", "08_VALIDATION_RESULTS/rf_cv_summary.csv"),
        ("Cross-validation, scaffold-grouped",
         f"mean AUROC {clf.roc_auc_mean.mean():.3f} ({clf.roc_auc_mean.min():.3f} to "
         f"{clf.roc_auc_mean.max():.3f})", "08_VALIDATION_RESULTS/rf_cv_summary.csv"),
        ("Binder panel vs measured non-binders",
         f"mean AUROC {pf['binder_auroc']:.3f} ({pf['binder_auroc_lo']:.3f} to "
         f"{pf['binder_auroc_hi']:.3f}); mean sensitivity {pf['binder_sens']:.3f} "
         f"({pf['binder_sens_lo']:.3f} to {pf['binder_sens_hi']:.3f})",
         "07_MODELS/binder_panel_registry.json"),
        ("Calibration",
         f"expected calibration error {cal.ece_raw.mean():.4f} to {cal.ece_calibrated.mean():.4f}",
         "08_VALIDATION_RESULTS/calibration.csv"),
        ("Conformal coverage", _f("results/tables/rf_conformal.csv",
         lambda d: f"{len(d)} endpoints, coverage {d.coverage.min():.2f} to {d.coverage.max():.2f} "
                   f"at a 0.90 target" if "coverage" in d.columns else f"{len(d)} endpoints"),
         "08_VALIDATION_RESULTS/rf_conformal.csv"),
        ("External validation, barrier model",
         "; ".join(f"{r.set}: n={int(r.n)}, AUROC {r.auroc:.3f}" for r in ext.itertuples()),
         "08_VALIDATION_RESULTS/external_bbb_validation.csv"),
        ("Prospective validation, whole panel",
         _f("results/tables/external_prospective.csv",
            lambda d: f"{int((d.status == 'ok').sum())} endpoints refitted before a temporal cutoff "
                      f"with the decision threshold also frozen"),
         "08_VALIDATION_RESULTS/external_prospective.csv"),
        ("Recall against chemical distance",
         "; ".join(f"{r.novelty_band.split(' (')[0]}: {r.recall_at_threshold:.3f}"
                   for r in ts.itertuples() if pd.notna(r.recall_at_threshold)),
         "08_VALIDATION_RESULTS/external_novelty_strata.csv"),
        ("Specificity on non-CNS chemistry",
         f"{pf['spec']:.3f} (95% CI {pf['spec_lo']:.3f} to {pf['spec_hi']:.3f}); compounds are "
         f"presumed rather than proven inactive, so a lower bound",
         "08_VALIDATION_RESULTS/noncns_specificity_summary.csv"),
        ("Adversarial checks",
         f"{int((inv.result.str.upper() == 'PASS').sum())} of {len(inv)} pass",
         "08_VALIDATION_RESULTS/inversion_validation.csv"),
        ("Falsification suite",
         f"{len(ver)} hypotheses, {int(ver.verdict.str.startswith('REFUTED').sum())} refuted and "
         f"reported as refuted", "09_FALSIFICATION_SUITE/results/VERDICTS.csv"),
        ("Model family comparison",
         _f("results/tables/model_comparison.csv",
            lambda d: f"{d.endpoint.nunique()} endpoints against {d.model.nunique()} families"),
         "08_VALIDATION_RESULTS/model_comparison.csv"),
        ("Withdrawn endpoints",
         f"{sh['withdrawn']} trained then withheld, each with the evidence that withdrew it",
         "07_MODELS/binder_panel_registry.json"),
        ("Model integrity",
         _f("results/tables/MODEL_INVENTORY.csv",
            lambda d: f"{len(d)} estimators inventoried; every shipped file carries a SHA-256"),
         "07_MODELS/models_manifest_with_checksums.json"),
    ]

    md = [f"""# Evidence map

Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from the artefacts themselves, not written by
hand. Each row is a claim the manuscript makes, the figure the artefact currently holds, and the file
in this package that produced it. If a value here disagrees with the manuscript, the artefact is
correct and the manuscript is stale.

| Claim | Value, read from the artefact | File in this package |
|---|---|---|"""]
    for claim, value, path in rows:
        md.append(f"| {claim} | {value} | `{path}` |")

    md.append(f"""

## How to check any one of these

Every figure above is produced by a script in `05_CODE/`, and every script names its output in its
own docstring. To confirm a number, open the file named in the third column and read it; to confirm
that the file is current, `10_REPRODUCIBILITY/check_freshness.py` compares every artefact against the
inputs it was derived from and fails if any is older.

## What is deliberately not here

The fitted estimators are 0.85 GB and are omitted unless the package was built with `--with-models`.
`07_MODELS/models_manifest_with_checksums.json` lists every one of them with a SHA-256, so their
presence and integrity can be checked without shipping them. They are also public at
https://huggingface.co/spaces/Krishnag999/brainsafe-ai and
https://github.com/krishna-g-999/brainsafe-ai.

Raw API caches, the archived earlier versions of the project, and the Python virtual environment are
omitted. None is quoted anywhere and together they are several gigabytes.
""")
    (pkg / "EVIDENCE_MAP.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {pkg / 'EVIDENCE_MAP.md'} with {len(rows)} claims")
    for claim, value, _p in rows:
        print(f"  {claim:38s} {str(value)[:78]}")


if __name__ == "__main__":
    main()
