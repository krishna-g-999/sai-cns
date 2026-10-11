"""Bring background_specificity.csv's reported columns back into line with the registry.

This table is the record of the background-specificity calibration run: for each endpoint it holds
the threshold before and after the run, the background false-positive rate before and after, the
sensitivity measured at the new threshold, and whether the endpoint was judged reliable.

Four of those six columns are still correct. `new_threshold` and `background_fpr_after` match
models_rf/binder_modes.json on all 47 deployed endpoints, and `old_threshold` and
`background_fpr_before` are historical: they describe the state before the run and exist nowhere
else, so they are carried forward untouched.

The other two went stale when the panel's sensitivity was corrected from the in-sample figure to the
held-out one. The table went on reporting a mean of 0.8983 and 46 of 47 endpoints reliable, against
the registry's 0.7638 and 41. That would be a private inconsistency except that a byte-identical copy
ships at submission_package/08_VALIDATION_RESULTS/background_specificity.csv, so a reviewer opening
the validation results found a table contradicting the panel it describes.

The freshness graph could not catch it, and cannot be made to. The artefact is declared against
models_rf/*_binder.joblib, and the fitted models genuinely did not change: only the figure computed
from them did. Redeclaring it against binder_modes.json would be worse, not better, for the reason
recorded at tools/check_freshness.py:110: the registry is a co-output of the same threshold sequence
that writes this table, so an edge from it can never be satisfied, and a check that cannot be
satisfied by doing the right thing teaches the reader to ignore it.

Staleness of this kind is a disagreement between two files, not a difference in their timestamps, so
it is pinned by a test instead:
tests/test_panel_app_consistency.py::TestReportedTablesAgreeWithRegistry.

Regenerating by re-running calibrate_background_specificity.py would also work and would be the
obvious move, but it rewrites the registry and recomputes every threshold to do it. Thresholds are
the operating parameters of a deployed panel; a reported column is not worth that risk.

Run:  python src/sai_cns/evaluation/refresh_background_specificity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "results" / "tables" / "background_specificity.csv"
REG = ROOT / "models_rf" / "binder_modes.json"
SHIPPED = ROOT / "submission_package" / "08_VALIDATION_RESULTS" / "background_specificity.csv"

CARRIED = ["old_threshold", "background_fpr_before"]      # historical, held nowhere else
FROM_REGISTRY = {"new_threshold": "threshold",
                 "background_fpr_after": "background_fpr_at_threshold",
                 "sensitivity_after": "sensitivity_at_threshold",
                 "reliable": "reliable_call"}


def main() -> None:
    modes = json.loads(REG.read_text(encoding="utf-8"))
    d = pd.read_csv(CSV)
    before_sens, before_rel = d.sensitivity_after.mean(), int(d.reliable.sum())

    missing = [t for t in d.target if t not in modes]
    if missing:
        raise SystemExit(f"targets absent from the registry: {missing}")

    for col, field in FROM_REGISTRY.items():
        d[col] = [modes[t].get(field) for t in d.target]
    d = d[["target", *CARRIED, *FROM_REGISTRY]]
    d.to_csv(CSV, index=False)

    print(f"{len(d)} endpoints")
    print(f"  mean sensitivity {before_sens:.4f} -> {d.sensitivity_after.mean():.4f}")
    print(f"  reliable         {before_rel} -> {int(d.reliable.sum())} of {len(d)}")

    if SHIPPED.exists():
        SHIPPED.write_bytes(CSV.read_bytes())
        print(f"  refreshed the shipped copy at {SHIPPED.relative_to(ROOT)}")
    else:
        print("  no shipped copy found; nothing else to update")
    print(f"\nwrote {CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
