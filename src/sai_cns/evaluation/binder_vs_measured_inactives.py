"""Rebuild the binder-versus-measured-inactives table from the registry that supersedes it.

This table existed as an orphan. It was written by an earlier version of the binder training, no
script regenerated it afterwards, and it was not declared in tools/check_freshness.py, so nothing
could notice when the panel was refitted and the file was not. It sat at a mean AUROC of 0.9765
where the registry holds 0.9113 over the same 47 targets, an inflation of about 0.065, and it
shipped in the institutional review package in that state.

Every column it carries is present in models_rf/binder_modes.json, which is the file the server, the
manuscript and the technical report all read. Regenerating it from the registry rather than deleting
it keeps provenance_audit.py working, and putting it in the freshness graph means the next retrain
cannot leave it behind.

Output: results/tables/binder_vs_measured_inactives.csv

Run:  python src/sai_cns/evaluation/binder_vs_measured_inactives.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
OUT = ROOT / "results" / "tables" / "binder_vs_measured_inactives.csv"

COLUMNS = [("n_positive", "n_positive"),
           ("n_measured_inactive_holdout", "n_measured_inactive_holdout"),
           ("scaffold_cv_auroc", "scaffold_cv_auroc"),
           ("auroc_vs_measured_inactives", "auroc_vs_measured_inactives"),
           ("threshold", "threshold"),
           ("sensitivity_at_threshold", "sensitivity_at_threshold"),
           ("background_fpr_held_out", "background_fpr_held_out")]


def main() -> None:
    modes = json.loads((ROOT / "models_rf" / "binder_modes.json").read_text(encoding="utf-8"))
    rows = []
    for target, rec in sorted(modes.items()):
        if not rec.get("deployed", True):
            continue
        row = {"target": target}
        row.update({out: rec.get(src) for out, src in COLUMNS})
        rows.append(row)

    d = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)

    a = pd.to_numeric(d.auroc_vs_measured_inactives, errors="coerce").dropna()
    s = pd.to_numeric(d.sensitivity_at_threshold, errors="coerce").dropna()
    print(f"{len(d)} deployed binder endpoints")
    print(f"  AUROC vs measured inactives : mean {a.mean():.4f}, "
          f"range {a.min():.3f} ({d.target[a.idxmin()]}) to {a.max():.3f} ({d.target[a.idxmax()]})")
    print(f"  sensitivity at threshold    : mean {s.mean():.4f}, "
          f"range {s.min():.3f} ({d.target[s.idxmin()]}) to {s.max():.3f} ({d.target[s.idxmax()]})")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
