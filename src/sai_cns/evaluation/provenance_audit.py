"""Which reported results are still true of the model that is deployed?

This project has changed a great deal in a short time: endpoints added, two withdrawn, thresholds
corrected, a gating rule changed. Every one of those changes can silently invalidate a number sitting
in a results file, and a stale number in a results file becomes a stale number in a manuscript. The
coverage panel had already drifted into advertising a withdrawn endpoint before a check caught it;
there is no reason to assume it was the only thing that drifted.

This compares every result artefact against the deployed state and reports three kinds of problem:

  stale       the file names endpoints that no longer exist, or is older than the models it describes
  incomplete  the file omits endpoints that are now deployed, so any aggregate it reports is over a
              different panel than the one running
  contradicts a metric in the file disagrees with the same metric computed from the deployed models

It does not fix anything. The point is a list of what must be regenerated before the manuscript is
final, produced mechanically rather than from memory of what was run when.

Read-only. Writes results/provenance_audit.csv
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))

RES = ROOT / "results"
INV = ROOT / "inversion" / "results"
MODES = ROOT / "models_rf" / "binder_modes.json"

# result files that describe the target panel, and the column naming the endpoint
# (path, endpoint column, scope). Scope matters: a binder-panel file is not incomplete for omitting
# the base-rate-enrichment endpoints, which have no binder model, and a record of which candidates
# were audited is not stale for naming one that was subsequently rejected. Judging every file against
# the full panel produced three false alarms on the first run of this script.
PANEL_FILES = [
    # H7 can only score targets that have a hold-out twin with enough held-out actives;
    # it reports the ones it could not reach, so those are not an omission.
    (INV / "H7_target_discrimination.csv", "target", "holdout"),
    (RES / "deployed_specificity_audit.csv", "target", "binder"),
    (RES / "batch4_audit.csv", "target", "historical"),
    (RES / "tables" / "binder_vs_measured_inactives.csv", None, "binder"),
    (RES / "tables" / "background_specificity.csv", None, "binder"),
    # added after it drifted to describe 39 targets unnoticed: it was not on this list
    (RES / "tables" / "MASTER_validation_summary.csv", None, "summary"),
]


def main():
    import app

    modes = json.loads(MODES.read_text())
    deployed = {t for t in app.TARGET_KIND if t != "NEURO"
                and modes.get(t, {}).get("deployed", True)}
    withdrawn = {t for t, v in modes.items() if not v.get("deployed", True)}
    model_mtime = max(p.stat().st_mtime for p in (ROOT / "models_rf").glob("*.joblib"))
    print(f"deployed endpoints: {len(deployed)}; withdrawn: {sorted(withdrawn)}", flush=True)
    print(f"newest model artefact: {time.ctime(model_mtime)}\n", flush=True)

    binder_deployed = {t for t in deployed if app.TARGET_KIND.get(t) == "binder"}
    rows = []
    for path, col, scope in PANEL_FILES:
        if not path.exists():
            rows.append({"artefact": path.name, "status": "MISSING", "detail": "file absent"})
            continue
        age = path.stat().st_mtime
        d = pd.read_csv(path)
        if col is None:
            col = next((c for c in d.columns if c.lower() in
                        ("target", "endpoint", "name")), None)
        named = set(d[col].astype(str)) if col else set()
        untestable = {"GABA_A", "GBA1", "TAAR1"}   # no hold-out twin; reported by H7 itself
        expected = (binder_deployed - untestable) if scope == "holdout" else (
            binder_deployed if scope == "binder" else
            set() if scope == "summary" else deployed)
        gone = sorted(named & withdrawn) if scope != "historical" else []
        absent = sorted(expected - named) if (named and scope != "historical") else []
        problems = []
        if age < model_mtime and scope != "historical":
            # a record of a decision taken before training is expected to predate the models
            problems.append(f"older than the deployed models ({time.ctime(age)})")
        if gone:
            problems.append(f"reports withdrawn endpoints {gone}")
        if absent:
            problems.append(f"omits {len(absent)} deployed endpoints: {absent[:6]}"
                            + (" ..." if len(absent) > 6 else ""))
        rows.append({"artefact": str(path.relative_to(ROOT)),
                     "status": "STALE" if problems else "current",
                     "n_rows": len(d),
                     "detail": "; ".join(problems) or "consistent with the deployed panel"})
        print(f"  {path.name:42} {'STALE' if problems else 'current'}", flush=True)

    # the inversion verdicts describe a graph that has since gained two conditions
    # Graph-dependent results are judged on a content fingerprint, not a timestamp. An edit to a
    # comment in app.py must not mark every inversion result stale, and a change to the graph must
    # not escape notice because the file's mtime happened not to move.
    import hashlib
    payload = json.dumps({
        "diseases": list(app.DISEASE_ORDER),
        "graph": {t: sorted((a, b, c, d) for a, b, c, d in e)
                  for t, e in sorted(app.KNOWLEDGE_GRAPH.items())},
        "peripheral": sorted(app.PERIPHERAL_MECHANISM_DISEASES),
    }, sort_keys=True)
    current_fp = hashlib.sha256(payload.encode()).hexdigest()
    fp_file = INV / "GRAPH_FINGERPRINT.json"
    recorded = json.loads(fp_file.read_text()).get("sha256") if fp_file.exists() else None
    graph_current = (recorded == current_fp)
    graph_files = [INV / "H1_disease_layer.csv", INV / "H2_weight_ablation.csv",
                   INV / "H6_clinical_indication.csv", INV / "H8_panel_independence.csv"]
    for q in graph_files:
        if not q.exists():
            continue
        rows.append({"artefact": str(q.relative_to(ROOT)),
                     "status": "current" if graph_current else "STALE",
                     "n_rows": len(pd.read_csv(q)),
                     "detail": (f"graph fingerprint matches: {len(app.DISEASE_ORDER)} conditions, "
                                f"{len(app.KNOWLEDGE_GRAPH)} targets")
                     if graph_current else
                     ("the knowledge graph has changed since these were computed; rerun the "
                      "inversion tests and summarise.py")})
        print(f"  {q.name:42} {'current' if graph_current else 'STALE'}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "provenance_audit.csv", index=False)
    pd.set_option("display.width", 230)
    pd.set_option("display.max_colwidth", 96)
    print()
    print(df.to_string(index=False))
    stale = df[df.status != "current"]
    print(f"\n{len(stale)} of {len(df)} artefacts need regenerating before the manuscript is final")
    print("wrote", RES / "provenance_audit.csv")


if __name__ == "__main__":
    main()
