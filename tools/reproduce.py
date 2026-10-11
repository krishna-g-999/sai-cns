"""One command that regenerates the results, on any platform.

The Makefile is the canonical description of what depends on what, but `make` is not present on a
default Windows install and this project is developed on Windows and deployed on Linux. This runs
the same sequences from Python so that "one command" is true everywhere rather than true on the
maintainer's machine.

Targets mirror the Makefile exactly. If the two ever disagree, the Makefile is the one to trust for
ordering and this file is the bug.

  reproduce   inventory, figures, manuscript, provenance. Minutes. Does not retrain.
  figures     every manuscript and supplementary figure
  check       freshness, tests, ledger
  thresholds  the four-step binder threshold sequence, which must run whole or not at all
  train       refit the panel. Hours. Read REPRODUCE.md first.

Every step is timed and its exit status recorded, and the run stops at the first failure rather than
continuing and producing a half-regenerated set that looks complete.

Run:  python tools/reproduce.py            (the default target: reproduce)
      python tools/reproduce.py check
      python tools/reproduce.py --list
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SRC = "src/sai_cns"

FIGURES = [f"{SRC}/figures/fig{n}.py" for n in (
    "01_architecture", "02_feature_vector", "03_cv_design", "04_pools_and_thresholds",
    "05_negative_class", "06_validation", "07_binder_panel", "08_use_case", "09_model_atlas")]

TARGETS: dict[str, list[list[str]]] = {
    "inventory": [[PY, f"{SRC}/analysis/build_model_inventory.py"]],
    "figures": [[PY, f] for f in FIGURES],
    "manuscript": [[PY, f"{SRC}/analysis/manuscript_tables.py"],
                   [PY, f"{SRC}/analysis/build_manuscript.py"]],
    "provenance": [[PY, "tools/build_provenance.py"]],
    "freshness": [[PY, "tools/check_freshness.py"]],
    "test": [[PY, "-m", "pytest", "tests/", "-q"]],
    "ledger": [[PY, "validation/repro/r03_ledger.py"]],
    # All four write models_rf/binder_modes.json and each depends on the one before it. Running one
    # alone silently reverts the later ones and can re-deploy a withdrawn endpoint.
    "thresholds": [[PY, f"{SRC}/models/final_thresholds.py"],
                   [PY, f"{SRC}/models/screening_thresholds.py"],
                   [PY, f"{SRC}/models/apply_specificity_decisions.py"],
                   [PY, f"{SRC}/models/calibrate_background_specificity.py"]],
}
COMPOSITE = {
    "reproduce": ["inventory", "figures", "manuscript", "provenance"],
    "check": ["freshness", "test", "ledger"],
}


def expand(target: str) -> list[list[str]]:
    if target in COMPOSITE:
        return [c for t in COMPOSITE[target] for c in expand(t)]
    if target in TARGETS:
        return TARGETS[target]
    raise SystemExit(f"unknown target {target!r}; try --list")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Regenerate results on any platform.")
    ap.add_argument("target", nargs="?", default="reproduce")
    ap.add_argument("--list", action="store_true", help="show the targets and exit")
    args = ap.parse_args(argv)

    if args.list:
        print("composite:", ", ".join(COMPOSITE))
        print("single   :", ", ".join(TARGETS))
        return

    commands = expand(args.target)
    print(f"target {args.target}: {len(commands)} step(s)")
    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"started {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    log, t_all = [], time.time()
    for i, cmd in enumerate(commands, 1):
        label = Path(cmd[-1]).name if cmd[-1].endswith(".py") else " ".join(cmd[-2:])
        print(f"[{i}/{len(commands)}] {label} ...", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        dur = time.time() - t0
        log.append({"step": label, "command": " ".join(cmd), "exit": r.returncode,
                    "seconds": round(dur, 1)})
        print(f"        exit {r.returncode} in {dur:.1f}s\n", flush=True)
        if r.returncode != 0:
            # Stop here. Continuing would leave a half-regenerated set that looks complete.
            print(f"FAILED at step {i}: {label}")
            _write(args.target, log, time.time() - t_all, ok=False)
            raise SystemExit(r.returncode)

    total = time.time() - t_all
    print(f"{args.target} complete in {total:.1f}s")
    if args.target == "reproduce":
        print("verify with: python tools/reproduce.py check")
    _write(args.target, log, total, ok=True)


def _write(target: str, log: list[dict], total: float, ok: bool) -> None:
    """Record what ran and how long it took, so runtimes are measured rather than estimated."""
    out = ROOT / "repro" / "run_log.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": target, "ok": ok,
        "finished": dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "total_seconds": round(total, 1),
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "cpu_count": __import__("os").cpu_count(),
        "steps": log,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main(sys.argv[1:])
