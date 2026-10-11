"""The panel: one place that says what this server is made of.

Every defect this module exists to prevent has the same shape. A script keeps its own idea of what
the panel contains, the panel changes, the script's idea does not, and nothing fails. It printed
DONE. Found in one audit:

  train_binders_hybrid.py         a hardcoded list of 37 names while 44 endpoints used its mode
  train_measured_label_holdout.py a hardcoded list of 2 names while 8 endpoints used its mode
  binder_cv_per_fold.py           a resume cache keyed on endpoint name, so after a retrain it
                                  reported the previous panel's cross-validation as complete
  apply_specificity_decisions.py  a hardcoded withdrawal list, applied whatever the models now do
  Makefile `train`                omitted one of the two binder trainers entirely

Between them, `make train` refitted 39 of 52 binders and said nothing, and a withdrawn endpoint that
had recovered would have stayed withdrawn for ever.

The fix is not more lists. It is that there is exactly one list, derived from the artefacts, and that
a caller states what it intends to cover and is checked against it.

Three independent views of the panel exist on disk, and the useful work of this module is
reconciling them rather than trusting any one:

  data/endpoints/*.csv        what there is data for
  models_rf/*_binder.joblib   what was actually fitted
  models_rf/binder_modes.json what the panel claims about itself

`verify()` compares all three and reports every disagreement, including a model older than the table
it was fitted from. That single check is what turns a silent partial retrain into a loud one.

Read-only. Nothing here writes, so it can be called from anywhere, including a test.

Run:  python src/sai_cns/panel.py          # print the reconciliation
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models_rf"
TABLES = ROOT / "data" / "endpoints"
MODES = MODELS / "binder_modes.json"

HYBRID = "hybrid_decoys_plus_measured_inactives"
MEASURED_LABEL = "measured_labels_holdout"

# The reliability gate: one definition, imported by everything that writes it, draws it or displays
# it. A call is reliable only above both floors, on actives withheld by scaffold and against compounds
# measured at that same target and found inactive.
#
# It previously lived in six places and disagreed with itself in three of them. The two training
# stages gated sensitivity at 0.60 and the two threshold stages that overwrite them at 0.50, so the
# deployed rule was 0.50 by accident of ordering rather than by decision. Figure 10 drew its line at
# 0.60 and labelled it the gate, placing three endpoints that clear the gate below a line that claimed
# to be it. app.py listed low-sensitivity endpoints at a third cut of 0.65 of its own, so the About
# page named ten endpoints while six failed the gate. None of those numbers was wrong on its own
# terms, which is the difficulty: a reader has no way to tell two measurements of different things
# apart from one measurement being wrong, and the project cannot ask them to.
MIN_SENSITIVITY = 0.50
MIN_AUROC = 0.75


def passes_gate(sensitivity: float | None, auroc: float | None) -> bool:
    """Whether an endpoint's numbers clear the reliability gate. Missing either is not a pass."""
    return (sensitivity is not None and auroc is not None
            and sensitivity >= MIN_SENSITIVITY and auroc >= MIN_AUROC)


@dataclass(frozen=True)
class Endpoint:
    """One binder endpoint, as the panel currently defines it."""
    name: str
    mode: str
    deployed: bool
    reliable: bool
    threshold: float | None
    auroc: float | None
    sensitivity: float | None
    withdrawn_reason: str | None
    withdrawn_evidence: str | None

    @property
    def model_path(self) -> Path:
        return MODELS / f"{self.name}_binder.joblib"

    @property
    def table_path(self) -> Path:
        return TABLES / f"{self.name}.csv"


def _modes() -> dict:
    if not MODES.exists():
        raise FileNotFoundError(f"no panel registry at {MODES}")
    return json.loads(MODES.read_text(encoding="utf-8"))


def binders(mode: str | None = None, deployed: bool | None = None) -> list[Endpoint]:
    """Every binder endpoint, optionally filtered by mode or deployment.

    This is the function every caller should use instead of writing a list. Filtering by mode is how
    a trainer asks "which endpoints am I responsible for"; filtering by deployment is how a report
    asks "which endpoints does the server actually offer".
    """
    out = []
    for name, rec in sorted(_modes().items()):
        if mode is not None and rec.get("mode") != mode:
            continue
        is_dep = bool(rec.get("deployed", True))
        if deployed is not None and is_dep != deployed:
            continue
        out.append(Endpoint(
            name=name, mode=rec.get("mode", ""), deployed=is_dep,
            reliable=bool(rec.get("reliable_call", True)),
            threshold=rec.get("threshold"),
            auroc=rec.get("auroc_vs_measured_inactives"),
            sensitivity=rec.get("sensitivity_at_threshold"),
            withdrawn_reason=rec.get("withdrawn_reason"),
            withdrawn_evidence=rec.get("withdrawn_evidence")))
    return out


def names(**kw) -> list[str]:
    """The endpoint names only, for callers that just want a list to iterate."""
    return [e.name for e in binders(**kw)]


def withdrawn() -> list[Endpoint]:
    """Endpoints trained and then withheld, with the reason each was withheld."""
    return binders(deployed=False)


def assert_covers(claimed, expected, what: str) -> None:
    """Fail loudly when a caller's idea of the panel disagrees with the panel.

    The argument for raising rather than warning: every defect above was survivable at the moment it
    happened and only became a wrong published number later. A partial retrain that stops is a
    nuisance; a partial retrain that completes is a paper reporting estimators that do not exist.
    """
    claimed, expected = set(claimed), set(expected)
    missing, extra = expected - claimed, claimed - expected
    if missing or extra:
        parts = [f"{what} does not match the panel registry:"]
        if missing:
            parts.append(f"  not covered ({len(missing)}): {', '.join(sorted(missing))}")
        if extra:
            parts.append(f"  covered but not in the panel ({len(extra)}): {', '.join(sorted(extra))}")
        parts.append(f"  the panel is defined by {MODES.relative_to(ROOT)}")
        raise AssertionError("\n".join(parts))


def verify() -> dict:
    """Reconcile the registry, the endpoint tables and the fitted models.

    Returns a dict of findings rather than printing, so a test can assert on it. Each key names a
    way the three views can disagree, and every one of them has actually happened:

      no_model          the registry claims an endpoint that was never fitted
      no_table          a fitted model whose training table has gone
      unregistered      a fitted model the registry does not know about
      stale_model       a model older than the table it was fitted from, i.e. a partial retrain
      no_mode           an endpoint with no training mode, so no trainer owns it
      withdrawn_silent  an endpoint withheld from users with no recorded reason
    """
    modes = _modes()
    registered = set(modes)
    fitted = {p.name[: -len("_binder.joblib")] for p in MODELS.glob("*_binder.joblib")}
    tabled = {p.stem for p in TABLES.glob("*.csv")}

    stale = []
    for ep in sorted(registered & fitted & tabled):
        m, t = MODELS / f"{ep}_binder.joblib", TABLES / f"{ep}.csv"
        if m.stat().st_mtime < t.stat().st_mtime:
            stale.append(ep)

    # A clone carries the registry and the endpoint tables but not the fitted models, which are
    # deposited separately because they are gigabytes. With no models on disk the three-way
    # reconciliation has only two views, and "every endpoint is missing its model" is a statement
    # about the clone rather than a disagreement in the panel. The model-dependent findings are
    # therefore reported as empty and `models_present` says why, so a caller can tell an absent
    # panel from an inconsistent one instead of failing on a fresh checkout.
    models_present = bool(fitted)
    if not models_present:
        return {
            "n_registered": len(registered), "n_fitted": 0,
            "n_tabled": len(tabled & registered), "models_present": False,
            "no_model": [], "no_table": [], "unregistered": [], "stale_model": [],
            "no_mode": sorted(ep for ep, r in modes.items() if not r.get("mode")),
            "withdrawn_silent": sorted(ep for ep, r in modes.items()
                                       if not r.get("deployed", True)
                                       and not (r.get("withdrawn_reason") or "").strip()),
            "by_mode": {m: len(binders(mode=m)) for m in (HYBRID, MEASURED_LABEL)},
            "n_deployed": len(binders(deployed=True)),
            "n_withdrawn": len(binders(deployed=False)),
        }

    return {
        "models_present": True,
        "n_registered": len(registered),
        "n_fitted": len(fitted),
        "n_tabled": len(tabled & registered),
        "no_model": sorted(registered - fitted),
        "no_table": sorted(fitted - tabled),
        "unregistered": sorted(fitted - registered),
        "stale_model": stale,
        "no_mode": sorted(ep for ep, r in modes.items() if not r.get("mode")),
        "withdrawn_silent": sorted(ep for ep, r in modes.items()
                                   if not r.get("deployed", True)
                                   and not (r.get("withdrawn_reason") or "").strip()),
        "by_mode": {m: len([e for e in binders(mode=m)]) for m in (HYBRID, MEASURED_LABEL)},
        "n_deployed": len(binders(deployed=True)),
        "n_withdrawn": len(binders(deployed=False)),
    }


def main() -> None:
    v = verify()
    print(f"panel registry : {MODES.relative_to(ROOT)}")
    print(f"  endpoints    : {v['n_registered']} registered, {v['n_fitted']} fitted, "
          f"{v['n_tabled']} with a training table")
    print(f"  deployment   : {v['n_deployed']} deployed, {v['n_withdrawn']} withdrawn")
    for mode, n in v["by_mode"].items():
        print(f"  mode         : {n:3d}  {mode}")
    problems = [k for k in ("no_model", "no_table", "unregistered", "stale_model", "no_mode",
                            "withdrawn_silent") if v[k]]
    print()
    if not v["models_present"]:
        print("no fitted models on disk, so only the registry and the endpoint tables were "
              "compared")
    if not problems:
        print("the registry, the endpoint tables and the fitted models agree"
              if v["models_present"] else "the registry and the endpoint tables agree")
        return
    for k in problems:
        print(f"  {k:17s} {len(v[k])}: {', '.join(v[k])}")
    raise SystemExit(f"\n{len(problems)} kind(s) of disagreement; the panel is not self-consistent")


if __name__ == "__main__":
    main()
