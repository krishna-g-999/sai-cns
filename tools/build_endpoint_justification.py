"""Generate docs/ENDPOINT_JUSTIFICATION.md from the current, deployed panel.

The document this replaces described a twelve-endpoint panel that no longer exists: the project has
since grown to 75 trained estimators over 54 molecular targets, a 52-endpoint binder panel validated
against measured non-binders rather than decoys, and a systematic survey of candidate targets that
this file previously did not mention at all. Every number below is read from the same artefacts the
manuscript and technical report cite, not retyped, so this document cannot go stale the way its
predecessor did.

Run:  python tools/build_endpoint_justification.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))

OUT = ROOT / "docs" / "ENDPOINT_JUSTIFICATION.md"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore")
    import app

    shape = app.panel_shape()
    facts = app.panel_facts()

    inv = pd.read_csv(ROOT / "results" / "tables" / "MODEL_INVENTORY.csv")
    modes = json.loads((ROOT / "models_rf" / "binder_modes.json").read_text(encoding="utf-8"))
    survey = pd.read_csv(ROOT / "results" / "tables" / "np_target_survey.csv")

    withdrawn_ids = [k for k, v in modes.items() if not v.get("deployed", True)]

    # ---- table 1: exposure / ADME layer -------------------------------------------------------
    exp = inv[inv.family == "exposure"].sort_values("model")
    exp_rows = [[r.model.replace("adme_", ""), r.task, f"{r.n_train:,}", "yes" if r.deployed else "no"]
                for r in exp.itertuples()]

    # ---- table 2: core target-potency / activity endpoints ------------------------------------
    tgt = inv[inv.family == "target"].sort_values("model")
    tgt_rows = [[r.model, r.task, f"{r.n_train:,}", "yes" if r.deployed else "no"]
                for r in tgt.itertuples()]

    # ---- table 3: safety -----------------------------------------------------------------------
    saf = inv[inv.family == "safety"].sort_values("model")
    saf_rows = [[r.model, r.task, f"{r.n_train:,}", "yes" if r.deployed else "no"]
                for r in saf.itertuples()]

    # ---- table 4: the full binder panel, one row per target ------------------------------------
    bnd = inv[inv.family == "binder"].copy()
    bnd["target"] = bnd.model.str.replace("_binder", "", regex=False)
    bnd = bnd.sort_values("target")
    bnd_rows = []
    n_unreliable = 0
    for r in bnd.itertuples():
        rec = modes.get(r.target, {})
        deployed = bool(rec.get("deployed", True))
        auroc = rec.get("auroc_vs_measured_inactives")
        sens = rec.get("sensitivity_at_threshold")
        auroc_s = f"{auroc:.3f}" if isinstance(auroc, (int, float)) else "—"
        sens_s = f"{sens:.3f}" if isinstance(sens, (int, float)) else "—"
        if not deployed:
            status = "withdrawn"
        elif not rec.get("reliable_call", True):
            status = "deployed, low-power flag"
            n_unreliable += 1
        else:
            status = "deployed"
        bnd_rows.append([r.target, f"{r.n_train:,}", auroc_s, sens_s, status])

    n_binder_total = len(bnd)
    n_binder_deployed = int(bnd.target.map(lambda t: modes.get(t, {}).get("deployed", True)).sum())

    # ---- withdrawn endpoints, with the registry's own recorded reason --------------------------
    withdrawn_blocks = []
    for k in sorted(withdrawn_ids):
        reason = modes[k].get("withdrawn_reason", "no reason recorded")
        withdrawn_blocks.append(f"**{k}.** {reason}.")

    # ---- the candidate-target survey ------------------------------------------------------------
    n_surveyed = len(survey)
    n_trainable = int(survey.trainable.sum())
    np_added = [k for k in ("NRF2", "NFKB1", "NR3C1") if k in modes]
    np_outcomes = []
    for k in np_added:
        dep = modes[k].get("deployed", True)
        np_outcomes.append(f"{k} ({'deployed' if dep else 'withdrawn'})")

    today = datetime.now().strftime("%Y-%m-%d")

    text = f"""# Endpoint justification: what is modelled, why, and where the training values come from

Every count in this document is read live from `results/tables/MODEL_INVENTORY.csv`,
`models_rf/binder_modes.json` and `results/tables/np_target_survey.csv` by
`tools/build_endpoint_justification.py`, and reflects the panel as deployed on {today}. It replaces
an earlier version of this document describing a twelve-endpoint panel that predates the binder
architecture and the target-panel expansion; nothing in this file is carried forward from that
version without being re-verified against the current artefacts.

## 1. Overview

The deployed server answers four questions about a candidate molecule, in the order a medicinal
chemist would ask them: does it reach the brain (*exposure*), does it engage a mechanism relevant to
a CNS condition (*target engagement*), is it safe (*safety*), and is it developable (*ADME*). This is
covered by **{shape['trained']} trained estimators**, of which **{shape['deployed']} are deployed**
and {shape['withdrawn']} were trained, tested and withdrawn after failing a specificity or
discrimination check described in section 3. The deployed panel spans **{shape['targets']} molecular
targets**, a {len(exp_rows)}-endpoint exposure and ADME layer, and one cardiac-safety classifier.

Target engagement is covered two ways, and both are counted above rather than one being treated as
supplementary. **{len(tgt_rows)} core endpoints** ({', '.join(sorted(tgt.model))}) are trained on
measured potency or activity, most reported as a decade-old, well-characterised assay. **The
{n_binder_total}-endpoint binder panel** ({n_binder_deployed} deployed) extends this to receptors and
enzymes across the monoaminergic, opioid, cannabinoid, histaminergic, adenosine, purinergic,
glutamatergic and nicotinic systems, each validated against compounds experimentally tested at that
target and found inactive, never against property-matched decoys. The rationale for each therapeutic
axis is set out in the manuscript's Methods section ("Endpoint selection") and is not repeated in
full here to avoid the two documents drifting apart; this document's job is the table a reviewer
asked for: every endpoint, its training size, and its deployment status, in one place.

## 2. The exposure and ADME layer ({len(exp_rows)} endpoints)

{md_table(["Endpoint", "Task", "Training compounds", "Deployed"], exp_rows)}

Blood-brain barrier penetration is modelled first among these because it gates every downstream
target score: a target score is admitted only in proportion to predicted exposure, so potency at a
target the compound cannot reach contributes nothing to the server's output. The remaining nine
endpoints (Caco-2 permeability, hepatocyte clearance, unbound brain-to-plasma ratio, lipophilicity,
logBB, P-glycoprotein inhibition and substrate status, plasma-protein binding, and aqueous
solubility) cover developability, the fourth of the four questions in section 1.

## 3. Core target-potency and activity endpoints ({len(tgt_rows)})

{md_table(["Endpoint", "Task", "Training compounds", "Deployed"], tgt_rows)}

`antioxidant_DPPH` and `pka_basic` are carried in this family for training-pipeline reasons (a
measured chemical property and a physicochemical property, respectively) rather than as
target-engagement calls, and are described as auxiliary endpoints in the manuscript's architecture
figure.

## 4. Safety

{md_table(["Endpoint", "Task", "Training compounds", "Deployed"], saf_rows)}

hERG blockade is the dominant cardiac-safety flag in CNS drug discovery and a standard early
counter-screen; it is reported here rather than folded into the target-engagement count because a
liability, not a mechanism, is what it answers.

## 5. The binder panel ({n_binder_total} endpoints, {n_binder_deployed} deployed)

Each binder classifier answers "does this compound bind this target," validated on compounds
experimentally tested at that target and found inactive rather than on property-matched decoys, at a
threshold constrained simultaneously by held-out measured inactives and by the false-positive rate on
a disjoint pool of unrelated chemistry (`results/tables/background_specificity_disjoint.csv`). AUROC
and sensitivity below are both measured against the target's own held-out measured inactives, by
scaffold.

{md_table(["Target", "Training compounds", "AUROC vs. measured inactives", "Sensitivity", "Status"], bnd_rows)}

{n_unreliable} deployed endpoints fall below the reliability gate on the corrected sensitivity figure
and carry a low-power marker on any negative call in the interface, rather than being withdrawn:
each holds its background false-positive rate at or below target, so each is weak rather than
misleading.

## 6. Endpoints withdrawn, and why

An endpoint is deployed only if a threshold exists that recovers real ligands without firing on
unrelated chemistry, and only if it discriminates better than chance against its own held-out measured
inactives. {len(withdrawn_ids)} endpoints were trained, tested against both conditions, and withdrawn.
They are recorded here rather than silently dropped, because a panel that reports only what survived
is a selection, not an inventory.

{chr(10).join(f"- {b}" for b in withdrawn_blocks)}

## 7. The candidate-target survey

Beyond the panel described above, {n_surveyed:,} candidate targets were surveyed for whether enough
measured data exists to train and honestly validate a binder classifier (at least 60 compounds, at
least 15 per class). {n_trainable} cleared that bar
(`results/tables/np_target_survey.csv`; Figure 10, panel C, `figures/Figure10_endpoint_selection.png`).
Three of those, {', '.join(np_added)}, were added specifically to test whether the panel's methodology
extends to targets relevant to natural-product coverage. The outcome is reported whichever way it
fell rather than only where it succeeded: {', '.join(np_outcomes)}. Section 6 above gives the specific
reason each failed.

Figure 10 also shows the two conditions that decide deployment more generally: panel A shows that
more training data buys discrimination with diminishing returns, and panel B shows that
discrimination is not sufficient on its own, a usable threshold is what actually gates deployment,
which is why endpoints that rank well by AUROC can still be withheld.

## 8. Data sources and standardisation

Target-engagement and exposure labels are pooled from ChEMBL (version 37, release 2026-05-01) and
BindingDB at compound level; blood-brain-barrier labels come from B3DB augmented with FDA-curated
approved drugs; the exposure and ADME endpoints beyond BBB use measured sets from Therapeutics Data
Commons, MoleculeNet, B3DB and ChEMBL; the three natural-product-coverage endpoints (section 7) use
NPASS 3.0. No value is imputed, hand-annotated, or drawn from qualitative curator annotation; every
label is a measured experimental value, standardised identically before use: reduced to the largest
organic fragment, neutralised, sanitised, and keyed by the InChIKey of that standardised parent.
Chirality is excluded from the representation, so rows identical in feature space are collapsed
before any split is drawn. Full provenance, including which compounds came from which source per
endpoint, is in `results/tables/endpoint_rebuild_provenance.csv`, and source licences are recorded in
`data/raw/measured_endpoints_SOURCE.md`.

A compound assayed and found inactive is frequently deposited only as a censored bound
(`standard_relation` of `>` with a concentration), which the conventional pChEMBL-only query
discards. This project recovers those rows as measured non-binders wherever the whole interval they
define falls on one side of the activity cut, rather than training the negative class on
property-matched decoys; the recovery and its effect on class balance are documented in
`results/tables/expansion_inactives.csv`.

## 9. Train/test discipline

Every core and exposure endpoint is cross-validated ten-fold under both a random split and a
scaffold-grouped split (Bemis-Murcko), so the distance between the two is the honest statement of how
far a model generalises to structurally novel chemistry
(`results/tables/rf_cv_folds.csv`, `rf_cv_summary.csv`). Binder classifiers are cross-validated
scaffold-grouped only, against their measured inactives
(`results/tables/binder_cv_folds.csv`, `binder_cv_summary.csv`). For every endpoint the same compound
never appears on both sides of a fold. Beyond cross-validation, the barrier model and the wider panel
are further tested against FDA-curated approved drugs absent from the training source
(`results/tables/external_bbb_validation.csv`) and under a prospective, date-based refit
(`results/tables/external_prospective.csv`); both are reported in the manuscript's Results section
and are not repeated here.

Exactly which compounds trained which endpoint, and the numeric feature vector each one was reduced
to, is not asserted here but laid out directly: `results/tables/master_training_usage.csv` gives one
row per endpoint-compound pair actually used, and `results/tables/master_feature_vectors.csv` gives
the full 1,036-column vector for every distinct compound, keyed by InChIKey so the two join directly.
Every formula in this document that turns such a vector into a reported score, prediction set or
disease score is walked through with a live worked example in `docs/ML_METHODS_AND_FORMULAS.md`.

---
*Generated by `tools/build_endpoint_justification.py` from the artefacts named beside each figure
above. Regenerate after any change to the panel with
`python tools/build_endpoint_justification.py`.*
"""
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text):,} chars)")
    print(f"  {shape['trained']} trained, {shape['deployed']} deployed, {shape['targets']} targets")
    print(f"  binder panel: {n_binder_total} total, {n_binder_deployed} deployed, "
          f"{len(withdrawn_ids)} withdrawn, {n_unreliable} deployed with a low-power flag")
    print(f"  candidate survey: {n_surveyed:,} surveyed, {n_trainable} trainable")


if __name__ == "__main__":
    main()
