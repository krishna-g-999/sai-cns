"""Act on the deployed-specificity audit.

The audit (results/deployed_specificity_audit.csv) scores every binder model at its calibrated
threshold against 600 random PubChem structures and against molecules no CNS target plausibly binds:
glucose, urea, acetate, ethanol, glycine, lactate, atenolol. A model that calls urea a sodium-channel
binder is not a marginal calibration issue, it is a false-positive generator inside a tool whose
purpose is to stop laboratories spending money on compounds that will not work.

Every decision here is re-derived from that audit whenever the models change, never inherited. A
withdrawal is a claim about a particular fit, and the panel has been refitted; carrying a decision
across a retrain would assert something about estimators that no longer exist. Re-deriving it after
the neutralised-representation retrain moved two entries and emptied a third category:

  Nav1_1     WITHDRAWN, as before but for freshly measured reasons. It fires on glucose, urea,
             glycine, lactate and atenolol at its calibrated threshold of 0.571, at a
             random-chemistry false-positive rate of 0.080. Holding 5 per cent would need 0.594,
             and sensitivity at the calibrated cut is already 0.120, so no cut separates a
             metabolite from a sodium-channel ligand. Its AUROC of 0.952 measures ranking, which is
             not the quantity a deployed threshold needs. Nav1.6 covers the axonal sodium-channel
             mechanism instead.

  GluA2      WITHDRAWN, and this is new: it passed before. It now fires on glucose and atenolol at
             a calibrated 0.629, reaching 0.719 on a trivial molecule, with a random-chemistry rate
             of 0.072 and a sensitivity of 0.103.

  Cav3_2     REINSTATED. It was rejected for an active band compressed near zero, a calibrated
             threshold of 0.065 and atenolol scoring 0.084 above it. On the refit the threshold is
             0.370, the highest any trivial control reaches is 0.048, random chemistry is 0.000, and
             it scores AUROC 0.982 at sensitivity 0.975. The compression belonged to the old fit,
             not to the target, and withholding it now would be an inherited decision rather than a
             measured one.

  NRF2       WITHDRAWN, natural-product coverage; random chemistry 0.057, above the 5 per cent the
  NFKB1      panel holds to, and NFKB1 fires on five trivial metabolites at sensitivity 0.000 while
  NR3C1      NR3C1 fails on discrimination instead, at AUROC 0.410 against its own held-out measured
             inactives. Reasons and evidence per endpoint are in WITHDRAW below.

  no         RETHRESHOLDING. All three overrides (a3b4nAChR 0.450, SIRT1 0.650, RIPK1 0.500) were
  overrides  removed: each was a remedy for an observation that does not reproduce on the current
             fits, and every one of the three now passes the audit at its calibrated threshold. An
             override whose justification has expired silently costs sensitivity.

Writes models_rf/binder_modes.json and results/specificity_decisions.csv
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
import panel  # noqa: E402
M = ROOT / "models_rf"
OUT = ROOT / "results"

# Each reason states the measurement that produced it, so a reason can be checked against a rerun of
# the audit rather than believed. See the module docstring for what moved and why.
WITHDRAW = {
    "Nav1_1": "fires on glucose, urea, glycine, lactate and atenolol at its calibrated threshold of "
              "0.571, with a random-chemistry false-positive rate of 0.080. Holding 5 per cent on "
              "random chemistry would need 0.594, and its sensitivity at the calibrated cut is "
              "already 0.120, so no threshold separates trivial metabolites from real ligands. Its "
              "AUROC of 0.952 measures ranking, which is not the quantity a deployed cut needs",
    "GluA2": "fires on glucose and atenolol at its calibrated threshold of 0.629, reaching 0.719 on "
             "a trivial molecule, with a random-chemistry false-positive rate of 0.072 and a "
             "sensitivity of 0.103. Withdrawn on the current fits, having previously passed",
    # The three below were added to test whether the panel's natural-product gap could be closed by
    # adding the targets natural products are actually assayed against. They fail for three different
    # reasons, which is why each carries its own: NRF2 on random-chemistry false positives, NFKB1 on
    # trivial metabolites, NR3C1 on discrimination alone. NRF2 in particular now passes the
    # reliability gate, so it is withdrawn on measured specificity rather than on a flag, and none of
    # the three can be dismissed as noise: each carries cross-validated signal it cannot convert into
    # a usable threshold. Assay composition: results/tables/np_endpoint_assay_composition.csv
    "NRF2": "added to test natural-product coverage. Its random-chemistry false-positive rate is "
            "0.057, above the 5 per cent the panel holds to, and reaching that rate would need a "
            "threshold of 0.303 against a calibrated 0.301. On the refit it discriminates better "
            "than it did (AUROC 0.789 against its own held-out measured inactives, sensitivity "
            "0.545) but it still cannot be given a cut that controls false positives. Of its "
            "labelled NPASS records 0.0 per cent are a direct binding constant (Ki or Kd); 1,019 "
            "of 1,029 are Potency, a pooled functional readout that does not define a binding "
            "class a ligand fingerprint can separate",
    "NFKB1": "added to test natural-product coverage. Fires on glucose, urea, acetate, glycine and "
             "lactate at its calibrated threshold of 0.416, and scores AUROC 0.459 against its own "
             "held-out measured inactives at a sensitivity of 0.000: it recovers no active while "
             "calling five trivial metabolites binders. Of its labelled NPASS records 0.3 per cent "
             "are a direct binding constant; 344 of 362 are Potency. Same cause as NRF2",
    "NR3C1": "added to test natural-product coverage. It passes the specificity audit, firing on no "
             "trivial molecule, and fails on discrimination instead: AUROC 0.410 against its own "
             "held-out measured inactives is below chance, at a sensitivity of 0.167. The only one "
             "of the three with real binding data (14.8 per cent Ki or Kd) and the one with too few "
             "compounds to fit: 140 after deduplication",
}
# Nav1_1 and GluA2 are caught by the deployed-specificity audit and that file is their evidence.
# The three natural-product endpoints are now measured by it too, since the audit no longer skips
# withdrawn endpoints, but the audit alone does not explain them: NR3C1 passes it outright and fails
# on discrimination, and all three are ultimately limited by what their labels are made of. Their
# evidence therefore also names the cross-validation and the assay composition.
WITHDRAW_EVIDENCE = {
    "NRF2": "results/tables/binder_cv_summary.csv, models_rf/binder_modes.json, "
            "results/tables/np_endpoint_assay_composition.csv",
    "NFKB1": "results/tables/binder_cv_summary.csv, models_rf/binder_modes.json, "
             "results/tables/np_endpoint_assay_composition.csv",
    "NR3C1": "results/tables/binder_cv_summary.csv, models_rf/binder_modes.json, "
             "results/tables/np_endpoint_assay_composition.csv",
}
# All three rethresholds were withdrawn after the retrain, because each was a remedy for a specific
# observation and none of those observations reproduces on the current fits. Re-measured by
# deployed_specificity_audit.py at the calibrated thresholds, before any override:
#
#   a3b4nAChR  calibrated 0.279; no trivial control fires; the highest any reaches is 0.193, and
#              ethanol, which drove the original raise to 0.450, is below it. Random-chemistry
#              false-positive rate 0.012.
#   SIRT1      calibrated 0.465; random-chemistry false-positive rate 0.033, already under the 5 per
#              cent the raise to 0.650 was meant to restore. No trivial control fires.
#   RIPK1      calibrated 0.050, the permitted floor; the highest probability any trivial control
#              reaches is 0.010, so acetate, glycine and lactate no longer sit above the cut. Random
#              chemistry 0.002.
#
# Keeping an override whose justification has expired is not caution, it is an unmeasured cost: each
# one trades sensitivity for a false positive that no longer occurs. They are removed rather than
# retained, and the audit is what would bring any of them back.
RETHRESHOLD: dict[str, tuple[float, str]] = {}


def main():
    # A decision naming an endpoint the panel does not contain is a decision that will never fire,
    # and it fails silently: the endpoint is simply absent from the loop. Both dicts are checked
    # against the registry before anything is written, so a rename or a removal is caught here
    # rather than by someone later wondering why a withdrawal stopped being applied.
    known = set(panel.names())
    for label, d in (("WITHDRAW", WITHDRAW), ("RETHRESHOLD", RETHRESHOLD)):
        unknown = sorted(set(d) - known)
        if unknown:
            raise SystemExit(
                f"{label} names {len(unknown)} endpoint(s) absent from the panel registry: "
                f"{', '.join(unknown)}. The panel is defined by "
                f"{panel.MODES.relative_to(panel.ROOT)}; fix the name or drop the entry.")

    p = M / "binder_modes.json"
    shutil.copy(p, M / "binder_modes.prespecificity.json")
    modes = json.loads(p.read_text())
    rows = []

    for ep, why in WITHDRAW.items():
        if ep not in modes:
            continue
        old = dict(modes[ep])
        modes[ep].update({"deployed": False, "reliable_call": False,
                          "withdrawn_reason": why,
                          "withdrawn_evidence": WITHDRAW_EVIDENCE.get(
                              ep, "results/deployed_specificity_audit.csv")})
        rows.append({"target": ep, "action": "withdrawn",
                     "old_threshold": old.get("threshold"), "new_threshold": None, "reason": why})
        print(f"[{ep}] WITHDRAWN: {why}", flush=True)

    for ep, (thr, why) in RETHRESHOLD.items():
        if ep not in modes:
            continue
        old = modes[ep].get("threshold")
        modes[ep]["threshold"] = thr
        modes[ep]["threshold_basis"] = "random_chemistry_specificity"
        modes[ep]["rethreshold_reason"] = why
        # the high-precision mode must never be looser than the standard one
        s = modes[ep].get("screening_threshold")
        if s is None or s < thr:
            modes[ep]["screening_threshold"] = thr
        modes[ep]["deployed"] = True
        rows.append({"target": ep, "action": "rethresholded", "old_threshold": old,
                     "new_threshold": thr, "reason": why})
        print(f"[{ep}] threshold {old} -> {thr}: {why}", flush=True)

    for ep, v in modes.items():
        v.setdefault("deployed", True)

    p.write_text(json.dumps(modes, indent=2))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "specificity_decisions.csv", index=False)
    live = [k for k, v in modes.items() if v.get("deployed", True)]
    print(f"\ndeployed binder endpoints after these decisions: {len(live)} of {len(modes)}")
    print("withdrawn:", ", ".join(k for k, v in modes.items() if not v.get("deployed", True)))
    print("wrote", p, "and", OUT / "specificity_decisions.csv")


if __name__ == "__main__":
    main()
