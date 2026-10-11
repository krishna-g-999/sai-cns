"""What does every DEPLOYED binder model do when shown chemistry it has no business recognising?

Two gaps in the existing evidence made this necessary.

The background-specificity calibration bounds the false-positive rate on a random sample of the
measured training library. That library is drug-like medicinal chemistry, so it is a weak test: a
model can pass it and still fire on a sugar. The inversion test H7 did use genuinely random PubChem
structures, but it scored the scaffold hold-out twins rather than the models the web server actually
serves, so its false-positive rates describe a model no user ever meets.

This closes both. Every deployed binder model is scored at its deployed threshold against the 600
random PubChem structures drawn for H4, and against a small set of molecules that no CNS target
plausibly binds: glucose, urea, acetate, ethanol and similar. A model calling urea a sodium-channel
binder is not a marginal calibration issue. It is a false-positive generator inside a tool whose
stated purpose is to stop laboratories spending money on compounds that will not work.

The trivial-control list is deliberately not chemistry any of these models was trained on, and is
small enough to be read individually rather than summarised away.

Read-only. Writes results/deployed_specificity_audit.csv
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import featurize  # noqa: E402
import panel  # noqa: E402

M = ROOT / "models_rf"
OUT = ROOT / "results"
MAX_RANDOM_FPR = 0.05     # the level the calibration intends, now measured on random chemistry

# Molecules no CNS target in this panel plausibly binds. Metabolites, solvents and a peripheral
# drug. If one of these is called a binder, the model is not discriminating, it is guessing.
TRIVIAL = {
    "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "urea": "NC(N)=O",
    "acetate": "CC(=O)O",
    "ethanol": "CCO",
    "glycine": "NCC(=O)O",
    "lactate": "CC(O)C(=O)O",
    "atenolol": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "metformin": "CN(C)C(=N)NC(N)=N",
}


def main():
    modes = json.loads((M / "binder_modes.json").read_text())
    bg = pd.read_csv(ROOT / "inversion" / "results" / "H4_distant_predictions.csv").smiles.tolist()
    Xbg, _ = featurize(bg)
    names = list(TRIVIAL)
    Xt, tm = featurize([TRIVIAL[n] for n in names])
    names = [n for n, k in zip(names, tm) if k]
    print(f"random PubChem structures: {Xbg.shape[0]}; trivial controls: {len(names)}", flush=True)

    rows = []
    for ep, v in sorted(modes.items()):
        p = M / f"{ep}_binder.joblib"
        if not p.exists():
            continue
        # Withdrawn endpoints ARE measured. This block used to `continue` here while its comment
        # claimed the opposite, so the reasons a withdrawal was made could never be rechecked. That
        # matters after a retrain: a withdrawal reason cites numbers, and a refitted model may no
        # longer produce them. They are scored and written out like any other endpoint, marked
        # `deployed=False`, and excluded from the deployed-panel summary so a withdrawn failure is
        # never counted as a failure of what the server offers.
        is_deployed = bool(v.get("deployed", True))
        thr = float(v.get("threshold", 0.40))
        mdl = joblib.load(p)
        pb = mdl.predict_proba(Xbg)[:, 1]
        pt = mdl.predict_proba(Xt)[:, 1]
        fired = [n for n, q in zip(names, pt) if q >= thr]
        fpr = float((pb >= thr).mean())
        rows.append({
            "target": ep, "deployed": is_deployed, "threshold": round(thr, 4),
            "random_fpr": round(fpr, 4),
            "n_trivial_fired": len(fired),
            "trivial_fired": ", ".join(fired),
            "max_trivial_probability": round(float(pt.max()), 4),
            # the threshold that would be needed to hold the intended rate on random chemistry
            "threshold_for_5pct": round(float(np.quantile(pb, 1 - MAX_RANDOM_FPR)), 4),
            "verdict": ("FAILS: fires on trivial molecules" if fired else
                        "FAILS: above 5 percent on random chemistry" if fpr > MAX_RANDOM_FPR
                        else "ok"),
        })
        flag = "" if not rows[-1]["verdict"].startswith("FAILS") else "  <-- " + rows[-1]["verdict"]
        mark = "" if is_deployed else "  (withdrawn)"
        print(f"  {ep:11} thr {thr:6.3f}  random FPR {fpr:6.3f}  trivial fired "
              f"{len(fired)}{flag}{mark}", flush=True)

    df = pd.DataFrame(rows).sort_values("random_fpr", ascending=False)
    df.to_csv(OUT / "deployed_specificity_audit.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 44)
    live = df[df.deployed]
    bad = live[live.verdict != "ok"]
    print()
    print(df.head(12).drop(columns=["trivial_fired"]).to_string(index=False))
    print(f"\nmedian random-chemistry FPR across the deployed panel: {live.random_fpr.median():.4f}")
    # Withdrawn endpoints are reported against their recorded reason, so a reason that a refit has
    # made untrue is visible rather than inherited.
    with_ = df[~df.deployed]
    if len(with_):
        print(f"\n{len(with_)} withdrawn endpoints, re-measured on the current fits:")
        for _, r in with_.iterrows():
            print(f"   {r.target:11} FPR {r.random_fpr:.3f} at threshold {r.threshold:.3f}; "
                  f"trivial fired {r.n_trivial_fired}; verdict {r.verdict}")
    if len(bad):
        print(f"\n{len(bad)} of {len(live)} deployed endpoints fail:")
        for _, r in bad.iterrows():
            extra = f" [{r.trivial_fired}]" if r.trivial_fired else ""
            print(f"   {r.target:11} FPR {r.random_fpr:.3f} at threshold {r.threshold:.3f}; "
                  f"{r.verdict}{extra}")
            print(f"{'':15}holding 5 percent on random chemistry would need "
                  f"{r.threshold_for_5pct:.4f}")
    else:
        print("\nEvery deployed endpoint holds under 5 percent on random chemistry and none fires "
              "on a trivial molecule.")
    print("\nwrote", OUT / "deployed_specificity_audit.csv")


if __name__ == "__main__":
    main()
