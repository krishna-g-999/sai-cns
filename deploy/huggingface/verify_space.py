"""Does an assembled Space run on its own, or does it lean on the repository it was built from?

Assembling a directory and reading the file list proves nothing. The failure this exists to catch is
a path that resolves during local testing because the repository happens to be importable, and then
fails on the Space where it is not. A missing data file behaves the same way: silently absent, and
discovered by the first visitor rather than by us.

The check therefore runs with the Space as the working directory and the repository's own source
roots removed from sys.path, then exercises what a first query touches: every model loaded, a
prediction made, the applicability domain assessed, and the novelty-strata table read, since the
interface quotes an expected recall from it and would drop that row without comment if it were
absent.

The repository's virtualenv is deliberately left on the path. site-packages lives inside the
repository directory, so removing everything under it would fail the import for a reason that has
nothing to do with whether the Space is self-contained.

Run:  python deploy/huggingface/verify_space.py ../sai-cns
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROBES = [("donepezil", "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"),
          ("atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"),
          ("withanolide A", "CC1=C(C)C(=O)OC(C1)C(C)C1CCC2(C)C3CCC4CC(O)CCC4(C)C3CCC12C")]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-1])
        return 2
    space = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    if not (space / "app.py").exists():
        print(f"no app.py in {space}; is that the assembled Space?")
        return 2

    os.chdir(space)
    ban = {repo, repo / "src", repo / "src" / "sai_cns"}
    sys.path = [p for p in sys.path if not p or Path(p).resolve() not in ban]
    sys.path.insert(0, str(space))
    sys.path.insert(0, str(space / "src" / "sai_cns"))
    print(f"working directory : {space}")
    print("the repository's source roots are not on the path\n")

    import app as A

    if not Path(A.__file__).resolve().is_relative_to(space):
        print(f"FAIL: imported the repository's app, not the Space's: {A.__file__}")
        return 1

    models = A.load_models()
    print(f"models loaded     : {len(models)}")
    print(f"binder targets    : {len(A.BINDER_TARGETS)} (derived from the registry)")

    failures = []
    for name, smi in PROBES:
        res = A.predict_all(smi, models)
        if res is None:
            failures.append(f"{name}: no prediction returned")
            continue
        bbb, _neuro, dz = A.disease_scores(res)
        ad = A.assess_domain(smi)
        if ad is None:
            failures.append(f"{name}: the applicability-domain reference did not load")
            continue
        er = ad.get("expected_recall")
        top = dz[0] if dz else {"disease": "-", "gated": 0.0}
        print(f"  {name:14s} BBB {bbb:.3f}  top {top['disease']} {top['gated']:.3f}  "
              f"AD {ad['max_sim']:.3f}  expected recall "
              f"{er['recall'] if er else 'UNAVAILABLE'}")
        if er is None:
            failures.append(f"{name}: results/tables/external_novelty_strata.csv did not travel, "
                            f"so the interface would drop the expected-recall row in silence")

    # The same failure shape as expected_recall above, and the reason this check exists:
    # family_cofire() returns {} when inversion/results/H8_family_correlation.csv does not travel,
    # and the co-firing badge then disappears from every result page without raising anything. The
    # directory was outside the shipping list until the interface began reading it.
    cofire = A.family_cofire()
    supported = sorted(f for f, (phi, _note) in cofire.items() if phi)
    print(f"  co-firing        {len(supported)} of {len(cofire)} families have a supported pair "
          f"{supported}")
    if not cofire:
        failures.append("inversion/results/H8_family_correlation.csv did not travel, so the "
                        "co-firing badge would vanish from result pages in silence")
    elif not supported:
        failures.append("the co-firing artefact travelled but no family has a pair meeting the "
                        "support floor, so every badge would be suppressed")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("the Space is self-contained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
