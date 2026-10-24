"""One partition of the background chemistry, so that no two roles can draw the same compound.

The binder pipeline asks the background library for three different things, and until now took all
three from the same place:

  decoy       compounds trained on as presumed-negative, to teach the model what a non-binder looks
              like when measured inactives are scarce
  threshold   compounds whose score distribution sets the decision threshold, as a quantile
  evaluation  compounds used to report the false-positive rate at that threshold

Taking the threshold and the evaluation sample from one pool makes the reported rate an arithmetic
consequence of the quantile rather than a measurement: the threshold is the 95th percentile of the
sample, so the false-positive rate on that same sample is 5 per cent whatever the model does. Taking
decoys from the same pool as either is worse, because the model was explicitly trained to score
those compounds as zero, and it is then congratulated for doing so.

The split is by a stable hash of the structure, not by shuffling. Two properties follow that a
shuffle does not give: the assignment does not depend on the order the library happens to be in, and
adding compounds later leaves every existing assignment untouched, so a threshold set today remains
comparable with a false-positive rate measured next year.

`external_negatives` answers a fourth need. The applicability-domain reference is built from the
training tables, so any sample drawn from it has a nearest neighbour at Tanimoto 1.000 and cannot
support a statement about novel chemistry. The DrugBank structures that are absent from that
reference can.
"""
from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import pandas as pd

# The applicability-domain reference has one guarded accessor; see ad_fingerprint.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "sai_cns"))
import ad_fingerprint as _adfp  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
M = ROOT / "models_rf"
AD_REFERENCE = M / "ad_reference.pkl"
EXTERNAL_DRUGS = ROOT / "data" / "external" / "processed" / "external_drugs.csv"

# Shares out of 100. Decoys need the most, being 3x the actives for every target; the other two need
# only enough for a stable quantile and a stable rate.
SHARES = {"decoy": 60, "threshold": 20, "evaluation": 20}
ROLES = tuple(SHARES)


def _band(smiles: str, salt: str = "brainsafe-background-v1") -> int:
    """Stable 0-99 band for a structure, independent of library order and of insertion time.

    DO NOT RENAME THE SALT. It carries the project's former name and is deliberately left alone.

    The salt is hashed with the structure, so it decides which pool every background compound
    lands in: which compounds supply decoys, which set each decision threshold, and which measure
    the false-positive rate that threshold is reported against. Every value in
    results/tables/final_thresholds.csv, and every background false-positive rate in the
    manuscript, was measured against the pools this exact string produces.

    Change it and nothing breaks loudly. The code runs, the pools stay disjoint, and every
    published threshold quietly stops describing the data it was computed from. A repo-wide
    find-and-replace of the old project name is the specific accident this warning exists to
    prevent; tests/test_pipeline_invariants.py pins the value so that accident fails the build.
    """
    digest = hashlib.blake2b(f"{salt}:{smiles}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 100


def role_of(smiles: str) -> str:
    """Which pool a structure belongs to. Total function: every structure has exactly one role."""
    b = _band(smiles)
    edge = 0
    for role, share in SHARES.items():
        edge += share
        if b < edge:
            return role
    return ROLES[-1]


def background_pools(with_fingerprints: bool = False):
    """Split the applicability-domain reference into the three disjoint role pools.

    Returns {role: [smiles]}, or {role: ([smiles], [fingerprint])} when fingerprints are wanted,
    preserving the pairing the reference file stores.
    """
    if not AD_REFERENCE.exists():
        raise FileNotFoundError(
            f"{AD_REFERENCE.relative_to(ROOT).as_posix()} is missing; build it with "
            "src/sai_cns/build_ad_reference.py before training binders."
        )
    smiles, fps = _adfp.load_reference(AD_REFERENCE)

    pools: dict[str, list] = {r: [] for r in ROLES}
    fp_pools: dict[str, list] = {r: [] for r in ROLES}
    for s, fp in zip(smiles, fps):
        r = role_of(str(s))
        pools[r].append(str(s))
        fp_pools[r].append(fp)
    if with_fingerprints:
        return {r: (pools[r], fp_pools[r]) for r in ROLES}
    return pools


def external_negatives() -> list[str]:
    """Approved and experimental drugs that are absent from the applicability-domain reference.

    Used where a specificity claim has to be about chemistry the model has not seen. A sample drawn
    from the reference itself cannot support one: every compound in it is its own nearest neighbour.
    """
    if not EXTERNAL_DRUGS.exists():
        raise FileNotFoundError(
            f"{EXTERNAL_DRUGS.relative_to(ROOT).as_posix()} is missing; build it with "
            "src/sai_cns/data/integrate_external.py."
        )
    with AD_REFERENCE.open("rb") as fh:
        ref = set(map(str, pickle.load(fh)[0]))
    drugs = pd.read_csv(EXTERNAL_DRUGS)
    col = "canonical_smiles" if "canonical_smiles" in drugs.columns else drugs.columns[0]
    return sorted({s for s in drugs[col].astype(str) if s not in ref})


def scaffold_holdout(groups, frac: float = 0.20, rng=None):
    """Boolean mask selecting whole scaffold groups to withhold from training.

    Whole groups rather than individual compounds, so a withheld active is a structurally different
    compound and not a close analogue of one the model was fitted to. Returns an all-False mask when
    the scaffold structure is degenerate (one group, or every compound its own), because withholding
    everything or nothing are both useless and silently doing either is worse.
    """
    import numpy as np
    rng = rng if rng is not None else np.random.default_rng(42)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return np.zeros(len(groups), dtype=bool)
    held = set(rng.permutation(uniq)[:max(1, int(round(frac * len(uniq))))])
    mask = np.array([g in held for g in groups])
    if mask.all() or not mask.any():
        return np.zeros(len(groups), dtype=bool)
    return mask


def write_holdout(endpoint: str, **sets) -> Path:
    """Persist what was withheld, so a later step cannot reach past it by re-reading the source.

    final_thresholds.py reads this instead of the endpoint table. Without it the threshold is set
    from every measured inactive, including the half that was trained on.
    """
    import json
    d = M / "holdout"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{endpoint}_binder_holdout.json"
    payload = {"endpoint": endpoint}
    payload.update({k: list(v) for k, v in sets.items()})
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def summary() -> pd.DataFrame:
    """Sizes of every pool, for the run log and for the record."""
    pools = background_pools()
    rows = [{"pool": r, "n": len(pools[r]), "share_pct": SHARES[r]} for r in ROLES]
    rows.append({"pool": "external_negatives", "n": len(external_negatives()), "share_pct": None})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(summary().to_string(index=False))
    pools = background_pools()
    overlap = set(pools["decoy"]) & set(pools["threshold"]) | \
              set(pools["decoy"]) & set(pools["evaluation"]) | \
              set(pools["threshold"]) & set(pools["evaluation"])
    print(f"\npairwise overlap between pools: {len(overlap)} (must be 0)")
