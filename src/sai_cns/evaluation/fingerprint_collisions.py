"""Measure how many distinct atomic environments share a bit in the folded fingerprint.

featurize.py described its 1,024-bit Morgan fingerprint as "a collision-free-by-construction numeric
encoding of substructure: bit k always means the same environment for every compound". That is the
opposite of what folding does. Folding is a modulo: RDKit hashes each atomic environment to a large
integer and the fingerprint stores hash % 1024, so any two environments whose hashes are congruent
modulo 1024 land on the same bit and become indistinguishable to the model.

The claim mattered beyond tidiness. The thesis discusses hash collisions as a limitation elsewhere,
so the codebase contradicted itself, and a reader who believed the docstring would conclude that a
bit's importance in a fitted tree names one substructure when it names about fifty.

Method. The environments are counted from the UNFOLDED fingerprint, which is what folding consumes:
GetSparseCountFingerprint returns the raw hashes, and the folded bit is the hash modulo the bit
count. This measures the property directly rather than inferring it from bit occupancy, and needs no
substructure SMILES, which cannot always be generated for an environment carrying an unspecified
double-bond stereocentre.

The sample is drawn from this project's own endpoint tables rather than a public benchmark, because
the quantity of interest is the collision rate on the chemistry these models were fitted to, and it
grows with library size and diversity.

Output: results/tables/fingerprint_collisions.csv

Run:  python src/sai_cns/evaluation/fingerprint_collisions.py
"""
from __future__ import annotations

import collections
import glob
import random
import statistics as st
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import MORGAN_BITS, MORGAN_RADIUS  # noqa: E402

OUT = ROOT / "results" / "tables" / "fingerprint_collisions.csv"
SAMPLE, SEED = 20000, 0


def main() -> None:
    smiles: list[str] = []
    for f in sorted(glob.glob(str(ROOT / "data" / "endpoints" / "*.csv"))):
        smiles += pd.read_csv(f, usecols=["smiles"], low_memory=False).smiles.astype(str).tolist()
    smiles = list(dict.fromkeys(smiles))
    population = len(smiles)
    random.Random(SEED).shuffle(smiles)
    smiles = smiles[:SAMPLE]

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS)   # unfolded
    envs: set[int] = set()
    parsed = 0
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        parsed += 1
        envs.update(gen.GetSparseCountFingerprint(m).GetNonzeroElements().keys())

    per_bit = collections.Counter(h % MORGAN_BITS for h in envs)
    counts = list(per_bit.values())
    rows = [
        ("distinct structures in the endpoint tables", population, ""),
        ("structures sampled", SAMPLE, f"seed {SEED}"),
        ("structures parsed", parsed, "an unparseable structure is skipped"),
        ("fingerprint bits", MORGAN_BITS, f"Morgan radius {MORGAN_RADIUS}"),
        ("distinct atomic environments", len(envs), "counted on the unfolded fingerprint"),
        ("bits occupied", len(per_bit), f"of {MORGAN_BITS}"),
        ("bits carrying more than one environment", sum(1 for c in counts if c > 1),
         "the collision count: a bit here is ambiguous"),
        ("environments per occupied bit, median", int(st.median(counts)), ""),
        ("environments per occupied bit, max", max(counts), ""),
        ("environments per occupied bit, min", min(counts), ""),
    ]
    d = pd.DataFrame(rows, columns=["measure", "value", "note"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)

    print(f"{parsed:,} structures parsed from a population of {population:,}")
    print(f"  {len(envs):,} distinct atomic environments mapped onto {MORGAN_BITS} bits")
    print(f"  {sum(1 for c in counts if c > 1)} of {len(per_bit)} occupied bits carry more than one")
    print(f"  median {int(st.median(counts))} environments per bit, max {max(counts)}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
