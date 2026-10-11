"""Turn a molecular structure into the numeric feature vector the models train on.

Every model input is numeric. A SMILES string is not fed to the model directly; it is converted
into two blocks of numbers:

  1. A 1024-bit ECFP-4 fingerprint (Morgan, radius 2). Each bit is 0/1 and records the presence of
     at least one local atomic environment whose hash falls in that bit.

     Bit k does NOT mean one environment. Folding is a modulo: RDKit hashes each environment to a
     large integer and the fingerprint stores hash % 1024, so environments congruent modulo 1024
     collide and become indistinguishable to the estimator. On 20,000 structures sampled from this
     project's own endpoint tables, 52,882 distinct atomic environments map onto the 1,024 bits, and
     every one of the 1,024 carries more than one: a median of 52 environments per bit and a maximum
     of 73 (results/tables/fingerprint_collisions.csv, from evaluation/fingerprint_collisions.py).

     This docstring previously called the encoding "collision-free-by-construction" and said bit k
     always means the same environment for every compound. That was the reverse of the truth, and it
     contradicted the thesis, which discusses hash collisions as a limitation. The practical
     consequence is for interpretation: a bit ranked important by a fitted tree names a set of
     environments, not a substructure, so SHAP attribution to a single bit cannot be read as
     attribution to a single chemical motif.

     Raising MORGAN_BITS would reduce the collision rate at the cost of a wider, sparser matrix. It
     has never been swept in this project; no hyper-parameter has.
  2. Twelve interpretable physicochemical descriptors (molecular weight, logP, polar surface area,
     H-bond donors/acceptors, rotatable bonds, aromatic rings, sp3 fraction, ring count, heavy-atom
     count, formal charge, drug-likeness QED).

The result is a fixed 1036-length numeric vector with stable, named columns, identical for every
endpoint, so nothing categorical or free-text ever reaches the estimator.

Any categorical metadata that is *not* a model feature (endpoint name, assay source) is encoded
separately and reversibly in encodings.py; it is never mixed into this feature matrix.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
MORGAN_BITS = 1024
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
# Built once: constructing it per call is the dominant cost of featurising a large table.
_UNCHARGER = rdMolStandardize.Uncharger()

# Ordered so the column layout is stable and reproducible across runs.
_DESCRIPTORS = {
    "mw": Descriptors.MolWt,
    "clogp": Crippen.MolLogP,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "hbd": rdMolDescriptors.CalcNumHBD,
    "hba": rdMolDescriptors.CalcNumHBA,
    "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
    "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings,
    "fraction_csp3": rdMolDescriptors.CalcFractionCSP3,
    "ring_count": rdMolDescriptors.CalcNumRings,
    "heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
    "formal_charge": Chem.GetFormalCharge,
    "qed": QED.qed,
}

N_FEATURES = MORGAN_BITS + len(_DESCRIPTORS)


def feature_names() -> list[str]:
    """Stable column names for the full feature vector (fingerprint bits then descriptors)."""
    return [f"ecfp4_{i}" for i in range(MORGAN_BITS)] + list(_DESCRIPTORS)


def parent_mol(smiles: str):
    """The largest fragment of a structure, desalted, neutralised and sanitised.

    Public because the scaffold grouping in training must desalt exactly as this does, or a salt
    and its free base become identical inputs assigned to different cross-validation folds.

    Neutralisation is not cosmetic. Stripping the counter-ion without it left the parent carrying
    the charge the salt gave it, so haloperidol hydrochloride written the way PubChem serves it,
    with a protonated amine and a chloride, scored BBB 0.613 against 0.993 for the free base and
    hERG 0.295 against 0.914. A user who pasted the salt form lost a cardiac liability flag on a
    compound that has one. The two forms are the same drug and must give the same answer.

    What is neutralised is only what a proton can move: a carboxylate becomes an acid, a protonated
    amine becomes an amine. A permanent charge is left alone, because it is real chemistry rather
    than an artefact of how a depositor wrote the structure. Choline and neostigmine keep their
    quaternary nitrogen and their +1, which is exactly the property that stops such compounds
    crossing the barrier; erasing it would teach the model that they should. `formal_charge` stays
    a descriptor and now means "permanently charged" rather than "however this row was written".

    Measured over the training library before it was adopted: of 170,617 unique structures, 198
    (0.12 per cent) change representation and 1,155 charged ones are correctly left untouched. The
    panel was retrained afterwards so that training and inference share this representation.
    """
    # RDKit parses the empty string into an empty molecule rather than returning None, so without
    # this guard featurize_one("") returns a well-formed 1,036-column vector of zeros and the
    # forests score it as though it were a compound. An empty field is not a molecule.
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:  # keep the largest organic fragment (salt / counter-ion removal)
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    # Uncharging can fail on exotic valences. A molecule that cannot be neutralised is still a
    # molecule, so the un-neutralised parent is returned rather than the compound being dropped:
    # the old behaviour is the fallback, never an error.
    try:
        neutral = _UNCHARGER.uncharge(Chem.Mol(mol))
        Chem.SanitizeMol(neutral)
        return neutral
    except Exception:
        return mol


def featurize_one(smiles: str) -> np.ndarray | None:
    """Return the numeric feature vector for one SMILES, or None if it cannot be parsed."""
    mol = parent_mol(smiles)
    if mol is None:
        return None
    fp = _MORGAN.GetFingerprintAsNumPy(mol).astype(np.float32)
    desc = np.array([fn(mol) for fn in _DESCRIPTORS.values()], dtype=np.float32)
    return np.concatenate([fp, desc])


def featurize(smiles_list) -> tuple[np.ndarray, np.ndarray]:
    """Featurize many SMILES.

    Returns (X, mask) where X has one row per *valid* structure and mask is a boolean array over the
    input marking which entries were parsed. The caller uses mask to align labels with rows.
    """
    rows, mask = [], np.zeros(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        v = featurize_one(smi)
        if v is not None:
            rows.append(v)
            mask[i] = True
    X = np.vstack(rows) if rows else np.empty((0, N_FEATURES), dtype=np.float32)
    return X, mask
