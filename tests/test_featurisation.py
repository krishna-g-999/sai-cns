"""The featuriser: the one component every model depends on.

If featurisation changes, every stored estimator becomes wrong without anything failing, because a
pickled forest will happily consume a vector of the right width and the wrong meaning. These tests
pin the properties the rest of the pipeline assumes:

  shape        1,036 columns, the same for every molecule, forever
  purity       a molecule's vector depends on that molecule and nothing else, not on what else is
               in the batch and not on input order
  parenting    salts are stripped and the largest organic fragment is used, so a salt and its free
               base give the same vector, which is why deduplication has to happen in feature space
  stereo       chirality is excluded, so enantiomers collide. This is a limitation, not an accident,
               and a test that pins it will fail loudly if someone "fixes" it without also changing
               the deduplication step that exists because of it
  determinism  the same input gives bit-identical output across calls

Run:  python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
from features.featurize import (MORGAN_BITS, N_FEATURES, featurize,  # noqa: E402
                                featurize_one, feature_names)

DONEPEZIL = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_SODIUM = "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"
L_ALANINE = "C[C@@H](N)C(=O)O"
D_ALANINE = "C[C@H](N)C(=O)O"


class TestShape(unittest.TestCase):
    def test_width_is_fixed_and_matches_the_declared_constant(self):
        v = featurize_one(DONEPEZIL)
        self.assertIsNotNone(v)
        self.assertEqual(v.shape, (N_FEATURES,))
        self.assertEqual(N_FEATURES, MORGAN_BITS + 12)

    def test_feature_names_match_the_vector_width(self):
        self.assertEqual(len(feature_names()), N_FEATURES)

    def test_fingerprint_block_is_binary(self):
        v = featurize_one(DONEPEZIL)
        bits = v[:MORGAN_BITS]
        self.assertTrue(set(np.unique(bits)).issubset({0.0, 1.0}),
                        "fingerprint block must be binary; a count fingerprint would silently "
                        "change what every stored model consumes")

    def test_unparseable_input_returns_none_rather_than_a_zero_vector(self):
        # A zero vector is a valid input to a forest and would be scored as though it were a
        # molecule. Returning None forces the caller to handle the failure.
        self.assertIsNone(featurize_one("not a molecule"))
        self.assertIsNone(featurize_one(""))


class TestPurity(unittest.TestCase):
    """A molecule's vector must not depend on anything except that molecule."""

    def test_alone_equals_in_batch(self):
        batch, mask = featurize([DONEPEZIL, ASPIRIN, L_ALANINE])
        self.assertTrue(mask.all())
        for i, smi in enumerate([DONEPEZIL, ASPIRIN, L_ALANINE]):
            np.testing.assert_array_equal(batch[i], featurize_one(smi))

    def test_order_does_not_matter(self):
        smis = [DONEPEZIL, ASPIRIN, L_ALANINE]
        fwd, _ = featurize(smis)
        rev, _ = featurize(list(reversed(smis)))
        for i in range(len(smis)):
            np.testing.assert_array_equal(fwd[i], rev[len(smis) - 1 - i])

    def test_repeated_calls_are_bit_identical(self):
        a, b = featurize_one(DONEPEZIL), featurize_one(DONEPEZIL)
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_mask_marks_failures_and_rows_stay_aligned(self):
        X, mask = featurize([DONEPEZIL, "garbage", ASPIRIN])
        self.assertEqual(list(mask), [True, False, True])
        self.assertEqual(len(X), 2, "failed rows must be absent, not zero-filled")
        np.testing.assert_array_equal(X[1], featurize_one(ASPIRIN))


class TestParentAndStereo(unittest.TestCase):
    def test_counter_ion_is_removed(self):
        """The sodium goes; what remains is the largest organic fragment."""
        from features.featurize import parent_mol
        from rdkit import Chem
        parent = Chem.MolToSmiles(parent_mol(ASPIRIN_SODIUM))
        self.assertNotIn("Na", parent)
        self.assertEqual(parent.count("."), 0, "the counter-ion should be gone")

    def test_a_salt_and_its_free_base_are_the_same_input(self):
        """The same drug written two ways must score identically.

        This was a defect, pinned here for a time as one. Stripping the counter-ion without
        neutralising the parent left aspirin sodium as the carboxyl*ate*, and haloperidol
        hydrochloride as a protonated amine. Measured on the models of the day, that moved
        haloperidol's BBB probability from 0.993 to 0.613 and its hERG probability from 0.914 to
        0.295: a user who pasted the salt form, which is what public databases serve, silently lost
        a cardiac liability flag on a compound that has one.

        The panel was retrained on the neutralised representation, so training and inference now
        agree. If this test fails, salt inputs have diverged from their free bases again.
        """
        for name, free, salt in (
                ("aspirin", ASPIRIN, ASPIRIN_SODIUM),
                ("haloperidol", "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1",
                                "OC1(CC[NH+](CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1.[Cl-]"),
                ("diclofenac", "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
                               "O=C([O-])Cc1ccccc1Nc1c(Cl)cccc1Cl.[Na+]")):
            with self.subTest(compound=name):
                np.testing.assert_array_equal(featurize_one(free), featurize_one(salt))

    def test_a_permanent_charge_survives_neutralisation(self):
        """Neutralisation must move protons, not erase real chemistry.

        A quaternary ammonium has no proton to lose. Its charge is the reason such compounds do not
        cross the barrier, so neutralising it would teach the model the opposite. The uncharger is
        only allowed to undo protonation states.
        """
        from features.featurize import parent_mol
        from rdkit import Chem
        for name, smi, want in (("choline", "C[N+](C)(C)CCO.[Cl-]", 1),
                                ("neostigmine", "CN(C)C(=O)Oc1cccc([N+](C)(C)C)c1", 1),
                                ("free amine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1", 0)):
            with self.subTest(compound=name):
                self.assertEqual(Chem.GetFormalCharge(parent_mol(smi)), want)

    def test_enantiomers_collide_which_is_a_known_limitation(self):
        # Pinned deliberately. If chirality is ever included, _dedup_features must be revisited,
        # because its whole purpose is to collapse rows this collision creates.
        np.testing.assert_array_equal(featurize_one(L_ALANINE), featurize_one(D_ALANINE))


if __name__ == "__main__":
    unittest.main(verbosity=2)
