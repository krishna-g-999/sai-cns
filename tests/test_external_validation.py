"""Invariants of the external-validation programme.

The prospective validation is only meaningful if it mirrors the deployed training exactly. Its whole
claim is "the same panel, fitted to less time", so any constant that drifts between the trainer and
the validator turns the comparison into a comparison of two different pipelines while continuing to
report a number labelled as the cost of prospectivity. Nothing would raise an error. The reported
figure would simply stop meaning what the report says it means, which is the failure mode this
project keeps finding and the reason this file exists.

Three properties are pinned:

  mirrored constants  the validator's active cut, decoy ratio, similarity ceiling, target
                      false-positive rate and forest configuration are the trainer's.
  no temporal leak    the year split is by earliest recorded year per structure, so a compound
                      measured before and after the cutoff counts as known. Taking the latest year
                      instead would move known chemistry into the test set and inflate every result.
  band coverage       the novelty bands tile [0, 1] without gap or overlap, so no test compound is
                      counted twice or silently dropped from the stratification.

Run:  python -m pytest tests/test_external_validation.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
sys.path.insert(0, str(ROOT / "src" / "sai_cns" / "evaluation"))

from models import train_binders_hybrid as TR                              # noqa: E402
from evaluation import external_prospective as EP                          # noqa: E402
from evaluation import external_novelty_strata as NS                       # noqa: E402


class TestValidatorMirrorsTheTrainer(unittest.TestCase):
    """Every constant that defines a binder model must be the same on both sides."""

    def test_scalar_constants_match(self):
        for name in ("ACTIVE_P", "DECOY_RATIO", "TAN_MAX", "TARGET_FPR"):
            with self.subTest(constant=name):
                self.assertEqual(
                    getattr(EP, name), getattr(TR, name),
                    f"{name} differs between the trainer and the prospective validator, so the "
                    f"'cost of prospectivity' compares two different pipelines")

    def test_forest_configuration_matches(self):
        self.assertEqual(
            EP.RF, TR.RF,
            "the prospective validator no longer fits the forest the deployed panel uses")

    def test_seed_matches(self):
        self.assertEqual(EP.SEED, 42, "the declared seed changed without the report being rebuilt")


class TestTemporalSplitCannotLeak(unittest.TestCase):
    """A compound known before the cutoff must never appear on the test side."""

    def _df(self):
        return pd.DataFrame({
            "smiles": ["A", "A", "B", "C", "D"],
            "year": [2010, 2022, 2022, 2008, None],
        })

    def test_earliest_year_decides(self):
        pre, post = EP._split_by_year(self._df(), ["A", "B", "C"], 2015)
        self.assertIn("A", pre, "a compound first measured in 2010 was moved into the future "
                                "because it was measured again later")
        self.assertIn("B", post)
        self.assertIn("C", pre)

    def test_undated_chemistry_is_treated_as_known(self):
        pre, post = EP._split_by_year(self._df(), ["D"], 2015)
        self.assertEqual((pre, post), (["D"], []),
                         "a compound of unknown date entered the test set, where it cannot be "
                         "shown to postdate the training data")

    def test_split_is_a_partition(self):
        names = ["A", "B", "C", "D"]
        pre, post = EP._split_by_year(self._df(), names, 2015)
        self.assertEqual(sorted(pre + post), sorted(names))
        self.assertEqual(set(pre) & set(post), set(), "a compound landed on both sides of the wall")


class TestNoveltyBands(unittest.TestCase):
    """The bands must tile the similarity range exactly once."""

    def test_bands_are_contiguous_and_cover_the_range(self):
        lows = [lo for lo, _, _ in NS.BINS]
        highs = [hi for _, hi, _ in NS.BINS]
        self.assertEqual(lows[0], 0.0)
        self.assertGreater(highs[-1], 1.0, "a Tanimoto of exactly 1.0 falls outside every band")
        for i in range(len(NS.BINS) - 1):
            self.assertEqual(highs[i], lows[i + 1],
                             f"bands {i} and {i + 1} leave a gap or overlap, so a compound is "
                             f"dropped from the stratification or counted twice")


class TestReportedTablesAgreeWithThemselves(unittest.TestCase):
    """Whatever the numbers are, the table must not contradict its own row counts."""

    @classmethod
    def setUpClass(cls):
        p = ROOT / "results" / "tables" / "external_prospective.csv"
        cls.d = pd.read_csv(p) if p.exists() else None

    def test_assessed_endpoints_have_test_compounds(self):
        if self.d is None:
            self.skipTest("external_prospective.csv not built")
        ok = self.d[self.d.status == "ok"]
        self.assertTrue((ok.time_n_test_actives >= EP.MIN_TEST_ACTIVES).all(),
                        "an endpoint was reported as assessed with fewer test actives than the "
                        "stated minimum")

    def test_random_control_is_size_matched(self):
        if self.d is None:
            self.skipTest("external_prospective.csv not built")
        ok = self.d[(self.d.status == "ok") & self.d.get("random_n_train_actives").notna()]
        if not len(ok):
            self.skipTest("no random control rows")
        # Featurisation can drop a handful of structures on either side; a match to within one per
        # cent is a size match, a larger gap means the control is not controlling for size.
        rel = (ok.random_n_train_actives - ok.time_n_train_actives).abs() / ok.time_n_train_actives
        self.assertTrue((rel <= 0.01).all(),
                        "the random control trains on a materially different number of actives, so "
                        "the difference between the splits is not the temporal wall alone")


if __name__ == "__main__":
    unittest.main()
