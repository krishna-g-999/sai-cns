"""Invariants the scientific results depend on, each of which has been violated at least once.

These are not tests of "does the code run". Each one pins a property whose loss would change a
published number without raising an error, and most correspond to a defect the audit found:

  deduplication   collapses rows identical in feature space, drops contradictory groups rather than
                  voting on them, and does so BEFORE any split. Losing this reintroduces the leak
                  where a stereoisomer of a test compound sits in training.
  pools           the three background pools are disjoint and assignment is a pure function of the
                  structure, so a threshold set on one pool and measured on another stays honest
                  across runs and machines.
  label rule      a censored bound settles a label only when the whole interval lies on one side of
                  the cut. Feeding a bound to the exact-value rule silently discarded 253 measured
                  non-binders for AChE alone.
  determinism     the declared seed actually fixes the result.

Run:  python -m pytest tests/ -v
"""
from __future__ import annotations

import importlib
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
sys.path.insert(0, str(ROOT / "src" / "sai_cns" / "data"))
from data.activity_labels import (ACTIVE_CUT, INACTIVE_CUT,                # noqa: E402
                                  bound_settles_inactive, label_from)
from features.featurize import N_FEATURES                                  # noqa: E402
from models.pools import SHARES, _band, role_of                            # noqa: E402
from models.train_rf import SEED, _dedup_features                          # noqa: E402


def _rows(patterns):
    """Build a feature matrix from short bit patterns, padded to the real width."""
    X = np.zeros((len(patterns), N_FEATURES), dtype=np.float32)
    for i, p in enumerate(patterns):
        X[i, :len(p)] = p
    return X


class TestDeduplication(unittest.TestCase):
    def test_identical_rows_collapse_to_one(self):
        X = _rows([[1, 0, 1], [1, 0, 1], [0, 1, 0]])
        y = np.array([1, 1, 0])
        g = np.array(["a", "a", "b"])
        Xd, yd, gd, sd, rep = _dedup_features(X, y, g, ["s1", "s2", "s3"], "classification")
        self.assertEqual(len(Xd), 2)
        self.assertEqual(rep["duplicate_rows_removed"], 1)
        self.assertEqual(rep["rows_in"], 3)

    def test_contradictory_duplicates_are_dropped_not_voted(self):
        """Two identical inputs with opposite labels cannot both be learned from.

        Voting would pick one and present an arbitrary choice as data. Dropping is the honest
        response, and the count is reported so the loss is visible.
        """
        X = _rows([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
        y = np.array([1, 0, 1])
        g = np.array(["a", "a", "b"])
        Xd, yd, gd, sd, rep = _dedup_features(X, y, g, ["s1", "s2", "s3"], "classification")
        self.assertEqual(rep["conflicting_groups_dropped"], 1)
        self.assertEqual(len(Xd), 1, "the contradictory pair must be gone, not resolved")
        self.assertEqual(int(yd[0]), 1)

    def test_regression_takes_the_group_median(self):
        X = _rows([[1, 0], [1, 0], [1, 0], [0, 1]])
        y = np.array([5.0, 7.0, 9.0, 2.0])
        g = np.array(["a", "a", "a", "b"])
        Xd, yd, gd, sd, rep = _dedup_features(X, y, g, ["a", "b", "c", "d"], "regression")
        self.assertEqual(len(Xd), 2)
        self.assertIn(7.0, list(yd), "the median of 5, 7, 9 is 7")

    def test_nothing_is_lost_when_there_are_no_duplicates(self):
        X = _rows([[1, 0], [0, 1], [1, 1]])
        y = np.array([1, 0, 1])
        g = np.array(["a", "b", "c"])
        Xd, _, _, _, rep = _dedup_features(X, y, g, ["a", "b", "c"], "classification")
        self.assertEqual(len(Xd), 3)
        self.assertEqual(rep["duplicate_rows_removed"], 0)


class TestBackgroundPools(unittest.TestCase):
    """The partition that keeps a threshold honest."""

    SMILES = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCCCC", "CN1CCC1",
              "OCC1OC(O)C(O)C(O)C1O", "CC(C)Cc1ccc(cc1)C(C)C(O)=O"]

    def test_assignment_is_a_pure_function_of_the_structure(self):
        first = [role_of(s) for s in self.SMILES]
        second = [role_of(s) for s in reversed(self.SMILES)][::-1]
        self.assertEqual(first, second, "pool must not depend on order or call history")

    def test_a_compound_belongs_to_exactly_one_pool(self):
        for s in self.SMILES:
            self.assertIn(role_of(s), SHARES)

    def test_bands_cover_exactly_one_hundred(self):
        self.assertEqual(sum(SHARES.values()), 100)

    def test_hash_band_is_stable_and_in_range(self):
        for s in self.SMILES:
            b = _band(s)
            self.assertEqual(b, _band(s))
            self.assertTrue(0 <= b < 100)

    def test_the_partition_is_not_degenerate(self):
        # A hash that sent everything to one pool would satisfy every test above.
        rng = np.random.default_rng(SEED)
        smis = ["C" * (1 + int(rng.integers(1, 25))) for _ in range(600)]
        roles = {r: 0 for r in SHARES}
        for s in smis:
            roles[role_of(s)] += 1
        for role, n in roles.items():
            self.assertGreater(n, 0, f"no compound landed in the {role} pool")

    def test_the_salt_is_the_one_every_threshold_was_measured_against(self):
        """The salt is data, not branding. Renaming it silently invalidates the thresholds.

        _band hashes "{salt}:{smiles}", so the salt decides which pool every background compound
        lands in: which compounds supply decoys, which set each threshold, and which measure the
        false-positive rate that threshold is reported against. Those three pools must stay
        disjoint and must stay fixed, because "thresholds measured on a pool disjoint from the one
        that set them" is a claim about THESE pools.

        Change the salt and the code still runs, the partition is still disjoint, and every number
        in final_thresholds.csv and every background false-positive rate quietly stops describing
        the data they were computed from. Nothing else in this suite pins the value: when the salt
        was mutated during an audit, the only failure came from the degeneracy test above, and that
        was luck, not coverage.

        The string contains the project's former name. It is deliberately NOT renamed with the rest
        of the project: a repo-wide find-and-replace of the old name is exactly the change this
        test exists to stop.
        """
        import inspect

        from models.pools import _band
        salt = inspect.signature(_band).parameters["salt"].default
        self.assertEqual(
            salt, "brainsafe-background-v1",
            "the background-pool salt has changed. Every deployed threshold and every background "
            "false-positive rate was measured against the pools this exact string produces. If "
            "this was a rename, revert it. If the pools are genuinely being rebuilt, re-derive "
            "every threshold and every rate, then update this test in the same commit.")

    def test_a_different_salt_would_move_compounds_between_pools(self):
        """Demonstrates the cost, so the test above reads as a real constraint rather than fussiness."""
        import hashlib

        def band(smiles: str, salt: str) -> int:
            digest = hashlib.blake2b(f"{salt}:{smiles}".encode("utf-8"), digest_size=8).digest()
            return int.from_bytes(digest, "big") % 100

        def role_for(b: int) -> str:
            # the same cumulative banding role_of applies, derived from SHARES rather than repeated
            edge = 0
            for name, share in SHARES.items():
                edge += share
                if b < edge:
                    return name
            return list(SHARES)[-1]

        rng = np.random.default_rng(SEED)
        smis = ["C" * (1 + int(rng.integers(1, 25))) for _ in range(400)]
        moved = sum(role_of(s) != role_for(band(s, "some-other-salt")) for s in smis)
        self.assertGreater(moved, len(smis) // 10,
                           "a changed salt should reshuffle a large share of the background library")


class TestCensoredLabelRule(unittest.TestCase):
    """A bound settles a label only when the whole interval lies on one side of the cut.

    This exercises the shipped rule in data/activity_labels.py. An earlier version of this class
    re-implemented the rule as a local `settles()` helper and asserted against that, so it would
    have passed unchanged had the production rule broken; the property under test is the rule the
    fetchers actually call, not a copy of it that happens to live beside the assertions.

    The defect this pins lost 253 measured non-binders for AChE by passing a bound to the
    exact-value rule, which treats it as a potency and discards the ambiguous 5 to 6 band.
    """

    def test_a_weak_bound_settles_the_compound_as_inactive(self):
        # "IC50 > 10 uM" is pChEMBL 5.0: everything below, so unambiguously inactive
        self.assertEqual(label_from(5.0, ">"), 0)
        self.assertEqual(label_from(4.2, ">"), 0)

    def test_a_bound_spanning_both_classes_is_undecidable(self):
        # "IC50 > 100 nM" is pChEMBL 7.0: the true value could be active or inactive
        self.assertIsNone(label_from(7.0, ">"))
        self.assertIsNone(label_from(5.5, ">"))

    def test_the_boundary_case_is_included(self):
        self.assertEqual(label_from(INACTIVE_CUT, ">"), 0,
                         "a bound exactly at the cut still places the true value below it")

    def test_a_strong_bound_settles_the_compound_as_active(self):
        self.assertEqual(label_from(ACTIVE_CUT, "<"), 1)
        self.assertEqual(label_from(9.0, "<"), 1)
        self.assertIsNone(label_from(5.5, "<"), "a '<' bound below the active cut settles nothing")

    def test_an_exact_value_in_the_ambiguous_band_is_discarded(self):
        # Literal values, not the constants. Asserting label_from(ACTIVE_CUT, "=") == 1 is true
        # for any cut whatsoever and so pins nothing; these pin the band to 5-6 specifically.
        self.assertEqual(label_from(6.0, "="), 1)
        self.assertEqual(label_from(6.1, "="), 1)
        self.assertIsNone(label_from(5.9, "="), "just below the active cut is ambiguous")
        self.assertIsNone(label_from(5.5, "="))
        self.assertIsNone(label_from(5.1, "="), "just above the inactive cut is ambiguous")
        self.assertEqual(label_from(5.0, "="), 0)
        self.assertEqual(label_from(4.9, "="), 0)

    def test_the_cuts_are_the_documented_values(self):
        """The two cuts are load-bearing, so they are pinned to literals here.

        Every label in data/endpoints was produced against 6.0 and 5.0, and the manuscript
        describes the 5 to 6 band as the ambiguous one. Moving either constant silently
        re-labels the training data without re-fitting anything.
        """
        self.assertEqual(ACTIVE_CUT, 6.0)
        self.assertEqual(INACTIVE_CUT, 5.0)

    def test_inclusive_bounds_are_read_as_bounds_not_as_potencies(self):
        """The divergence that made a single source of truth worth having.

        One fetcher matched only a bare '>' and '<', so a '>=' fell through to the exact-value
        branch and a bound at pChEMBL 7 was labelled ACTIVE. ChEMBL's standard_relation emits both
        forms, so this was a live trap on any re-fetch.
        """
        self.assertIsNone(label_from(7.0, ">="))
        self.assertEqual(label_from(4.0, ">="), 0)
        self.assertIsNone(label_from(5.5, "<="))
        self.assertEqual(label_from(7.0, "<="), 1)

    def test_the_relation_is_normalised_before_it_is_read(self):
        """pChEMBL 7.0 is the value that tells the two branches apart.

        As a '>' bound it settles nothing, because the true potency lies somewhere below 7 and
        could be either class. Read as an exact potency it is comfortably ACTIVE. A padded or
        tab-terminated symbol that is not normalised falls through to the exact branch and turns
        an undecidable bound into a confident active, so testing normalisation at a value like
        4.0, which is inactive under either reading, would prove nothing.
        """
        for rel in (" > ", ">\t", " >", ">"):
            with self.subTest(relation=rel):
                self.assertIsNone(label_from(7.0, rel))
        for rel in (None, "", "=", "~"):
            with self.subTest(relation=rel):
                self.assertEqual(label_from(7.0, rel), 1,
                                 "an absent or non-bound relation is read as an exact value")

    def test_a_non_numeric_measurement_cannot_settle_a_class(self):
        nan = float("nan")
        self.assertIsNone(label_from(nan, ">"))
        self.assertIsNone(label_from(nan, "="))
        self.assertFalse(bound_settles_inactive(nan))

    def test_the_two_halves_of_the_rule_agree(self):
        """bound_settles_inactive must stay the '>' branch of label_from, not drift from it."""
        for b in (0.0, 4.2, 4.999, 5.0, 5.001, 5.5, 6.0, 7.0, 12.0, float("nan")):
            with self.subTest(bound=b):
                self.assertEqual(bound_settles_inactive(b), label_from(b, ">") == 0)


class TestTheLabelRuleHasOneDefinition(unittest.TestCase):
    """The cuts must be declared in exactly one place.

    They were previously re-declared in six fetchers plus this test file. The values agreed, but
    the rules did not, and nothing would have caught them drifting apart.
    """

    def test_no_other_module_declares_the_cuts(self):
        src = ROOT / "src" / "sai_cns"
        pattern = re.compile(r"^\s*(ACTIVE_CUT|INACTIVE_CUT)\s*[:,=]", re.M)
        offenders = []
        for path in src.rglob("*.py"):
            if path.name == "activity_labels.py":
                continue
            text = path.read_text(encoding="utf-8")
            body = "\n".join(l for l in text.splitlines()
                             if not l.lstrip().startswith(("from ", "import ")))
            if pattern.search(body):
                offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [],
                         "these modules declare the activity cuts locally instead of importing "
                         "them from data/activity_labels.py: " + ", ".join(offenders))

    def test_every_fetcher_uses_the_shared_rule_object(self):
        import activity_labels
        for mod_name in ("build_np_endpoints", "fetch_natural_products",
                         "ingest_npass", "survey_np_targets"):
            with self.subTest(module=mod_name):
                mod = importlib.import_module(mod_name)
                self.assertIs(mod.label_from, activity_labels.label_from)


class TestDeterminism(unittest.TestCase):
    def test_declared_seed_is_the_one_the_pipeline_uses(self):
        self.assertEqual(SEED, 42)

    def test_a_forest_with_the_fixed_seed_reproduces_itself(self):
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(SEED)
        X = rng.random((120, 8))
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
        a = RandomForestClassifier(n_estimators=25, random_state=SEED).fit(X, y)
        b = RandomForestClassifier(n_estimators=25, random_state=SEED).fit(X, y)
        np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


class TestPanelRegistryIsConsistent(unittest.TestCase):
    """The registry, the endpoint tables and the fitted models must describe one panel.

    Three views of the panel exist on disk and nothing used to compare them, which is how `make
    train` refitted 39 of 52 binders and reported success. panel.verify() reconciles them; this test
    is what makes the reconciliation binding rather than advisory.
    """

    def setUp(self):
        import panel
        self.panel = panel
        self.v = panel.verify()

    def test_the_three_views_agree(self):
        for kind in ("no_model", "no_table", "unregistered", "no_mode"):
            with self.subTest(disagreement=kind):
                self.assertEqual(self.v[kind], [], f"{kind}: {self.v[kind]}")

    def test_no_model_is_older_than_the_table_it_was_fitted_from(self):
        """A partial retrain is exactly this and nothing else reports it."""
        self.assertEqual(self.v["stale_model"], [],
                         f"these models predate their training tables, so the panel is part-way "
                         f"through a retrain: {self.v['stale_model']}")

    def test_no_endpoint_is_withheld_without_a_recorded_reason(self):
        self.assertEqual(self.v["withdrawn_silent"], [],
                         "an endpoint withheld from users with no reason recorded is an "
                         "undocumented scientific decision")

    def test_every_mode_is_owned_by_exactly_one_trainer(self):
        counts = self.v["by_mode"]
        self.assertEqual(sum(counts.values()), self.v["n_registered"],
                         f"some endpoint uses a mode no trainer claims: {counts} against "
                         f"{self.v['n_registered']} registered")

    def test_assert_covers_rejects_a_list_that_has_fallen_behind(self):
        """The guard itself must fail when it should, or it is decoration."""
        full = self.panel.names()
        with self.assertRaises(AssertionError):
            self.panel.assert_covers(full[:-1], full, "a trainer that lost an endpoint")
        with self.assertRaises(AssertionError):
            self.panel.assert_covers(full + ["NotAnEndpoint"], full, "a trainer with a stale name")
        self.panel.assert_covers(full, full, "the complete list")   # must not raise


class TestEveryEndpointIsRetrainable(unittest.TestCase):
    """`make train` must refit the whole panel, not the part of it that existed when it was written.

    Both binder trainers iterated a hardcoded TARGETS list. The panel grew past both: the measured
    label trainer still named two endpoints while eight used its mode, and the hybrid trainer named
    37 while 44 used its. A full retrain therefore refitted 39 of 52 binders and left 13 carrying
    weights fitted to an older featurisation, with nothing in the output saying so. The lists are
    now derived from the panel registry, and this test is what keeps them derived.
    """

    def setUp(self):
        import importlib.util, json
        self.modes = json.loads((ROOT / "models_rf" / "binder_modes.json").read_text())

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, ROOT / path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        self.hybrid = load("_h", "src/sai_cns/models/train_binders_hybrid.py")
        self.holdout = load("_q", "src/sai_cns/models/train_measured_label_holdout.py")

    def test_every_binder_endpoint_is_claimed_by_exactly_one_trainer(self):
        a, b = set(self.hybrid._targets()), set(self.holdout._targets())
        panel = set(self.modes)
        self.assertEqual(panel - (a | b), set(),
                         "these endpoints are in the panel but no trainer would refit them")
        self.assertEqual(a & b, set(),
                         "these endpoints would be refitted twice, by two different procedures")

    def test_each_trainer_covers_exactly_its_own_mode(self):
        for mod, name in ((self.hybrid, "hybrid"), (self.holdout, "measured-label")):
            want = {ep for ep, rec in self.modes.items() if rec.get("mode") == mod.MODE}
            with self.subTest(trainer=name):
                self.assertEqual(set(mod._targets()), want,
                                 f"{name} trainer has drifted from the endpoints using its mode")


class TestThresholdSequenceIsAtomic(unittest.TestCase):
    """The four threshold scripts must be treated as one unit, by the code and by the checker.

    Running `final_thresholds.py` alone reverts the tightening that
    `calibrate_background_specificity.py` applied and re-deploys endpoints that
    `apply_specificity_decisions.py` withdrew. It has happened twice: once found by the audit, and
    once again while clearing a freshness report that could not be cleared, because the checker
    declared these tables to depend on a file the same sequence rewrites. Both failures are silent.
    """

    def setUp(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import check_freshness
        self.cf = check_freshness

    def test_every_member_of_the_sequence_rebuilds_the_whole_sequence(self):
        members = [n for n, _inputs, cmd in self.cf.GRAPH if cmd == self.cf.THRESHOLD_SEQUENCE]
        self.assertGreaterEqual(len(members), 3,
                                "the threshold tables should share one rebuild command")
        for step in ("final_thresholds.py", "screening_thresholds.py",
                     "apply_specificity_decisions.py", "calibrate_background_specificity.py"):
            self.assertIn(step, self.cf.THRESHOLD_SEQUENCE,
                          f"{step} missing from the sequence; running the rest alone reverts it")

    def test_no_threshold_table_depends_on_a_file_the_sequence_rewrites(self):
        """An output of the sequence must never be an input to another output of it.

        Such an edge can never be satisfied: the later steps rewrite the file after the earlier
        steps wrote the table, so the table is reported stale immediately after a correct run, and
        the natural way to clear that report is to re-run one step alone.
        """
        rewritten = {"models_rf/binder_modes.json"}
        for name, inputs, cmd in self.cf.GRAPH:
            if cmd != self.cf.THRESHOLD_SEQUENCE:
                continue
            for src in inputs:
                self.assertNotIn(
                    src, rewritten,
                    f"{name} depends on {src}, which the same sequence rewrites; this edge is "
                    f"unsatisfiable and drove a re-run of one step in isolation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
