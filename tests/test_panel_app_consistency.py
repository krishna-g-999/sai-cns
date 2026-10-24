"""The served panel must be the registered panel, and the graph must only name models that exist.

The panel registry reconciles three views: the endpoint tables, the fitted models, and the registry
itself. It does not see app.py, and that gap let a defect sit undetected. `BINDER_TARGETS` was a
hardcoded list and had drifted from the registry in both directions at once:

  GluA2  had been withdrawn from the panel and was still in the list, so it was still scored, still
         carried an Epilepsy edge in the knowledge graph at weight 0.6, and could therefore drive a
         reported condition from a model the project had judged unfit to deploy.
  Cav3_2 is deployed and validated and was absent from the list, so it was never scored at all.

Neither raised an error. Nothing in the freshness graph or the panel check could see either, because
both compare the registry against the models rather than against the code that serves them. These
tests close that gap from the other side.

Run:  python -m pytest tests/test_panel_app_consistency.py -v
"""
from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
sys.path.insert(0, str(ROOT / "src" / "sai_cns" / "figures"))

import panel  # noqa: E402


class TestServedPanelMatchesRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app
        cls.app = app
        cls.deployed = set(panel.names(deployed=True))
        cls.withdrawn = {e.name for e in panel.withdrawn()}

    def test_binder_targets_are_exactly_the_registry_less_the_regressors(self):
        expected = self.deployed - set(self.app.RECEPTOR_REGRESSORS)
        self.assertEqual(
            set(self.app.BINDER_TARGETS), expected,
            "the served binder list has drifted from the registry; extra entries are scored "
            "without being deployed, missing ones are deployed without being scored")

    def test_no_withdrawn_endpoint_is_scored(self):
        served = set(self.app.BINDER_TARGETS) | set(self.app.TARGET_CLASSIFIERS)
        leaked = sorted(self.withdrawn & served)
        self.assertEqual(leaked, [], f"withdrawn endpoints are still being scored: {leaked}")

    def test_no_withdrawn_endpoint_holds_a_disease_edge(self):
        leaked = sorted(self.withdrawn & set(self.app.KNOWLEDGE_GRAPH))
        self.assertEqual(
            leaked, [],
            f"withdrawn endpoints still claim a mechanism in the knowledge graph: {leaked}. A "
            f"model judged unfit to deploy must not be able to drive a reported condition")

    def test_every_graph_target_has_a_scoring_path(self):
        scorable = (set(self.app.BINDER_TARGETS) | set(self.app.TARGET_CLASSIFIERS)
                    | set(self.app.RECEPTOR_REGRESSORS) | {"NEURO"})
        orphans = sorted(set(self.app.KNOWLEDGE_GRAPH) - scorable)
        self.assertEqual(
            orphans, [],
            f"graph targets with no model behind them: {orphans}. Before the guard in "
            f"target_signal this raised a KeyError in the middle of a live query")

    def test_every_served_binder_has_a_binder_scoring_kind(self):
        wrong = sorted(t for t in self.app.BINDER_TARGETS
                       if self.app.TARGET_KIND.get(t) != "binder")
        self.assertEqual(
            wrong, [],
            f"served binders scored by the wrong branch: {wrong}. The enrichment branch reads a "
            f"base rate these endpoints do not have, and would fall through to a KeyError")


class TestManuscriptCountsMatchThePanel(unittest.TestCase):
    """The manuscript states the panel's shape in prose, and prose does not regenerate.

    A prose pass found the abstract claiming 63 molecular targets where the server serves 54, and
    three places saying the pathway graph holds 52 targets when it holds 51, the graph having lost
    GluA2 when that endpoint was withdrawn from scoring. Neither number is produced by a training
    run, so nothing rebuilt them and nothing complained.

    check_manuscript_numbers.py cannot catch this. It asks whether a computed value appears
    somewhere in the documents, and a bare integer like 51 matches inside 0.951 or 151, so a small
    count is effectively unverifiable by substring search. This asserts the counts directly.
    """

    @classmethod
    def setUpClass(cls):
        import app
        cls.app = app
        # Both manuscripts are checked. The condensed draft is what would go to NAR at 4 to 5
        # printed pages; the full draft is the extended version the technical report draws on. Two
        # documents stating the same counts is two places for one of them to fall behind.
        cls.text = "\n".join(
            (ROOT / "manuscript" / f).read_text(encoding="utf-8")
            for f in ("NAR_WebServer_SAI-CNS_draft.md", "NAR_condensed_draft.md")
            if (ROOT / "manuscript" / f).exists())
        if not cls.text:
            # manuscript/ is not part of this repository (it lives in the private working copy
            # this tool is built from), so there is nothing here for either count to be checked
            # against; the check belongs where the manuscript does.
            raise unittest.SkipTest("manuscript/ is not present in this checkout")

    def test_pathway_graph_target_count(self):
        import re
        n = len(self.app.KNOWLEDGE_GRAPH)
        stated = set(int(m) for m in re.findall(r"the (\d+) targets in the pathway graph", self.text))
        self.assertTrue(stated, "the manuscript no longer states a pathway-graph target count")
        self.assertEqual(
            stated, {n},
            f"the manuscript states {sorted(stated)} pathway-graph targets; the graph holds {n}")

    def test_molecular_target_count(self):
        import re
        served = (set(self.app.TARGET_CLASSIFIERS) | set(self.app.BINDER_TARGETS)
                  | set(self.app.RECEPTOR_REGRESSORS)) - {"BBB"}
        stated = re.search(r"engagement of (\d+) molecular targets", self.text)
        self.assertIsNotNone(stated, "the abstract no longer states a molecular-target count")
        self.assertEqual(
            int(stated.group(1)), len(served),
            f"the abstract claims {stated.group(1)} molecular targets; the server serves "
            f"{len(served)} excluding the barrier model")


class TestPanelCountsReconcile(unittest.TestCase):
    """The panel was being counted six ways, and the counts must add up.

    The interface quoted 47, 52, 55, 63, 70 and 75 in different sections. Each was correct for a
    different question and none was reconciled, which reads as carelessness whatever the arithmetic.
    panel_shape() is now the single source, and these assert the two identities that make its
    figures a partition rather than a list.
    """

    @classmethod
    def setUpClass(cls):
        import app
        cls.sh = app.panel_shape()
        cls.app = app
        if "targets" not in cls.sh:
            # panel_shape() reads results/tables/MODEL_INVENTORY.csv and returns {} rather than
            # raising when it is absent, which is the right behaviour for the running app (callers
            # there already tolerate a partial shape) but leaves nothing here for these identities
            # to check: results/ is not part of this repository.
            raise unittest.SkipTest("results/tables/MODEL_INVENTORY.csv is not present in this checkout")

    def test_quantities_partition_into_targets_exposure_and_other(self):
        s = self.sh
        self.assertEqual(
            s["targets"] + s["exposure"] + s["other"], s["quantities"],
            "the panel breakdown does not sum to the number of distinct predicted quantities")

    def test_estimators_exceed_quantities_only_by_dual_modelled_proteins(self):
        s = self.sh
        self.assertEqual(
            s["quantities"] + len(s["dual_model"]), s["deployed"],
            f"{s['deployed']} deployed estimators cover {s['quantities']} quantities, which is "
            f"reconcilable only if exactly {s['deployed'] - s['quantities']} proteins carry two "
            f"models; {len(s['dual_model'])} do: {s['dual_model']}")

    def test_trained_is_deployed_plus_withdrawn(self):
        s = self.sh
        self.assertEqual(s["deployed"] + s["withdrawn"], s["trained"])

    def test_barrier_model_is_not_counted_as_a_target(self):
        """BBB is exposure. Counting it among the targets is how 54 became 55."""
        served = (set(self.app.TARGET_CLASSIFIERS) | set(self.app.BINDER_TARGETS)
                  | set(self.app.RECEPTOR_REGRESSORS))
        self.assertIn("BBB", served, "the barrier model is no longer served")
        self.assertEqual(
            self.sh["targets"], len(served) - 1,
            "the molecular-target count must exclude the barrier model, which is an exposure term")


class TestReliabilityGateIsStatedConsistently(unittest.TestCase):
    """One gate, one definition, and no script allowed a copy of it.

    reliable_call was written by four scripts and read by two more. The two training stages gated
    sensitivity at 0.60 and the two threshold stages that overwrite them at 0.50, so 0.50 was the
    deployed rule by accident of ordering. Figure 10 drew 0.60 and labelled it the gate, placing
    a7nAChR, KEAP1 and Nav1_6 below a line they clear. app.py listed low-sensitivity endpoints at a
    third cut of 0.65, naming ten where six carried the low-power marker. Every one of those numbers
    was right about what it measured, which is why none of them failed. The gate now has a single
    definition in panel.py and these tests fail if a literal copy of it reappears anywhere.
    """

    GATE_HOLDERS = ["src/sai_cns/models/final_thresholds.py",
                    "src/sai_cns/models/calibrate_background_specificity.py",
                    "src/sai_cns/models/calibrate_binder_thresholds.py",
                    "src/sai_cns/models/train_binders_hybrid.py",
                    "src/sai_cns/models/train_measured_label_holdout.py",
                    "src/sai_cns/figures/fig10_endpoint_selection.py",
                    "app.py"]

    def test_no_script_keeps_its_own_copy_of_the_gate(self):
        """A bare 0.60 or 0.65 beside reliable_call is the defect this class exists to prevent."""
        for rel in self.GATE_HOLDERS:
            src = (ROOT / rel).read_text(encoding="utf-8")
            body = "\n".join(ln for ln in src.splitlines()
                             if not ln.lstrip().startswith("#"))
            for literal in ("0.60", "0.65"):
                self.assertNotIn(
                    literal, body,
                    f"{rel} contains a bare {literal}; the gate belongs to panel.py alone")

    def test_figure_draws_the_gate_it_imports(self):
        import fig10_endpoint_selection as fig
        self.assertEqual((fig.SENS_FLOOR, fig.AUROC_FLOOR),
                         (panel.MIN_SENSITIVITY, panel.MIN_AUROC),
                         "Figure 10 draws floors the deployed gate does not use")

    def test_registry_reliable_call_follows_the_gate(self):
        wrong = [e.name for e in panel.binders(deployed=True)
                 if e.reliable != panel.passes_gate(e.sensitivity, e.auroc)]
        self.assertEqual(wrong, [], "reliable_call disagrees with the gate it is defined by")

    def test_about_page_list_is_the_low_power_set(self):
        """The About page and the marker on a result must be one statement, not two."""
        import app
        listed = {t for t, _why in app.coverage_low()}
        marked = {app.MECH_LABEL.get(e.name, e.name) for e in panel.binders(deployed=True)
                  if app.low_power_target(e.name)}
        self.assertEqual(listed, marked,
                         "the About page names a different set than the low-power marker")

    def test_missing_metrics_do_not_pass_the_gate(self):
        """An unmeasured AUROC used to be read as 1.0, which is a pass by omission."""
        self.assertFalse(panel.passes_gate(0.99, None))
        self.assertFalse(panel.passes_gate(None, 0.99))

    def test_sensitivity_is_measured_where_it_claims_to_be(self):
        """The label said held out while the number was measured over the training compounds."""
        import json
        modes = json.loads((ROOT / "models_rf" / "binder_modes.json").read_text(encoding="utf-8"))
        bad = [k for k, v in modes.items() if v.get("deployed", True)
               and v.get("sensitivity_basis") != "held_out_actives_by_scaffold"]
        self.assertEqual(bad, [], "a deployed endpoint reports sensitivity on an undeclared basis")


class TestReportedTablesAgreeWithRegistry(unittest.TestCase):
    """A table that restates the registry must not disagree with it.

    background_specificity.csv reported a mean sensitivity of 0.8983 and 46 of 47 endpoints reliable
    for as long as the registry said 0.7638 and 41, because the sensitivity correction changed the
    figure computed from the fitted models without changing the models themselves. The freshness
    graph is a timestamp check and could not see it; worse, the artefact cannot be declared against
    the registry at all, because the registry is a co-output of the same threshold sequence that
    writes the table (tools/check_freshness.py:110). A byte-identical copy ships in the submission
    package, so the disagreement was visible to reviewers before it was visible to us.

    Refresh with src/sai_cns/evaluation/refresh_background_specificity.py.
    """

    def _rows(self):
        import csv
        with (ROOT / "results" / "tables" / "background_specificity.csv").open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_sensitivity_and_reliability_match_the_registry(self):
        import json
        if not (ROOT / "results" / "tables" / "background_specificity.csv").exists():
            self.skipTest("results/tables/background_specificity.csv is not present in this checkout")
        modes = json.loads((ROOT / "models_rf" / "binder_modes.json").read_text(encoding="utf-8"))
        wrong = []
        for r in self._rows():
            rec = modes.get(r["target"])
            if rec is None:
                wrong.append(f"{r['target']}: absent from the registry")
                continue
            if abs(float(r["sensitivity_after"]) - float(rec["sensitivity_at_threshold"])) > 1e-9:
                wrong.append(f"{r['target']}: sensitivity {r['sensitivity_after']} "
                             f"vs registry {rec['sensitivity_at_threshold']}")
            if (r["reliable"].strip().lower() == "true") != bool(rec["reliable_call"]):
                wrong.append(f"{r['target']}: reliable {r['reliable']} "
                             f"vs registry {rec['reliable_call']}")
        self.assertEqual(wrong, [], "background_specificity.csv disagrees with the registry")

    def test_shipped_copy_is_not_left_behind(self):
        """The copy a reviewer opens is the one that matters."""
        shipped = ROOT / "submission_package" / "08_VALIDATION_RESULTS" / "background_specificity.csv"
        if not shipped.exists():
            self.skipTest("no submission package in this checkout")
        live = ROOT / "results" / "tables" / "background_specificity.csv"
        self.assertEqual(shipped.read_bytes(), live.read_bytes(),
                         "the shipped table differs from the one the project maintains")


class TestPipelineDiagramMatchesThePipeline(unittest.TestCase):
    """The 3.1.1 diagram described a pipeline nobody built.

    It showed one activity cut for both stages, pChEMBL >= 7, where the table builder admits at 6.0
    and only the binder trainer asks for 7.0, and it placed the binder trainer's Tanimoto-0.35 decoy
    rule inside the table-building stage, which contains no decoy logic at all. It also stated the
    reliability gate as 0.60, the floor used by the training stages, where the deployed gate is the
    0.50 written last. A reader applying the diagram would have expected a training table about forty
    per cent smaller than the one that exists.

    A diagram is prose, so no numeric check reached it. These tests do.
    """

    @classmethod
    def setUpClass(cls):
        # The technical-report generator is an internal authoring script and is excluded from the
        # published repository, so on a fresh clone and in CI it is simply absent. Skip rather than
        # error there, the same way the integration suite skips when models_rf/ holds no binaries:
        # a missing authoring tool is the normal state of a published checkout, not a defect.
        gen_path = ROOT / "src" / "sai_cns" / "analysis" / "build_technical_report.py"
        if not gen_path.exists():
            raise unittest.SkipTest(
                "src/sai_cns/analysis/build_technical_report.py absent; "
                "the diagram check runs only in the working repository")
        cls.gen = gen_path.read_text(encoding="utf-8")
        cls.s1 = cls.gen.split("Stage 1:")[1].split("Stage 2:")[0]
        # Stage 2 runs from its heading to the end of its mermaid fence, not to the next section:
        # everything after that fence is unrelated prose that would swamp a negative assertion.
        after = cls.gen.split("Stage 2:")[1]
        cls.s2 = after[:after.index("```", after.index("```mermaid") + 10)]

    def test_stage_one_states_the_table_cut(self):
        self.assertIn("pChEMBL >= 6.0 : active", self.s1)
        self.assertNotIn("pChEMBL >= 7", self.s1,
                         "stage 1 shows the binder trainer's cut, not the table builder's")

    def test_stage_one_has_no_decoy_logic(self):
        """rebuild_endpoints.py contains no Tanimoto, decoy or 0.35 anywhere."""
        src = (ROOT / "src" / "sai_cns" / "data"
               / "rebuild_endpoints.py").read_text(encoding="utf-8").lower()
        for token in ("tanimoto", "decoy", "0.35"):
            self.assertNotIn(token, src, f"rebuild_endpoints.py now mentions {token}")
            self.assertNotIn(token, self.s1.lower(),
                             f"stage 1 shows {token}, which belongs to the binder trainer")

    def test_stage_two_states_the_binder_cut_and_decoy_rule(self):
        active_p = self._const("src/sai_cns/models/train_binders_hybrid.py", "ACTIVE_P")
        self.assertIn(f"pChEMBL >= {active_p}", self.s2)
        self.assertIn("Tanimoto 0.35", self.s2)

    def test_diagram_gate_is_the_deployed_gate(self):
        self.assertIn(f"sensitivity >= {panel.MIN_SENSITIVITY:.2f}", self.s2)
        self.assertIn(f"AUROC vs measured inactives >= {panel.MIN_AUROC:.2f}", self.s2)
        self.assertNotIn("0.60", self.s2, "the diagram states a gate the panel does not use")

    def test_the_band_between_the_cuts_is_reported_correctly(self):
        """The figure that proves the two cuts differ must match the tables."""
        import glob
        import pandas as pd
        n = 0
        for f in glob.glob(str(ROOT / "data" / "endpoints" / "*.csv")):
            d = pd.read_csv(f, usecols=lambda c: c in ("pchembl", "label"), low_memory=False)
            if "pchembl" not in d or "label" not in d:
                continue
            p = pd.to_numeric(d["pchembl"], errors="coerce")
            n += int(((pd.to_numeric(d["label"], errors="coerce") == 1) & (p >= 6.0) & (p < 7.0)).sum())
        # The count used to be a literal in the generator, so this asserted against the source.
        # It is now computed by the generator at build time, which is the stronger guarantee: it
        # cannot drift when an endpoint is added, which is exactly how it drifted before. So the
        # assertion moves to the rendered document, which is what a reader actually sees.
        report = ROOT / "docs" / "TECHNICAL_REPORT.md"
        if not report.exists():
            self.skipTest("docs/TECHNICAL_REPORT.md absent; run the generator first")
        self.assertIn(f"{n:,}", report.read_text(encoding="utf-8"),
                      "the stated count of actives between the two cuts is not what the tables hold")

    @staticmethod
    def _const(path: str, name: str) -> str:
        import re
        text = (ROOT / path).read_text(encoding="utf-8")
        m = re.search(rf"^{re.escape(name)}[^=\n]*=\s*([^\n#]+)", text, flags=re.M)
        assert m, f"{name} not assigned in {path}"
        return m.group(1).split(",")[0].strip()


class TestFingerprintIsNotDescribedAsCollisionFree(unittest.TestCase):
    """Folding is a modulo, so the encoding cannot be collision-free, and it is not.

    featurize.py and docs/METHODS.md both asserted the opposite: that bit k always denotes the same
    substructure. The measurement says every one of the 1,024 bits carries more than one atomic
    environment. The claim survived because it was prose in a docstring, which no numeric check
    reads, and because the thesis discusses collisions as a limitation elsewhere, so the two halves
    of the project were never compared.
    """

    FILES = ["src/sai_cns/features/featurize.py", "docs/METHODS.md",
             "submission_package/04_TECHNICAL_REPORT/methods.md"]

    def test_no_document_claims_the_encoding_is_collision_free(self):
        for rel in self.FILES:
            path = ROOT / rel
            if not path.exists():
                # submission_package/ is a local, gitignored copy built by
                # tools/build_submission_package.py, not a tracked source file, so it is absent on
                # a fresh checkout (a clean clone, or CI) exactly the way models_rf/ is: this failed
                # every CI run since the check was added, on every OS, for that reason alone.
                continue
            text = path.read_text(encoding="utf-8").lower()
            for phrase in ("collision-free by construction", "collision-free-by-construction"):
                # The corrected text may quote the old claim to say it was wrong; only an assertion
                # that the encoding *is* collision-free should fail.
                for hit in range(text.count(phrase)):
                    start = text.find(phrase, 0 if hit == 0 else start + 1)
                    window = text[max(0, start - 90):start]
                    self.assertTrue(
                        any(m in window for m in ("not ", "previously", "earlier edition", "reverse")),
                        f"{rel} asserts the fingerprint encoding is collision-free")

    def test_the_measurement_contradicts_the_old_claim(self):
        import pandas as pd
        p = ROOT / "results" / "tables" / "fingerprint_collisions.csv"
        if not p.exists():
            self.skipTest("results/tables/fingerprint_collisions.csv is not present in this checkout")
        d = pd.read_csv(p).set_index("measure")
        bits = int(d.loc["fingerprint bits", "value"])
        colliding = int(d.loc["bits carrying more than one environment", "value"])
        occupied = int(d.loc["bits occupied", "value"])
        self.assertEqual(colliding, occupied,
                         "some occupied bit is unambiguous; the docstring's wording may be revisitable")
        self.assertGreater(int(d.loc["distinct atomic environments", "value"]), bits,
                           "more bits than environments would make collisions avoidable")

    def test_featurize_quotes_the_artefact(self):
        import pandas as pd
        p = ROOT / "results" / "tables" / "fingerprint_collisions.csv"
        if not p.exists():
            self.skipTest("results/tables/fingerprint_collisions.csv is not present in this checkout")
        d = pd.read_csv(p).set_index("measure")
        text = (ROOT / "src" / "sai_cns" / "features" / "featurize.py").read_text(encoding="utf-8")
        for key in ("distinct atomic environments", "environments per occupied bit, median",
                    "environments per occupied bit, max"):
            v = int(d.loc[key, "value"])
            self.assertIn(f"{v:,}" if v >= 1000 else str(v), text,
                          f"featurize.py does not state the measured {key}")


class TestOrphanTargetDegradesToSilence(unittest.TestCase):
    """The guard must return zero rather than raise, whatever the graph says."""

    def test_unknown_target_scores_zero(self):
        import app
        r = {"targets": {}, "receptor_binder": {}}
        self.assertEqual(app.target_signal(r, 0.0, "a_target_that_does_not_exist"), 0.0)


if __name__ == "__main__":
    unittest.main()
