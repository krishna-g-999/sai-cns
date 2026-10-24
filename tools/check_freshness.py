"""Is any derived artefact older than something it was derived from?

Every stale-artefact incident in this project has had the same shape. A file is produced from an
input, the input is later regenerated, and nothing notices that the output is now describing
something that no longer exists. The failure is silent by construction: the stale file is internally
consistent, so every content check passes, and every command exits 0.

Content checks cannot catch this. `check_manuscript_numbers.py` compares numbers against the tables,
and it is satisfied when the manuscript agrees with a table that is itself six days out of date.
Freshness is a different property and needs a different check.

This declares the dependency graph explicitly and compares modification times. If an output is older
than any of its inputs it is STALE, the command that rebuilds it is printed, and the exit code is
non-zero so a pipeline or a hook can refuse to proceed.

What it deliberately does NOT do: rebuild anything. Deciding what to regenerate is a scientific
judgement, because rebuilding an artefact changes what published numbers refer to. This reports and
stops.

Staleness is judged on content, not on timestamps. Timestamps alone produce false alarms that train
a reader to ignore the tool: re-running the threshold sequence rewrote binder_modes.json with
byte-identical thresholds, and a timestamp check called nine downstream artefacts stale when nothing
had changed. A check that cries wolf is worse than no check.

Each input therefore gets a fingerprint. Small text artefacts are fingerprinted by SHA-256 of their
content. Large binary trees, where hashing 0.78 GB on every run would make the check too slow to use,
are fingerprinted by file count and total size, which is stated here because it is weaker: a retrain
that produced exactly the same number of files at exactly the same total size would not be caught.

`--accept` records the current fingerprints in tools/freshness_state.json, and means "I have checked
that everything is consistent as of now". Later runs compare against that record, so an artefact is
stale only when an input's fingerprint has actually moved since it was last accepted.

models_manifest.json is verified by recomputing every SHA-256 it records, which is stronger still.

Run:  python tools/check_freshness.py
      python tools/check_freshness.py --accept    record the current state as consistent
      python tools/check_freshness.py --json      machine-readable
      python tools/check_freshness.py --quiet     only the stale ones
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Four steps that all mutate models_rf/binder_modes.json, in the only order that is correct. See the
# ORDERING HAZARD note in GRAPH below.
THRESHOLD_SEQUENCE = (
    "python src/sai_cns/models/final_thresholds.py && "
    "python src/sai_cns/models/screening_thresholds.py && "
    "python src/sai_cns/models/apply_specificity_decisions.py && "
    "python src/sai_cns/models/calibrate_background_specificity.py"
)

# output -> (inputs, command that rebuilds it)
# Inputs may be files, directories or globs. A directory is represented by its newest file, which is
# what "this directory changed" means for these purposes.
GRAPH: list[tuple[str, list[str], str]] = [
    # ---- models are derived from the endpoint tables -----------------------------------------
    ("models_rf/BBB.joblib", ["data/endpoints/*.csv"],
     "python src/sai_cns/models/train_rf.py"),
    ("results/tables/rf_cv_summary.csv", ["models_rf/BBB.joblib"],
     "python src/sai_cns/models/train_rf.py"),
    ("results/tables/rf_cv_folds.csv", ["models_rf/BBB.joblib"],
     "python src/sai_cns/models/train_rf.py"),

    # The label-permutation null for the eight core classifiers. Written after a thesis audit found
    # the technical report asserting mean AUROCs of 0.4938 random and 0.4921 scaffold under permuted
    # labels with those numbers hard-coded in a prose block and no file holding them. It is the
    # evidence that the cross-validated figures are not obtainable without the labels, and that
    # scaffold grouping alone confers nothing, so it belongs in the graph beside the figures it
    # supports. It depends on the endpoint tables, because it refits from them.
    ("results/tables/permutation_null.csv", ["data/endpoints/*.csv"],
     "python src/sai_cns/evaluation/permutation_null.py"),

    # H10, the falsification suite's test of the barrier model itself. Added after a thesis audit
    # found that the component the architecture is named for was the one component no hypothesis
    # tested. It compares the deployed forest against descriptor rules and descriptor-only models on
    # the same Bemis-Murcko folds and on the external approved-drug set, so it depends on the served
    # model and on the stored out-of-fold predictions.
    # The two library-composition figures behind the natural-product limitation. They were measured
    # inside build_technical_report.py and written to no file, so no other document could cite them
    # and nothing checked them. The artefact reproduces the generator's values exactly.
    ("results/tables/library_sp3_coverage.csv", ["data/endpoints/*.csv"],
     "python src/sai_cns/evaluation/library_sp3_coverage.py"),

    # Whether the panel's two different "how many compounds" figures, 170,619 distinct SMILES and
    # 169,341 unique compounds keyed by the InChIKey of the desalted parent, are both real. Neither
    # the thesis nor the manuscript had re-derived the InChIKey figure; three separate outstanding-
    # items entries said so. It reproduces to the digit against the manuscript's stated value.
    ("results/tables/unique_compound_count.csv", ["data/endpoints/*.csv"],
     "python src/sai_cns/evaluation/unique_compound_count.py"),

    # SHAP attribution for the barrier model and hERG, quoted in the manuscript's Attribution
    # paragraph. No script computed these values and no table held them; the technical report has no
    # SHAP section at all. Depends on the served model, not only the table, because TreeExplainer
    # reads the fitted forest.
    ("results/tables/shap_attribution.csv", ["models_rf/BBB.joblib", "models_rf/hERG.joblib"],
     "python src/sai_cns/evaluation/shap_attribution.py"),

    # The full estimator inventory behind Figure 2 (panel atlas) and the manuscript's own summary
    # counts. It read from no declared input, so a retrained panel could leave it silently behind;
    # this is the same gap already closed for model_comparison.csv and background_specificity.csv.
    ("results/tables/MODEL_INVENTORY.csv", ["models_rf/binder_modes.json", "models_rf/*.joblib"],
     "python src/sai_cns/analysis/build_model_inventory.py"),

    # The external approved-drug set and its novelty flags. It was declared only as an INPUT, never
    # as an output, and it went stale in the way that omission allows. novel_to_model is computed by
    # featurising every external compound and asking whether its vector already exists in the BBB
    # training table, so it is a function of the featuriser, not of the data. When parent_mol began
    # neutralising, fourteen compounds became feature-identical to a training row and the stored flag
    # did not notice. The external AUROC was reported as 0.7934 on 241 compounds where the honest
    # figure on the 227 genuinely novel is 0.7666, which is level with the 0.7645 obtained without
    # any novelty filtering at all. The dependency on featurize.py is the one that matters here.
    ("data/external/processed/external_bbb_test.csv",
     ["src/sai_cns/features/featurize.py", "data/endpoints/BBB.csv"],
     "python src/sai_cns/data/integrate_external.py"),

    # The family comparison and the paired test on it. Neither was declared: model_comparison.csv
    # appeared only as an INPUT to the technical report and the significance table was absent
    # altogether, so the table derived from the comparison had no edge recording that it was derived
    # from it. That is the condition that let the technical report go stale, and it applied here to a
    # table whose conclusion, "the forest is not distinguishable from the boosters", is quoted in the
    # thesis, the viva pack and two decks.
    ("results/tables/model_comparison.csv", ["data/endpoints/*.csv"],
     "python src/sai_cns/evaluation/model_comparison.py"),
    ("results/tables/model_family_significance.csv", ["results/tables/model_comparison.csv"],
     "python src/sai_cns/evaluation/model_family_significance.py"),

    # The folded fingerprint's collision rate. featurize.py and docs/METHODS.md both asserted the
    # encoding was collision-free by construction, which is the reverse of what folding does, and
    # neither had a measurement behind it. It depends on the endpoint tables because the rate grows
    # with the size and diversity of the library actually being encoded.
    ("results/tables/fingerprint_collisions.csv", ["data/endpoints/*.csv"],
     "python src/sai_cns/evaluation/fingerprint_collisions.py"),

    ("inversion/results/H10_barrier_necessity.csv",
     ["models_rf/BBB.joblib", "data/processed/cv_predictions/BBB_scaffold_oof.csv",
      "data/external/processed/external_bbb_test.csv"],
     "python inversion/inv_barrier_necessity.py"),

    # ---- thesis figures ------------------------------------------------------------------------
    # Four figures explaining the model itself, drawn in the manuscript house style so the thesis
    # and the paper are one visual system. T1 and T2 score real compounds through app at build time,
    # so they depend on the served models; T3 and T4 read tables.
    ("thesis/figures/FigureT1_scoring_pipeline.png",
     ["models_rf/binder_modes.json", "models_rf/BBB.joblib", "app.py"],
     "python src/sai_cns/figures/thesis_T1_scoring_pipeline.py"),
    ("thesis/figures/FigureT2_enrichment_and_gate.png",
     ["models_rf/binder_modes.json", "models_rf/endpoint_context.json", "app.py"],
     "python src/sai_cns/figures/thesis_T2_enrichment_and_gate.py"),
    ("thesis/figures/FigureT3_uncertainty_stack.png",
     ["results/tables/calibration.csv", "results/tables/rf_conformal.csv",
      "results/tables/integrity_calibration_per_target.csv"],
     "python src/sai_cns/figures/thesis_T3_uncertainty_stack.py"),
    ("thesis/figures/FigureT4_falsification.png",
     ["inversion/results/VERDICTS.csv", "results/tables/permutation_null.csv",
      "inversion/results/H10_barrier_necessity.csv"],
     "python src/sai_cns/figures/thesis_T4_falsification.py"),

    # ---- calibration sits on the core models --------------------------------------------------
    # The generator is a declared input, not just the rebuild command. This table was built on
    # 2026-08-19 and the expected-calibration-error function was corrected on 2026-10-24; with only
    # models_rf/BBB.joblib declared, correcting the metric could not make the table stale, so the
    # README went on quoting a figure computed with the broken metric. The out-of-fold predictions
    # it actually reads are declared too. Any entry here whose generator changed since its output
    # was built has the same blind spot.
    ("results/tables/calibration.csv",
     ["models_rf/BBB.joblib", "src/sai_cns/models/calibrate.py",
      "data/processed/cv_predictions/*_random_oof.csv"],
     "python tools/recompute_calibration_ece.py --write   (calibrate.py also refits the eight "
     "deployment models, which moves their checksums)"),

    # ---- the binder panel ----------------------------------------------------------------------
    ("models_rf/binder_modes.json", ["data/endpoints/*.csv"],
     "python src/sai_cns/models/train_binders_hybrid.py"),
    # ORDERING HAZARD. These four steps all write models_rf/binder_modes.json and each depends on
    # the one before it. Running any of them alone silently reverts the later ones: re-running
    # final_thresholds.py by itself undid the background-specificity tightening for HT1A, HT2A and
    # Nav1_5, and running it without apply_specificity_decisions.py afterwards would re-deploy two
    # endpoints that were withdrawn for firing on glucose and atenolol. The whole sequence is
    # therefore quoted as the rebuild command for every member of it.
    # These three depend on the fitted binder models, NOT on binder_modes.json, even though every
    # step of the sequence reads it. binder_modes.json is a co-output of this same sequence: steps
    # three and four rewrite it after step one has written final_thresholds.csv, so an edge from it
    # to these tables can never be satisfied. A correct, complete run left them reported stale, and
    # re-running to clear the report re-ran step one alone, which is precisely the ordering hazard
    # above. A check that cannot be satisfied by doing the right thing teaches the reader to ignore
    # it. The binder .joblib files carry the same signal without the cycle: they change when the
    # panel is retrained, which is when these thresholds genuinely must be re-derived, and the
    # threshold sequence never writes them.
    ("results/tables/final_thresholds.csv", ["models_rf/*_binder.joblib"], THRESHOLD_SEQUENCE),
    ("results/tables/screening_thresholds.csv", ["models_rf/*_binder.joblib"],
     THRESHOLD_SEQUENCE),
    ("results/tables/background_specificity.csv", ["models_rf/*_binder.joblib"],
     THRESHOLD_SEQUENCE),

    # The background false-positive rate the manuscript actually claims: the CURRENTLY deployed
    # threshold, scored on the evaluation pool, which is disjoint from both the decoy pool a binder
    # trained against and the threshold pool that set any version of its cut. Three other fields in
    # binder_modes.json answer a related but different question: background_fpr_at_threshold is
    # circular (calibrate_background_specificity.py sets its threshold as a quantile of the same
    # sample it then measures on), and background_fpr_held_out is genuinely disjoint but reflects the
    # threshold train_binders_hybrid.py set before final_thresholds.py and
    # calibrate_background_specificity.py were free to raise it further, so it is silent on whatever
    # the server currently runs. This depends on the binder models for scoring and on
    # binder_modes.json for the threshold actually deployed, not on the threshold sequence's own
    # tables, precisely because those tables are what this artefact exists to check against.
    ("results/tables/background_specificity_disjoint.csv",
     ["models_rf/*_binder.joblib", "models_rf/binder_modes.json"],
     "python src/sai_cns/evaluation/background_specificity_disjoint.py"),

    # ---- the evaluation layer, all of it downstream of the models ------------------------------
    # Declares the external set as well as the model. Both this table and H10 below filter on
    # novel_to_model, and neither said so, which is why a stale flag reached the headline external
    # figure without anything going red.
    ("results/tables/external_bbb_validation.csv",
     ["models_rf/BBB.joblib", "data/external/processed/external_bbb_test.csv"],
     "python src/sai_cns/evaluation/external_validation.py"),
    ("results/tables/rf_conformal.csv", ["models_rf/BBB.joblib"],
     "python src/sai_cns/evaluation/rf_conformal_temporal.py"),
    # Two of its six checks read the external BBB set directly (the "ranks permeable drugs" and
    # "domain flag separates" checks both draw their comparator from novel_to_model), the same
    # dependency external_validation.py already declares two entries above. Missing here until this
    # pass, which is the same class of gap Finding A closed for the manuscript's own external figure.
    # ad_reference.pkl belongs here: this check measures similarity against the deployed
    # applicability-domain reference, so a rebuilt reference makes its numbers stale. The
    # dependency was undeclared until the chirality-aware reference was published, at which point
    # the artefact went stale silently while the graph reported everything current.
    ("results/tables/inversion_validation.csv",
     ["models_rf/BBB.joblib", "data/external/processed/external_bbb_test.csv",
      "models_rf/ad_reference.pkl"],
     "python src/sai_cns/evaluation/validate_inversion.py"),
    ("results/tables/integrity_audit.csv", ["models_rf/BBB.joblib"],
     "python src/sai_cns/evaluation/integrity_audit.py"),

    # The applicability-domain outputs, absent from this graph until 2026-08-26 and stale by six
    # days when a thesis audit compared them with the file next to them. Both were built on
    # 13 August against the pre-retrain barrier model and never rebuilt, so
    # applicability_bbb_validation.csv reported AUROC 0.7608 on the 306 approved drugs where
    # external_bbb_validation.csv, generated from the same model on the same compounds, reported
    # 0.7645. Two artefacts describing one test set disagreed, and nothing complained, because
    # neither was declared here. Rebuilt, they agree exactly. This is the third artefact in this
    # project found stale for want of a declaration rather than for want of a rebuild, so the rule
    # is worth stating: an output that no entry here names is unprotected, whatever else is true
    # of it.
    ("results/tables/applicability_bbb_validation.csv", ["models_rf/BBB.joblib"],
     "python src/sai_cns/evaluation/applicability_domain.py"),
    ("results/tables/applicability_coverage.csv", ["models_rf/BBB.joblib"],
     "python src/sai_cns/evaluation/applicability_domain.py"),

    # Per-target calibration for the binder panel. The headline calibration figure covers the eight
    # isotonically calibrated core classifiers; this covers the 38 Platt-scaled binder endpoints,
    # whose expected calibration error is about five times larger. It is the artefact behind the
    # only honest statement of how well the panel's probabilities are calibrated as a whole.
    ("results/tables/integrity_calibration_per_target.csv", ["models_rf/binder_modes.json"],
     "python src/sai_cns/evaluation/integrity_audit.py"),

    # Reconciles the four figures this project reports for deployed binder sensitivity. Written
    # after a thesis audit found the registry storing a value computed over every active in the
    # endpoint table under a label saying held-out-by-scaffold, which made the published mean
    # 0.898 where the held-out figure is 0.764. It depends on the registry and on the held-out
    # definitions, because it is a claim about the relationship between them.
    ("results/tables/sensitivity_reconciliation.csv",
     ["models_rf/binder_modes.json", "models_rf/holdout"],
     "python src/sai_cns/evaluation/sensitivity_reconciliation.py"),


    # Two supporting analyses that were absent from this graph and had gone stale unnoticed. Both
    # predate the neutralisation retrain by a month, and both are quoted in the technical report, so
    # nothing would have caught them except someone checking file dates by hand. Being outside the
    # graph is exactly what let them drift, which is the defect this checker exists to prevent.
    ("results/tables/feature_block_ablation.csv", ["models_rf/BBB.joblib"],
     "python src/sai_cns/evaluation/feature_analysis.py"),

    # The falsification suite. Nine hypotheses, four of them refuted, quoted in the technical
    # report and in the manuscript. Four of the original eight had gone a month stale against the
    # retrained panel before this entry existed, for the same reason as the two above: nothing named
    # them. H9 is listed separately below because it is an input to the summary, not a sibling of it.
    ("inversion/results/VERDICTS.csv",
     ["models_rf/binder_modes.json", "models_rf/holdout",
      "inversion/results/H9_disease_discrimination.csv"],
     "python inversion/inv_disease_layer.py && python inversion/inv_clinical_indication.py && "
     "python inversion/inv_distant_specificity.py && python inversion/inv_readacross_value.py && "
     "python inversion/inv_target_discrimination.py && python inversion/inv_panel_independence.py "
     "&& python inversion/inv_disease_discrimination.py && python inversion/summarise.py"),

    # H9 scores the disease layer through app.py's own graph, so it is stale if either the models or
    # the graph move. H6's predictions are its input population.
    ("inversion/results/H9_disease_discrimination.csv",
     ["models_rf/binder_modes.json", "inversion/results/H6_clinical_indication_predictions.csv"],
     "python inversion/inv_disease_discrimination.py"),

    # An orphan found by a reviewer-style audit: written by an earlier binder training, regenerated
    # by nothing, declared nowhere, and therefore three weeks behind the panel while shipping in the
    # institutional review package. It read a mean AUROC of 0.9765 where the registry holds 0.9113
    # over the same 47 targets. It is now derived from the registry and declared here.
    ("results/tables/binder_vs_measured_inactives.csv", ["models_rf/binder_modes.json"],
     "python src/sai_cns/evaluation/binder_vs_measured_inactives.py"),

    # The two artefacts the LIVE SERVER quotes on every query, and the two that were outside this
    # graph the longest. Both were found stale by a reviewer-style audit of the deployed interface,
    # three weeks behind the models they describe.
    #
    # endpoint_context.json is the more serious of the two. It carries the training base rate for
    # each classifier, and the interface reports engagement as enrichment over that rate, so a
    # stale prior silently mis-scores every query. Its base rates predated the neutralisation
    # retrain, and were high by up to 0.185: a compound scoring 0.70 at AChE was reported as no
    # engagement at all when the correct figure is 0.26. Its generator's docstring said "Run once",
    # which is precisely why nobody ran it again.
    ("models_rf/endpoint_context.json",
     ["models_rf/BBB.joblib", "models_rf/binder_modes.json"],
     "python src/sai_cns/build_endpoint_context.py"),

    # temporal_by_domain.csv backs the reliability banner shown above every result, which states the
    # AUROC and rank correlation to expect at the query's own distance from training chemistry.
    ("results/tables/temporal_by_domain.csv", ["models_rf/BBB.joblib"],
     "python src/sai_cns/analysis/temporal_diagnosis.py"),

    # The worked example in section 5.1 of the technical report. It is the output of the deployed
    # models on three named compounds, so it is stale the moment any of them is refitted. Quoting a
    # prediction in a document without declaring it here is how the earlier supporting analyses
    # drifted, and this one is quoted line by line.
    ("results/tables/worked_example.csv",
     ["models_rf/binder_modes.json", "results/tables/external_novelty_strata.csv"],
     "python src/sai_cns/analysis/worked_example.py"),

    # The external-validation programme. These refit the panel rather than scoring it, so they do
    # not depend on the fitted models for their result; they depend on binder_modes.json only
    # because each row quotes the deployed figure beside the prospective one. They depend on the
    # endpoint tables, which is what actually invalidates them.
    ("results/tables/external_prospective.csv",
     ["data/endpoints/*.csv", "models_rf/binder_modes.json"],
     "python src/sai_cns/evaluation/external_prospective.py"),
    ("results/tables/external_prospective_core.csv", ["data/endpoints/*.csv"],
     "python src/sai_cns/evaluation/external_prospective_core.py"),
    ("results/tables/external_cross_source.csv",
     ["data/endpoints/*.csv", "models_rf/binder_modes.json"],
     "python src/sai_cns/evaluation/external_cross_source.py"),
    # The stratification is downstream of the compound-level output of both, and is the analysis the
    # headline numbers are meaningless without.
    ("results/tables/external_novelty_strata.csv",
     ["results/tables/external_prospective.csv", "results/tables/external_cross_source.csv"],
     "python src/sai_cns/evaluation/external_novelty_strata.py"),

    # The technical report itself. It was the one generated document outside this graph, and it went
    # stale exactly as the graph exists to prevent: it computes its cross-provenance sensitivity from
    # external_cross_source.csv, that artefact was regenerated after the sensitivity correction, and
    # the report went on stating 0.868 where the artefact said 0.716 because nothing declared the
    # dependency. Every table its `read()` helper opens is listed, so a regenerated input now fails
    # the check until the report is rebuilt from it.
    ("docs/TECHNICAL_REPORT.md",
     ["models_rf/binder_modes.json", "inversion/results/VERDICTS.csv",
      "inversion/results/H9_disease_discrimination.csv",
      "results/tables/adme_cv_summary.csv", "results/tables/applicability_measures.csv",
      "results/tables/calibration.csv", "results/tables/external_bbb_validation.csv",
      "results/tables/external_cross_source.csv", "results/tables/external_novelty_strata.csv",
      "results/tables/external_prospective.csv", "results/tables/external_prospective_core.csv",
      "results/tables/feature_block_ablation.csv", "results/tables/inversion_validation.csv",
      "results/tables/learning_curve.csv", "results/tables/model_comparison.csv",
      "results/tables/noncns_specificity_summary.csv", "results/tables/permutation_null.csv",
      "results/tables/rf_conformal.csv", "results/tables/rf_cv_summary.csv",
      "results/tables/rf_temporal.csv", "results/tables/scaffold_holdout_results.csv",
      "results/tables/stereochemistry_audit.csv", "results/tables/worked_example.csv"],
     "python src/sai_cns/analysis/build_technical_report.py"),

    # The two analyses written to answer foreseeable reviewer criticisms. Both are quoted verbatim in
    # section 8.2 of the technical report, which is precisely the condition that let earlier
    # supporting analyses drift: quoted in a document, named by no dependency.
    ("results/tables/stereochemistry_audit.csv", ["data/endpoints/*.csv"],
     "python src/sai_cns/evaluation/stereochemistry_audit.py"),
    # The three applicability-domain artefacts are DERIVED, and until now this graph named
    # ad_reference.pkl only as an input while naming ad_per_endpoint.pkl and readacross_index.pkl
    # nowhere at all. This file states the governing rule itself: an output that no entry here
    # names is unprotected, whatever else is true of it. All three are built from the endpoint and
    # ADME tables and from the fingerprint definition, so adding an endpoint table or changing the
    # fingerprint makes all three stale. Leaving them undeclared is what allowed the per-endpoint
    # domain and the read-across index to be served for weeks built with one fingerprint and
    # queried with another, which reported morphine, a compound that is in the index, at a maximum
    # similarity of 0.53 against a true 1.00.
    ("models_rf/ad_reference.pkl",
     ["data/endpoints/*.csv", "data/adme/*.csv", "src/sai_cns/ad_fingerprint.py"],
     "python src/sai_cns/build_ad_reference.py"),
    ("models_rf/ad_per_endpoint.pkl",
     ["data/endpoints/*.csv", "models_rf/binder_modes.json",
      "src/sai_cns/ad_fingerprint.py"],
     "python src/sai_cns/build_ad_per_endpoint.py"),
    ("models_rf/readacross_index.pkl",
     ["data/endpoints/*.csv", "src/sai_cns/ad_fingerprint.py"],
     "python src/sai_cns/build_readacross_index.py"),
    # specificity_background_comparison.csv carries the server's headline caveat, specificity 0.925
    # on the curated library against 0.8817 on 15,000 unseen compounds. It had no generator anywhere
    # in the tree and no entry here, and it is cited as the source for those figures by the thesis
    # and the viva notes. Its two similarity columns were measured with the stereochemistry-blind
    # fingerprint; when the fingerprint changed, the manuscripts, the thesis and the viva notes were
    # all updated to the re-measured median of 0.400 and 49.4 per cent below Tanimoto 0.40, and the
    # table they cite was not, so anyone checking the citation found 0.417 and 43.3 per cent. This
    # is the same failure mode the comment above describes: quoted in a document, named by no
    # dependency. The firing columns do not depend on the reference and are not recomputed.
    ("results/tables/specificity_background_comparison.csv",
     ["results/tables/background_pubchem_other.csv", "models_rf/ad_reference.pkl",
      "src/sai_cns/ad_fingerprint.py"],
     "python tools/recompute_background_ad.py --write"),
    ("results/tables/applicability_measures.csv",
     ["models_rf/ad_reference.pkl", "data/external/processed/external_bbb_test.csv"],
     "python src/sai_cns/evaluation/applicability_measures.py"),
    ("results/tables/learning_curve.csv", ["models_rf/BBB.joblib"],
     "python src/sai_cns/evaluation/learning_curve.py"),
    ("results/tables/noncns_specificity_summary.csv",
     ["models_rf/BBB.joblib", "models_rf/binder_modes.json"],
     "python src/sai_cns/evaluation/noncns_specificity.py"),
    ("results/tables/binder_cv_summary.csv", ["models_rf/binder_modes.json"],
     "python src/sai_cns/evaluation/binder_cv_per_fold.py"),

    # The two-step hold-out. This pair is the incident that motivated the whole file: the panel was
    # re-run and the report was not, so the inputs were current and the output was a day stale.
    ("models_rf/holdout", ["models_rf/binder_modes.json"],
     "python src/sai_cns/evaluation/scaffold_holdout_panel.py"),
    ("results/tables/scaffold_holdout_results.csv", ["models_rf/holdout"],
     "python src/sai_cns/evaluation/scaffold_holdout_report.py"),

    # ---- the reviewer package -------------------------------------------------------------------
    ("reviewer_package/model_outputs/TRAINING_MATRIX.csv", ["data/endpoints/*.csv"],
     "python src/sai_cns/analysis/build_reviewer_matrix.py"),
    ("reviewer_package/model_outputs/SAI-CNS_models_inputs_and_folds.xlsx",
     ["reviewer_package/model_outputs/TRAINING_MATRIX.csv", "results/tables/rf_cv_folds.csv"],
     "python src/sai_cns/analysis/build_reviewer_workbook_full.py"),

    # ---- figures read the tables ----------------------------------------------------------------
    ("manuscript/figures/Figure3_cv_design.png", ["results/tables/rf_cv_summary.csv"],
     "python src/sai_cns/figures/fig03_cv_design.py"),
    ("manuscript/figures/Figure4_pools_and_thresholds.png", ["models_rf/binder_modes.json"],
     "python src/sai_cns/figures/fig04_pools_and_thresholds.py"),
    ("manuscript/figures/Figure5_negative_class.png", ["results/tables/rf_cv_summary.csv"],
     "python src/sai_cns/figures/fig05_negative_class.py"),
    ("manuscript/figures/Figure6_validation.png",
     ["results/tables/calibration.csv", "results/tables/scaffold_holdout_results.csv",
      "results/tables/noncns_specificity_summary.csv",
      "results/tables/external_bbb_validation.csv", "results/tables/inversion_validation.csv"],
     "python src/sai_cns/figures/fig06_validation.py"),
    ("manuscript/figures/Figure7_binder_panel.png",
     ["models_rf/binder_modes.json", "results/tables/background_specificity_disjoint.csv"],
     "python src/sai_cns/figures/fig07_binder_panel.py"),
    # Five figures were outside this graph entirely, and one of them mattered. Figure 8 shows real
    # predictions for eight drugs, so it runs through disease_scores and therefore through the
    # training base rates in endpoint_context.json. That file was corrected on 25 August, and the
    # figure predated it by six days, which means the worked example in the manuscript was drawn
    # with the base rates that suppressed real signal. The others are schematics or panel summaries,
    # but a schematic that quotes a panel size goes stale the same way.
    ("manuscript/figures/Figure8_use_case.png",
     ["models_rf/binder_modes.json", "models_rf/endpoint_context.json"],
     "python src/sai_cns/figures/fig08_use_case.py"),
    # The separate file NAR's Web Server Issue requires as a submission item. It quotes the panel
    # size and scores a live donepezil query through the same entry point Figure 4 uses, so it goes
    # stale the same way that figure does; replaces figures/graphical_abstract.png, which quoted
    # 64,474 training records against the 228,200 the tables have held since this session began.
    ("manuscript/figures/GraphicalAbstract.png",
     ["models_rf/binder_modes.json", "models_rf/endpoint_context.json"],
     "python src/sai_cns/figures/fig_graphical_abstract.py"),
    ("manuscript/figures/Figure9_model_atlas.png",
     ["models_rf/binder_modes.json", "results/tables/rf_cv_summary.csv",
      "results/tables/MODEL_INVENTORY.csv"],
     "python src/sai_cns/analysis/build_model_inventory.py && "
     "python src/sai_cns/figures/fig09_model_atlas.py"),
    ("manuscript/figures/Figure10_endpoint_selection.png", ["models_rf/binder_modes.json"],
     "python src/sai_cns/figures/fig10_endpoint_selection.py"),
    ("manuscript/figures/Figure1_architecture.png",
     ["models_rf/binder_modes.json", "results/tables/rf_cv_folds.csv",
      "results/tables/binder_cv_folds.csv", "results/tables/adme_cv_folds.csv"],
     "python src/sai_cns/figures/fig01_architecture.py"),
    ("manuscript/figures/Figure2_feature_vector.png", ["models_rf/binder_modes.json"],
     "python src/sai_cns/figures/fig02_feature_vector.py"),

    ("manuscript/figures/Figure11_external_validation.png",
     ["results/tables/external_prospective.csv", "results/tables/external_novelty_strata.csv",
      "results/tables/external_cross_source.csv"],
     "python src/sai_cns/figures/fig11_external_validation.py"),

    # ---- the manuscript is downstream of everything it quotes -----------------------------------
    ("manuscript/tables_generated.md",
     ["results/tables/rf_cv_summary.csv", "models_rf/binder_modes.json"],
     "python src/sai_cns/analysis/manuscript_tables.py"),
    ("manuscript/NAR_WebServer_SAI-CNS_built.md",
     ["manuscript/NAR_WebServer_SAI-CNS_draft.md", "manuscript/tables_generated.md",
      "manuscript/references_verified.json", "manuscript/references_links.json",
      "manuscript/figures/Figure6_validation.png"],
     "python src/sai_cns/analysis/build_manuscript.py"),

    # The condensed, NAR-length draft states the same headline numbers in its own words, so it can
    # go stale independently of the full draft, and did: it went two full audit passes over the full
    # draft without a corresponding pass over this one. Declaring it here is what makes that a build
    # failure instead of a second document quietly disagreeing with the first.
    ("manuscript/NAR_condensed_built.md",
     ["manuscript/NAR_condensed_draft.md", "manuscript/references_verified.json",
      "manuscript/references_links.json"],
     "python src/sai_cns/analysis/build_manuscript_condensed.py"),

    # Supplementary_Information.docx embeds Figure9_model_atlas.png, Figure6_validation.png and
    # Figure11_external_validation.png as S1-S3; it went stale relative to two of the three, twice,
    # because nothing declared the dependency and check_freshness had no way to catch it.
    ("manuscript/Supplementary_Information.docx",
     ["manuscript/figures/Figure9_model_atlas.png", "manuscript/figures/Figure6_validation.png",
      "manuscript/figures/Figure11_external_validation.png"],
     "python src/sai_cns/analysis/build_supplementary_information.py"),

    # references.md used to be a hand-maintained second copy of the bibliography, numbered by year
    # rather than by citation order, so it silently disagreed with every number actually printed in
    # the manuscript. cite.py now writes it as a byproduct of the same resolution the built manuscript
    # uses, from the same three inputs, so the two cannot diverge again; this entry is what makes a
    # forgotten rebuild after any of the three changes into a freshness failure instead of a stale
    # file nobody notices.
    ("manuscript/references.md",
     ["manuscript/NAR_WebServer_SAI-CNS_draft.md", "manuscript/references_verified.json",
      "manuscript/references_links.json"],
     "python src/sai_cns/analysis/cite.py"),

    # ENDPOINT_JUSTIFICATION.md and DATA_MANIFEST.md previously described a twelve-endpoint,
    # 61,317-compound panel that predates the binder architecture, for months, because nothing
    # declared them and no build ever touched them after the panel grew past that snapshot. Both are
    # now generated from the same registries the manuscript cites, so the documents a reviewer reads
    # cannot drift from the panel a rebuild would find again.
    ("docs/ENDPOINT_JUSTIFICATION.md",
     ["models_rf/binder_modes.json", "results/tables/MODEL_INVENTORY.csv",
      "results/tables/np_target_survey.csv"],
     "python tools/build_endpoint_justification.py"),
    ("docs/DATA_MANIFEST.md",
     ["models_rf/binder_modes.json", "results/tables/unique_compound_count.csv"],
     "python tools/build_data_manifest.py"),

    # The actual numeric input every model trains on was computed on the fly and never written to
    # a file anywhere, so a reviewer asking "what did the model see" could not be shown it. This is
    # a ~10 minute, ~800 MB regeneration (every distinct compound across every endpoint table,
    # featurised once), which is exactly why it needs a freshness entry rather than being trusted
    # to stay in sync by memory alone.
    ("results/tables/master_feature_vectors.csv",
     ["data/endpoints", "data/adme"],
     "python tools/build_master_training_vectors.py"),
    ("results/tables/master_training_usage.csv",
     ["data/endpoints", "data/adme"],
     "python tools/build_master_training_vectors.py"),

    # Every formula quoted in the manuscript, checked against the deployed models rather than
    # restated from memory; the BBB worked example specifically reconciles the 5-fold calibrated
    # ensemble's own internal arithmetic against app.py's live output.
    ("docs/ML_METHODS_AND_FORMULAS.md",
     ["models_rf/BBB_calibrated.joblib", "results/tables/rf_conformal.csv"],
     "python tools/build_ml_formulas.py"),
]

# Checked by content rather than by timestamp, because it carries checksums of what it describes.
CHECKSUMMED = "models_manifest.json"
STATE = ROOT / "tools" / "freshness_state.json"
HASH_LIMIT = 32 * 1024 * 1024        # above this, fingerprint by size rather than content


def fingerprint(pattern: str) -> str | None:
    """A content fingerprint for a file, glob or directory.

    Small files are hashed. Large binary trees are described by file count and total size, which is
    weaker and is documented as such: it would miss a retrain that happened to produce the same
    number of files at the same total size.
    """
    p = ROOT / pattern
    if "*" in pattern:
        hits = sorted(ROOT.glob(pattern))
    elif p.is_dir():
        hits = sorted(q for q in p.rglob("*") if q.is_file())
    elif p.exists():
        hits = [p]
    else:
        return None
    total = sum(q.stat().st_size for q in hits)
    if len(hits) == 1 and total <= HASH_LIMIT:
        h = hashlib.sha256()
        with open(hits[0], "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()
    if total <= HASH_LIMIT:
        h = hashlib.sha256()
        for q in hits:
            h.update(q.relative_to(ROOT).as_posix().encode())
            h.update(q.read_bytes())
        return "sha256:" + h.hexdigest()
    return f"count-size:{len(hits)}:{total}"


def newest(pattern: str):
    """The newest file matching a path, glob or directory, and which file it was."""
    p = ROOT / pattern
    if "*" in pattern:
        hits = sorted(ROOT.glob(pattern))
    elif p.is_dir():
        hits = [q for q in p.rglob("*") if q.is_file()]
    elif p.exists():
        hits = [p]
    else:
        hits = []
    if not hits:
        return None, None
    best = max(hits, key=lambda q: q.stat().st_mtime)
    return best.stat().st_mtime, best


def stamp(ts) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "missing"


def _repackage_command() -> str:
    """The repackage command, with the NEXT version rather than a hardcoded one.

    This was written as a literal "package_models.py 1.1". Following it once the manifest had
    moved to 1.2 would have rewritten models_manifest.json back to 1.1 and written a v1.1 archive
    over the v1.2 one, which is the opposite of what a remediation line should do. The version is
    read from the manifest and incremented.
    """
    path = ROOT / CHECKSUMMED
    nxt = "1.1"
    try:
        v = str(json.loads(path.read_text(encoding="utf-8")).get("version", "1.0"))
        major, _, minor = v.partition(".")
        nxt = f"{int(major)}.{int(minor or 0) + 1}"
    except Exception:
        pass
    return f"python src/sai_cns/models/package_models.py {nxt}"


def check_manifest() -> dict:
    """Verify every checksum the manifest records, which is stronger than any timestamp."""
    path = ROOT / CHECKSUMMED
    if not path.exists():
        return {"artefact": CHECKSUMMED, "state": "MISSING", "detail": "no manifest",
                "rebuild": "python src/sai_cns/models/package_models.py 1.0"}
    man = json.loads(path.read_text(encoding="utf-8"))
    files = man.get("files", {})
    ok = bad = missing = 0
    for rel, rec in files.items():
        q = ROOT / rel
        if not q.exists():
            missing += 1
            continue
        want = rec["sha256"] if isinstance(rec, dict) else rec
        h = hashlib.sha256()
        with open(q, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        ok += 1 if h.hexdigest() == want else 0
        bad += 0 if h.hexdigest() == want else 1
    state = "OK" if (bad == 0 and missing == 0) else "STALE"
    return {"artefact": CHECKSUMMED, "state": state,
            "detail": f"{ok} verified, {bad} checksum mismatch, {missing} missing, "
                      f"of {len(files)} entries",
            "rebuild": _repackage_command()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Report artefacts older than their inputs.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="print only stale artefacts")
    ap.add_argument("--accept", action="store_true",
                    help="record the current fingerprints as a consistent state")
    args = ap.parse_args(argv)

    recorded = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}

    # A clone of the public repository carries the server and the science, but not the
    # paper-authoring scripts nor the manuscript/ and docs/ trees they write into. For such a
    # checkout an entry whose builder AND artefact are both absent describes a pipeline that is
    # simply not present, and reporting it as MISSING would tell a reader to run a script the
    # repository does not contain. An entry is still checked whenever either side exists, so a
    # full working tree sees exactly what it saw before and nothing can hide by deleting itself.
    def _builder_present(cmd: str) -> bool:
        m = re.search(r"(\S+\.py)", cmd or "")
        return not m or (ROOT / m.group(1)).exists()

    graph = [(o, i, c) for (o, i, c) in GRAPH
             if _builder_present(c) or newest(o)[0] is not None]
    skipped = len(GRAPH) - len(graph)

    current: dict[str, str] = {}
    for _out, ins, _cmd in graph:
        for src in ins:
            if src not in current:
                fp = fingerprint(src)
                if fp:
                    current[src] = fp

    results = []
    for out, ins, cmd in graph:
        out_ts, _ = newest(out)
        if out_ts is None:
            results.append({"artefact": out, "state": "MISSING", "detail": "not built",
                            "rebuild": cmd})
            continue
        stale_by = []
        for src in ins:
            if src not in current:
                stale_by.append(f"{src} (missing)")
                continue
            # Staleness needs both conditions. An input touched after the output was built is
            # suspicious, but only actually stale if its content moved: re-running a step that
            # rewrites a file identically must not condemn everything downstream. Conversely a
            # content change that predates the rebuild is already incorporated, which is the case
            # every time an input is edited and its output regenerated straight afterwards.
            in_ts, in_file = newest(src)
            touched_after_build = bool(in_ts and in_ts > out_ts)
            if not touched_after_build:
                continue
            was = recorded.get(src)
            if was is None:
                stale_by.append(f"{in_file.relative_to(ROOT).as_posix()} newer "
                                f"@ {stamp(in_ts)} (no accepted baseline)")
            elif was != current[src]:
                stale_by.append(f"{src} changed after this was built")
        results.append({
            "artefact": out,
            "state": "STALE" if stale_by else "OK",
            "detail": (f"built {stamp(out_ts)}; " + "; ".join(stale_by)) if stale_by
                      else f"built {stamp(out_ts)}",
            "rebuild": cmd,
        })
    results.append(check_manifest())

    bad = [r for r in results if r["state"] != "OK"]
    if args.accept:
        STATE.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        print(f"recorded {len(current)} input fingerprints in "
              f"{STATE.relative_to(ROOT).as_posix()}")
        print("later runs will call an artefact stale only when an input's content has moved")
        raise SystemExit(0)
    if args.json:
        print(json.dumps({"stale": len(bad), "results": results}, indent=2))
    else:
        width = max(len(r["artefact"]) for r in results)
        for r in results:
            if args.quiet and r["state"] == "OK":
                continue
            mark = {"OK": "  ok  ", "STALE": " STALE", "MISSING": "MISSING"}[r["state"]]
            print(f"[{mark}] {r['artefact']:<{width}}  {r['detail']}")
        print()
        if bad:
            print(f"{len(bad)} artefact(s) are stale or missing. Rebuild, in this order:")
            # One command per line. Several rebuild recipes are && chains, which is fine in bash and
            # a parse error in Windows PowerShell 5.1, the shell this project is developed in: it
            # rejects && as a statement separator outright, so a pasted chain fails before the first
            # script runs. Splitting the chain keeps the ordering visible and the lines runnable in
            # either shell.
            seen = set()
            for r in bad:
                if r["rebuild"] in seen:
                    continue
                seen.add(r["rebuild"])
                for step in r["rebuild"].split("&&"):
                    print(f"    {step.strip()}")
        else:
            print("every declared artefact is newer than its inputs")

    # Timestamps answer "is this file older than its inputs". They cannot answer "does the panel
    # still consist of what everything claims it consists of", and that is the question `make train`
    # got wrong when it refitted 39 of 52 binders and reported success. Every model file was newer
    # than its table, so this checker was satisfied; thirteen endpoints were simply never visited.
    # The registry reconciliation is the missing half, and it runs here because this is what the
    # pre-commit hook already calls.
    panel_bad = []
    try:
        sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
        import panel
        v = panel.verify()
        panel_bad = [k for k in ("no_model", "no_table", "unregistered", "stale_model", "no_mode",
                                 "withdrawn_silent") if v[k]]
        if not args.json:
            print()
            if panel_bad:
                print("the panel is not self-consistent:")
                for k in panel_bad:
                    print(f"    {k:17s} {len(v[k])}: {', '.join(v[k])}")
                print("    run: python src/sai_cns/panel.py")
            else:
                print(f"panel consistent: {v['n_registered']} endpoints, {v['n_deployed']} deployed, "
                      f"registry and fitted models agree")
    except Exception as exc:                       # never let the extra check mask the primary one
        if not args.json:
            print(f"\npanel check could not run: {type(exc).__name__}: {exc}")

    raise SystemExit(1 if (bad or panel_bad) else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
