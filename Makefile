# SAI-CNS. One command per intention.
#
# The targets are split by cost and by consequence, not by convenience. `reproduce` regenerates
# everything downstream of the trained models and takes minutes. `train` refits them and takes
# hours, so it is never a dependency of anything: refitting is a deliberate act, and a Makefile that
# retrains because a figure looked stale would be a trap.
#
# Ordering inside `reproduce` is the dependency order from tools/check_freshness.py. Where a group
# of scripts all write models_rf/binder_modes.json, the whole group runs or none of it does; running
# one alone silently reverts the others.
#
# Windows: use `make` from Git Bash, or run the commands under each target directly. PY is
# overridable so the same file works against a virtualenv on any platform:
#   make reproduce PY=python3

PY ?= ./brainsafe_env/Scripts/python.exe
SRC := src/sai_cns

.DEFAULT_GOAL := help
.PHONY: help setup test test-fast coverage check freshness accept provenance \
        reproduce figures manuscript inventory ledger train thresholds docker-build docker-run clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------- environment --------
setup:  ## install pinned dependencies and the test tools
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install pytest pytest-cov

# ---------------------------------------------------------------------------- tests --------------
test:  ## run the full test suite, including the end-to-end smoke test
	$(PY) -m pytest tests/ -v

test-fast:  ## unit tests only; skips the smoke test that loads 0.84 GB of models
	$(PY) -m pytest tests/ -v --ignore=tests/test_integration_smoke.py

coverage:  ## test the scientific core and report coverage on it
	$(PY) -m pytest tests/ --cov=$(SRC)/features --cov=$(SRC)/models \
		--cov-report=term-missing --cov-report=html:repro/coverage

# ---------------------------------------------------------------------------- integrity ----------
freshness:  ## report any artefact older than something it was derived from
	$(PY) tools/check_freshness.py

accept:  ## record the current artefact state as consistent
	$(PY) tools/check_freshness.py --accept

ledger:  ## rebuild the reproduction ledger against the manuscript
	$(PY) validation/repro/r03_ledger.py

check: freshness test ledger  ## everything that should pass before a commit or a submission

# ---------------------------------------------------------------------------- reproduction -------
inventory:  ## the dated inventory of every deployed estimator
	$(PY) $(SRC)/analysis/build_model_inventory.py

provenance:  ## regenerate the provenance map from the dependency graph
	$(PY) tools/build_provenance.py

figures:  ## every manuscript and supplementary figure, from artefacts
	$(PY) $(SRC)/figures/fig01_architecture.py
	$(PY) $(SRC)/figures/fig02_feature_vector.py
	$(PY) $(SRC)/figures/fig03_cv_design.py
	$(PY) $(SRC)/figures/fig04_pools_and_thresholds.py
	$(PY) $(SRC)/figures/fig05_negative_class.py
	$(PY) $(SRC)/figures/fig06_validation.py
	$(PY) $(SRC)/figures/fig07_binder_panel.py
	$(PY) $(SRC)/figures/fig08_use_case.py
	$(PY) $(SRC)/figures/fig09_model_atlas.py

manuscript:  ## rebuild the manuscript, resolving citations and injecting tables
	$(PY) $(SRC)/analysis/manuscript_tables.py
	$(PY) $(SRC)/analysis/build_manuscript.py

reproduce: inventory figures manuscript provenance  ## regenerate metrics, figures and manuscript
	@echo ""
	@echo "reproduce complete. Verify with: make check"

# ---------------------------------------------------------------------------- training -----------
# Hours, and never a dependency. See REPRODUCE.md before running these.
thresholds:  ## re-derive binder thresholds. All four steps, in this order, or none
	$(PY) $(SRC)/models/final_thresholds.py
	$(PY) $(SRC)/models/screening_thresholds.py
	$(PY) $(SRC)/models/apply_specificity_decisions.py
	$(PY) $(SRC)/models/calibrate_background_specificity.py

train:  ## refit the whole panel from the endpoint tables. Hours. Read REPRODUCE.md first
	$(PY) $(SRC)/models/train_rf.py
	$(PY) $(SRC)/models/calibrate.py
	$(PY) $(SRC)/models/train_binders_hybrid.py
# The binder panel is trained by TWO scripts, not one. 44 endpoints are decoy-aware and come from
# train_binders_hybrid.py; the other 8 have enough measured inactives to be trained on labels alone
# and come from train_measured_label_holdout.py. Omitting the second is silent: the panel still
# loads, still scores, and 8 endpoints keep weights fitted to whatever representation the previous
# run used. It was omitted here until the featurisation changed underneath it.
	$(PY) $(SRC)/models/train_measured_label_holdout.py
	$(PY) $(SRC)/adme/train_adme.py
	$(MAKE) thresholds

# ---------------------------------------------------------------------------- container ----------
docker-build:  ## build the server image
	docker build -t sai-cns:latest .

docker-run:  ## run the server: interface on 8501, API on 8000
	docker run --rm -p 8501:8501 -p 8000:8000 sai-cns:latest

clean:  ## remove caches and coverage output, never results or models
	rm -rf .pytest_cache repro/coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
