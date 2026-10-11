# Reproducing SAI-CNS

Everything in the manuscript regenerates from this repository. This page is the quickstart; the
edge-by-edge map of what produces what is `repro/PROVENANCE.md`, generated from the same dependency
declaration the freshness check enforces.

---

## Quickstart

```bash
git clone https://github.com/krishna-g-999/sai-cns && cd sai-cns
python -m venv brainsafe_env && . brainsafe_env/bin/activate   # Windows: brainsafe_env\Scripts\activate
pip install -r requirements.txt
python model_fetch.py                 # trained estimators, verified by SHA-256
python tools/reproduce.py             # metrics, figures, manuscript. ~75 s
python tools/reproduce.py check       # freshness, tests, reproduction ledger
```

`make reproduce` and `make check` do the same thing where `make` is available. `tools/reproduce.py`
exists because `make` is absent from a default Windows install and this work is developed on Windows
and deployed on Linux; the two are kept in step, and the Makefile is the authority on ordering.

---

## What each command does

| Command | Does | Cost |
|---|---|---|
| `python tools/reproduce.py` | model inventory, all nine figures, manuscript, provenance map | ~75 s |
| `python tools/reproduce.py check` | freshness, 33 tests, reproduction ledger | ~2 min |
| `python tools/reproduce.py figures` | the figures only | ~70 s |
| `python tools/reproduce.py thresholds` | re-derive binder thresholds, all four steps | ~25 min |
| `python tools/reproduce.py train` | refit the entire panel | **hours** |
| `python tools/check_freshness.py` | report any artefact older than its inputs | ~40 s |
| `python -m pytest tests/ -v` | the test suite | ~90 s |

`reproduce` deliberately does **not** retrain. Training is hours, and refitting is a scientific act
rather than a build step: a command that retrained because a figure looked stale would be a trap.

---

## Measured runtimes

On the development machine: Windows 11, 24 logical CPU, Python 3.13.13. Recorded by
`tools/reproduce.py` into `repro/run_log.json` on every run, so these are measurements rather than
estimates.

| Stage | Wall clock |
|---|---|
| Full reproduction (13 steps) | 74.4 s |
| Test suite, 33 tests | 84 s |
| Test suite under coverage instrumentation | 980 s (16 min 20 s) |
| Core cross-validation, 13 endpoints, independent re-run | 962 s |
| Binder panel retrain, 52 endpoints | ~95 min |
| Non-CNS specificity, 1,000 compounds through the full pipeline | ~28 min |
| Container build | not measured; see Blockers |

---

## Test coverage, and an honest denominator

```
python -m pytest tests/ -v
33 passed, 33 subtests passed in 83.72s
```

| Scope | Coverage |
|---|---|
| `features/featurize.py`, the module every model depends on | **95%** |
| `models/train_rf.py`, including the deduplication step | 39% |
| `models/pools.py`, the background partition | 36% |
| All of `models/` including one-shot training scripts | 9% |

The 9% figure is reported for completeness and is the wrong denominator to judge by. Most files
under `models/` are single-purpose training entry points that run for hours and write estimators;
covering them would mean retraining inside the test suite. What the tests target is the code whose
silent failure would change a published number: featurisation, deduplication, the pool partition, the
censored-label rule, and the assembled prediction path. The uncovered lines in `featurize.py` are the
two-line branch for a molecule that sanitises to nothing.

---

## Determinism

Every seed is 42, declared at 78 sites across 31 source files and recorded in
`validation/repro/environment.json`.

The independent reproduction re-ran the entire core cross-validation from the endpoint tables and
scored it with separately written metric code: **all 26 values reproduced exactly**, maximum
deviation 4.8 × 10⁻⁵, which is the rounding in the stored summary and nothing else.

One caveat, measured rather than assumed. Repeated scoring of the same compound by the deployed
models agrees to about 1 × 10⁻¹⁶ but is **not bit-identical**, because the forests run with
`n_jobs=-1` and floating-point addition is not associative, so the order in which tree votes are
summed is not fixed. The test suite asserts agreement to 1 × 10⁻¹² rather than bit-equality, since
asserting bit-equality would assert something the deployment does not provide and would fail on a
machine with a different core count.

---

## Pinned versions

`requirements.txt` pins every scientific dependency. The pins matter: scikit-learn records its own
version inside each pickle, so loading the deployed estimators under a different minor version raises
`InconsistentVersionWarning`. That warning was checked rather than assumed harmless, by comparing 630
model outputs across 10 structurally unrelated compounds between 1.8.0 and 1.9.0; they agreed exactly
except for one value differing by 1 × 10⁻¹².

| Package | Version |
|---|---|
| Python | 3.13.13 |
| scikit-learn | 1.8.0 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| rdkit | 2026.03.2 |
| scipy | 1.17.1 |
| xgboost | 3.3.0 |

Verify an environment with `python src/sai_cns/evaluation/app_health.py`, which exits non-zero if
the deployed panel does not behave as recorded.

---

## Cross-platform notes

- **Windows.** `make` is usually absent; use `tools/reproduce.py`. Run shell snippets from Git Bash
  rather than `cmd`. Long paths must be enabled for the deepest artefact directories.
- **Linux and macOS.** `make` targets work directly. RDKit's molecule drawing needs
  `libxrender1 libxext6 libsm6`, which the Dockerfile installs; without them structure images fail
  at run time rather than at import.
- **CI.** Tests run on `ubuntu-latest` and `windows-latest`, because a path-separator defect that
  appears on only one of them is a real class of bug in this codebase.
- **Models are not in git.** They are 0.84 GB. `model_fetch.py` downloads and verifies the archive
  checksum and then every extracted file, and is fatal on mismatch.

---

## The integrity checks

Three, each answering a different question. Run them all with `python tools/reproduce.py check`.

**Freshness** (`tools/check_freshness.py`) reports any artefact older than something it was derived
from. This exists because four separate incidents had the same shape: a derived file left describing
an input that had since changed, passing every content check because it was internally consistent
while every command exited 0. A pre-commit hook installed by `tools/install_hooks.py` refuses a
commit when anything is stale.

**Tests** (`tests/`) pin the invariants whose loss would change a published number without raising an
error: featurisation shape and purity, deduplication before splitting, disjoint background pools, the
censored-label rule, and an end-to-end check that the deployed panel still returns the
pharmacologically correct driver for four reference drugs.

**Ledger** (`validation/REPRO_LEDGER.csv`) compares every reported number against an independently
produced one, records the evidence tier, and verifies that the value it attributes to the manuscript
actually appears in the manuscript.

---

## Known blockers

**Container build not verified here.** Docker is not installed on the development machine, so
`docker build` has not been run locally. The Dockerfile is exercised by the `container` job in
`.github/workflows/ci.yml`. To verify:

```bash
docker build --build-arg SKIP_MODEL_FETCH=1 -t sai-cns:ci .
```

`SKIP_MODEL_FETCH=1` builds without the 0.84 GB archive, which is what CI needs; the resulting image
starts and serves the interface but cannot answer a query, so it is for build verification only.

**Deposit URL absent.** `models_manifest.json` carries the SHA-256 of the archive and of all 246
files but an empty `doi` and `urls`, because the archive is not yet published. `model_fetch.py`
therefore fails honestly rather than downloading the superseded pre-audit models. Publishing the
archive and filling in those two fields is the last step before a fresh clone can reproduce without
a local copy of `models_rf/`.

**Salt handling.** Counter-ions are stripped but the remaining fragment is not neutralised, so a
compound submitted as a salt scores differently from its free base. Measured on the deployed models,
haloperidol hydrochloride returns BBB 0.613 where haloperidol returns 0.993. This is pinned by a test
named as a known defect so that changing it is deliberate and visible, and it is not changed here
because doing so alters predictions for every salt input and requires re-validation.
