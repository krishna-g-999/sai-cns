# SAI-CNS

**Repository:** https://github.com/krishna-g-999/sai-cns
**Server:** https://huggingface.co/spaces/Krishnag999/sai-cns

**SAI-CNS: A Calibrated, Exposure-Gated Web Server for Small-Molecule Target Engagement and
Pharmacokinetics in the Human Brain.**

SAI-CNS predicts, from chemical structure alone, whether a compound reaches the brain and what
it is likely to do once there. It couples the two questions that most CNS drug-discovery tools answer
separately: a target score is admitted only in proportion to the compound's predicted brain exposure,
so potency at a target the compound cannot reach contributes nothing to the output. A submitted
structure returns engagement across **54 molecular targets** spanning the principal neurodegenerative,
psychiatric, neuroinflammatory, analgesic and sleep-related mechanisms, predicted blood-brain barrier
penetration, two cardiac safety liabilities (hERG, Nav1.5), and a nine-endpoint ADME and exposure
layer including a directly modelled unbound brain-to-plasma ratio (Kp,uu). Engaged targets are traced
through a curated pathway graph to the conditions they touch.

Every endpoint is trained on **measured** experimental values only, never on qualitative annotation.
The panel holds **228,200 measured compound-endpoint records** over **170,619 distinct structures**
from ChEMBL, BindingDB and B3DB.

> **Research use, pending peer review.** This tool predicts molecular target engagement and
> physicochemical properties, not clinical efficacy, and has not undergone wet-lab or clinical
> validation. It is not for medical, diagnostic, or treatment decisions.

## Model and validation (honest, multi-regime)

Deployed model: a **random forest** per endpoint, chosen after a like-for-like comparison against
XGBoost, histogram gradient boosting, logistic regression, a nearest-neighbour read-across, and a
graph neural network on a subset of endpoints. Features are a 1,024-bit ECFP-4 fingerprint plus 12
physicochemical descriptors. **75 estimators are trained, 70 deployed**: a 12-endpoint target-potency
layer, a 52-endpoint binder panel (47 deployed), a 10-endpoint exposure and ADME layer including the
barrier model, and the hERG safety classifier.

| Regime | Measured-label classifier AUROC |
|---|---|
| Random 10-fold | 0.899–0.976 (mean 0.958) |
| Scaffold-grouped 10-fold (GroupKFold) | 0.878–0.965 (mean 0.925) |
| Temporal (compounds published after a frozen cutoff) | 0.713–0.913 |
| External (306 FDA-approved drugs absent from B3DB) | 0.764, or 0.767 on the 227 also distinguishable from training in feature space |

Binder classifiers, validated against compounds measured and found inactive at the same target
rather than against decoys: mean AUROC **0.917** across the 47 deployed (range 0.719–0.985).

Isotonic-calibrated probabilities (mean expected calibration error **0.0801 → 0.0147**); conformal
prediction on the eight core classifiers achieves empirical coverage **0.876–0.933** against a 0.90
target; an applicability-domain distance to the nearest measured analogue on every prediction. On
1,000 compounds with no recorded activity at any modelled target, specificity is **0.925** (95% CI
0.907–0.940). Six adversarial checks, each written so that it could fail, all pass; a falsification
analysis withdrew two endpoints that could not separate a real ligand from a trivial metabolite. Full
detail: [`manuscript/NAR_WebServer_SAI-CNS_built.md`](manuscript/NAR_WebServer_SAI-CNS_built.md),
[`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).

## Repository structure

Standard research layout (`data/`, `src/sai_cns/{data,features,models,evaluation,figures,gnn,adme}`,
`results/`, `docs/`, `manuscript/`, `thesis/`, `inversion/`). `tools/check_freshness.py` declares
every derived artefact and its inputs, and a pre-commit hook refuses a commit that leaves one stale.

Key entry points: **`app.py`** (the interactive server), `src/sai_cns/models/train_rf.py`
(core-panel training), `src/sai_cns/models/train_binders_hybrid.py` (binder-panel training),
`src/sai_cns/evaluation/` (validation, comparison, conformal/temporal, adversarial checks),
`src/sai_cns/adme/` (ADME and exposure, including Kp,uu), `src/sai_cns/figures/` (every
manuscript figure, regenerated from the current models on every run), `inversion/` (the ten-hypothesis
falsification suite).

## Reproducibility

Trained-model binaries and large structure libraries are **not** stored in git; they are regenerated
by the released scripts, or restored from the checksummed archive named in
[`manuscript/NAR_WebServer_SAI-CNS_built.md`](manuscript/NAR_WebServer_SAI-CNS_built.md)'s Data
Availability section. In brief:

```bash
python src/sai_cns/data/rebuild_endpoints.py          # pool ChEMBL + BindingDB
python src/sai_cns/models/train_rf.py                 # core panel, RF + 10-fold
python src/sai_cns/models/train_binders_hybrid.py      # binder panel
python src/sai_cns/models/calibrate.py                # isotonic calibration
python src/sai_cns/evaluation/rf_conformal_temporal.py
python src/sai_cns/adme/fetch_adme.py && python src/sai_cns/adme/train_adme.py
python src/sai_cns/evaluation/validate_inversion.py   # adversarial checks
python tools/check_freshness.py                         # confirm nothing is stale
streamlit run app.py                                    # interactive tool
```

## Using the web server

Three ways in, all from the same models:

| Mode | What it is for |
|---|---|
| **Compound Search** | one compound, full report: exposure, mechanism network, target engagement profile, disease relevance, ADME, read-across, CNS MPO and applicability domain |
| **Batch Screening** | up to 300 compounds pasted or uploaded as CSV, TSV or plain text, returned as one ranked row each with a CSV download |
| **Export** | every result downloadable as a tidy CSV, a self-contained HTML report, a structured JSON object, or the mechanism network as vector SVG |

The HTML report inlines its own figures, structure image and styling, so it opens offline in any
browser and prints to PDF without a network connection. The JSON carries thresholds, the screening
mode and the caveats, so a stored answer stays interpretable away from the interface.

No registration or login is required. No submitted structure is written to disk, logged, or retained
beyond the lifetime of its request, and none is used to train or update any model; the one exception
is a compound entered by name, which is resolved to a structure through a single call to PubChem.
Curated example compounds are provided so the server can be tried without preparing any input.

## Running it

Locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

As a container, which is how it should be deployed:

```bash
docker build -t sai-cns . && docker run -p 8501:8501 sai-cns
```

The image runs as an unprivileged user, reads only from disk, and exposes Streamlit's health
endpoint at `/_stcore/health` so an orchestrator can restart a wedged container. Server settings and
the theme live in `.streamlit/config.toml`.

Before deploying, verify the environment and the application in one step:

```bash
python src/sai_cns/evaluation/app_health.py
```

It checks that every declared dependency resolves and matches its pin, that the registered model
artefacts load, that the knowledge graph is internally consistent, that chemically unrelated
compounds produce distinct and directionally correct profiles, that every export format is well
formed and self-contained, and that no red hue has entered the palette. It exits non-zero on any
failure, so it can gate a release.

## Environment

Python 3.13, RDKit 2026.03.2, scikit-learn 1.8.0, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1,
matplotlib 3.10.9, Plotly 6.7.0. See `requirements.txt`. Random seed 42 throughout.

## Data sources

ChEMBL 37 and BindingDB (target activity), B3DB (BBB; Meng et al., 2021), Therapeutics Data Commons
and MoleculeNet (ADME endpoints), and ChEMBL DPPH radical-scavenging assays (antioxidant). Structures
for user-entered compounds are resolved via PubChem.

## Citation

See [`CITATION.cff`](CITATION.cff) for how to cite the software, and
[`manuscript/NAR_WebServer_SAI-CNS_built.md`](manuscript/NAR_WebServer_SAI-CNS_built.md) for the
manuscript (submitted to *Nucleic Acids Research*, Web Server Issue; a condensed, submission-length
draft is at
[`manuscript/NAR_condensed_built.md`](manuscript/NAR_condensed_built.md)).

## License

See [`LICENSE`](LICENSE). Underlying data retain the licences of their own sources; see the
manuscript's Data Availability section for detail.
