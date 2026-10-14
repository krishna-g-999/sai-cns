"""SAI-CNS: interactive predictor (validated RF + ADME/exposure models).

A single-page Streamlit app over the deployed random-forest models: target engagement (calibrated
probabilities), receptor potency, an hERG safety flag, the ADME / exposure (ADMET) panel, the
directly-modelled unbound brain exposure (K_p,uu) with a free-brain-exposure verdict, BBB-gated
per-disease target-engagement scores, a compound-target-disease network, and an applicability-domain
confidence with the nearest measured analogue. Every number is a model output on measured public data
(228,200 records); this is a research triage tool, not a clinical or diagnostic device.

Visual identity (navy + gold, SAI-Net / SSSIHL) matches the earlier SAI-CNS app; the science is the
current validated model set, now published under the name SAI-CNS. Run:  streamlit run app.py
"""
from __future__ import annotations

import base64
import json
import pickle
import sys
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# RDKit's 2D drawing links against system graphics libraries (libXrender, libXext, libSM) that a
# minimal Linux host may not carry. Streamlit Community Cloud does not, and the resulting ImportError
# happened at module import, so the entire server failed to start over a picture. packages.txt now
# requests those libraries, and this import is guarded as well: every number the tool produces is
# independent of the drawing layer, so losing structure images must degrade one panel rather than
# take down the whole application.
try:
    from rdkit.Chem import Draw
    _DRAW_AVAILABLE = True
except (ImportError, OSError):
    Draw = None
    _DRAW_AVAILABLE = False

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
import panel as _panel  # noqa: E402
from features.featurize import featurize, featurize_one, feature_names  # noqa: E402

MODELS = ROOT / "models_rf"
ASSETS = ROOT / "assets"

TARGET_CLASSIFIERS = {
    "BBB": "Blood-brain barrier penetration",
    "AChE": "Acetylcholinesterase (Alzheimer's)",
    "BChE": "Butyrylcholinesterase (Alzheimer's)",
    "BACE1": "Beta-secretase 1 / amyloid (Alzheimer's)",
    "GSK3B": "GSK-3 beta / tau (Alzheimer's)",
    "MAO_A": "Monoamine oxidase A (mood)",
    "MAO_B": "Monoamine oxidase B (Parkinson's)",
    "hERG": "hERG cardiac-safety liability",
}
LIABILITY_TARGETS = {"hERG"}
RECEPTOR_REGRESSORS = {
    "D2": "Dopamine D2 receptor", "A2A": "Adenosine A2A receptor",
    "HT2A": "Serotonin 5-HT2A receptor", "SERT": "Serotonin transporter",
}
ADME = {
    "kpuu": ("Unbound brain/plasma (Kp,uu)", "ratio", "10^"),
    "logbb": ("Brain penetration (logBB, total)", "log ratio", None),
    "caco2_permeability": ("Passive permeability (Caco-2)", "log Papp cm/s", None),
    "pgp_substrate": ("P-gp substrate (efflux)", "probability", "proba"),
    "pgp_inhibition": ("P-gp inhibition", "probability", "proba"),
    "solubility": ("Aqueous solubility (logS)", "log mol/L", None),
    "lipophilicity": ("Lipophilicity (logD)", "logD", None),
    # The training labels run 10.09 to 99.95, so this endpoint is percent bound, not a fraction.
    # It was labelled "fraction bound", which made every value read as impossible: a fraction cannot
    # be 33. The model was always correct; only the unit was wrong.
    "plasma_protein_binding": ("Plasma-protein binding", "% bound", None),
    "clearance_hepatocyte": ("Hepatocyte clearance", "uL/min/1e6", None),
}
# Target readout kind: 'enrich' = classifier engagement as enrichment over the endpoint base rate
# (a calibrated probability is evidence of engagement only when it exceeds prevalence);
# 'binder' = decoy-aware receptor binder probability; 'pct' = antioxidant/neuroprotection percentile.
# Extended binder-only targets (decoy-aware classifiers), trained from ChEMBL, all binder-heavy.
# Derived from the panel registry, never written by hand. This list was hardcoded until an audit
# found it had drifted from the registry in both directions at once: it still carried GluA2, which
# had been withdrawn and was therefore being scored and allowed to drive an Epilepsy call, and it
# omitted Cav3_2, which is deployed and validated and was never being scored at all. Neither raised
# an error, and the panel-consistency check could not see it because it compares the registry with
# the fitted models, not with this file. Deriving the list closes that gap by construction.
def _deployed_binder_targets():
    """Deployed binder endpoints, less the four that have their own regressor entry above."""
    return sorted(set(_panel.names(deployed=True)) - set(RECEPTOR_REGRESSORS))


BINDER_TARGETS = _deployed_binder_targets()
TARGET_KIND = {"AChE": "enrich", "BChE": "enrich", "BACE1": "enrich", "GSK3B": "enrich",
               "MAO_A": "enrich", "MAO_B": "enrich", "SERT": "binder", "D2": "binder",
               "HT2A": "binder", "NEURO": "pct", **{t: "binder" for t in BINDER_TARGETS}}

# Curated target -> pathway -> disease knowledge graph. Each entry is (pathway, source_id, disease,
# weight).
#
# On the weights: an ablation over the 7,008 compounds of the disease-layer evaluation set found that
# curated, uniform and randomly permuted weights give top-3 disease accuracy of 0.7901, 0.7897 and
# 0.7874 respectively (inversion/results/H2_weight_ablation.csv, n from H1_disease_layer.csv). This
# comment previously reported 0.7917, 0.7911 and 0.7899 over 15,609 compounds, none of which the
# ablation ever used: 15,609 is the pooled-recall denominator of the scaffold hold-out, a different
# experiment, recorded at audit/AUDIT_REPORT.md:824 as 12,325/15,609 = 0.790. That the wrong
# denominator sat beside a right-looking accuracy for so long is explained by the coincidence that
# the two quantities agree to three decimals. Audit item BS-M-04, closed in the manuscript and until
# now left open here. The predictive information is carried by the graph's
# TOPOLOGY, which target connects to which disease, and not by these numbers. They are retained as a
# mechanistic prior expressing which link is the more direct, and they still scale the reported
# score, but they are NOT tuned parameters and must not be described as such.
#
# Pathway and source anchors: KEGG cholinergic (hsa04725), serotonergic (hsa04726) and dopaminergic
# (hsa04728) synapse maps; Alzheimer (hsa05010) and Parkinson (hsa05012) disease maps; Reactome
# KEAP1-NFE2L2 pathway (R-HSA-9755511). Every assignment was checked against the source database's
# own gene annotation.
#
# A disease takes the STRONGEST engaged target rather than an average, so unrelated targets cannot
# dilute a real signal, and the result is then scaled by predicted BBB penetration. That scaling is
# identical across diseases and therefore cannot reorder them; it decides only whether anything
# clears the reporting threshold.
KNOWLEDGE_GRAPH = {
    "AChE":  [("Cholinergic synapse", "hsa04725", "Alzheimer's disease", 1.0),
              ("Cholinergic synapse", "hsa04725", "Cognition (cholinergic)", 1.0)],
    "BChE":  [("Cholinergic (cholinesterase)", "IUPHAR:BChE", "Alzheimer's disease", 0.7),
              ("Cholinergic (cholinesterase)", "IUPHAR:BChE", "Cognition (cholinergic)", 0.8)],
    "BACE1": [("Amyloid-beta processing", "hsa05010", "Alzheimer's disease", 1.0)],
    "GSK3B": [("Tau / GSK-3 signalling", "hsa05010", "Alzheimer's disease", 0.8)],
    "MAO_B": [("Monoamine catabolism", "hsa04728", "Parkinson's disease", 1.0)],
    "MAO_A": [("Monoamine catabolism", "hsa04726", "Depression / anxiety", 0.8)],
    "SERT":  [("Serotonergic synapse", "hsa04726", "Depression / anxiety", 1.0)],
    "D2":    [("Dopaminergic synapse", "hsa04728", "Psychosis / schizophrenia", 1.0)],
    "HT2A":  [("Serotonergic synapse", "hsa04726", "Psychosis / schizophrenia", 0.8)],
    "NEURO": [("Antioxidant / Nrf2 response", "R-HSA-9755511", "Neuroprotection / oxidative stress", 1.0),
              ("Antioxidant / Nrf2 response", "R-HSA-9755511", "Alzheimer's disease", 0.4),
              ("Antioxidant / Nrf2 response", "R-HSA-9755511", "Parkinson's disease", 0.4)],
    # expanded ChEMBL-trained receptor/transporter/kinase targets
    "HT1A":   [("Serotonergic synapse", "hsa04726", "Depression / anxiety", 1.0)],
    "HT6":    [("Serotonergic synapse", "hsa04726", "Cognition (cholinergic)", 0.7)],
    "HT7":    [("Serotonergic synapse", "hsa04726", "Depression / anxiety", 0.7),
               ("Serotonergic synapse", "hsa04726", "Sleep / wakefulness", 0.7)],
    "H3":     [("Histaminergic signalling", "hsa04080", "Sleep / wakefulness", 1.0),
               ("Histaminergic signalling", "hsa04080", "Cognition (cholinergic)", 0.8)],
    "DAT":    [("Dopaminergic synapse", "hsa04728", "Addiction", 1.0),
               ("Dopaminergic synapse", "hsa04728", "ADHD", 0.9)],
    "NET":    [("Noradrenergic signalling", "IUPHAR:NET", "ADHD", 0.9),
               ("Noradrenergic signalling", "IUPHAR:NET", "Depression / anxiety", 0.8)],
    "Sigma1": [("Sigma-1 receptor chaperone", "IUPHAR:Sigma1", "Neuroprotection / oxidative stress", 0.8),
               ("Sigma-1 receptor chaperone", "IUPHAR:Sigma1", "Depression / anxiety", 0.6)],
    "CB1":    [("Retrograde endocannabinoid signalling", "hsa04723", "Chronic pain", 0.8),
               ("Retrograde endocannabinoid signalling", "hsa04723", "Depression / anxiety", 0.6)],
    "OPRK1":  [("Opioid signalling", "IUPHAR:OPRK1", "Chronic pain", 0.9),
               ("Opioid signalling", "IUPHAR:OPRK1", "Depression / anxiety", 0.6)],
    "OPRM1":  [("Opioid signalling", "hsa05032", "Chronic pain", 1.0),
               ("Opioid signalling", "hsa05032", "Addiction", 0.9)],
    "D3":     [("Dopaminergic synapse", "hsa04728", "Addiction", 0.8),
               ("Dopaminergic synapse", "hsa04728", "Psychosis / schizophrenia", 0.7)],
    "A1":     [("Adenosine / purinergic signalling", "hsa04080", "Epilepsy", 0.8),
               ("Adenosine / purinergic signalling", "hsa04080", "Neuroprotection / oxidative stress", 0.6)],
    "a7nAChR": [("Cholinergic synapse", "hsa04725", "Cognition (cholinergic)", 0.8),
                ("Cholinergic synapse", "hsa04725", "Alzheimer's disease", 0.5)],
    "LRRK2":  [("Parkinson disease pathway", "hsa05012", "Parkinson's disease", 1.0)],
    # --- second expansion ---
    # neuroinflammation, a mechanism shared across neurodegeneration
    "NLRP3":  [("Neuroinflammation (inflammasome)", "hsa04621", "Neuroinflammation", 1.0),
               ("Neuroinflammation (inflammasome)", "hsa04621", "Amyotrophic lateral sclerosis", 0.7),
               ("Neuroinflammation (inflammasome)", "hsa04621", "Alzheimer's disease", 0.4)],
    "P2X7":   [("Neuroinflammation (inflammasome)", "hsa04621", "Neuroinflammation", 0.9),
               ("Neuroinflammation (inflammasome)", "hsa04621", "Amyotrophic lateral sclerosis", 0.6)],
    "COX2":   [("Prostaglandin inflammation", "hsa00590", "Neuroinflammation", 0.8)],
    "CSF1R":  [("Cytokine receptor signalling", "hsa04060", "Neuroinflammation", 0.9),
               ("Cytokine receptor signalling", "hsa04060", "Amyotrophic lateral sclerosis", 0.6)],
    # Huntington's disease
    "PDE10A": [("Striatal cyclic-nucleotide signalling", "hsa04024", "Huntington's disease", 1.0)],
    "HDAC1":  [("Histone acetylation", "hsa05016", "Huntington's disease", 0.8)],
    "HDAC6":  [("Histone deacetylation", "hsa05014", "Amyotrophic lateral sclerosis", 0.7)],
    # excitotoxicity: ALS (riluzole axis), epilepsy, ketamine-type antidepressant action
    "GluN2B": [("Glutamatergic excitotoxicity", "hsa05014", "Amyotrophic lateral sclerosis", 0.9),
               ("Glutamatergic excitotoxicity", "hsa04724", "Epilepsy", 0.8),
               ("Glutamatergic excitotoxicity", "hsa04724", "Depression / anxiety", 0.6)],
    "mGluR5": [("Glutamatergic excitotoxicity", "hsa04724", "Epilepsy", 0.7),
               ("Glutamatergic excitotoxicity", "hsa04724", "Depression / anxiety", 0.5)],
    "GABA_A": [("GABAergic inhibition", "hsa04727", "Epilepsy", 1.0),
               ("GABAergic inhibition", "hsa04727", "Depression / anxiety", 0.7),
               ("GABAergic inhibition", "hsa04727", "Sleep / wakefulness", 0.7)],
    # sleep and circadian
    "OX1":    [("Orexin signalling", "hsa04080", "Sleep / wakefulness", 0.9)],
    "OX2":    [("Orexin signalling", "hsa04080", "Sleep / wakefulness", 1.0)],
    "MT1":    [("Melatonergic / circadian", "hsa04713", "Sleep / wakefulness", 0.9)],
    # proteostasis, oxidative stress: ALS and general neuroprotection
    "mTOR":   [("Autophagy and proteostasis", "hsa05014", "Amyotrophic lateral sclerosis", 0.8),
               ("Autophagy and proteostasis", "hsa04140", "Neuroprotection / oxidative stress", 0.7)],
    "SIRT1":  [("Antioxidant / Nrf2 response", "R-HSA-9755511", "Neuroprotection / oxidative stress", 0.7)],
    "KEAP1":  [("Antioxidant / Nrf2 response", "R-HSA-9755511", "Neuroprotection / oxidative stress", 0.9),
               ("Antioxidant / Nrf2 response", "R-HSA-9755511", "Amyotrophic lateral sclerosis", 0.7)],
    # Parkinson genetics
    "GBA1":   [("Lysosomal function", "hsa04142", "Parkinson's disease", 0.8)],
    # cognition
    "PDE4B":  [("Cyclic-AMP signalling", "hsa04024", "Cognition (cholinergic)", 0.6),
               ("Cyclic-AMP signalling", "hsa04024", "Neuroinflammation", 0.5)],
    # --- third expansion ---
    # Nav1.7 is the principal peripheral pain sodium channel; Nav1.1 underlies genetic epilepsy
    "Nav1_7": [("Voltage-gated sodium channels", "IUPHAR:Nav1.7", "Chronic pain", 0.9)],
    # --- fourth expansion: the epilepsy and pain mechanisms the panel was missing ---
    # Added because the clinical-indication test (inversion H6) measured epilepsy and chronic pain as
    # the two conditions served worst, recovering the approved indication for 10 and 23 percent of
    # drugs, and the data audit found these clear both the volume and the source-diversity bar.
    # Nav1.1 previously carried the epilepsy sodium-channel edge and has been withdrawn: it scored
    # every trivial metabolite at 0.80 to 0.82 against a 0.796 threshold (see
    # results/deployed_specificity_audit.csv). Nav1.6 replaces it with a random-chemistry
    # false-positive rate of 0.000.
    # KEGG membership verified: hsa04725 contains CHRNA3, CHRNA4, CHRNB2 and CHRNB4; hsa05033
    # contains CHRNA4 and CHRNB2 but NOT CHRNA3 or CHRNB4, so the alpha3beta4 addiction edge is
    # cited to the receptor annotation rather than to a pathway it does not belong to.
    "Nav1_6": [("Voltage-gated sodium channels", "IUPHAR:Nav1.6", "Epilepsy", 0.85)],
    "Nav1_8": [("Voltage-gated sodium channels", "IUPHAR:Nav1.8", "Chronic pain", 0.85)],
    "a4b2nAChR": [("Cholinergic synapse", "hsa04725", "Cognition (cholinergic)", 0.7),
                  ("Nicotine addiction", "hsa05033", "Addiction", 0.8)],
    "a3b4nAChR": [("Cholinergic synapse", "hsa04725", "Cognition (cholinergic)", 0.4),
                  ("Ganglionic nicotinic signalling", "IUPHAR:alpha3beta4", "Addiction", 0.7)],
    # --- fifth expansion: two conditions that were absent for want of an endpoint, not of data ---
    # The coverage audit found migraine and multiple sclerosis were listed as unmodelled while both
    # have well-measured targets. Amyotrophic lateral sclerosis was already scored but through seven
    # general mechanisms with nothing specific to the disease; RIPK1 closes that.
    # KEGG membership verified: hsa04080 contains CALCRL, hsa00240 contains DHODH, and both hsa04217
    # (necroptosis) and hsa04668 (TNF signalling) contain RIPK1.
    "CGRP":  [("Neuroactive ligand-receptor interaction", "hsa04080", "Migraine", 0.9),
              ("Trigeminovascular signalling", "IUPHAR:CGRP", "Chronic pain", 0.5)],
    "DHODH": [("Pyrimidine metabolism", "hsa00240", "Multiple sclerosis", 0.9)],
    "RIPK1": [("Necroptosis", "hsa04217", "Amyotrophic lateral sclerosis", 0.7),
              ("TNF signalling", "hsa04668", "Neuroinflammation", 0.8),
              ("Necroptosis", "hsa04217", "Multiple sclerosis", 0.6)],
    # TAAR1 agonism is the mechanism of the non-D2 antipsychotic class
    "TAAR1":  [("Trace-amine signalling", "IUPHAR:TAAR1", "Psychosis / schizophrenia", 0.7)],
    # AMPA receptor modulation in excitotoxicity and seizure control
    # GluA2 held an Epilepsy edge here while being withdrawn from the panel, so a model the
    # project had judged unfit to deploy was still able to drive a reported condition. The
    # edge is removed rather than left inert, because a withdrawn endpoint has no business
    # claiming a mechanism. Epilepsy retains GABA_A (1.0), Nav1_6 (0.85), A1 (0.8), GluN2B
    # (0.8) and mGluR5 (0.7), every one weighted above the 0.6 this carried, and the score is
    # a maximum, so removing it can only lower a score it seldom drove.
}
DISEASE_ORDER = ["Alzheimer's disease", "Parkinson's disease", "Amyotrophic lateral sclerosis",
                 "Huntington's disease", "Depression / anxiety", "Psychosis / schizophrenia",
                 "Cognition (cholinergic)", "Addiction", "ADHD", "Chronic pain",
                 "Sleep / wakefulness", "Epilepsy", "Migraine", "Multiple sclerosis",
                 "Neuroinflammation", "Neuroprotection / oxidative stress"]
# derived: disease -> [(target, weight)]
DISEASE_CONTRIB = {}
for _t, _entries in KNOWLEDGE_GRAPH.items():
    for _pw, _src, _dis, _w in _entries:
        DISEASE_CONTRIB.setdefault(_dis, []).append((_t, _w))

MECH_LABEL = {"AChE": "AChE", "BChE": "BChE", "BACE1": "BACE1", "GSK3B": "GSK-3β",
              "MAO_A": "MAO-A", "MAO_B": "MAO-B", "NEURO": "antioxidant",
              "D2": "D2", "HT2A": "5-HT2A", "SERT": "SERT", "A2A": "A2A",
              "HT1A": "5-HT1A", "HT6": "5-HT6", "HT7": "5-HT7", "H3": "H3", "DAT": "DAT",
              "NET": "NET", "Sigma1": "Sigma-1", "CB1": "CB1", "OPRK1": "kappa-opioid",
              "OPRM1": "mu-opioid", "D3": "D3", "A1": "A1", "a7nAChR": "alpha7-nAChR", "LRRK2": "LRRK2",
              "NLRP3": "NLRP3", "P2X7": "P2X7", "COX2": "COX-2", "CSF1R": "CSF1R", "PDE10A": "PDE10A",
              "HDAC1": "HDAC1", "HDAC6": "HDAC6", "GluN2B": "NMDA GluN2B", "mGluR5": "mGluR5",
              "GABA_A": "GABA-A", "OX1": "orexin OX1", "OX2": "orexin OX2", "MT1": "melatonin MT1",
              "mTOR": "mTOR", "SIRT1": "SIRT1", "KEAP1": "KEAP1/Nrf2", "GBA1": "GBA1", "PDE4B": "PDE4B",
              "Nav1_5": "Nav1.5", "Nav1_7": "Nav1.7", "TAAR1": "TAAR1",
              "GluA2": "AMPA GluA2", "Nav1_6": "Nav1.6", "Nav1_8": "Nav1.8",
              "a4b2nAChR": "alpha4beta2-nAChR", "a3b4nAChR": "alpha3beta4-nAChR",
              "CGRP": "CGRP receptor", "DHODH": "DHODH", "RIPK1": "RIPK1",
              # Cav3_2 and hERG were served without a display label, so the interface printed their
              # internal identifiers beside properly typeset names like Nav1.5. GluA2's label is
              # retained deliberately: the endpoint is withdrawn and no longer scored, but a stored
              # report or exported CSV from before the withdrawal still names it, and a label map
              # that forgets it would render those as a bare identifier.
              "Cav3_2": "Cav3.2", "hERG": "hERG", "BBB": "blood-brain barrier",
              # Withdrawn endpoints still appear in the scope list, which names them, so they need
              # labels too; without these the panel printed Nav1_1 beside AMPA GluA2.
              "Nav1_1": "Nav1.1", "NFKB1": "NF-kB p105/p50", "NR3C1": "glucocorticoid receptor",
              "NRF2": "NRF2"}

# ---- SAI-CNS visual identity (navy + gold, matching the earlier app) ----
NAVY   = "#0D2137"
NAVY_DK = "#071626"
NAVY_MD = "#1A3A5C"
GOLD   = "#F0A500"
# Tab bar, in the brand gold that the wordmark and the logo already use, so the page carries one
# accent colour rather than three. The label colour is measured rather than assumed: white on this
# gold is 2.08:1, which is not readable at any size, while the navy ink is 7.82:1. The resting tabs
# therefore take the navy, and the selected tab inverts to a navy chip with white type at 16.3:1,
# which both marks the active page unmistakably and puts white type where it is legible.
TAB_BAR = GOLD          # navy on this: 7.82:1
TAB_ACTIVE = NAVY       # white on this: 16.28:1
BG     = "#EEF2F9"
SURF   = "#FFFFFF"
LINE   = "#E6ECF5"
CARDBD = "#E2E8F2"
INK    = "#0D2137"
MUTE   = "#5A6B82"
MUTE2  = "#94A3B8"
# Status palette. There is deliberately no red anywhere in this interface. Red-green is the
# commonest form of colour-vision deficiency, affecting roughly one man in twelve, and a scientific
# result that changes meaning depending on who reads it is a defect. The adverse state is carried by
# a deep violet, which separates cleanly from both the green and the amber under deuteranopia and
# protanopia, and every status is additionally labelled in words so that colour is never the only
# channel carrying the information.
GREEN  = "#15803D"   # favourable / high engagement / in-domain
AMBER  = "#B45309"   # borderline / caution
ADVERSE = "#6B21A8"  # limited exposure / safety liability / out of domain
BLUE   = "#1D4ED8"   # engagement accent

_DESC_NAMES = feature_names()[-12:]
# The applicability-domain fingerprint is defined once, in src/sai_cns/ad_fingerprint.py, and
# imported here rather than restated. It used to be restated in twelve places, all of them
# stereochemistry-blind, which is how a stereoisomer of a reference compound came to be reported
# as that compound at maximum similarity.
import ad_fingerprint as _adfp  # noqa: E402
_AD_GEN = _adfp.generator()


# --------------------------------- models / inference ---------------------------------
@st.cache_resource
def _ensure_models_present():
    """Fetch the published model archive if this deployment does not carry the binaries.

    The models are 0.78 GB and are not in the repository, so a container built from a clone has the
    code and none of the science. This is a no-op on any tree that already has them, and it verifies
    every file against a recorded checksum before returning, because a partial download would
    otherwise surface as predictions that are quietly wrong rather than as an error."""
    try:
        from model_fetch import ensure_models
        return ensure_models(verbose=True)
    except Exception:
        return True          # never let the fetch layer break a working local deployment


@st.cache_resource
def load_models():
    if not _ensure_models_present():
        st.error("Model files are unavailable and could not be downloaded. The server cannot "
                 "produce predictions. See models_manifest.json.")
        st.stop()
    m = {}
    for ep in TARGET_CLASSIFIERS:
        cal = MODELS / f"{ep}_calibrated.joblib"
        m[ep] = joblib.load(cal if cal.exists() else MODELS / f"{ep}.joblib")
    for ep in RECEPTOR_REGRESSORS:
        m[ep] = joblib.load(MODELS / f"{ep}.joblib")
        bp = MODELS / f"{ep}_binder.joblib"
        if bp.exists():
            m[f"{ep}_binder"] = joblib.load(bp)
    for ep in BINDER_TARGETS:
        bp = MODELS / f"{ep}_binder.joblib"
        if bp.exists():
            m[f"{ep}_binder"] = joblib.load(bp)
    m["antioxidant_DPPH"] = joblib.load(MODELS / "antioxidant_DPPH.joblib")
    for ep in ADME:
        m[f"adme_{ep}"] = joblib.load(MODELS / "adme" / f"{ep}.joblib")
    return m


@st.cache_resource
def load_ad_reference():
    """(smiles, fingerprints) of measured training chemistry for the applicability domain.

    The reference carries the fingerprint spec it was built with, and a mismatch against the spec
    this process would query it with is refused rather than tolerated. Querying a reference with a
    different fingerprint does not fail: it returns similarities that are quietly wrong, which is
    the worst of the available outcomes.
    """
    path = MODELS / "ad_reference.pkl"
    if not path.exists():
        return None
    with path.open("rb") as fh:
        obj = pickle.load(fh)
    # A two-element reference predates the spec field, which means it was necessarily built with
    # the legacy stereochemistry-blind fingerprint. That is a known state, not an unknown one, so
    # it is read as LEGACY_SPEC rather than refused.
    if len(obj) == 3:
        ref_smiles, ref_fps, spec = obj
    else:
        ref_smiles, ref_fps = obj[0], obj[1]
        spec = _adfp.LEGACY_SPEC
    if spec != _adfp.SPEC:
        st.error(f"The applicability-domain reference was built with fingerprint '{spec}' but this "
                 f"server computes '{_adfp.SPEC}'. Querying a reference with a different "
                 f"fingerprint returns similarities that are wrong rather than missing, so the "
                 f"domain is disabled. Rebuild it with src/sai_cns/build_ad_reference.py and "
                 f"republish the model archive; see src/sai_cns/ad_fingerprint.py.")
        return None
    return ref_smiles, ref_fps


@st.cache_resource
def load_ad_per_endpoint():
    """Per-endpoint fingerprints of measured training chemistry (endpoint-specific domain)."""
    p = MODELS / "ad_per_endpoint.pkl"
    if not p.exists():
        return None
    with p.open("rb") as fh:
        return pickle.load(fh)


@st.cache_resource
def load_context():
    """Per-endpoint training context: classifier base rates + regressor distributions."""
    p = MODELS / "endpoint_context.json"
    return json.loads(p.read_text()) if p.exists() else {"classifiers": {}, "regressors": {}}


def base_rate(ep):
    return load_context().get("classifiers", {}).get(ep, {}).get("base_rate")


def enrichment(ep, p):
    """Signed enrichment of a calibrated probability over the endpoint base rate, in [-1, 1].
    +1 = certain given a rare label; 0 = exactly at base rate; negative = below chance."""
    br = base_rate(ep)
    if br is None:
        return (p - 0.5) * 2
    return (p - br) / (1 - br) if p >= br else (p - br) / br


def engaged_signal(ep, p):
    """Positive engagement signal: how far a prediction sits ABOVE the base rate (0 if at/below)."""
    return max(0.0, enrichment(ep, p))


def reg_percentile(ep, val):
    """Percentile of a regression prediction against the measured training distribution."""
    q = load_context().get("regressors", {}).get(ep)
    if not q:
        return None
    pts = sorted((float(k), v) for k, v in q["quantiles"].items())
    xs = [q["min"]] + [v for _, v in pts] + [q["max"]]
    ys = [0.0] + [pr for pr, _ in pts] + [1.0]
    return float(np.clip(np.interp(val, xs, ys), 0.0, 1.0))


def neuro_signal(r):
    """Neuroprotection/antioxidant signal: 0 at median antioxidant capacity, ~1 by the 90th percentile."""
    pct = reg_percentile("antioxidant", r["antioxidant"])
    return 0.0 if pct is None else float(np.clip((pct - 0.5) / 0.4, 0.0, 1.0))


def predict_all(smiles, m):
    x = featurize_one(smiles)
    if x is None:
        return None
    xr = x.reshape(1, -1)
    out = {"targets": {}, "receptors": {}, "receptor_binder": {}, "adme": {}, "descriptors": {}}
    for ep in TARGET_CLASSIFIERS:
        out["targets"][ep] = float(m[ep].predict_proba(xr)[0, 1])
    for ep in RECEPTOR_REGRESSORS:
        out["receptors"][ep] = float(m[ep].predict(xr)[0])
        if f"{ep}_binder" in m:
            out["receptor_binder"][ep] = float(m[f"{ep}_binder"].predict_proba(xr)[0, 1])
    for ep in BINDER_TARGETS:
        if f"{ep}_binder" in m:
            out["receptor_binder"][ep] = float(m[f"{ep}_binder"].predict_proba(xr)[0, 1])
    out["antioxidant"] = float(m["antioxidant_DPPH"].predict(xr)[0])
    for ep, (_, _, tf) in ADME.items():
        v = float(m[f"adme_{ep}"].predict(xr)[0]) if tf != "proba" else float(m[f"adme_{ep}"].predict_proba(xr)[0, 1])
        out["adme"][ep] = 10 ** v if tf == "10^" else v
    out["descriptors"] = {name: float(v) for name, v in zip(_DESC_NAMES, x[-12:])}
    return out


def predict_all_batch(smiles_list, m, chunk=512):
    """predict_all for many compounds at once, returning one dict per input or None.

    Identical arithmetic to predict_all, which it is checked against: a random forest scores each
    row independently, so stacking N rows into one matrix and calling each estimator once gives the
    same numbers as calling it N times with one row.

    What changes is the overhead. The per-compound path pays scikit-learn's call dispatch, array
    validation and tree-traversal setup once per estimator PER COMPOUND, which is about seventy
    times per compound across the panel. Batched, that cost is paid seventy times for the whole
    run. Measured on this panel the per-compound path runs at about 4.1 s/compound, which puts a
    300-compound batch, the limit the interface advertises, at roughly twenty minutes; long enough
    that a reviewer running a test set would hit a timeout before seeing a result.

    Compounds that fail to featurise keep their place in the output as None, so the caller can zip
    results back to inputs without tracking indices.
    """
    smiles_list = list(smiles_list)
    results = [None] * len(smiles_list)

    for start in range(0, len(smiles_list), chunk):
        block = smiles_list[start:start + chunk]
        X, mask = featurize(block)
        if X is None or len(X) == 0:
            continue
        mask = np.asarray(mask)
        idx = np.flatnonzero(mask)                 # positions within this block that parsed
        n = len(idx)
        if n == 0:
            continue

        cols = {}
        for ep in TARGET_CLASSIFIERS:
            cols[("targets", ep)] = m[ep].predict_proba(X)[:, 1]
        for ep in RECEPTOR_REGRESSORS:
            cols[("receptors", ep)] = m[ep].predict(X)
            if f"{ep}_binder" in m:
                cols[("receptor_binder", ep)] = m[f"{ep}_binder"].predict_proba(X)[:, 1]
        for ep in BINDER_TARGETS:
            if f"{ep}_binder" in m:
                cols[("receptor_binder", ep)] = m[f"{ep}_binder"].predict_proba(X)[:, 1]
        anti = m["antioxidant_DPPH"].predict(X)
        adme_cols = {}
        for ep, (_, _, tf) in ADME.items():
            v = (m[f"adme_{ep}"].predict_proba(X)[:, 1] if tf == "proba"
                 else m[f"adme_{ep}"].predict(X))
            adme_cols[ep] = (10 ** v) if tf == "10^" else v

        for j in range(n):
            out = {"targets": {}, "receptors": {}, "receptor_binder": {}, "adme": {},
                   "descriptors": {}}
            for (grp, ep), arr in cols.items():
                out[grp][ep] = float(arr[j])
            out["antioxidant"] = float(anti[j])
            for ep, arr in adme_cols.items():
                out["adme"][ep] = float(arr[j])
            out["descriptors"] = {name: float(v)
                                  for name, v in zip(_DESC_NAMES, X[j][-12:])}
            results[start + int(idx[j])] = out
    return results


@st.cache_data(show_spinner=False)
def _recall_bands():
    """Measured recall as a function of distance from training chemistry.

    The prospective validation (section 6.5 of the technical report) found that this panel's recall
    depends on how far a compound sits from the chemistry it was fitted to, and not on when that
    compound was published: conditioned on distance, a time-split and a random-split test set behave
    the same. Because the distance is computed for every query anyway, the expected sensitivity for
    the compound in hand is knowable at the moment it is submitted, rather than only as a panel
    average measured on a held-out set that is four fifths close analogues.

    Read from the artefact rather than restated here, so the figure the interface shows and the
    figure the report prints cannot drift apart. Returns [] if the table is absent, in which case
    the interface says nothing rather than guessing.
    """
    p = ROOT / "results" / "tables" / "external_novelty_strata.csv"
    if not p.exists():
        return []
    try:
        d = pd.read_csv(p)
        d = d[(d.source == "prospective") & (d.split == "time") & d.recall_at_threshold.notna()]
        out = []
        for r in d.itertuples():
            lo = float(str(r.novelty_band).split()[0]) if "below" not in str(r.novelty_band) else 0.0
            out.append((lo, float(r.recall_at_threshold), int(r.n_actives)))
        return sorted(out)
    except Exception:
        return []


def expected_recall(max_sim: float):
    """The measured recall band this query falls in, or None if the artefact is unavailable."""
    bands = _recall_bands()
    if not bands:
        return None
    lo, rec, n = bands[0]
    for b_lo, b_rec, b_n in bands:
        if max_sim >= b_lo:
            lo, rec, n = b_lo, b_rec, b_n
    return {"recall": rec, "n": n, "band_low": lo}


def assess_domain(smiles):
    """Max Tanimoto to measured training chemistry + the nearest analogue (applicability domain)."""
    ref = load_ad_reference()
    if ref is None:
        return None
    ref_smiles, ref_fps = ref
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    q = _AD_GEN.GetFingerprint(mol)
    sims = DataStructs.BulkTanimotoSimilarity(q, ref_fps)
    i = int(np.argmax(sims))
    mx = float(sims[i])
    tier = ("In domain", GREEN, "high") if mx >= 0.5 else \
           ("Near domain", AMBER, "moderate") if mx >= 0.3 else \
           ("Out of domain", ADVERSE, "low")
    return {"max_sim": mx, "nearest_smiles": ref_smiles[i], "n_ref": len(ref_smiles),
            "tier": tier[0], "colour": tier[1], "confidence": tier[2],
            "expected_recall": expected_recall(mx)}


def assess_per_endpoint(smiles):
    """Max Tanimoto of the query to each endpoint's own training chemistry (endpoint-specific domain)."""
    ref = load_ad_per_endpoint()
    if ref is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    q = _AD_GEN.GetFingerprint(mol)
    return {ep: float(max(DataStructs.BulkTanimotoSimilarity(q, fps))) for ep, fps in ref.items() if fps}


# Homologous target families, and how often their members fire together.
#
# Measured on 400 approved drugs (inversion/results/H8_family_correlation.csv). The panel reports an
# engaged-target count and draws one network edge per target, which invites reading two engaged
# targets as two independent findings. For homologues that is wrong: across the drug set 36 targets
# ever fire but they span only 14 independent directions, so the panel is roughly 2.6-fold redundant.
# Co-firing is not itself an error, since a genuinely promiscuous ligand should hit both homologues
# and that is a true fact about the compound. Presenting it as corroboration is the error, so where
# two members of a family are engaged the interface says so and the measured conditional probability
# is quoted rather than left for the reader to assume independence.
TARGET_FAMILIES = {
    "Dopamine D2-like": ["D2", "D3"],
    "Opioid": ["OPRM1", "OPRK1"],
    "Monoamine transporters": ["SERT", "DAT", "NET"],
    "Serotonin receptors": ["HT1A", "HT2A", "HT6", "HT7"],
    "Neuronal nicotinic": ["a7nAChR", "a4b2nAChR", "a3b4nAChR"],
    "Voltage-gated sodium": ["Nav1_5", "Nav1_6", "Nav1_7", "Nav1_8"],
}
_TARGET_TO_FAMILY = {t: f for f, ts in TARGET_FAMILIES.items() for t in ts}

# Below this many approved drugs engaging both targets, a conditional probability is noise. Without
# the guard the nicotinic family reports "fires for 100% of compounds", from one compound.
MIN_JOINT_DRUGS = 10


@st.cache_data(show_spinner=False)
def family_cofire():
    """Strongest measured co-firing pair per family, read from the falsification suite's artefact.

    These figures were written down, and had drifted from the artefact on five of five correlations
    and three of four conditional probabilities. The monoamine entry was the clearest failure: it
    displayed DAT to NET at r = 0.64 and 84%, where the artefact measures that pair at 0.425 and
    61.8%, and where the family's strongest pair is not DAT to NET at all but SERT to NET at 0.732.
    A badge reading "correlated, r = 0.81" renders beside two engaged targets, so the reader deciding
    how much independent evidence the pair represents was given a number no artefact supported.

    Deriving them ends the class of defect rather than the instance: the numbers cannot drift from
    inversion/results/H8_family_correlation.csv because they are read from it.

    Returns {family: (phi, note)}. A family with no adequately supported pair returns (None, note),
    which suppresses the badge and says why.
    """
    src = ROOT / "inversion" / "results" / "H8_family_correlation.csv"
    out = {}
    try:
        rows = pd.read_csv(src).to_dict("records")
    except Exception:
        return {}
    best = {}
    for r in rows:
        fam = _TARGET_TO_FAMILY.get(str(r.get("target_a")))
        if fam is None or fam != _TARGET_TO_FAMILY.get(str(r.get("target_b"))):
            continue
        phi, joint = r.get("phi_correlation"), r.get("n_joint")
        # An artefact without the count columns predates the guard; treat support as unknown and
        # decline to report, rather than repeating the one-compound certainty this guard exists for.
        if phi is None or phi != phi or joint is None or joint != joint:
            continue
        if int(joint) < MIN_JOINT_DRUGS:
            continue
        if fam not in best or float(phi) > float(best[fam].get("phi_correlation", -1)):
            best[fam] = r
    for fam in TARGET_FAMILIES:
        r = best.get(fam)
        if r is None:
            out[fam] = (None, "too few approved drugs engage both members to measure co-firing")
            continue
        a, b = str(r["target_a"]), str(r["target_b"])
        out[fam] = (float(r["phi_correlation"]),
                    f'{MECH_LABEL.get(b, b)} fires for {float(r["p_b_given_a"]):.0%} of the '
                    f'{int(r["n_a"])} approved drugs engaging {MECH_LABEL.get(a, a)}')
    return out


def independent_mechanisms(engaged):
    """Count distinct mechanism families among engaged targets, not raw targets.

    Two homologous receptors engaged by the same ligand are close to one observation. Counting
    families rather than targets is a blunt correction, but it errs towards understating the
    evidence, which is the safer direction for a tool whose purpose is to avoid wasted experiments.
    """
    fams, singles = set(), 0
    for t in engaged:
        f = _TARGET_TO_FAMILY.get(t)
        if f:
            fams.add(f)
        else:
            singles += 1
    return len(fams) + singles


def grouped_engagement(engaged):
    """Engaged targets arranged by family, with the families that carry more than one member."""
    by_fam = {}
    for t in engaged:
        by_fam.setdefault(_TARGET_TO_FAMILY.get(t, ""), []).append(t)
    multi = {f: ts for f, ts in by_fam.items() if f and len(ts) > 1}
    return by_fam, multi


BINDER_FLOOR = 0.40  # calibrated binder probability below this is treated as no engagement

# Minimum BBB-gated score at which a disease signal is reported as actionable. Set from the
# prospective external test: at 0.30 every peripheral negative control (losartan, allopurinol,
# omeprazole, insulin glargine and the rest) falls silent while genuine signals are retained.
# Below this value a score reflects weak, non-specific engagement and is not shown as a finding.
MIN_ACTIONABLE_SCORE = 0.30

# The smallest engagement signal reported as an engaged target, anywhere in the interface.
#
# This was a local constant inside the mechanism map while every other view filtered on `> 0`, so
# one compound reported different mechanisms depending on which panel a reader opened: the map drew
# five targets for haloperidol, and the batch table drew six, the extra one a 5-HT7 signal of 0.02.
# The export field was labelled "targets with a signal above threshold" while applying a threshold
# of zero. A signal this small is a probability that has barely cleared its own calibrated cut, and
# presenting it beside a 0.93 as though the two were comparable findings overstates it.
#
# Disease scoring deliberately does NOT apply this floor. It consumes the raw signal and is gated
# afterwards by MIN_ACTIONABLE_SCORE, so a weak mechanism can still contribute to a condition
# without being listed as an engaged target in its own right.
MIN_ENGAGEMENT_SIGNAL = 0.06


@st.cache_resource
def load_binder_modes():
    """How each binder-panel model was trained: against property-matched decoys, or against
    experimentally measured inactives. The two produce probabilities on different scales and must
    not be thresholded identically."""
    p = MODELS / "binder_modes.json"
    return json.loads(p.read_text()) if p.exists() else {}


def target_signal(r, neuro, tgt):
    """Engagement signal for one target, used for disease scoring and the mechanism map.

    Three readouts, each converted to a common 0-1 engagement scale:
      * measured-label classifiers (including the core target panel and the four binder-panel
        models trained against real measured inactives) use ENRICHMENT over the endpoint base rate,
        because a probability below prevalence is evidence of inactivity, not of engagement;
      * decoy-aware models use the calibrated binder probability above a floor, so sub-threshold
        off-target calls within the measured background false-positive rate contribute nothing;
      * the antioxidant model uses its percentile against measured chemistry.
    The displayed tables always show the raw calibrated probability, not this transformed signal."""
    kind = TARGET_KIND.get(tgt, "enrich")
    if kind == "pct":
        return neuro
    # A target named in the graph but carrying no deployed model contributes nothing. This cannot
    # happen while the registry and the graph agree, and the test suite pins that they do; the guard
    # exists so that a future withdrawal degrades to silence rather than to a KeyError mid-query.
    if kind == "enrich" and tgt not in r.get("targets", {}):
        return 0.0
    if kind == "binder":
        p = r.get("receptor_binder", {}).get(tgt, 0.0)
        thr = binder_threshold(tgt)
        return max(0.0, (p - thr) / (1.0 - thr)) if p >= thr else 0.0
    return engaged_signal(tgt, r["targets"][tgt])


def _ramp(x, hi_good, lo_good):
    """Monotonically decreasing desirability: 1 at or below hi_good, 0 at or above lo_good."""
    if x <= hi_good:
        return 1.0
    if x >= lo_good:
        return 0.0
    return float((lo_good - x) / (lo_good - hi_good))


@st.cache_resource
def load_pka_model():
    """Most-basic-pKa regressor, loaded only if it passed its deployment threshold."""
    p, mp = MODELS / "pka_basic.joblib", MODELS / "pka_basic_meta.json"
    if not (p.exists() and mp.exists()):
        return None, None
    meta = json.loads(mp.read_text())
    if not meta.get("deployed"):
        return None, meta
    return joblib.load(p), meta


def predict_pka(smiles):
    """Predicted most basic pKa, or None when no model is deployed or the structure has no basic
    centre (in which case the CNS MPO pKa term is satisfied by definition)."""
    mdl, _ = load_pka_model()
    if mdl is None:
        return None
    x = featurize_one(smiles)
    if x is None:
        return None
    return float(mdl.predict(x.reshape(1, -1))[0])


def cns_mpo(r, smiles=None):
    """CNS multiparameter optimisation desirability, after Wager et al. (ACS Chem Neurosci 2010).

    The published score sums six desirability functions, each mapping a physicochemical property to
    [0, 1]: cLogP, cLogD at pH 7.4, molecular weight, topological polar surface area, hydrogen-bond
    donor count, and the most basic pKa. Higher totals associate with better CNS exposure.

    cLogP, molecular weight, TPSA and donor count come from the RDKit descriptors already in the
    feature vector; cLogD is taken from the measured-data lipophilicity model rather than a rule of
    thumb. The most basic pKa comes from a model trained here on 6,384 ChEMBL compounds that carry a
    basic centre and no acid indication, which reaches a scaffold-split R2 of 0.456 and a mean
    absolute error of 1.17 pKa units. That error is not negligible against a desirability transition
    that runs from 8 to 10, so the pKa term is the least certain of the six and is reported as such.
    When no pKa model is deployed the score falls back to the five-component form on a 0 to 5 scale.
    """
    d = r.get("descriptors", {})
    mw, clogp = d.get("mw"), d.get("clogp")
    tpsa, hbd = d.get("tpsa"), d.get("hbd")
    logd = r.get("adme", {}).get("lipophilicity")
    if None in (mw, clogp, tpsa, hbd) or logd is None:
        return None
    # TPSA is the one non-monotonic term: desirable in a window, penalised on both sides.
    if tpsa < 20:
        t_tpsa = 0.0
    elif tpsa < 40:
        t_tpsa = (tpsa - 20) / 20.0
    elif tpsa <= 90:
        t_tpsa = 1.0
    else:
        t_tpsa = _ramp(tpsa, 90, 120)
    parts = {
        "cLogP": _ramp(clogp, 3.0, 5.0),
        "cLogD (predicted)": _ramp(logd, 2.0, 4.0),
        "MW": _ramp(mw, 360.0, 500.0),
        "TPSA": t_tpsa,
        "HBD": _ramp(hbd, 0.5, 3.5),
    }
    pka = predict_pka(smiles) if smiles else None
    if pka is not None:
        parts["most basic pKa (predicted)"] = _ramp(pka, 8.0, 10.0)
    return {"total": float(sum(parts.values())), "max": float(len(parts)),
            "parts": parts, "pka": pka}


@st.cache_resource
def load_readacross():
    """Measured actives with fingerprints and the targets they hit, for similarity read-across."""
    p = MODELS / "readacross_index.pkl"
    if not p.exists():
        return None
    with p.open("rb") as fh:
        return pickle.load(fh)


def read_across(smiles, k=5, min_sim=0.35):
    """Nearest measured actives and what they are active at.

    Evidence about the neighbours, not about the query. Similarity is returned with every hit so the
    user can weigh it, and nothing is returned below min_sim, where structural inference is not
    defensible."""
    idx = load_readacross()
    if idx is None:
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    q = _AD_GEN.GetFingerprint(mol)
    sims = np.array(DataStructs.BulkTanimotoSimilarity(q, idx["fps"]))
    order = np.argsort(-sims)[:k]
    return [{"smiles": idx["smiles"][i], "similarity": float(sims[i]), "targets": idx["targets"][i]}
            for i in order if sims[i] >= min_sim]


READ_ACROSS_POOL = 60      # neighbours considered when accumulating evidence per target
READ_ACROSS_MIN_SIM = 0.35


def read_across_targets(smiles, pool=READ_ACROSS_POOL, min_sim=READ_ACROSS_MIN_SIM):
    """Aggregate read-across evidence per target, weighting by how many neighbours support it.

    A single neighbour at Tanimoto 0.70 is weaker evidence than five independent neighbours at 0.55,
    yet a list of nearest neighbours presents them alike. Each neighbour active at a target is
    therefore treated as an independent, imperfect observation and combined with a noisy-OR:

        evidence(T) = 1 - product over supporting neighbours of (1 - s_i)

    where s_i is the neighbour's Tanimoto similarity. The form has the properties the problem
    requires: it is monotonic in both similarity and neighbour count, it cannot exceed 1, and one
    very close neighbour can still carry a target on its own. Supporting counts are returned so the
    interface can show whether a score rests on one analogue or on many.
    """
    idx = load_readacross()
    if idx is None:
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    q = _AD_GEN.GetFingerprint(mol)
    sims = np.array(DataStructs.BulkTanimotoSimilarity(q, idx["fps"]))
    order = [i for i in np.argsort(-sims)[:pool] if sims[i] >= min_sim]
    acc = {}
    for i in order:
        s = float(sims[i])
        # Temper each neighbour's contribution before combining. Raw Tanimoto in a noisy-OR
        # saturates almost immediately: twenty neighbours at 0.7 reach 1.00, which asserts certainty
        # that moderate analogy cannot support and destroys the ranking between targets. Rescaling
        # similarity above the inclusion floor and squaring it keeps the ordering while reserving
        # scores near 1 for genuinely close matches.
        w = ((s - min_sim) / (1.0 - min_sim)) ** 2 if s > min_sim else 0.0
        for t in idx["targets"][i]:
            a = acc.setdefault(t, {"target": t, "n": 0, "best_sim": 0.0, "prod": 1.0})
            a["n"] += 1
            a["best_sim"] = max(a["best_sim"], s)
            a["prod"] *= (1.0 - w)
    out = []
    for t, a in acc.items():
        a["evidence"] = 1.0 - a["prod"]
        del a["prod"]
        out.append(a)
    out.sort(key=lambda x: -x["evidence"])
    return out


def low_power_target(tgt):
    """True when a binder model fails the reliability gate, so a negative call from it carries little.

    The gate is sensitivity of at least 0.50 on scaffold-held-out actives at the deployed threshold,
    and AUROC of at least 0.75 against compounds measured at that same target and found inactive. It
    is not a fixed false-positive rate, which is what sets the threshold rather than what judges it.
    """
    info = load_binder_modes().get(tgt, {})
    return info.get("reliable_call") is False


def screening_mode():
    """True when the user has selected the high-precision mode for low-prevalence screening."""
    return bool(st.session_state.get("screening_mode", False))


def binder_threshold(tgt):
    """Decision threshold for a binder-panel target. Where enough experimentally measured inactives
    exist, this is the probability at which 10% of them would be called binders; otherwise a global
    fallback. Using a per-target threshold matters because the default 0.5 cut, learned against
    property-matched decoys, is far too permissive on real tested chemistry."""
    info = load_binder_modes().get(tgt, {})
    # In screening mode use the stricter cut calibrated to a 1% background false-positive rate,
    # which is what keeps precision usable when real actives are rare.
    t = info.get("screening_threshold") if screening_mode() else None
    if t is None:
        t = info.get("threshold")
    if t is None:
        br = info.get("base_rate")
        return float(br) if (info.get("mode") == "measured_labels" and br) else BINDER_FLOOR
    return float(t)


# Conditions whose principal drug mechanism acts OUTSIDE the blood-brain barrier, and which must
# therefore not be filtered by predicted barrier penetration.
#
# The gate encodes an assumption that the target sits behind the barrier. For most conditions here it
# does. For two it does not, and applying the gate would suppress correct calls:
#
#   Migraine              the calcitonin-gene-related-peptide receptor acts at the trigeminal
#                         ganglion and the dural vasculature, which lie outside the barrier. That is
#                         precisely why the gepants and the anti-CGRP antibodies work without central
#                         penetration. Rimegepant carries a predicted barrier probability of 0.33 and
#                         would be silenced by the gate despite engaging the receptor at 0.99.
#   Multiple sclerosis    dihydroorotate dehydrogenase inhibition acts on proliferating peripheral
#                         lymphocytes. Teriflunomide's therapeutic effect is immunological and
#                         extracerebral.
#
# For these the reported score is the mechanism score itself. This does not weaken the exposure
# statement: predicted penetration is still shown, and for a compound intended to act centrally it
# still matters. It stops the tool from requiring central exposure of a drug class that does not need
# it. RIPK1 is deliberately not exempted: its relevance to amyotrophic lateral sclerosis and to
# neuroinflammation is central, so the gate is appropriate there.
PERIPHERAL_MECHANISM_DISEASES = {"Migraine", "Multiple sclerosis"}


def disease_scores(r):
    """Brain-relevance per condition, from the STRONGEST engaged target in the knowledge graph,
    scaled by predicted BBB penetration. Returns (bbb, neuro, rows).

    The BBB term multiplies every gated disease identically, so it cannot change their relative order
    (verified in inversion/results/H3_gating.csv). It is an exposure filter: it decides whether
    anything is reported at all, not which disease is reported. Describing it as though it sharpens
    the disease call would overstate what it does. Conditions in PERIPHERAL_MECHANISM_DISEASES are
    exempt, because for them the assumption the gate encodes is false."""
    bbb = r["targets"]["BBB"]
    neuro = neuro_signal(r)
    rows = []
    for disease in DISEASE_ORDER:
        best, driver = 0.0, None
        for tgt, w in DISEASE_CONTRIB.get(disease, []):
            s = target_signal(r, neuro, tgt)
            if w * s > best:
                best, driver = w * s, (tgt, s)
        gate = 1.0 if disease in PERIPHERAL_MECHANISM_DISEASES else bbb
        rows.append({"disease": disease, "signal": best, "gated": gate * best, "driver": driver,
                     "peripheral": disease in PERIPHERAL_MECHANISM_DISEASES})
    rows.sort(key=lambda d: d["gated"], reverse=True)
    return bbb, neuro, rows


def verdict(kpuu, bbb):
    if kpuu >= 0.3:
        return "Favourable", GREEN, "Predicted to reach a meaningful free concentration in the brain."
    if kpuu >= 0.1:
        return "Borderline", AMBER, "Some free brain exposure predicted; interpret with caution."
    return "Limited", ADVERSE, "Low predicted free brain exposure (poor penetration or active efflux)."


# --------------------------------- presentation helpers ---------------------------------
def _b64(path: Path) -> str:
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except Exception:
        return ""


def _mol_data_uri(mol, size=(340, 300)) -> str:
    """Structure image, or an empty string where the drawing libraries are unavailable.

    Callers already emit an <img> whose src may be empty, which renders as nothing rather than as a
    broken layout. No prediction depends on this."""
    if not _DRAW_AVAILABLE or mol is None:
        return ""
    try:
        img = Draw.MolToImage(mol, size=size)
        buf = BytesIO(); img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def _risk_colour(p): return ADVERSE if p >= 0.5 else (AMBER if p >= 0.2 else GREEN)
def _score_colour(p): return GREEN if p >= 0.5 else (AMBER if p >= 0.25 else MUTE2)


def _kpi(label, value, sub, colour):
    return (f'<div class="bs-kpi"><div class="bs-kpi-label">{label}</div>'
            f'<div class="bs-kpi-val" style="color:{colour}">{value}</div>'
            f'<div class="bs-kpi-sub">{sub}</div></div>')


def _table(headers, rows, aligns):
    amap = {"l": "left", "r": "right", "c": "center"}
    thead = "".join(f'<th style="text-align:{amap[a]}">{h}</th>' for h, a in zip(headers, aligns))
    body = "".join("<tr>" + "".join(f'<td style="text-align:{amap[a]}">{c}</td>'
                   for c, a in zip(row, aligns)) + "</tr>" for row in rows)
    return f'<table class="bs-table"><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>'


def _badge(text, colour, filled=True):
    if filled:
        return (f'<span class="bs-badge" style="background:{colour}14;color:{colour};border-color:{colour}33">'
                f'<span class="bs-dot" style="background:{colour}"></span>{text}</span>')
    return f'<span class="bs-badge bs-badge-out"><span class="bs-dot" style="background:{MUTE2}"></span>{text}</span>'


def _bar(frac, colour, label_left, label_right):
    pct = max(0.0, min(1.0, frac)) * 100
    return (f'<div class="bs-barrow"><div class="bs-barlabels"><span>{label_left}</span>'
            f'<span class="bs-barval">{label_right}</span></div>'
            f'<div class="bs-bartrack"><div class="bs-barfill" style="width:{pct:.1f}%;background-color:{colour}"></div></div></div>')


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600;14..32,700;14..32,800;14..32,900&family=JetBrains+Mono:wght@400;500&display=swap');

        /* ------------------------------------------------------------------
           Design tokens. Everything downstream is expressed in these, so the
           interface can be retuned from one place and stays internally
           consistent. Type and spacing follow a modular scale rather than
           ad-hoc pixel values, which is what stops a dense scientific readout
           from looking assembled by accident.
           ------------------------------------------------------------------ */
        :root {{
            --ink:{INK}; --mute:{MUTE}; --mute2:{MUTE2}; --line:{LINE}; --surf:{SURF};
            --navy:{NAVY}; --navy-md:{NAVY_MD}; --gold:{GOLD};
            --ok:{GREEN}; --warn:{AMBER}; --adverse:{ADVERSE}; --accent:{BLUE};

            /* fluid type scale, 1.200 minor third, clamped so it never collapses on mobile
               nor bloats on a 4K panel */
            --t-3xs:.625rem; --t-2xs:.6875rem; --t-xs:.75rem; --t-sm:.8125rem;
            --t-base:.875rem; --t-md:clamp(.95rem,.9rem + .2vw,1.02rem);
            --t-lg:clamp(1.1rem,1rem + .5vw,1.3rem);
            --t-xl:clamp(1.35rem,1.15rem + .9vw,1.75rem);
            --t-2xl:clamp(1.6rem,1.25rem + 1.6vw,2.35rem);

            /* 4px base spacing scale */
            --s1:4px; --s2:8px; --s3:12px; --s4:16px; --s5:22px; --s6:30px; --s7:42px;

            --r-sm:8px; --r-md:12px; --r-lg:16px; --r-xl:22px; --r-pill:999px;

            /* layered elevation: a tight contact shadow plus a wide soft one reads as
               physical depth, where a single blurred shadow reads as a smudge */
            --e1:0 1px 2px rgba(13,33,55,.05), 0 1px 3px -1px rgba(13,33,55,.06);
            --e2:0 1px 2px rgba(13,33,55,.04), 0 12px 28px -16px rgba(13,33,55,.20);
            --e3:0 2px 4px rgba(13,33,55,.05), 0 26px 50px -24px rgba(13,33,55,.28);
            --shadow:var(--e2);
            --ring:0 0 0 3px rgba(240,165,0,.34);
            --ease:cubic-bezier(.22,.61,.36,1);
        }}

        /* Derived tints. color-mix keeps every wash in exact relation to its source colour, so a
           palette change propagates instead of leaving stale hard-coded rgba values behind. */
        @supports (color: color-mix(in srgb, red, blue)) {{
            :root {{
                --wash-ok:color-mix(in srgb, {GREEN} 8%, white);
                --wash-warn:color-mix(in srgb, {AMBER} 8%, white);
                --wash-adverse:color-mix(in srgb, {ADVERSE} 7%, white);
                --wash-navy:color-mix(in srgb, {NAVY} 4%, white);
            }}
        }}

        html, body, [class*="css"], .stApp {{
            font-family:'Inter',system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; color:var(--ink);
            -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
            font-feature-settings:"tnum" 1, "ss01" 1, "cv05" 1, "calt" 1;
            font-optical-sizing:auto; }}
        .stApp {{ background:
            radial-gradient(1200px 520px at 50% -200px, #F5F8FD 0%, transparent 70%),
            linear-gradient(180deg, {BG} 0%, #E9EFF7 100%) fixed; }}
        ::selection {{ background:{GOLD}40; color:{NAVY}; }}
        [data-testid="stMainBlockContainer"] {{ max-width:1320px; padding:0 var(--s5) var(--s7); }}
        [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer,
        [data-testid="stStatusWidget"], [data-testid="stHeader"] {{ display:none !important; }}

        /* Headings set with balanced wrapping so a two-line title never leaves one orphan word,
           and body copy set pretty so paragraphs do not end on a single short word. */
        h1,h2,h3,.bs-h,.about-h,.header-title {{ text-wrap:balance; }}
        p,.bs-note,.bs-verdict-note {{ text-wrap:pretty; }}

        /* Visible, keyboard-only focus. The default browser ring is invisible against navy. */
        *:focus-visible {{ outline:none; box-shadow:var(--ring); border-radius:var(--r-sm); }}

        @media (prefers-reduced-motion:reduce) {{
            *,*::before,*::after {{ animation-duration:.001ms !important; transition-duration:.001ms !important; }}
        }}

        /* ---- site header (navy + gold) ---- */
        .site-header {{ background:linear-gradient(135deg,{NAVY_DK} 0%,#0A1929 40%,{NAVY} 72%,#112A47 100%);
            padding:26px 40px; margin:8px -2rem 16px; border-bottom:3px solid {GOLD};
            border-radius:0 0 16px 16px; box-shadow:0 8px 30px rgba(0,0,0,.28); position:relative; }}
        .header-wrap {{ display:flex; align-items:center; gap:26px; }}
        /* The mark is a photorealistic illustration drawn for a light ground: measured over its
           opaque pixels it is 29 per cent very light and 66 per cent mid-tone, and its soft edge
           averages luminance 0.68. Laid straight onto the navy header that pale edge reads as haze
           and the brain washes out, so it sits on a white plate, which is its intended ground and
           is how the institute mark beside it is already treated. It is also given more room: at
           96px only about 3,900 pixels carried the whole illustration and the detail turned to
           noise. Contain, never cover: the gold arc runs to the edge of the artwork and a circular
           crop cuts it. */
        .header-logo {{ height:112px; width:112px; object-fit:contain; flex-shrink:0;
            background:#FFFFFF; border-radius:18px; padding:8px;
            border:1px solid rgba(240,165,0,.55);
            box-shadow:0 0 20px rgba(240,165,0,.22), 0 4px 14px rgba(0,0,0,.40); }}
        /* Matched to the SAI-CNS mark opposite it: same white plate, same corner radius, so the
           two read as a pair rather than as two unrelated badges. Slightly smaller, because it is
           the institutional credit and not the identity of the tool. */
        .header-inst {{ height:96px; width:96px; border-radius:18px;
            border:1px solid rgba(255,255,255,.55); background:#FFFFFF; object-fit:contain;
            padding:8px; flex-shrink:0; box-shadow:0 4px 14px rgba(0,0,0,.35); }}
        .header-div {{ width:2px; height:76px; background:linear-gradient(to bottom,transparent,{GOLD},transparent); flex-shrink:0; }}
        .header-textblk {{ flex:1; min-width:0; }}
        .header-title {{ color:#fff; font-size:2.3rem; font-weight:800; margin:0; letter-spacing:-.5px; line-height:1.12; }}
        .header-title span {{ color:{GOLD}; }}
        .header-sub {{ color:#88AECF; font-size:.95rem; margin-top:5px; font-weight:500; }}
        .header-tags {{ margin-top:10px; display:flex; gap:7px; flex-wrap:wrap; }}
        .header-tag {{ font-size:.72rem; font-weight:600; color:#BFD6EA; background:rgba(255,255,255,.07);
            border:1px solid rgba(240,165,0,.25); padding:3px 10px; border-radius:20px; }}
        .preview-banner {{ background:#FFF8E6; border:1px solid #F0C674; border-left:5px solid {GOLD};
            border-radius:8px; padding:10px 16px; margin:0 0 14px; font-size:.84rem; color:#7A5B00; }}

        /* ---- cards ---- */
        .bs-card {{ background:var(--surf); border:1px solid rgba(13,33,55,.07); border-radius:var(--r-lg);
            padding:var(--s5) var(--s5) calc(var(--s5) + 2px); box-shadow:var(--e2); height:100%;
            box-sizing:border-box; position:relative; overflow:hidden;
            transition:box-shadow .28s var(--ease), border-color .28s var(--ease); }}
        /* a hairline of gold along the top edge, so a grid of cards has a rhythm without borders
           heavy enough to fragment the page */
        .bs-card::before {{ content:""; position:absolute; inset:0 0 auto 0; height:2px;
            background:linear-gradient(90deg, transparent, {GOLD}66 18%, {GOLD}66 82%, transparent);
            opacity:0; transition:opacity .28s var(--ease); }}
        .bs-card:hover {{ box-shadow:var(--e3); border-color:rgba(13,33,55,.11); }}
        .bs-card:hover::before {{ opacity:1; }}
        .bs-h {{ position:relative; font-size:var(--t-2xs); font-weight:800; letter-spacing:.13em;
            text-transform:uppercase; color:{NAVY}; display:flex; align-items:baseline; gap:var(--s2);
            margin:0 0 var(--s4); padding:0 0 var(--s3); border-bottom:1px solid #EDF1F8; }}
        .bs-h::before {{ content:""; width:3px; height:12px; border-radius:2px; align-self:center;
            background:linear-gradient(180deg,{GOLD},#D98E00); flex:0 0 auto; }}
        .bs-h-sub {{ margin-left:auto; font-size:var(--t-3xs); font-weight:600; letter-spacing:.02em;
            color:var(--mute2); text-transform:none; text-align:right; }}
        .bs-summary {{ display:grid; grid-template-columns:300px 1fr; gap:24px; align-items:stretch; }}
        .bs-mol {{ border:1px solid var(--line); border-radius:12px; background:#FBFCFE;
            display:flex; align-items:center; justify-content:center; padding:10px; }}
        .bs-mol img {{ max-width:100%; height:auto; display:block; }}
        .bs-summary-main {{ display:flex; flex-direction:column; gap:16px; min-width:0; }}
        .bs-verdict {{ border-left:5px solid var(--vc); background:linear-gradient(90deg,var(--vc)12,transparent);
            border-radius:10px; padding:13px 16px; }}
        .bs-verdict-eyebrow {{ font-size:.68rem; font-weight:800; text-transform:uppercase; letter-spacing:1.2px; color:var(--mute2); }}
        .bs-verdict-label {{ font-size:1.5rem; font-weight:800; color:var(--vc); line-height:1.15; margin:2px 0 4px; }}
        .bs-verdict-note {{ font-size:.83rem; color:var(--mute); line-height:1.45; }}
        .bs-kpis {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
        .bs-kpi {{ background:linear-gradient(180deg,#FCFDFF,#F5F9FE); border:1px solid #E7EDF6;
            border-radius:var(--r-md); padding:var(--s4) var(--s4) var(--s3); position:relative; }}
        .bs-kpi-label {{ font-size:var(--t-3xs); color:var(--mute); font-weight:700;
            text-transform:uppercase; letter-spacing:.08em; }}
        .bs-kpi-val {{ font-size:clamp(1.5rem,1.2rem + .9vw,1.85rem); font-weight:800; line-height:1;
            margin:var(--s2) 0 var(--s1); font-variant-numeric:tabular-nums lining-nums;
            letter-spacing:-.025em; }}
        .bs-kpi-sub {{ font-size:var(--t-3xs); color:var(--mute2); line-height:1.4; }}
        .bs-table {{ width:100%; border-collapse:separate; border-spacing:0; font-size:var(--t-sm); }}
        .bs-table th {{ font-size:var(--t-3xs); font-weight:800; text-transform:uppercase;
            letter-spacing:.07em; color:var(--mute2); padding:0 var(--s3) var(--s2);
            border-bottom:1.5px solid #E7EDF6; position:sticky; top:0; background:var(--surf); z-index:1; }}
        .bs-table td {{ padding:var(--s3) var(--s3); border-bottom:1px solid #F2F5FA; color:var(--ink);
            font-variant-numeric:tabular-nums lining-nums; vertical-align:middle; }}
        .bs-table tbody tr:last-child td {{ border-bottom:0; }}
        .bs-table tbody tr {{ transition:background .16s var(--ease); }}
        .bs-table tbody tr:hover td {{ background:#F6FAFE; }}
        /* zebra only on long tables, where the eye needs help tracking a row across */
        .bs-table tbody:has(tr:nth-child(9)) tr:nth-child(even) td {{ background:#FCFDFF; }}
        .bs-table tbody:has(tr:nth-child(9)) tr:hover td {{ background:#F2F7FD; }}
        .bs-num {{ font-weight:700; letter-spacing:-.01em; }}
        .bs-ctx {{ color:var(--mute); font-size:.76rem; }}
        .bs-badge {{ display:inline-flex; align-items:center; gap:5px; padding:3px 10px; border-radius:6px;
            font-size:.66rem; font-weight:700; letter-spacing:.02em; border:1px solid transparent; white-space:nowrap; }}
        .bs-badge-out {{ background:#F1F4F9; color:var(--mute2); border-color:var(--line); }}
        .bs-dot {{ width:6px; height:6px; border-radius:50%; flex:0 0 auto; }}
        .bs-delta {{ display:inline-flex; align-items:center; gap:3px; padding:2px 8px; border-radius:6px;
            font-size:.7rem; font-weight:800; font-variant-numeric:tabular-nums; line-height:1.4; }}
        .bs-delta .ar {{ font-size:.6em; line-height:1; position:relative; top:-.5px; }}
        .bs-delta.up {{ background:rgba(21,128,61,.10); color:{GREEN}; }}
        .bs-delta.mid {{ background:rgba(180,83,9,.12); color:{AMBER}; }}
        .bs-delta.down {{ background:rgba(148,163,184,.16); color:{MUTE}; }}
        .bs-basetxt {{ font-size:.6rem; color:var(--mute2); margin-top:3px; letter-spacing:.02em; }}
        .bs-barrow {{ margin:0 0 14px; }}
        .bs-barlabels {{ display:flex; justify-content:space-between; font-size:.82rem; margin-bottom:5px; }}
        .bs-barval {{ font-weight:800; font-variant-numeric:tabular-nums; }}
        .bs-bartrack {{ height:8px; border-radius:var(--r-pill); background:#E9EEF6; overflow:hidden;
            box-shadow:inset 0 1px 2px rgba(13,33,55,.07); }}
        .bs-barfill {{ height:100%; border-radius:var(--r-pill);
            background-image:linear-gradient(90deg, rgba(255,255,255,.32), rgba(255,255,255,0));
            transform-origin:left center; animation:bs-grow .62s var(--ease) both; }}
        @keyframes bs-grow {{ from {{ transform:scaleX(.02); opacity:.35 }} to {{ transform:scaleX(1); opacity:1 }} }}
        .bs-barsub {{ font-size:.68rem; color:var(--mute2); margin-top:3px; }}
        .bs-chip {{ font-size:.66rem; padding:2px 8px; border-radius:6px; background:#EEF2F9; color:var(--mute); }}
        .bs-note {{ font-size:.78rem; color:var(--mute); line-height:1.6; }}
        .bs-note b {{ color:var(--ink); }}
        .bs-foot {{ margin-top:22px; padding-top:16px; border-top:1px solid var(--line);
            font-size:.72rem; color:var(--mute2); line-height:1.6; }}
        .bs-empty {{ border:1px dashed {CARDBD}; border-radius:14px; background:var(--surf);
            padding:38px 28px; text-align:center; color:var(--mute); font-size:.9rem; }}
        .bs-empty .k {{ font-size:34px; }}
        /* input card */
        .bs-searchcard p.t {{ font-size:var(--t-md); font-weight:700; color:{NAVY}; margin:0 0 var(--s1);
            letter-spacing:-.01em; }}
        .bs-searchcard p.d {{ font-size:var(--t-sm); color:#4A5568; margin:0 0 var(--s2); line-height:1.6; }}

        /* ---- tabs: a segmented control rather than a browser tab strip ---- */
        /* Orange bar, white type, with the selected tab a deeper tint of the same hue so the bar
           reads as one control and the active page is obvious. The tints are chosen by measurement:
           at 13px bold, white needs 4.5:1, which the brand gold (2.1:1) and the logo orange (2.9:1)
           do not reach. The bar is 5.2:1 and the selected tab 7.3:1. Resting labels sit at 82 per
           cent white, still above 4.5:1 on this ground, so the selected tab stands out without the
           unselected ones becoming hard to read. */
        /* The bar spans the content column and the tabs divide it evenly. It was inline-flex, so it
           shrank to its labels and left the rest of a 1320px row empty beside it, which read as a
           control that had failed to load rather than as navigation. */
        [data-baseweb="tab-list"] {{ gap:var(--s1) !important; background:{TAB_BAR}; padding:var(--s1);
            border-radius:var(--r-md); border:1px solid {TAB_BAR}; margin-bottom:var(--s4);
            display:flex !important; width:100%; flex-wrap:nowrap !important; overflow-x:auto; }}
        /* The label lives in a child node that Streamlit colours itself, so colouring only the
           button leaves the text unchanged however many !important flags it carries. Every
           descendant is targeted, and -webkit-text-fill-color is set alongside color because the
           theme sets that too and it wins over color where both are present. */
        [data-baseweb="tab-list"] button[data-baseweb="tab"] {{ font-weight:800 !important;
            font-size:.95rem !important; letter-spacing:.01em; border-radius:var(--r-sm) !important;
            padding:var(--s3) var(--s4) !important; transition:all .2s var(--ease);
            flex:1 1 0; min-width:0; justify-content:center !important; text-align:center;
            white-space:nowrap; }}
        [data-baseweb="tab-list"] button[data-baseweb="tab"],
        [data-baseweb="tab-list"] button[data-baseweb="tab"] * {{
            color:{NAVY} !important; -webkit-text-fill-color:{NAVY} !important; }}
        [data-baseweb="tab-list"] button[data-baseweb="tab"]:hover {{
            background:rgba(255,255,255,.28) !important; }}
        /* The selected tab is painted with an inset shadow rather than a background colour. The
           theme sets background-color on these buttons through a generated class that survives an
           !important background rule, so the colour never reached the element; box-shadow is not
           contested and covers the whole button. The white underline is a second, redundant cue for
           anyone who cannot separate the two oranges. */
        /* The active tab is filled by a pseudo-element rather than a background or a shadow. The
           theme's generated classes reset background-color and box-shadow on these buttons and win
           against !important, but they do not style ::before, so this is the one fill that holds.
           It sits behind the label, which is why the button gets its own stacking context. */
        [data-baseweb="tab-list"] button[data-baseweb="tab"] {{ position:relative; z-index:0; }}
        [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"]::before {{
            content:""; position:absolute; inset:0; z-index:-1; background:{TAB_ACTIVE};
            border-radius:var(--r-sm); }}
        [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"],
        [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] * {{
            color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; }}
        [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"],
        [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] * {{
            color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; }}
        button[data-baseweb="tab"]:focus-visible {{ outline:2px solid #FFFFFF !important;
            outline-offset:-3px; }}
        /* BaseWeb's own selected-tab indicator, kept rather than hidden. Painting the selected tab
           with a background or an inset shadow does not survive the theme's generated classes, but
           this element is the mechanism the component already uses to mark the active tab, so it is
           simply recoloured to white against the orange bar. The border strip below it is still
           removed: on a coloured bar it reads as a stray line. */
        [data-baseweb="tab-highlight"] {{ display:none !important; }}
        [data-baseweb="tab-border"] {{ display:none !important; }}

        /* ---- inputs ---- */
        [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {{
            font-size:var(--t-base) !important; border-radius:var(--r-sm) !important; }}
        [data-testid="stTextArea"] textarea {{ font-family:'JetBrains Mono',ui-monospace,Consolas,monospace;
            font-size:var(--t-sm) !important; line-height:1.7; }}
        [data-testid="stTextInput"] input {{ padding:var(--s3) var(--s4) !important; }}
        [data-baseweb="input"], [data-baseweb="textarea"] {{ border-radius:var(--r-sm) !important;
            border-color:#D6DFEC !important; transition:border-color .2s var(--ease), box-shadow .2s var(--ease); }}
        [data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within {{
            border-color:{GOLD} !important; box-shadow:var(--ring); }}

        /* ---- buttons ---- */
        div.stButton > button, div.stDownloadButton > button {{ border-radius:var(--r-sm);
            font-weight:700; font-size:var(--t-sm); padding:var(--s3) var(--s4);
            transition:transform .16s var(--ease), box-shadow .2s var(--ease), background .2s var(--ease); }}
        div.stButton > button:active, div.stDownloadButton > button:active {{ transform:translateY(1px); }}
        div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {{
            background:{NAVY}; border:1px solid {NAVY}; color:#fff; box-shadow:var(--e1); }}
        div.stButton > button[kind="primary"]:hover, div.stDownloadButton > button[kind="primary"]:hover {{
            background:{NAVY_MD}; border-color:{NAVY_MD}; box-shadow:var(--e2); }}
        div.stDownloadButton > button {{ background:var(--surf); border:1px solid #DCE4F0;
            color:{NAVY}; width:100%; }}
        div.stDownloadButton > button:hover {{ border-color:{GOLD}; background:#FFFDF7;
            box-shadow:var(--e2); color:{NAVY}; }}
        .bs-export {{ background:linear-gradient(180deg,#FDFEFF,#F7FAFE); }}

        /* ---- alerts: Streamlit paints errors red by default, which this interface does not use.
           Warnings become amber and errors become the adverse violet, each with a written label so
           the state never depends on colour alone. ---- */
        [data-testid="stAlert"] {{ border-radius:var(--r-md); border-width:1px; border-style:solid;
            box-shadow:var(--e1); font-size:var(--t-sm); }}
        [data-testid="stAlertContentError"], div[data-baseweb="notification"][kind="negative"] {{
            background:var(--wash-adverse,#F7F2FC) !important; border-color:{ADVERSE}3d !important;
            color:{ADVERSE} !important; }}
        [data-testid="stAlertContentWarning"] {{ background:var(--wash-warn,#FDF7EE) !important;
            border-color:{AMBER}3d !important; color:{AMBER} !important; }}
        [data-testid="stAlertContentSuccess"] {{ background:var(--wash-ok,#F1F9F3) !important;
            border-color:{GREEN}3d !important; color:{GREEN} !important; }}
        [data-testid="stAlertContentInfo"] {{ background:var(--wash-navy,#F4F8FD) !important;
            border-color:{NAVY}26 !important; color:{NAVY} !important; }}
        [data-testid="stAlert"] svg {{ fill:currentColor !important; }}

        /* ---- data grid, uploader, progress ---- */
        [data-testid="stDataFrame"] {{ border:1px solid #E3EAF4; border-radius:var(--r-md);
            overflow:hidden; box-shadow:var(--e1); }}
        [data-testid="stFileUploaderDropzone"] {{ border-radius:var(--r-md); border:1.5px dashed #CBD6E6;
            background:#FAFCFF; transition:border-color .2s var(--ease), background .2s var(--ease); }}
        [data-testid="stFileUploaderDropzone"]:hover {{ border-color:{GOLD}; background:#FFFDF7; }}
        [data-testid="stProgress"] > div > div > div {{ background:{NAVY} !important; }}
        [data-testid="stProgress"] > div > div {{ background:#E4EBF5 !important; border-radius:var(--r-pill); }}
        [role="checkbox"][aria-checked="true"] {{ background:{NAVY} !important; border-color:{NAVY} !important; }}
        [data-testid="stCaptionContainer"] p {{ font-size:var(--t-xs); color:var(--mute2); }}
        /* expanders carry secondary reading, so they sit quietly until opened */
        [data-testid="stExpander"] {{ border:1px solid #E3EAF4; border-radius:var(--r-md);
            background:var(--surf); box-shadow:var(--e1); overflow:hidden; }}
        [data-testid="stExpander"] summary {{ font-weight:700; font-size:var(--t-sm);
            color:{NAVY}; padding:var(--s3) var(--s4); }}
        [data-testid="stExpander"] summary:hover {{ background:#F7FAFE; }}
        [data-testid="stExpander"] .bs-card {{ box-shadow:none; border:0; }}
        [data-testid="stExpander"] .bs-card::before {{ display:none; }}

        /* ---- print: the browser's own print path is the most reliable PDF route ---- */
        @media print {{
            .stApp {{ background:#fff; }}
            [data-baseweb="tab-list"], div.stButton, div.stDownloadButton,
            [data-testid="stFileUploaderDropzone"] {{ display:none !important; }}
            .bs-card {{ box-shadow:none; border:1px solid #DDE4EF; break-inside:avoid; }}
            .site-header {{ box-shadow:none; }}
        }}
        /* about */
        .about-h {{ font-size:1.65rem; font-weight:800; color:{NAVY}; margin:0 0 14px; letter-spacing:-.01em; }}
        .about-h span {{ color:{GOLD}; }}
        .about-eyebrow {{ font-size:.9rem; font-weight:800; color:{AMBER}; letter-spacing:.06em; text-transform:uppercase; margin:0 0 6px; }}
        .quote {{ border-left:4px solid {GOLD}; background:#FFFBF0; border-radius:0 10px 10px 0; padding:14px 18px; margin:0 0 12px; }}
        .quote p {{ font-size:1.0rem; font-style:italic; color:#5A4000; margin:0 0 8px; line-height:1.6; }}
        .quote cite {{ font-size:.72rem; font-weight:700; color:{AMBER}; letter-spacing:.06em; text-transform:uppercase; font-style:normal; }}
        @media (max-width:1040px) {{ .bs-kpis {{ grid-template-columns:repeat(2,1fr); }} }}
        @media (max-width:820px) {{
            [data-testid="stMainBlockContainer"] {{ padding:0 var(--s3) var(--s6); }}
            .bs-summary {{ grid-template-columns:1fr; }}
            .bs-kpis {{ grid-template-columns:1fr; }}
            .header-inst,.header-div {{ display:none; }}
            .site-header {{ padding:var(--s5) var(--s4); margin:var(--s2) calc(-1 * var(--s3)) var(--s4); }}
            [data-baseweb="tab-list"] {{ display:flex !important; width:100%; overflow-x:auto; }}
            button[data-baseweb="tab"] {{ padding:var(--s2) var(--s3) !important; white-space:nowrap; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    # The SAI-CNS mark: a barrier with a single channel through it, and one compound in the
    # channel. The wordmark is set as HTML text below, so the header carries the mark alone rather
    # than a second copy of the name.
    logo = _b64(ASSETS / "sai_cns_mark.png")
    inst = _b64(ASSETS / "sssihl_logo.png")
    logo_html = (f'<img class="header-logo" src="data:image/png;base64,{logo}"/>' if logo else
                 f'<div class="header-logo" style="display:flex;align-items:center;justify-content:center;'
                 f'background:{NAVY_MD};color:{GOLD};font-size:2rem;font-weight:800;">S</div>')
    inst_html = f'<div class="header-div"></div><img class="header-inst" src="data:image/png;base64,{inst}"/>' if inst else ""
    # Counted from what is actually served, not written by hand. This list named eight endpoints,
    # all of them core classifiers, and omitted the 43-endpoint binder panel entirely, so the
    # landing page advertised about a seventh of the tool. Deriving the counts also means they
    # cannot fall behind the registry the way a hand-written list did.
    _sh = panel_shape()
    n_targets = _sh.get("targets", "-")
    tags = [f"{n_targets} molecular targets", "BBB and K(p,uu) exposure", "hERG safety",
            "Receptor potency", "Neuroprotection", f"{len(DISEASE_ORDER)} conditions",
            "Calibrated + conformal", "Applicability domain"]
    tagbar = "".join(f'<span class="header-tag">{t}</span>' for t in tags)
    st.markdown(
        f"""
        <div class="site-header"><div class="header-wrap">
          {logo_html}
          <div class="header-div"></div>
          <div class="header-textblk">
            <div class="header-title">SAI<span>-CNS</span></div>
            <div class="header-sub">Calibrated, exposure-gated prediction of small-molecule action in the human brain</div>
            <div class="header-tags">{tagbar}</div>
          </div>
          {inst_html}
        </div></div>
        <div class="preview-banner"><b>Research preview: pending peer review.</b> SAI-CNS is a
        computational decision-support tool for research prioritisation and hypothesis generation. It predicts
        <i>molecular target engagement and physicochemical properties</i>, not clinical efficacy, and has not
        undergone wet-lab or clinical validation. Not for medical, diagnostic, or treatment decisions.</div>
        <div class="bs-note" style="margin:-4px 0 12px;font-size:.78rem">
        Free to use without registration, for any purpose including commercial, under the
        <a href="https://opensource.org/licenses/MIT" target="_blank" rel="noopener">MIT licence</a>.
        Source, trained models and every validation artefact:
        <a href="https://github.com/krishna-g-999/sai-cns" target="_blank" rel="noopener">github.com/krishna-g-999/sai-cns</a>.
        Underlying data retain their own licences (ChEMBL CC BY-SA 3.0, BindingDB, B3DB, Therapeutics
        Data Commons, MoleculeNet; pathway annotations from KEGG, Reactome and IUPHAR/BPS).
        Submitted structures are processed in memory and are not stored, logged or shared.</div>
        """,
        unsafe_allow_html=True,
    )


def render_summary(mol, r):
    kpuu = r["adme"]["kpuu"]; bbb = r["targets"]["BBB"]; herg = r["targets"]["hERG"]
    label, colour, note = verdict(kpuu, bbb)
    # Both tiles state which quantity they show. The panel reports two different numbers for the
    # same endpoint: the calibrated probability, and the enrichment of that probability over the
    # training base rate, which is what the disease layer consumes and what the engagement chart
    # plots. For hERG on donepezil they are 73 per cent and 65 per cent, and a reader given both
    # without a label sees a contradiction rather than two answers to two questions.
    herg_e = enrichment("hERG", herg)
    kpis = (_kpi("Predicted K<sub>p,uu</sub>", f"{kpuu:.2f}", "unbound brain / plasma", colour)
            + _kpi("BBB penetration", f"{bbb:.0%}",
                   f"calibrated probability &middot; base rate {base_rate('BBB'):.0%}", BLUE)
            + _kpi("hERG liability", f"{herg:.0%}",
                   f"calibrated probability &middot; enrichment {herg_e:.0%} over a "
                   f"{base_rate('hERG'):.0%} base rate", _risk_colour(herg)))
    st.markdown(
        f"""
        <div class="bs-card bs-summary" style="--vc:{colour}">
          <div class="bs-mol"><img src="{_mol_data_uri(mol)}" alt="molecule"/></div>
          <div class="bs-summary-main">
            <div class="bs-verdict" style="--vc:{colour}">
              <div class="bs-verdict-eyebrow">Free brain exposure</div>
              <div class="bs-verdict-label">{label}</div>
              <div class="bs-verdict-note">{note}</div>
            </div>
            <div class="bs-kpis">{kpis}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hex_rgba(hex_colour, alpha):
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha:.2f})"


PATHWAY_SHORT = {
    "Cholinergic synapse": "Cholinergic", "Serotonergic synapse": "Serotonergic",
    "Dopaminergic synapse": "Dopaminergic", "Amyloid-beta processing": "Amyloid-β",
    "Tau / GSK-3 signalling": "Tau / GSK-3", "Monoamine catabolism": "Monoamine",
    "Antioxidant / Nrf2 response": "Antioxidant", "Retrograde endocannabinoid signalling": "Endocannabinoid",
    "Opioid signalling": "Opioid", "Noradrenergic signalling": "Noradrenergic",
    "Histaminergic signalling": "Histaminergic", "Adenosine / purinergic signalling": "Adenosine",
    "Sigma-1 receptor chaperone": "Sigma-1", "Parkinson disease pathway": "Parkinson",
    "Cardiac ion channel": "Cardiac",
}


def _short_disease(d):
    return d.split(" (")[0].split(" / ")[0]


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_network_svg(r, name="Compound"):
    """Layered mechanism diagram (compound to target to pathway to disease) as clean SVG.
    Nodes are ordered barycentrically to minimise edge crossings; only engaged targets are drawn."""
    bbb, neuro, dz = disease_scores(r)
    dzmap = {d["disease"]: d for d in dz}
    ENG = MIN_ENGAGEMENT_SIGNAL

    targets = [t for t in KNOWLEDGE_GRAPH if target_signal(r, neuro, t) >= ENG]
    herg = engaged_signal("hERG", r["targets"]["hERG"])
    show_herg = herg >= ENG

    # collect edges (target, pathway, disease, weight) for engaged targets and scored diseases
    edges = []
    for t in targets:
        for pw, src_id, dis, w in KNOWLEDGE_GRAPH[t]:
            if dzmap[dis]["gated"] >= MIN_ACTIONABLE_SCORE:
                edges.append((t, pw, dis, w, target_signal(r, neuro, t)))
    if show_herg:
        edges.append(("hERG", "Cardiac ion channel", "CARD", 1.0, herg))
    nav = r.get("receptor_binder", {}).get("Nav1_5", 0.0)
    nav_thr = binder_threshold('Nav1_5')
    nav_s = max(0.0, (nav - nav_thr) / (1.0 - nav_thr)) if nav >= nav_thr else 0.0
    if nav_s >= ENG:
        edges.append(("Nav1_5", "Cardiac ion channel", "CARD", 0.8, nav_s))

    if not edges:
        # A silent result must still be an informative one. Report the strongest sub-threshold
        # mechanisms and the exposure numbers, so a peripheral drug is visibly distinguishable
        # from another peripheral drug rather than every null result looking identical.
        ranked = sorted(((t, target_signal(r, neuro, t)) for t in KNOWLEDGE_GRAPH),
                        key=lambda x: -x[1])[:4]
        near = [(t, s) for t, s in ranked if s > 0]
        if near:
            chips = "".join(
                f'<span class="bs-badge bs-badge-out" style="margin:3px">'
                f'{MECH_LABEL.get(t, t)} {s:.0%}</span>' for t, s in near)
            detail = (f'<div class="bs-note" style="margin-top:10px">Closest mechanisms, all below '
                      f'the {MIN_ACTIONABLE_SCORE:.0%} reporting threshold:</div>'
                      f'<div style="margin-top:6px">{chips}</div>')
        else:
            detail = ('<div class="bs-note" style="margin-top:10px">No modelled target scored above '
                      'its own background rate.</div>')
        return (
            f'<div style="padding:26px 14px;text-align:center;color:{MUTE}">'
            f'<div style="font-size:1.02rem;font-weight:700;color:{NAVY}">'
            f'No CNS target engagement above the reporting threshold</div>'
            f'<div class="bs-note" style="margin-top:8px;margin-left:auto;'
            f'margin-right:auto">The analysis ran across all {len(KNOWLEDGE_GRAPH)} modelled '
            f'targets. This is the expected result for a peripherally acting compound. It is not '
            f'evidence of inactivity, for two reasons. The molecule may act through a mechanism this '
            f'tool does not model (see the Coverage panel in About). It may also act through a '
            f'modelled target but fall under its reporting threshold: those thresholds are set for '
            f'precision, holding the false-positive rate near 0.2% on random chemistry, and the '
            f'measured cost is a median sensitivity of 0.77 across the panel, falling to 0.26 at the '
            f'strictest endpoints (inversion/results/H7_target_discrimination.csv). Exposure was '
            f'still assessed, and is shown above: predicted BBB penetration {bbb:.0%} and unbound '
            f'brain exposure K<sub>p,uu</sub> {r["adme"]["kpuu"]:.2f}.</div>{detail}</div>')

    T = list(dict.fromkeys(e[0] for e in edges))
    P = list(dict.fromkeys(e[1] for e in edges))
    D = list(dict.fromkeys(e[2] for e in edges))
    dscore = {d: (herg if d == "CARD" else dzmap[d]["gated"]) for d in D}
    D.sort(key=lambda d: -dscore[d])
    Dpos = {d: i for i, d in enumerate(D)}
    P.sort(key=lambda p: float(np.mean([Dpos[e[2]] for e in edges if e[1] == p])))
    Ppos = {p: i for i, p in enumerate(P)}
    T.sort(key=lambda t: float(np.mean([Ppos[e[1]] for e in edges if e[0] == t])))

    NW, NH, ROW, TOP, BOT = 150, 34, 56, 46, 22
    colx = [22, 212, 408, 604]
    W = colx[3] + NW + 22
    H = max(len(T), len(P), len(D), 1) * ROW + TOP + BOT

    def ycenters(n):
        block = n * ROW
        start = TOP + (H - TOP - BOT - block) / 2
        return [start + i * ROW + ROW / 2 for i in range(n)]

    cy = {("c", "C"): H / 2}
    for i, t in enumerate(T):
        cy[("t", t)] = ycenters(len(T))[i]
    for i, p in enumerate(P):
        cy[("p", p)] = ycenters(len(P))[i]
    for i, d in enumerate(D):
        cy[("d", d)] = ycenters(len(D))[i]

    def bez(x0, y0, x1, y1, width, colr, alpha):
        dx = (x1 - x0) * 0.42
        return (f'<path d="M{x0:.1f} {y0:.1f} C {x0+dx:.1f} {y0:.1f}, {x1-dx:.1f} {y1:.1f}, {x1:.1f} {y1:.1f}" '
                f'fill="none" stroke="{colr}" stroke-opacity="{alpha:.2f}" stroke-width="{width:.1f}" stroke-linecap="round"/>')

    safety = {"hERG", "Nav1_5"}
    paths = []
    # compound -> target
    for t in T:
        s = herg if t == "hERG" else target_signal(r, neuro, t)
        col = ADVERSE if t in safety else (GREEN if t == "NEURO" else BLUE)
        paths.append(bez(colx[0] + NW, cy[("c", "C")], colx[1], cy[("t", t)], 1.5 + 5 * s, col, 0.22 + 0.5 * s))
    # target -> pathway (dedup)
    seen = set()
    for t, pw, dis, w, s in edges:
        if (t, pw) not in seen:
            seen.add((t, pw))
            col = ADVERSE if t in safety else NAVY_MD
            paths.append(bez(colx[1] + NW, cy[("t", t)], colx[2], cy[("p", pw)], 1.5 + 5 * s, col, 0.20 + 0.45 * s))
    # pathway -> disease (max contribution per (pathway, disease))
    pd = {}
    for t, pw, dis, w, s in edges:
        pd[(pw, dis)] = max(pd.get((pw, dis), 0.0), w * s)
    for (pw, dis), c in pd.items():
        col = ADVERSE if dis == "CARD" else NAVY_MD
        paths.append(bez(colx[2] + NW, cy[("p", pw)], colx[3], cy[("d", dis)], 1.5 + 5 * c, col, 0.20 + 0.5 * c))

    def node(x, ycc, label, sub, fill, stroke, textcol, sub_col=None):
        y = ycc - NH / 2
        t1 = f'<tspan font-weight="700" fill="{textcol}">{_esc(label)}</tspan>'
        t2 = (f'<tspan font-weight="800" fill="{sub_col or stroke}"> {_esc(sub)}</tspan>' if sub else "")
        return (f'<g><rect x="{x}" y="{y:.1f}" rx="9" width="{NW}" height="{NH}" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="1.3"/>'
                f'<text x="{x+NW/2}" y="{ycc:.1f}" dy=".33em" text-anchor="middle" '
                f'font-family="Inter,sans-serif" font-size="12">{t1}{t2}</text></g>')

    nodes = [node(colx[0], cy[("c", "C")], name, "", NAVY, NAVY, "#FFFFFF", "#FFFFFF")]
    for t in T:
        s = herg if t == "hERG" else target_signal(r, neuro, t)
        c = ADVERSE if t in safety else (GREEN if t == "NEURO" else _score_colour(s))
        nodes.append(node(colx[1], cy[("t", t)], MECH_LABEL.get(t, t), f"{s:.0%}",
                          _hex_rgba(c, 0.12), _hex_rgba(c, 0.55), INK, c))
    for p in P:
        c = ADVERSE if p == "Cardiac ion channel" else NAVY_MD
        nodes.append(node(colx[2], cy[("p", p)], PATHWAY_SHORT.get(p, p), "",
                          _hex_rgba(c, 0.08), _hex_rgba(c, 0.45), INK))
    for d in D:
        g = dscore[d]
        c = ADVERSE if d == "CARD" else _score_colour(g)
        lab = "Cardiac liability" if d == "CARD" else _short_disease(d)
        nodes.append(node(colx[3], cy[("d", d)], lab, f"{g:.0%}",
                          _hex_rgba(c, 0.12), _hex_rgba(c, 0.55), INK, c))

    heads = ["Compound", "Target", "Pathway", "Disease"]
    headers = "".join(
        f'<text x="{colx[i]+NW/2}" y="22" text-anchor="middle" font-family="Inter,sans-serif" '
        f'font-size="10.5" font-weight="800" letter-spacing="1.2" fill="{MUTE2}">{h.upper()}</text>'
        for i, h in enumerate(heads))

    return (f'<div style="width:100%;overflow-x:auto"><svg viewBox="0 0 {W} {H:.0f}" width="100%" '
            f'style="min-width:660px;max-width:{W}px;display:block;margin:0 auto" '
            f'xmlns="http://www.w3.org/2000/svg">{headers}{"".join(paths)}{"".join(nodes)}</svg></div>')


def render_network(r, name="Compound"):
    st.markdown(
        '<div class="bs-card"><div class="bs-h">Compound, target, pathway and disease'
        '<span class="bs-h-sub">mechanistic map; line weight shows engagement</span></div>'
        + build_network_svg(r, name)
        + '<div class="bs-note" style="margin-top:10px">Pathway anchors: KEGG cholinergic (hsa04725), '
        'serotonergic (hsa04726), dopaminergic (hsa04728), glutamatergic (hsa04724), GABAergic (hsa04727), '
        'endocannabinoid (hsa04723) and opioid (hsa05032) maps, neuroactive ligand-receptor interaction '
        '(hsa04080), and the Alzheimer (hsa05010), Parkinson (hsa05012), Huntington (hsa05016) and '
        'amyotrophic lateral sclerosis (hsa05014) disease maps; Reactome KEAP1-NFE2L2 pathway '
        '(R-HSA-9755511); IUPHAR Guide to Pharmacology. Every target-to-pathway assignment was verified '
        'against the source database gene annotation. Assignments not supported by pathway membership are '
        'cited to the target annotation instead. The edge weights are a curated mechanistic prior, not '
        'tuned parameters: an ablation on held-out compounds found uniform and randomly permuted '
        'weights give the same disease ranking, so the predictive content lies in the graph topology. '
        'The BBB term scales every disease equally and therefore sets whether anything is reported, '
        'not which condition ranks first.'
        '</div></div>',
        unsafe_allow_html=True)


def render_independence(r):
    """How many independent mechanisms were engaged, as opposed to how many targets."""
    neuro = neuro_signal(r)
    engaged = [t for t in TARGET_KIND
               if t != "NEURO" and target_signal(r, neuro, t) >= MIN_ENGAGEMENT_SIGNAL]
    if not engaged:
        return
    by_fam, multi = grouped_engagement(engaged)
    n_ind = independent_mechanisms(engaged)

    chips = ""
    for fam in sorted(by_fam):
        ts = sorted(by_fam[fam], key=lambda t: -target_signal(r, neuro, t))
        labels = " + ".join(f'{MECH_LABEL.get(t, t)} <b>{target_signal(r, neuro, t):.0%}</b>'
                            for t in ts)
        if not fam:
            chips += (f'<div style="margin:6px 0"><span class="bs-chip">unrelated</span> '
                      f'<span style="font-size:.82rem">{labels}</span></div>')
            continue
        phi, note = family_cofire().get(fam, (None, ""))
        tag = (f'<span class="bs-badge" style="background:{AMBER}14;color:{AMBER};'
               f'border-color:{AMBER}33">correlated, r = {phi:.2f}</span>'
               if (len(ts) > 1 and phi) else "")
        chips += (f'<div style="margin:6px 0"><span class="bs-chip">{fam}</span> '
                  f'<span style="font-size:.82rem">{labels}</span> {tag}'
                  + (f'<div class="bs-note" style="margin-left:4px">{note}</div>'
                     if len(ts) > 1 else "") + '</div>')

    if multi:
        head = (f'{len(engaged)} targets engaged, but approximately <b>{n_ind}</b> independent '
                f'mechanism{"s" if n_ind != 1 else ""}')
        body = ('Members of the same family are homologous and are engaged together by the same '
                'ligands, so their joint engagement is close to one observation rather than several. '
                'The counts below are grouped for that reason, and the correlation measured across '
                '400 approved drugs is stated wherever two members of a family both fire. Across '
                'that set 36 targets ever fire but span only 14 independent directions, so a raw '
                'target count overstates the evidence by roughly a factor of two and a half '
                '(inversion/results/H8_family_correlation.csv).')
    else:
        head = f'{len(engaged)} target{"s" if len(engaged) != 1 else ""} engaged, no two homologous'
        body = ('No two engaged targets belong to the same homologous family, so each is an '
                'independent observation. Where homologues do fire together the interface says so, '
                'because their joint engagement would otherwise read as corroboration it is not.')

    st.markdown(
        f'<div class="bs-card"><div class="bs-h">Independent mechanisms'
        f'<span class="bs-h-sub">grouped by homology, not counted raw</span></div>'
        f'<div style="font-size:.92rem;margin-bottom:8px">{head}</div>{chips}'
        f'<div class="bs-note" style="margin-top:10px">{body}</div></div>',
        unsafe_allow_html=True)


def render_profile(r):
    """The panel-wide view. The network shows only what fired; this shows everything that was asked,
    including what came close, which is the context needed to read a negative result."""
    svg = build_profile_svg(r)
    if not svg:
        return
    st.markdown(
        '<div class="bs-card"><div class="bs-h">Target engagement profile'
        '<span class="bs-h-sub">distance above each endpoint\'s own reporting threshold</span></div>'
        f'<div style="overflow-x:auto">{svg}</div>'
        '<div class="bs-note" style="margin-top:10px">Raw probabilities are not comparable between '
        'endpoints, because each carries its own training base rate and its own threshold, so 0.8 '
        'means engagement at one target and nothing at another. Plotted here is the distance each '
        'target sits above its own cut, which is the quantity the disease layer consumes. Targets '
        'below the cut show how close they came, as a percentage of their threshold; that is context '
        'for a negative result, not evidence of activity.</div></div>',
        unsafe_allow_html=True)


def interpret_profile(r, bbb, neuro, rows):
    """One honest sentence describing the compound's predicted brain profile."""
    top = rows[0] if rows else None
    enriched = [ep for ep in TARGET_CLASSIFIERS if ep not in ("BBB",) and enrichment(ep, r["targets"][ep]) >= 0.2]
    gate = "and is predicted to reach the brain" if bbb >= 0.5 else f"but predicted BBB penetration is low ({bbb:.0%})"
    if top and top["gated"] >= 0.4:
        return f'Strong predicted signal for <b>{top["disease"]}</b> (via {MECH_LABEL.get(top["driver"][0], top["driver"][0])}).'
    if enriched:
        extra = "; also a neuroprotective/antioxidant signal" if neuro >= 0.35 else ""
        return f'Engages <b>{", ".join(MECH_LABEL.get(e, e) for e in enriched)}</b> above base rate{extra}.'
    if neuro >= 0.35:
        return f'Predominantly a <b>neuroprotective / antioxidant</b> profile, {gate}.'
    if neuro >= 0.2:
        return f'Modest <b>neuroprotective / antioxidant</b> signal, {gate}; no distinctive target engagement.'
    return ('No distinctive engagement among the <b>currently modelled</b> mechanisms. This compound sits '
            'near the training base rates for every endpoint. Absence of signal is not proof of inactivity; '
            'its real target may not yet be modelled (see the Coverage panel in About).')


def render_disease(r):
    bbb, neuro, rows = disease_scores(r)
    bars = ""
    for d in rows:
        c = _score_colour(d["gated"])
        drv = d["driver"]
        sub = (f'strongest mechanism: {MECH_LABEL.get(drv[0], drv[0])} ({drv[1]:.0%}) × BBB {bbb:.0%}'
               if drv and d["signal"] > 0 else f'no engaged mechanism · BBB {bbb:.0%}')
        if drv and d["signal"] > 0 and low_power_target(drv[0]):
            sub += (f' &middot; <span style="color:{AMBER};font-weight:700">the '
                    f'{MECH_LABEL.get(drv[0], drv[0])} model is below the reliability gate, so this '
                    f'may be under-called</span>')
        if d["gated"] < MIN_ACTIONABLE_SCORE:
            sub += (f' &middot; <span style="color:{MUTE2}">below the actionable threshold of '
                    f'{MIN_ACTIONABLE_SCORE:.0%}, treat as no finding</span>')
            c = MUTE2
        bars += _bar(d["gated"], c, d["disease"], f'{d["gated"]:.0%}')
        bars += f'<div class="bs-barsub">{sub}</div>'
    profile = interpret_profile(r, bbb, neuro, rows)
    st.markdown(f'<div class="bs-card"><div class="bs-h">Brain-relevance signal'
                f'<span class="bs-h-sub">strongest engaged mechanism, scaled by BBB exposure</span></div>'
                f'<div class="bs-note" style="margin:-4px 0 12px;padding:9px 12px;background:#F7FAFE;'
                f'border-left:3px solid {NAVY_MD};border-radius:0 8px 8px 0">{profile}</div>{bars}'
                f'<div class="bs-note" style="margin-top:6px">Signal = enrichment of each mechanism <i>over its '
                f'training base rate</i> (raw probability alone is not engagement), then scaled by predicted BBB '
                f'penetration. Research prioritisation only, not clinical efficacy.</div></div>',
                unsafe_allow_html=True)


def render_targets(r):
    rows = []
    for ep, ctx in TARGET_CLASSIFIERS.items():
        p = r["targets"][ep]; br = base_rate(ep); e = enrichment(ep, p)
        engaged = e >= 0.2
        if ep in LIABILITY_TARGETS:
            call = _badge("liability", ADVERSE) if engaged else _badge("clear", GREEN)
        elif engaged:
            call = _badge("engaged", BLUE)
        elif e >= 0.05:
            call = _badge("weak", AMBER, filled=False)
        else:
            call = _badge("inactive", MUTE2, filled=False)
        base_txt = f"{br:.0%}" if br is not None else "n/a"
        dcls = "up" if e >= 0.2 else ("mid" if e >= 0 else "down")
        ar = "▲" if e >= 0 else "▼"
        vs = (f'<span class="bs-delta {dcls}"><span class="ar">{ar}</span>{e:+.0%}</span>'
              f'<div class="bs-basetxt">base rate {base_txt}</div>')
        rows.append([f'<b>{ep}</b>', f'<span class="bs-ctx">{ctx}</span>',
                     f'<span class="bs-num">{p:.0%}</span>', vs, call])
    html = _table(["Endpoint", "Context", "P(active)", "vs base rate", "Call"], rows, ["l", "l", "r", "r", "c"])
    st.markdown(f'<div class="bs-card"><div class="bs-h">Target engagement'
                f'<span class="bs-h-sub">enrichment over base rate</span></div>{html}'
                f'<div class="bs-note" style="margin-top:8px"><b>Call</b> reflects enrichment over each endpoint\'s '
                f'measured base rate rather than the raw probability. Where the active fraction is high '
                f'(BACE1 91%, GSK-3β 93%), only a substantial exceedance is counted as engagement.</div></div>',
                unsafe_allow_html=True)


def render_adme(r):
    rows = []
    for ep, (name, unit, tf) in ADME.items():
        v = r["adme"][ep]; val = f"{v:.0%}" if tf == "proba" else f"{v:.2f}"
        rows.append([name, f'<span class="bs-num">{val}</span>', f'<span class="bs-ctx">{unit}</span>'])
    html = _table(["Property", "Value", "Units"], rows, ["l", "r", "l"])
    st.markdown(f'<div class="bs-card"><div class="bs-h">ADME / exposure (ADMET)'
                f'<span class="bs-h-sub">absorption, distribution, clearance</span></div>{html}</div>', unsafe_allow_html=True)


def render_receptors(r):
    has_binder = bool(r.get("receptor_binder"))
    rows = []
    for ep in RECEPTOR_REGRESSORS:
        v = r["receptors"][ep]; bp = r.get("receptor_binder", {}).get(ep)
        if bp is not None:
            thr = binder_threshold(ep)
            call = _badge("binder", BLUE) if bp >= thr else _badge("non-binder", MUTE2, filled=False)
            bpc = f'<span class="bs-num">{bp:.0%}</span>'
            pki = f'<span class="bs-ctx">{v:.1f}</span>' if bp >= thr else '<span class="bs-ctx">n/a</span>'
        else:
            call = _badge("n/a", MUTE2, filled=False)
            pct = reg_percentile(ep, v)
            bpc = "n/a"; pki = f'<span class="bs-ctx">{v:.2f} (p{pct:.0%})</span>' if pct is not None else f"{v:.2f}"
        rows.append([RECEPTOR_REGRESSORS[ep], bpc, pki, call])
    apct = reg_percentile("antioxidant", r["antioxidant"])
    rows.append(["Antioxidant (DPPH)", "n/a", f'<span class="bs-num">{r["antioxidant"]:.2f}</span>',
                 f'<span class="bs-ctx">p{apct:.0%}</span>' if apct is not None else "n/a"])
    html = _table(["Target / assay", "Binder P", "pKi / value", "Call"], rows, ["l", "r", "r", "c"])
    note = ("<b>Binder probability</b> is a classifier trained to distinguish measured binders "
            "(pChEMBL &ge; 7) from property-matched non-binders. It is validated against compounds "
            "experimentally tested on the same target and found inactive, which were never used in "
            "training: mean AUROC 0.917 across the 47 deployed targets. Each model is fitted on "
            "property-matched "
            "decoys together with half of the measured inactives and is validated and thresholded on "
            "the held-out half, so no compound used for fitting is used to judge it. Thresholds hold "
            "the false-positive rate near 10% on those measured inactives AND no more than 5% on "
            "unrelated chemistry, whichever is stricter, giving a mean sensitivity of 0.90; the "
            "conventional 0.5 "
            "cut would be far too permissive. Predicted "
            "pK<sub>i</sub> is shown only for compounds called binders, as potency among binders. "
            "The numeric pK<sub>i</sub> and the antioxidant value are weak priors rather than affinity "
            "predictions: on a temporal (future-compound) split these regressions reach only R&sup2; 0.01 "
            "to 0.34, so read them qualitatively. "
            "<b>Read the predicted pK<sub>i</sub> as an upper bound.</b> Measured against 291 "
            "compounds that are absent from the training set and from the applicability-domain "
            "reference, and that this panel calls binders, the predicted value runs "
            "<b>1.04 log units too potent on average</b> (mean absolute error 1.05), a factor of "
            "about eleven. The over-prediction does not shrink for compounds close to known "
            "chemistry: it is roughly constant across the whole similarity range (Spearman "
            "&minus;0.07 between error and distance), so a high applicability-domain score is no "
            "protection against it. Rank order is better preserved than the absolute value "
            "(Spearman 0.19 to 0.28 against measured potency), so use these numbers to order "
            "compounds, not to estimate affinity."
            if has_binder else
            "Receptor potency models are trained on binder-only assays; interpret the value as a weak prior "
            "rather than a definitive binding call.")
    st.markdown(f'<div class="bs-card"><div class="bs-h">Receptor binding and antioxidant'
                f'<span class="bs-h-sub">binder classification</span></div>{html}'
                f'<div class="bs-note" style="margin-top:8px">{note}</div></div>', unsafe_allow_html=True)


def render_readacross(smiles, r):
    """Evidence available when no model fires: what the nearest measured compounds are active at."""
    nn = read_across(smiles)
    if not nn:
        body = ('<div class="bs-note">No measured compound lies within a Tanimoto of 0.35 of this '
                'structure. The chemistry is outside the measured library, so no similarity-based '
                'inference is defensible. This is a genuine absence of evidence rather than evidence '
                'of absence.</div>')
    else:
        agg = read_across_targets(smiles)
        if agg:
            arows = []
            for a in agg[:6]:
                e = a["evidence"]
                col = GREEN if e >= 0.8 else (AMBER if e >= 0.5 else MUTE2)
                arows.append([
                    (f'<b>{MECH_LABEL.get(a["target"].rstrip("*"), a["target"].rstrip("*"))}</b>'
                     + ('<span class="bs-ctx"> (not modelled)</span>' if a["target"].endswith("*") else '')),
                    f'<span class="bs-num" style="color:{col}">{e:.2f}</span>',
                    f'{a["n"]}',
                    f'{a["best_sim"]:.2f}'])
            body_top = (
                '<div class="bs-note" style="font-weight:700;color:' + NAVY + ';margin-bottom:6px">'
                'Aggregated evidence per target</div>'
                + _table(["Target", "Evidence", "Supporting neighbours", "Closest"],
                         arows, ["l", "r", "c", "r"])
                + '<div class="bs-note" style="margin:8px 0 14px">Evidence combines every supporting '
                  'neighbour as an independent observation, 1 minus the product of (1 minus each '
                  'similarity), so five neighbours at 0.55 outweigh a single one at 0.70 while one '
                  'very close analogue can still carry a target alone.</div>')
        else:
            body_top = ""
        rows = []
        for h in nn:
            tg = ", ".join(MECH_LABEL.get(t.rstrip("*"), t.rstrip("*")) + ("*" if t.endswith("*") else "")
                           for t in h["targets"][:5])
            s = h["similarity"]
            col = GREEN if s >= 0.7 else (AMBER if s >= 0.5 else MUTE2)
            rows.append([
                f'<span class="bs-delta {"up" if s >= 0.7 else "mid" if s >= 0.5 else "down"}">'
                f'{s:.2f}</span>',
                f'<span class="bs-num" style="color:{col}">{tg or "no annotated target"}</span>',
                f'<span class="bs-ctx" style="word-break:break-all">{_esc(h["smiles"][:46])}</span>'])
        body = body_top + '<div class="bs-note" style="font-weight:700;color:' + NAVY + '">Nearest measured neighbours</div>' + _table(["Tanimoto", "Measured active at", "Structure"], rows, ["c", "l", "l"])
    st.markdown(
        f'<div class="bs-card"><div class="bs-h">Read-across evidence'
        f'<span class="bs-h-sub">what the nearest measured compounds do</span></div>{body}'
        f'<div class="bs-note" style="margin-top:8px">These are statements about the <b>neighbours</b>, '
        f'not about the query. Read-across is weaker than a validated model and fails at activity '
        f'cliffs, where one substitution abolishes binding, so the similarity is shown for every row. '
        f'It is most useful when no model fires: a compound with no direct engagement may still sit in '
        f'a chemical neighbourhood with known brain pharmacology. Index: 153,038 measured actives '
        f'across 72 targets, of which 22 are marked <i>not modelled</i> or asterisked because this '
        f'project trains no model for them; for those, similarity is the only evidence offered.</div></div>', unsafe_allow_html=True)


def render_cns_mpo(r, smiles=None):
    m = cns_mpo(r, smiles)
    if m is None:
        return
    frac = m["total"] / m["max"]
    col = GREEN if frac >= 0.7 else (AMBER if frac >= 0.5 else ADVERSE)
    bars = "".join(
        _bar(v, GREEN if v >= 0.7 else (AMBER if v >= 0.4 else MUTE2), k, f"{v:.2f}")
        for k, v in m["parts"].items())
    if m.get("pka") is not None:
        pka_txt = (f'The most basic pK<sub>a</sub> is predicted at <b>{m["pka"]:.1f}</b> by a model '
                   f'trained here on 6,384 measured ChEMBL compounds carrying a basic centre '
                   f'(scaffold-split R&sup2; 0.456, mean absolute error 1.17 pK<sub>a</sub> units). '
                   f'That error is appreciable against a desirability transition running from 8 to '
                   f'10, so this is the least certain of the six terms, and its prospective '
                   f'behaviour is weaker still at a temporal R&sup2; of 0.12.')
    else:
        pka_txt = ('The most basic pK<sub>a</sub> term is omitted because no pK<sub>a</sub> model is '
                   'deployed, so the total is on a 0 to 5 scale rather than the published 0 to 6. '
                   'Without it, strongly basic compounds are over-scored.')
    st.markdown(
        f'<div class="bs-card"><div class="bs-h">CNS-likeness (CNS MPO)'
        f'<span class="bs-h-sub">physicochemical desirability, no target required</span></div>'
        f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:12px">'
        f'<span style="font-size:2rem;font-weight:800;color:{col};line-height:1">'
        f'{m["total"]:.2f}</span><span class="bs-ctx">of {m["max"]:.0f}</span></div>{bars}'
        f'<div class="bs-note" style="margin-top:6px">Desirability functions after Wager et al. '
        f'(ACS Chemical Neuroscience, 2010). cLogD is taken from the measured-data lipophilicity '
        f'model. {pka_txt}<br><br>'
        f'<b>Read this alongside the exposure models, not in place of them.</b> CNS MPO is a '
        f'physicochemical heuristic that carries no information about transporters, so an '
        f'actively effluxed or poorly absorbed compound can still score well. Where this score '
        f'and the predicted K<sub>p,uu</sub> disagree, the measured-data exposure models are the '
        f'stronger evidence.</div></div>',
        unsafe_allow_html=True)


def render_physchem(r):
    d = r["descriptors"]
    show = [("MW", d.get("mw"), "{:.0f}"), ("clogP", d.get("clogp"), "{:.2f}"), ("TPSA", d.get("tpsa"), "{:.0f}"),
            ("HBD", d.get("hbd"), "{:.0f}"), ("HBA", d.get("hba"), "{:.0f}"),
            ("Rot. bonds", d.get("rotatable_bonds"), "{:.0f}"), ("Arom. rings", d.get("aromatic_rings"), "{:.0f}"),
            ("QED", d.get("qed"), "{:.2f}")]
    rows = [[k, f'<span class="bs-num">{fmt.format(v)}</span>'] for k, v, fmt in show if v is not None]
    html = _table(["Descriptor", "Value"], rows, ["l", "r"])
    st.markdown(f'<div class="bs-card"><div class="bs-h">Physicochemical profile'
                f'<span class="bs-h-sub">RDKit descriptors</span></div>{html}</div>', unsafe_allow_html=True)


def render_decision_guidance():
    """State what acting on a positive call costs, because that is what decides wet-lab spend.

    Discrimination is prevalence-free; a laboratory is not. At the deployed operating point
    (sensitivity 0.90, false-positive rate 0.125 measured on 1000 non-CNS compounds) precision
    depends entirely on how common real actives are in the set being examined, and it collapses when
    they are rare. Presenting a positive call without that context invites exactly the wasted animal
    and reagent spend this tool exists to prevent."""
    st.markdown(
        f"""
        <div class="bs-card" style="border-left:5px solid {AMBER};margin-bottom:16px">
          <div style="font-weight:800;color:{NAVY};font-size:.95rem;margin-bottom:6px">
            How to act on a positive call</div>
          <div class="bs-note">
            This tool ranks well but is <b>not</b> a substitute for a binary go or no-go decision when
            real actives are rare. At the deployed operating point, sensitivity 0.90 and a measured
            false-positive rate of 12.5%, the precision of a positive call depends on how common
            actives are in the set you are screening:
            <ul style="margin:8px 0 0;padding-left:18px;line-height:1.7">
              <li>focused set, roughly half active: precision <b>88%</b>, about 0.1 false leads per true hit</li>
              <li>enriched set, roughly one in ten: precision <b>44%</b>, about 1.3 false leads per true hit</li>
              <li>diverse library, roughly one in a hundred: precision <b>7%</b>, about
                  <b>14 false leads per true hit</b></li>
            </ul>
            <br>Use it to <b>prioritise</b> a ranked list and test from the top, which is where the
            0.79 prospective recall pays off. A negative call is the more dependable direction: the
            negative predictive value exceeds 0.98 whenever actives are uncommon, so this is a better
            instrument for deprioritising than for selecting.
            <br><br>For a large diverse library, switch on <b>high-precision screening mode</b> above.
            It raises every threshold to a 1% background false-positive rate, which lifts precision at
            1% prevalence from 7% to <b>37%</b> and cuts the cost from about 14 wasted experiments per
            true hit to <b>1.7</b>. Mean sensitivity falls from 0.90 to 0.69 in exchange, so the
            stricter mode finds fewer of the real actives but wastes far less effort on the ones it
            reports. Neither setting is the correct one in general: the choice depends on whether a
            missed active or a wasted experiment is the more costly error in your campaign.
          </div>
        </div>
        """, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _domain_reliability():
    """Prospective performance by applicability-domain stratum, read from the artefact.

    These three figures are shown above every result, and they were hardcoded as 0.83, 0.74 and 0.57
    until an audit of the deployed server found the table behind them three weeks older than the
    models it described. Quoting a constant in the interface is the same defect as quoting a stale
    file: neither can be checked, and neither changes when the panel is refitted. The spread across
    endpoints is carried too, because averaging seven endpoints whose in-domain AUROC ranges from
    0.70 to 0.93 into a single number states more precision than the data holds.
    """
    p = ROOT / "results" / "tables" / "temporal_by_domain.csv"
    if not p.exists():
        return {}
    try:
        d = pd.read_csv(p)
        out = {}
        for s in d.stratum.unique():
            c = d[(d.stratum == s) & (d.task == "classification")].auroc.dropna()
            r = d[(d.stratum == s) & (d.task == "regression")].spearman.dropna()
            if not len(c):
                continue
            out[s] = {"auroc": float(c.mean()), "auroc_lo": float(c.min()),
                      "auroc_hi": float(c.max()),
                      "spearman": float(r.mean()) if len(r) else float("nan"),
                      "n_endpoints": int(len(c))}
        return out
    except Exception:
        return {}


def render_reliability_banner(smiles):
    """Prospective-reliability statement based on the query's applicability domain.

    Temporal validation (train on past compounds, test on future ones) shows that predictive power
    is retained inside the domain and degrades outside it. The figures are read from
    results/tables/temporal_by_domain.csv rather than written here, because they move whenever the
    panel is refitted and a number in a docstring does not.

    Two honesty notes travel with the banner. The per-endpoint spread is quoted alongside the mean,
    since in-domain AUROC ranges roughly 0.70 to 0.93 across seven classifiers and a single average
    overstates what is known. And the gradient is not monotone for every endpoint: BACE1 and GSK3B
    score higher near the domain edge than inside it, on strata of a few dozen compounds. The
    banner describes a tendency, not a law."""
    ad = assess_domain(smiles)
    if ad is None:
        return
    s = ad["max_sim"]
    stratum = "in_domain (T>=0.5)" if s >= 0.5 else               "near_domain (0.3-0.5)" if s >= 0.3 else "out_domain (T<0.3)"
    st_ = _domain_reliability().get(stratum)
    if st_ is None:                       # artefact absent; say less rather than quote a constant
        auroc_txt = rho_txt = None
    else:
        auroc_txt = (f"an AUROC of about {st_['auroc']:.2f} "
                     f"(per endpoint {st_['auroc_lo']:.2f} to {st_['auroc_hi']:.2f})")
        rho_txt = f"a rank correlation of about {st_['spearman']:.2f}"

    if s >= 0.5:
        col, head = GREEN, "Inside the applicability domain"
        body = ("This compound resembles measured training chemistry (max Tanimoto "
                f"{s:.2f}).")
        if auroc_txt:
            body += (f" On future compounds in this regime the classifiers hold {auroc_txt} and the "
                     f"potency models {rho_txt}.")
    elif s >= 0.3:
        col, head = AMBER, "Near the edge of the applicability domain"
        body = (f"This compound is only moderately similar to training chemistry (max Tanimoto {s:.2f}).")
        if auroc_txt:
            body += (f" On future compounds in this regime the classifiers fall to {auroc_txt} and the "
                     f"potency models to {rho_txt}.")
        body += " Treat the profile as indicative."
    else:
        col, head = ADVERSE, "Outside the applicability domain"
        body = (f"This compound is structurally unlike the measured training data (max Tanimoto {s:.2f}).")
        if auroc_txt:
            body += (f" On future compounds in this regime the classifiers approach chance, {auroc_txt}, "
                     f"and the potency models retain {rho_txt}.")
        body += (" Use the qualitative direction only, and confirm experimentally.")

    # Two different domain stratifications exist in this project and a reader meeting both without
    # warning sees a contradiction. This banner reports ranking quality (AUROC and rank correlation)
    # in the three bands the deployed applicability flag uses, at Tanimoto 0.3 and 0.5. The
    # prospective validation reports recall in four narrower bands, at 0.40, 0.55 and 0.70, because
    # recall turned out to vary within what this banner treats as one band. They measure different
    # things on different cuts, and both are quoted here rather than leaving the user to reconcile
    # a number in the interface with a number in the paper.
    er = ad.get("expected_recall")
    if er:
        body += (f" Measured recall for compounds at this distance is {er['recall']:.2f} "
                 f"(n={er['n']:,}), from a separate prospective analysis that bands distance more "
                 f"finely than the three tiers used here.")
    if st_ and st_.get("n_endpoints"):
        body += (f" Averaged over {st_['n_endpoints']} classifiers; the gradient is a tendency and "
                 f"not monotone for every endpoint.")
    st.markdown(
        f'<div class="bs-card" style="border-left:5px solid {col};margin-bottom:16px">'
        f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:5px">'
        f'<span class="bs-dot" style="background:{col};width:9px;height:9px"></span>'
        f'<span style="font-weight:800;color:{col};font-size:.95rem">{head}</span></div>'
        f'<div class="bs-note" style="margin:0">{body}</div></div>', unsafe_allow_html=True)


def render_confidence(smiles):
    ad = assess_domain(smiles)
    if ad is None:
        st.markdown('<div class="bs-card"><div class="bs-h">Applicability &amp; confidence</div>'
                    '<div class="bs-note">Reference library unavailable.</div></div>', unsafe_allow_html=True)
        return
    nn_mol = Chem.MolFromSmiles(ad["nearest_smiles"])
    nn_img = _mol_data_uri(nn_mol, size=(220, 150)) if nn_mol else ""

    # per-endpoint domain grid
    pe = assess_per_endpoint(smiles)
    grid = ""
    if pe:
        labels = {**{e: e for e in TARGET_CLASSIFIERS},
                  **{e: MECH_LABEL.get(e, e) for e in list(RECEPTOR_REGRESSORS) + BINDER_TARGETS},
                  "antioxidant": "antiox"}
        order = list(TARGET_CLASSIFIERS) + list(RECEPTOR_REGRESSORS) + BINDER_TARGETS + ["antioxidant"]
        chips = ""
        for ep in order:
            s = pe.get(ep)
            if s is None:
                continue
            col = GREEN if s >= 0.5 else (AMBER if s >= 0.3 else MUTE2)
            chips += (f'<span class="bs-badge" style="background:{col}1a;color:{col};border-color:{col}55;'
                      f'margin:2px" title="max Tanimoto {s:.2f}">{labels.get(ep, ep)} {s:.2f}</span>')
        grid = (f'<div style="margin-top:12px;padding-top:12px;border-top:1px solid {LINE}">'
                f'<div class="bs-note" style="font-weight:700;color:{NAVY};margin-bottom:6px">'
                f'Per-endpoint applicability (max Tanimoto to that endpoint\'s chemistry)</div>'
                f'<div class="bs-chips" style="line-height:2">{chips}</div>'
                f'<div class="bs-note" style="margin-top:6px">Green in-domain (&ge;0.5), amber near-domain '
                f'(0.3 to 0.5), grey out-of-domain (&lt;0.3). Read an endpoint\'s prediction with more caution '
                f'when its chip is grey.</div></div>')

    st.markdown(
        f"""
        <div class="bs-card"><div class="bs-h">Applicability and confidence
          <span class="bs-h-sub">distance to measured training chemistry</span></div>
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <span class="bs-badge" style="background:{ad['colour']}1a;color:{ad['colour']};border-color:{ad['colour']}55">
              {ad['tier']} · {ad['confidence']} confidence</span>
            <span class="bs-ctx">global max Tanimoto {ad['max_sim']:.2f} to {ad['n_ref']:,} measured compounds</span>
          </div>
          <div style="display:flex;gap:14px;align-items:center">
            <img src="{nn_img}" style="border:1px solid {LINE};border-radius:8px;background:#fff"/>
            <div class="bs-note"><b>Nearest measured analogue</b><br>
              Tanimoto {ad['max_sim']:.2f}. Predictions are most reliable when this is high (&ge;0.5).<br>
              <span class="bs-ctx" style="word-break:break-all">{ad['nearest_smiles']}</span></div>
          </div>{grid}
        </div>
        """,
        unsafe_allow_html=True,
    )


COVERAGE_YES = [
    ("BBB penetration", "B3DB"), ("Cholinesterases AChE, BChE", "ChEMBL"),
    ("Amyloid BACE1, tau GSK-3β", "ChEMBL"), ("Monoamine oxidase MAO-A, MAO-B", "ChEMBL"),
    ("Serotonin 5-HT1A, 5-HT2A, 5-HT6, 5-HT7", "ChEMBL"),
    ("Dopamine D2, D3; transporters DAT, NET, SERT", "ChEMBL"),
    ("Opioid mu, kappa; cannabinoid CB1; sigma-1", "ChEMBL"),
    ("Histamine H3; adenosine A1, A2A; alpha7-nAChR", "ChEMBL"),
    ("Glutamate NMDA GluN2B, mGluR5; GABA-A", "ChEMBL"),
    ("Orexin OX1, OX2; melatonin MT1", "ChEMBL"),
    ("Neuroinflammation NLRP3, P2X7, COX-2, CSF1R", "ChEMBL"),
    ("Epigenetic and proteostasis HDAC1, HDAC6, SIRT1, mTOR, KEAP1", "ChEMBL"),
    ("Parkinson's LRRK2, GBA1; Huntington's PDE10A; cognition PDE4B", "ChEMBL"),
    ("Sodium channels Nav1.7 (pain), Nav1.1 (epilepsy), Nav1.5 (cardiac safety)", "ChEMBL"),
    ("Safety hERG; measured antioxidant (DPPH)", "ChEMBL"),
    ("ADMET: Kp,uu, logBB, Caco-2, P-gp, solubility, logD, PPB, clearance", "measured"),
]
# The low-sensitivity list is computed, never written down.
#
# It was previously hand-maintained and drifted badly: it claimed TAAR1 at sensitivity 0.54 and AMPA
# GluA2 at 0.58 when the deployed values were 0.774 and 0.660, while omitting the one endpoint that
# genuinely qualified. A coverage statement that is wrong about its own models is worse than none,
# because it is read as an audit. Everything below is therefore derived from binder_modes.json at
# render time and cannot disagree with what is deployed.
#
# It then acquired a second defect of the same family. Selection used a 0.65 cut of its own, while
# low_power_target() marked results using the 0.50 reliability gate, so this page named ten endpoints
# where six carried the marker. Both numbers were correct about what they measured and the page gave
# a reader no way to see that. Selection is now the gate, so the list on this page and the marker on
# a result are the same statement made twice.


def coverage_low():
    """Deployed endpoints that fail the reliability gate, so a negative call from them carries little.

    Selected on reliable_call, the field low_power_target() reads, and explained in terms of whichever
    floor the endpoint actually missed rather than assuming it was sensitivity.
    """
    out = []
    for ep, v in sorted(load_binder_modes().items()):
        if not v.get("deployed", True) or v.get("reliable_call") is not False:
            continue
        s, a = v.get("sensitivity_at_threshold"), v.get("auroc_vs_measured_inactives")
        missed = []
        if s is not None and s < _panel.MIN_SENSITIVITY:
            missed.append(f"sensitivity {s:.2f} on actives withheld by scaffold")
        if a is not None and a < _panel.MIN_AUROC:
            missed.append(f"AUROC {a:.2f} against compounds measured inactive")
        why = " and ".join(missed) if missed else "below the reliability gate"
        if v.get("sensitivity_note"):
            why += ", by necessity rather than miscalibration"
        out.append((MECH_LABEL.get(ep, ep), why))
    return out


def coverage_modelled():
    """Every deployed mechanism, grouped for reading, built from the live panel.

    Grouping labels are curated for legibility but membership is not: a target appears here if and
    only if it is scorable and not withdrawn, so an endpoint cannot be advertised after it has been
    removed, nor omitted after it has been added."""
    modes = load_binder_modes()
    live = {t for t in TARGET_KIND if t != "NEURO"
            and modes.get(t, {}).get("deployed", True)}
    groups = [
        ("Blood-brain barrier and exposure", ["BBB"], "B3DB"),
        ("Cholinesterases and cognition", ["AChE", "BChE", "a7nAChR", "H3", "HT6", "PDE4B"], "ChEMBL"),
        ("Amyloid and tau", ["BACE1", "GSK3B"], "ChEMBL"),
        ("Monoamine oxidases", ["MAO_A", "MAO_B"], "ChEMBL"),
        ("Serotonin receptors", ["HT1A", "HT2A", "HT6", "HT7"], "ChEMBL"),
        ("Dopamine and transporters", ["D2", "D3", "DAT", "NET", "SERT", "TAAR1"], "ChEMBL"),
        ("Opioid, cannabinoid, sigma", ["OPRM1", "OPRK1", "CB1", "Sigma1"], "ChEMBL"),
        ("Adenosine and nicotinic", ["A1", "A2A", "a7nAChR", "a4b2nAChR", "a3b4nAChR"], "ChEMBL"),
        ("Glutamate and GABA", ["GluN2B", "mGluR5", "GABA_A"], "ChEMBL"),
        ("Sleep and circadian", ["OX1", "OX2", "MT1"], "ChEMBL"),
        ("Neuroinflammation", ["NLRP3", "P2X7", "COX2", "CSF1R"], "ChEMBL"),
        ("Epigenetic and proteostasis", ["HDAC1", "HDAC6", "SIRT1", "mTOR", "KEAP1"], "ChEMBL"),
        ("Parkinson's and Huntington's", ["LRRK2", "GBA1", "PDE10A"], "ChEMBL"),
        ("Ion channels", ["Nav1_5", "Nav1_6", "Nav1_7", "hERG"], "ChEMBL"),
        ("Pain-selective channels", ["Nav1_8"], "ChEMBL"),
        ("Migraine and multiple sclerosis", ["CGRP", "DHODH"], "ChEMBL"),
        ("Necroptosis", ["RIPK1"], "ChEMBL"),
    ]
    out, seen = [], set()
    for label, members, src in groups:
        present = [m for m in members if m in live]
        if not present:
            continue
        seen |= set(present)
        out.append((f"{label}: " + ", ".join(MECH_LABEL.get(m, m) for m in present), src))
    missing = sorted(live - seen)
    if missing:
        # never silently drop a deployed endpoint from its own coverage statement
        out.append(("Other deployed endpoints: "
                    + ", ".join(MECH_LABEL.get(m, m) for m in missing), "ChEMBL"))
    out.append(("Antioxidant capacity (DPPH)", "ChEMBL"))
    out.append(("ADMET: Kp,uu, logBB, Caco-2, P-gp substrate and inhibition, solubility, logD, "
                "plasma-protein binding, clearance", "measured"))
    return out


def coverage_withdrawn():
    """Endpoints trained and then removed, with the reason, so a removal is visible not silent."""
    return [(MECH_LABEL.get(ep, ep), v.get("withdrawn_reason", "withdrawn"))
            for ep, v in sorted(load_binder_modes().items())
            if not v.get("deployed", True)]
# What is not modelled, and why, with the measured evidence behind each claim.
#
# Every count below comes from a systematic ChEMBL query (results/unmodelled_target_data.csv) or,
# where stated, from results/tables/np_target_survey.csv, rather than from recollection, because
# "there is not enough data" ages badly as ChEMBL grows. That claim was audited against the files
# and four entries failed it: a kainate count attributed to GluK3 that the query records against a
# generic receptor, an assay described as thioflavin-S that the query records only as a fibril
# counterscreen, activity counts for SARM1 and KCNQ2 that appear in no table here, and 4 measured
# inactives for GABA-A alpha5 where the endpoint table holds 12. All four are corrected. A number
# in this list that cannot be traced to a file is worse than no number, because the list exists to
# be trusted about limits. The reasons
# are not interchangeable: too few ligands is a different problem from volume without diversity,
# which is different again from a binding site that the target annotation does not separate, and
# only the first is fixed by waiting. A health check asserts that nothing named here is in fact
# deployed, which is how the previous version of this list was found to be wrong.
COVERAGE_NO = [
    "Kainate receptors: GluK1 has 244 measured activities and GluK2 has 139, far below the 800 "
    "required to train an endpoint that survives a scaffold split. A further 34 are annotated to "
    "a generic kainate-receptor target that does not resolve to a subunit at all",
    "The NMDA channel-blocker site used by ketamine and phencyclidine. ChEMBL annotates activity to "
    "the protein rather than the binding site, so channel blockers, glycine-site ligands and GluN2B "
    "allosteric modulators are pooled despite unrelated structure-activity relationships. The GluN2B "
    "allosteric site modelled here can be isolated; the channel site cannot without assay-level "
    "curation",
    "Vesicular monoamine transporter VMAT2, the target of tetrabenazine: 149 activities from 12 "
    "sources, too few and too narrow to train",
    "Calcium channel subtype Cav2.2: 566 measured activities, below the bar for an endpoint that "
    "survives a scaffold split. Cav3.2 is modelled and deployed",
    "Aggregation of alpha-synuclein, tau and huntingtin. These are phenotypic assays with no defined "
    "binding site. Tau carries 95,345 potency values, more than any deployed endpoint, but 86 per "
    "cent come from a single high-throughput fibril-formation counterscreen and a 1,000-activity "
    "sample draws on "
    "one document; huntingtin is 98 per cent one assay. A model trained on either learns the "
    "screening library, not the protein",
    "ALS genetics: SOD1 has 29 measured activities, TDP-43 has 9, and C9orf72 has no ChEMBL target "
    "record at all. Amyotrophic lateral sclerosis itself is scored, through eight mechanisms "
    "including RIPK1-mediated necroptosis, but none of them is one of the genetic lesions that "
    "define the familial forms",
    "Axon-degeneration and excitability targets proposed for ALS: SARM1 is recorded in this "
    "project's target survey with a single compound and five measurements, and KCNQ2 was never "
    "surveyed, so neither is anywhere near the bar and neither carries a count this repository "
    "can support",
    "GABA-A subtype alpha5-beta3-gamma2: 672 actives were fetched and audited but only 12 measured "
    "inactives exist, so no threshold could be set honestly. The pooled GABA-A endpoint is deployed "
    "in its place",
    "Stroke and cerebral ischaemia, the one condition on this list with no comparable small-molecule "
    "target set, consistent with four decades of failed neuroprotection trials. Migraine and multiple "
    "sclerosis were on this list and have been removed: both are now scored, through the "
    "calcitonin-gene-related-peptide receptor and dihydroorotate dehydrogenase respectively",
]

# The mechanisms the prose above asserts are absent, named in canonical form so the claim can be
# tested rather than read. Prose is not testable: an entry explaining that the pooled GABA-A endpoint
# stands in for an unmodelled subtype necessarily contains the string "GABA-A", which a text search
# cannot distinguish from a claim that GABA-A itself is missing.
COVERAGE_NO_MECHANISMS = {
    "GluK1", "GluK2", "GluK3", "NMDA_PCP_site", "VMAT2", "Cav2_2",
    "SNCA", "MAPT", "HTT", "SOD1", "TARDBP", "C9orf72", "GABAA_a5", "SARM1", "KCNQ2",
}

# A drug class the panel cannot represent, which is a different kind of gap from a missing target and
# is stated separately because no further endpoint will close it.
COVERAGE_CLASS_LIMIT = (
    "Use-dependent, state-dependent channel blockers. The panel defines engagement as high-affinity "
    "binding at pChEMBL 7 or better, roughly 100 nM. The classic antiepileptic drugs act one to two "
    "orders of magnitude weaker, and ChEMBL's own measurements label them inactive at the sodium "
    "channels through which they work: carbamazepine 4.49 at Nav1.7, lamotrigine 4.77, mexiletine "
    "4.93, lacosamide 6.74. Silence on carbamazepine is therefore correct with respect to the "
    "measured data and unhelpful with respect to the clinical question. Covering this class needs a "
    "different kind of endpoint, modelling use-dependent block or a phenotypic outcome, not another "
    "binding target.")


RESULTS = ROOT / "results" / "tables"
# --------------------------------- export ---------------------------------
# A prediction a scientist cannot save is a prediction they cannot act on: it cannot go into a
# notebook, a grant, a supplementary table, or a colleague's inbox, and it cannot be re-checked
# months later against what the bench actually found. Every export below is generated from one tidy
# frame, so the CSV, the JSON and the printable report can never disagree with the screen or with
# each other.

APP_VERSION = "1.0"


def result_frame(smiles, name, r):
    """Every number the app computes, as one tidy table. This is the single source for all exports."""
    bbb, neuro, dz = disease_scores(r)
    rows = []

    def add(section, endpoint, description, value, unit="", context=""):
        rows.append({"section": section, "endpoint": endpoint, "description": description,
                     "value": value, "unit": unit, "context": context})

    for ep, desc in TARGET_CLASSIFIERS.items():
        p = r["targets"][ep]
        br = base_rate(ep)
        add("Target and property models", ep, desc, round(p, 4), "calibrated probability",
            f"training base rate {br:.3f}; enrichment {enrichment(ep, p):+.3f}" if br else "")

    for ep in RECEPTOR_REGRESSORS:
        v = r["receptors"][ep]
        pct = reg_percentile(ep, v)
        add("Receptor potency", ep, RECEPTOR_REGRESSORS[ep], round(v, 3), "pKi / pIC50",
            f"{pct:.0%} of measured training actives are weaker" if pct is not None else "")

    for ep, p in sorted(r.get("receptor_binder", {}).items()):
        thr = binder_threshold(ep)
        add("Binder panel", ep, MECH_LABEL.get(ep, ep), round(p, 4), "calibrated probability",
            f"reporting threshold {thr:.3f}; {'engaged' if p >= thr else 'below threshold'}")

    add("Neuroprotection", "antioxidant", "Antioxidant capacity (DPPH)",
        round(r["antioxidant"], 3), "model units",
        f"signal {neuro:.3f} on the 0 to 1 scale used by the disease layer")

    for ep, (label, unit, _tf) in ADME.items():
        add("ADME and exposure", ep, label, round(r["adme"][ep], 4), unit)

    for k, v in r["descriptors"].items():
        add("Physicochemical descriptors", k, k, round(float(v), 4), "")

    for d in dz:
        drv = d["driver"]
        add("Disease relevance", d["disease"], "BBB-gated mechanism score", round(d["gated"], 4),
            "0 to 1",
            f"strongest mechanism {MECH_LABEL.get(drv[0], drv[0])} at signal {drv[1]:.3f}"
            if drv else "no modelled mechanism engaged")

    m = cns_mpo(r, smiles)
    if m:
        add("CNS likeness", "CNS_MPO", "CNS multiparameter optimisation score",
            round(m["total"], 2), f"of {m['max']:.0f}",
            "; ".join(f"{k} {v:.2f}" for k, v in m["parts"].items()))

    neuro_sig = neuro_signal(r)
    engaged = [t for t in TARGET_KIND
               if t != "NEURO" and target_signal(r, neuro_sig, t) >= MIN_ENGAGEMENT_SIGNAL]
    _by_fam, multi = grouped_engagement(engaged)
    add("Mechanism independence", "n_targets_engaged",
        f"Targets with an engagement signal of at least {MIN_ENGAGEMENT_SIGNAL:.2f}",
        len(engaged), "count",
        "; ".join(f"{f}: {', '.join(MECH_LABEL.get(t, t) for t in ts)}" for f, ts in multi.items())
        or "no two engaged targets are homologous")
    add("Mechanism independence", "n_independent_mechanisms",
        "Distinct homology families among engaged targets", independent_mechanisms(engaged), "count",
        "homologues are engaged together by the same ligands, so a raw target count overstates the "
        "evidence; see inversion/results/H8_family_correlation.csv")

    ad = assess_domain(smiles)
    if ad:
        er = ad.get("expected_recall")
        note = f"{ad['tier']}; nearest analogue {ad['nearest_smiles']}"
        if er:
            # The strength of a negative depends on the recall at this distance, so the sentence
            # has to depend on it too. Saying "weak evidence" for a compound the panel recovers
            # 86 per cent of the time understates a real negative exactly as badly as saying
            # "strong evidence" would overstate one at 0.16.
            r_ = er["recall"]
            strength = ("reasonably strong" if r_ >= 0.75 else
                        "moderate" if r_ >= 0.50 else "weak")
            note += (f"; measured recall for compounds at this distance {r_:.2f} "
                     f"(n={er['n']:,}), so a silent endpoint here is {strength} evidence of "
                     f"inactivity")
        add("Applicability domain", "max_tanimoto", "Similarity to nearest measured training compound",
            round(ad["max_sim"], 4), "Tanimoto", note)
        if er:
            add("Expected sensitivity at this distance", "expected_recall",
                "Measured recall of the panel on compounds this far from its training chemistry",
                round(er["recall"], 4), "fraction",
                "prospective validation, results/tables/external_novelty_strata.csv; recall is a "
                "function of chemical distance rather than of publication date")

    df = pd.DataFrame(rows)
    df.insert(0, "compound", name)
    df.insert(1, "smiles", smiles)
    return df


def result_json(smiles, name, r):
    """Structured result for programmatic use, with provenance so a stored answer stays traceable."""
    bbb, neuro, dz = disease_scores(r)
    ad = assess_domain(smiles)
    m = cns_mpo(r, smiles)
    kpuu = r["adme"]["kpuu"]
    label, _colour, note = verdict(kpuu, bbb)
    return {
        "sai_cns_version": APP_VERSION,
        "screening_mode": "high-precision" if screening_mode() else "standard",
        "compound": {"name": name, "smiles": smiles},
        "exposure": {"bbb_probability": round(bbb, 4), "kpuu": round(kpuu, 4),
                     "verdict": label, "note": note},
        "targets": {k: round(v, 4) for k, v in r["targets"].items()},
        "receptors": {k: round(v, 3) for k, v in r["receptors"].items()},
        "binders": {k: {"probability": round(v, 4), "threshold": round(binder_threshold(k), 4),
                        "engaged": bool(v >= binder_threshold(k))}
                    for k, v in sorted(r.get("receptor_binder", {}).items())},
        "adme": {k: round(v, 4) for k, v in r["adme"].items()},
        "descriptors": {k: round(float(v), 4) for k, v in r["descriptors"].items()},
        "antioxidant": {"value": round(r["antioxidant"], 3), "signal": round(neuro, 4)},
        "diseases": [{"disease": d["disease"], "score": round(d["gated"], 4),
                      "driver": (MECH_LABEL.get(d["driver"][0], d["driver"][0]) if d["driver"] else None),
                      "reported": bool(d["gated"] >= MIN_ACTIONABLE_SCORE)} for d in dz],
        "cns_mpo": ({"score": round(m["total"], 2), "max": int(m["max"]),
                     "components": {k: round(v, 3) for k, v in m["parts"].items()}} if m else None),
        "applicability_domain": ({"max_tanimoto": round(ad["max_sim"], 4), "tier": ad["tier"],
                                  "confidence": ad["confidence"],
                                  "nearest_analogue": ad["nearest_smiles"]} if ad else None),
        "mechanism_independence": (lambda eng: {
            "n_targets_engaged": len(eng),
            "n_independent_mechanisms": independent_mechanisms(eng),
            "correlated_families": {f: [MECH_LABEL.get(t, t) for t in ts]
                                    for f, ts in grouped_engagement(eng)[1].items()},
        })([t for t in TARGET_KIND
            if t != "NEURO" and target_signal(r, neuro, t) >= MIN_ENGAGEMENT_SIGNAL]),
        "caveats": [
            "Research triage tool. Not for medical, diagnostic or treatment use.",
            "Homologous targets are engaged together by the same ligands, so the number of engaged "
            "targets overstates the number of independent observations. Across 400 approved drugs "
            "36 targets ever fire but span only 14 independent directions.",
            "A silent result is not evidence of inactivity. Deployed sensitivity has a median of "
            "0.77 across the panel and falls to 0.26 at the strictest endpoints.",
            "The BBB term multiplies every disease equally and cannot change which disease ranks "
            "first; it is an exposure filter, not a discriminator.",
        ],
    }


def build_profile_svg(r, top_n=18):
    """Target engagement, with engaged mechanisms and near-misses in separate blocks.

    A bar chart of raw probabilities would be misleading, because each endpoint carries a different
    threshold and a different training base rate, so 0.8 means engagement at one target and nothing
    at another. This plots the distance each target sits above its own cut, which is the quantity
    the disease layer consumes and the only one comparable across endpoints.

    The two classes are drawn apart rather than ranked together. A single ordered list put TAAR1 at
    "95 per cent of cut" second for donepezil, above D2 and BACE1, purely because it came closest to
    a threshold it never crossed. Position reads as magnitude whatever the caption says, so the
    engaged targets get the chart and the near-misses get a separate, visibly quieter block beneath
    a rule, under a heading that states they are not findings.
    """
    neuro = neuro_signal(r)
    engaged, near = [], []
    for t in sorted(TARGET_KIND):
        if t == "NEURO":
            continue
        s = target_signal(r, neuro, t)
        p = r.get("receptor_binder", {}).get(t)
        if p is None:
            p = r["targets"].get(t)
        if p is None:
            continue
        thr = binder_threshold(t) if TARGET_KIND.get(t) == "binder" else 1.0
        frac = float(p) / thr if thr > 0 else 0.0
        (engaged if s > 0 else near).append((t, s, float(p), frac))
    engaged.sort(key=lambda x: -x[1])
    near.sort(key=lambda x: -x[3])
    near = near[:max(0, top_n - len(engaged))][:8]
    if not engaged and not near:
        return ""

    W, rowh, pad_l, pad_t = 720, 22, 150, 34
    gap = 34 if (engaged and near) else 0
    H = pad_t + rowh * (len(engaged) + len(near)) + gap + 34
    track_x, track_w = pad_l, W - pad_l - 74
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;display:block" '
           f'xmlns="http://www.w3.org/2000/svg" role="img" '
           f'aria-label="Target engagement profile">']
    head = "ENGAGED MECHANISMS" if engaged else "NO MECHANISM REACHED ITS THRESHOLD"
    out.append(f'<text x="0" y="14" font-size="11" font-weight="700" fill="{NAVY}" '
               f'letter-spacing="1.1">{head}</text>')

    def gridlines(y0, y1):
        for g in range(5):
            x = track_x + track_w * g / 4
            out.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" '
                       f'stroke="#E9EEF6" stroke-width="1"/>')

    y = pad_t
    if engaged:
        gridlines(pad_t - 8, pad_t + rowh * len(engaged) - 4)
        for t, s, _p, _f in engaged:
            col = ADVERSE if t in ("hERG", "Nav1_5") else (GREEN if s >= 0.5 else BLUE)
            out.append(f'<text x="{pad_l - 10}" y="{y + 11}" font-size="11" fill="{INK}" '
                       f'text-anchor="end">{MECH_LABEL.get(t, t)}</text>')
            out.append(f'<rect x="{track_x}" y="{y + 4}" width="{track_w}" height="9" rx="4.5" '
                       f'fill="#EEF2F8"/>')
            out.append(f'<rect x="{track_x}" y="{y + 4}" width="{max(2.0, track_w * s):.1f}" '
                       f'height="9" rx="4.5" fill="{col}"/>')
            out.append(f'<text x="{track_x + track_w + 8}" y="{y + 12}" font-size="10.5" '
                       f'font-weight="700" fill="{col}">{s:.0%}</text>')
            y += rowh
        for g in range(5):
            x = track_x + track_w * g / 4
            out.append(f'<text x="{x:.1f}" y="{y + 12}" font-size="9.5" fill="{MUTE2}" '
                       f'text-anchor="middle">{g * 25}%</text>')

    if near:
        if engaged:
            y += gap - 12
            out.append(f'<line x1="0" y1="{y - 8}" x2="{W}" y2="{y - 8}" stroke="#DDE4EE" '
                       f'stroke-width="1" stroke-dasharray="3 3"/>')
        out.append(f'<text x="0" y="{y + 6}" font-size="10" font-weight="700" fill="{MUTE2}" '
                   f'letter-spacing="1.1">BELOW THRESHOLD &#183; NOT ENGAGED, SHOWN FOR CONTEXT'
                   f'</text>')
        y += 16
        for t, _s, _p, frac in near:
            out.append(f'<text x="{pad_l - 10}" y="{y + 11}" font-size="10.5" fill="{MUTE2}" '
                       f'text-anchor="end">{MECH_LABEL.get(t, t)}</text>')
            out.append(f'<rect x="{track_x}" y="{y + 7}" '
                       f'width="{max(1.5, track_w * min(1.0, frac) * 0.5):.1f}" height="3" rx="1.5" '
                       f'fill="#CBD5E4"/>')
            out.append(f'<text x="{track_x + track_w + 8}" y="{y + 12}" font-size="10" '
                       f'font-weight="500" fill="{MUTE2}">{frac:.0%} of cut</text>')
            y += rowh
    out.append("</svg>")
    return "".join(out)


def build_html_report(smiles, name, r):
    """A self-contained report the user can keep: no network, no fonts, no scripts, opens anywhere.

    Everything is inlined, including the structure image and both figures, so the file still renders
    identically in five years on a machine with no internet connection."""
    bbb, neuro, dz = disease_scores(r)
    kpuu = r["adme"]["kpuu"]
    label, colour, note = verdict(kpuu, bbb)
    mol = Chem.MolFromSmiles(smiles)
    img = _mol_data_uri(mol, size=(300, 250)) if mol else ""
    ad = assess_domain(smiles)
    m = cns_mpo(r, smiles)
    df = result_frame(smiles, name, r)

    def esc(x):
        return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    sections = ""
    for sec in df.section.unique():
        sub = df[df.section == sec]
        body = "".join(
            f"<tr><td>{esc(x.endpoint)}</td><td>{esc(x.description)}</td>"
            f"<td class='n'>{esc(x.value)}</td><td class='u'>{esc(x.unit)}</td>"
            f"<td class='c'>{esc(x.context)}</td></tr>" for x in sub.itertuples())
        sections += (f"<h2>{esc(sec)}</h2><table><thead><tr><th>Endpoint</th><th>Description</th>"
                     f"<th>Value</th><th>Unit</th><th>Context</th></tr></thead>"
                     f"<tbody>{body}</tbody></table>")

    reported = [d for d in dz if d["gated"] >= MIN_ACTIONABLE_SCORE]
    dz_txt = ("".join(
        f"<li><b>{esc(d['disease'])}</b> at {d['gated']:.2f}, driven by "
        f"{esc(MECH_LABEL.get(d['driver'][0], d['driver'][0]) if d['driver'] else 'no mechanism')}</li>"
        for d in reported) if reported else
        "<li>No modelled mechanism reached the reporting threshold. This is not evidence of "
        "inactivity: see the limitations below.</li>")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAI-CNS report: {esc(name)}</title><style>
*{{box-sizing:border-box}}
body{{font-family:ui-sans-serif,system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 color:{INK};margin:0;padding:38px 30px 60px;background:#fff;line-height:1.55;
 -webkit-font-smoothing:antialiased}}
.wrap{{max-width:980px;margin:0 auto}}
h1{{font-size:1.7rem;margin:0 0 2px;letter-spacing:-.02em}}
h1 span{{color:{GOLD}}}
h2{{font-size:.72rem;text-transform:uppercase;letter-spacing:1.3px;color:{NAVY};
 margin:30px 0 10px;padding-bottom:8px;border-bottom:2px solid {GOLD}}}
.sub{{color:{MUTE};font-size:.86rem;margin:0 0 22px}}
.top{{display:flex;gap:26px;align-items:flex-start;border:1px solid #E4EAF4;border-radius:14px;
 padding:20px;background:#FBFCFE;margin-bottom:8px}}
.top img{{border:1px solid #E4EAF4;border-radius:10px;background:#fff}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;flex:1}}
.kpi{{border:1px solid #E7EDF6;border-radius:11px;padding:12px 14px;background:#fff}}
.kpi .l{{font-size:.63rem;text-transform:uppercase;letter-spacing:.07em;color:{MUTE};font-weight:700}}
.kpi .v{{font-size:1.5rem;font-weight:800;margin-top:5px;font-variant-numeric:tabular-nums}}
.verdict{{border-left:5px solid {colour};background:linear-gradient(90deg,{colour}12,transparent);
 border-radius:9px;padding:12px 16px;margin:14px 0 0}}
.verdict .h{{font-size:1.15rem;font-weight:800;color:{colour}}}
table{{width:100%;border-collapse:collapse;font-size:.8rem;margin-bottom:6px}}
th{{text-align:left;font-size:.6rem;text-transform:uppercase;letter-spacing:.06em;color:{MUTE2};
 padding:0 9px 7px;border-bottom:1.5px solid #E7EDF6}}
td{{padding:7px 9px;border-bottom:1px solid #F2F5FA;vertical-align:top}}
td.n{{font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap}}
td.u,td.c{{color:{MUTE};font-size:.75rem}}
ul{{margin:6px 0 0 18px;padding:0}} li{{margin:3px 0;font-size:.86rem}}
.fig{{border:1px solid #E4EAF4;border-radius:12px;padding:16px;margin:10px 0 0;background:#fff;
 overflow-x:auto}}
.foot{{margin-top:34px;padding-top:16px;border-top:1px solid #E6ECF5;font-size:.72rem;color:{MUTE}}}
@media print{{body{{padding:0}} .fig,.top{{break-inside:avoid}}}}
</style></head><body><div class="wrap">
<h1>SAI <span>CNS</span></h1>
<p class="sub">Multi-endpoint prediction of small-molecule effects on the human brain</p>
<div class="top">
  <img src="{img}" alt="structure of {esc(name)}" width="300" height="250">
  <div style="flex:1">
    <div style="font-size:1.25rem;font-weight:800;margin-bottom:3px">{esc(name)}</div>
    <div style="font-family:ui-monospace,Consolas,monospace;font-size:.74rem;color:{MUTE};
     word-break:break-all;margin-bottom:14px">{esc(smiles)}</div>
    <div class="kpis">
      <div class="kpi"><div class="l">BBB penetration</div><div class="v">{bbb:.0%}</div></div>
      <div class="kpi"><div class="l">Unbound brain (Kp,uu)</div><div class="v">{kpuu:.2f}</div></div>
      <div class="kpi"><div class="l">CNS MPO</div>
        <div class="v">{(f"{m['total']:.1f}/{m['max']:.0f}" if m else "n/a")}</div></div>
    </div>
    <div class="verdict"><div class="h">{esc(label)}</div>
      <div style="font-size:.82rem;color:{MUTE}">{esc(note)}</div></div>
  </div>
</div>
<h2>Reported disease relevance</h2><ul>{dz_txt}</ul>
<h2>Mechanism network</h2><div class="fig">{build_network_svg(r, name)}</div>
<h2>Target engagement profile</h2><div class="fig">{build_profile_svg(r)}</div>
{sections}
<h2>Applicability and limitations</h2>
<p style="font-size:.82rem">Nearest measured training compound at Tanimoto
{(f"{ad['max_sim']:.2f} ({esc(ad['tier'])}, {esc(ad['confidence'])} confidence)" if ad else "unknown")}.
Predictions are least reliable far from training chemistry.</p>
<ul>
<li>A silent result is not evidence of inactivity. Reporting thresholds are set for precision,
holding the false-positive rate near 0.2 percent on random chemistry, and the measured cost is a
median sensitivity of 0.77 across the panel, falling to 0.26 at the strictest endpoints.</li>
<li>The blood-brain barrier term multiplies every disease equally, so it cannot change which
condition ranks first. It is an exposure filter, not a discriminator.</li>
<li>The tool ranks mechanisms. It does not predict clinical efficacy, and it has not undergone
wet-lab or clinical validation.</li>
</ul>
<div class="foot">SAI-CNS version {APP_VERSION}, {"high-precision" if screening_mode() else "standard"}
screening mode. Calibrated random-forest models trained on measured public bioactivity and ADME data
(228,200 records; 169,341 unique compounds). Research decision-support for prioritisation and
hypothesis generation only. Not for medical, diagnostic or treatment decisions.</div>
</div></body></html>"""


def _slug(name):
    keep = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name)).strip("_")
    return (keep or "compound")[:60]


def render_exports(smiles, name, r):
    """Four formats, one frame behind all of them: a table to analyse, a record to archive, a
    structured object to automate against, and a figure to drop into a manuscript."""
    slug = _slug(name)
    st.markdown('<div class="bs-card bs-export"><div class="bs-h">Export this result'
                '<span class="bs-h-sub">every number on this page, in the format you need</span>'
                '</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("Data table (CSV)",
                           result_frame(smiles, name, r).to_csv(index=False).encode("utf-8"),
                           file_name=f"sai_cns_{slug}.csv", mime="text/csv",
                           use_container_width=True,
                           help="Every endpoint as one tidy row per prediction, with units and the "
                                "training context needed to interpret it.")
    with c2:
        st.download_button("Full report (HTML)",
                           build_html_report(smiles, name, r).encode("utf-8"),
                           file_name=f"sai_cns_{slug}_report.html", mime="text/html",
                           use_container_width=True,
                           help="Self-contained page with the structure, both figures and every "
                                "table. Opens offline in any browser and prints to PDF.")
    with c3:
        st.download_button("Structured (JSON)",
                           json.dumps(result_json(smiles, name, r), indent=2).encode("utf-8"),
                           file_name=f"sai_cns_{slug}.json", mime="application/json",
                           use_container_width=True,
                           help="Machine-readable result with thresholds, provenance and the "
                                "caveats that must travel with the numbers.")
    with c4:
        svg = build_network_svg(r, name)
        i = svg.find("<svg")
        st.download_button("Network figure (SVG)",
                           (svg[i:svg.rfind("</svg>") + 6] if i >= 0 else svg).encode("utf-8"),
                           file_name=f"sai_cns_{slug}_network.svg", mime="image/svg+xml",
                           use_container_width=True,
                           help="Vector figure at any size, for a manuscript or a slide.")
    st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------- batch screening ---------------------------------
BATCH_LIMIT = 300


def batch_row(smiles, name, models, pre=None):
    """One summary row per compound. Deliberately narrow: the columns a triage decision needs.

    `pre` is an already-computed prediction for this compound, which the batch screen supplies so
    that a run of N compounds costs one vectorised pass rather than N single-row passes. Passing
    None preserves the original behaviour exactly.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"input": name, "smiles": smiles, "status": "unparseable"}
    cs = Chem.MolToSmiles(mol)
    r = pre if pre is not None else predict_all(cs, models)
    if r is None:
        return {"input": name, "smiles": cs, "status": "not featurizable"}
    bbb, neuro, dz = disease_scores(r)
    top = [d for d in dz if d["gated"] >= MIN_ACTIONABLE_SCORE][:3]
    ad = assess_domain(cs)
    eng = [(t, target_signal(r, neuro, t)) for t in TARGET_KIND if t != "NEURO"]
    eng = sorted([e for e in eng if e[1] >= MIN_ENGAGEMENT_SIGNAL], key=lambda x: -x[1])[:5]
    m = cns_mpo(r, cs)
    return {
        "input": name, "smiles": cs, "status": "ok",
        "bbb_probability": round(bbb, 3),
        "kpuu": round(r["adme"]["kpuu"], 3),
        "exposure_verdict": verdict(r["adme"]["kpuu"], bbb)[0],
        "herg_probability": round(r["targets"]["hERG"], 3),
        "cns_mpo": round(m["total"], 2) if m else None,
        "n_targets_engaged": len(eng),
        # Homologous targets are engaged together, so a raw count overstates the evidence. Both are
        # reported: rank on the independent count, read the raw one as breadth of engagement.
        "n_independent_mechanisms": independent_mechanisms([t for t, _ in eng]),
        "top_mechanisms": ", ".join(f"{MECH_LABEL.get(t, t)} {s:.0%}" for t, s in eng),
        "top_diseases": ", ".join(f"{d['disease']} {d['gated']:.2f}" for d in top) or "none reported",
        "ad_max_tanimoto": round(ad["max_sim"], 3) if ad else None,
        "ad_tier": ad["tier"] if ad else None,
    }


def render_batch():
    st.markdown('<div class="bs-searchcard"><p class="t">Screen a set of compounds</p>'
                '<p class="d">Paste one compound per line, as SMILES or as a name, or upload a file '
                'with a <b>smiles</b> or <b>name</b> column. Every compound is profiled across the '
                'full panel and returned as one row. Use this to rank a library before committing '
                f'bench time. Limit {BATCH_LIMIT} compounds per run.</p></div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        text = st.text_area("Compounds", height=190, label_visibility="collapsed",
                            placeholder="donepezil\nfluoxetine\nCC(=O)Nc1ccc(O)cc1\n...")
    with c2:
        up = st.file_uploader("or upload CSV / TSV / plain text", type=["csv", "tsv", "txt", "smi"])
        st.caption("A CSV needs a column called smiles or name. A plain text file is read as one "
                   "compound per line.")
    run = st.button("Screen set", type="primary")

    entries = []
    if up is not None:
        try:
            if up.name.lower().endswith((".csv", ".tsv")):
                d = pd.read_csv(up, sep=None, engine="python")
                col = next((c for c in d.columns if c.strip().lower() in ("smiles", "smi")), None) \
                    or next((c for c in d.columns if c.strip().lower() in ("name", "compound")), None)
                if col is None:
                    st.warning("No smiles or name column found in that file.")
                else:
                    entries = [str(v) for v in d[col].dropna().tolist()]
            else:
                entries = up.getvalue().decode("utf-8", "replace").splitlines()
        except Exception as e:
            st.warning(f"Could not read that file: {e}")
    if text.strip():
        entries = [ln for ln in text.splitlines()] + entries
    entries = [e.strip() for e in entries if e and e.strip()]

    if not run:
        if entries:
            st.caption(f"{len(entries)} compounds ready. Press **Screen set** to run.")
        return
    if not entries:
        st.warning("Nothing to screen. Paste compounds or upload a file first.")
        return

    dropped = max(0, len(entries) - BATCH_LIMIT)
    entries = entries[:BATCH_LIMIT]
    models = load_models()
    prog = st.progress(0.0, text=f"Resolving {len(entries)} compounds")
    resolved = []
    for i, e in enumerate(entries, 1):
        smi, nm, err = resolve_query(e)
        resolved.append((e, smi, nm, err))
        prog.progress(0.35 * i / len(entries), text=f"Resolving {i} of {len(entries)}")

    # One vectorised pass over everything that resolved. Scoring each compound separately costs
    # roughly fifty times more, which at the 300-compound limit is the difference between about
    # twenty seconds and about twenty minutes.
    prog.progress(0.4, text=f"Scoring {sum(1 for r in resolved if not r[3])} compounds")
    canon = []
    for _e, smi, _nm, err in resolved:
        if err:
            canon.append(None); continue
        mol = Chem.MolFromSmiles(smi)
        canon.append(Chem.MolToSmiles(mol) if mol is not None else None)
    scorable = [c for c in canon if c is not None]
    preds = predict_all_batch(scorable, models) if scorable else []
    by_smiles = dict(zip(scorable, preds))

    prog.progress(0.75, text="Building profiles")
    rows = []
    for (e, smi, nm, err), cs in zip(resolved, canon):
        if err:
            rows.append({"input": e, "smiles": e, "status": err})
        elif cs is None:
            rows.append({"input": e, "smiles": smi, "status": "unparseable"})
        else:
            rows.append(batch_row(smi, nm, models, pre=by_smiles.get(cs)))
    prog.empty()

    df = pd.DataFrame(rows)
    ok = int((df.status == "ok").sum())
    if dropped:
        st.info(f"{ok} of {len(entries)} compounds profiled. {dropped} beyond the "
                f"{BATCH_LIMIT}-compound limit were not run.")
    else:
        st.success(f"{ok} of {len(entries)} compounds profiled.")

    st.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="sai_cns_batch.csv", mime="text/csv", type="primary")
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"smiles": st.column_config.TextColumn("SMILES", width="medium")})
    st.markdown('<div class="bs-note" style="margin-top:10px">Rank on the mechanisms and the '
                'exposure verdict together: a strong mechanism behind a poor K<sub>p,uu</sub> will '
                'not reach its target in vivo. Compounds reporting no mechanism are not thereby '
                'inactive, and those outside the applicability domain carry the widest uncertainty.'
                '</div>', unsafe_allow_html=True)


CLF_ORDER = ["BBB", "AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG"]


@st.cache_resource
def load_results():
    def _read(p):
        return pd.read_csv(p) if p.exists() else None
    return {"cv": _read(RESULTS / "rf_cv_summary.csv"),
            "cmp": _read(RESULTS / "model_comparison.csv"),
            "gnn": _read(ROOT / "results" / "gnn" / "gnn_vs_rf.csv")}


def render_model_comparison():
    res = load_results()
    cmp, gnn = res["cmp"], res["gnn"]
    st.markdown('<div class="bs-card"><div class="bs-h">Model selection and validation'
                '<span class="bs-h-sub">why random forest is the deployed model</span></div>'
                '<div class="bs-note" style="margin-bottom:10px">Every endpoint was trained under random and '
                'scaffold (grouped) 10-fold cross-validation and benchmarked across four model families. '
                'Random forest matched or exceeded gradient boosting and a graph neural network on the '
                'scaffold split, and was selected for deployment. All results below are on held-out folds. '
                'They come from a benchmark that refits every family under one fold assignment, so the '
                'random-forest column here differs from the cross-validation of the deployed model by up '
                'to 0.012: two independent partitions of the same deduplicated data, not two different '
                'results. The headline figures above are the deployed run.</div>',
                unsafe_allow_html=True)
    # The bar chart that used to sit here plotted scaffold AUROC per endpoint from
    # rf_cv_summary.csv, immediately above a table showing the same quantity per endpoint from
    # model_comparison.csv. The two disagreed, by up to 0.012 at MAO_A, because they are separate
    # cross-validation runs with independent fold assignments over the same deduplicated data. Both
    # are correct for their own purpose, and showing them three inches apart without saying so
    # presented one quantity as two. The chart carried nothing the table does not, except error
    # bars, so it is the chart that goes and the note below that stays.

    if cmp is not None:
        c = cmp[(cmp["task"] == "classification") & (cmp["split"] == "scaffold") & (cmp["metric"] == "roc_auc")]
        piv = c.pivot_table(index="endpoint", columns="model", values="mean")
        models = [m for m in ["RandomForest", "XGBoost", "HistGradientBoosting"] if m in piv.columns]
        rows = []
        for ep in CLF_ORDER:
            if ep not in piv.index:
                continue
            best = max(piv.loc[ep, m] for m in models)
            cells = []
            for m in models:
                v = piv.loc[ep, m]
                strong = abs(v - best) < 1e-9
                cells.append(f'<span class="bs-num" style="color:{GREEN if strong else INK}">{v:.3f}</span>'
                             if strong else f'{v:.3f}')
            rows.append([f"<b>{ep}</b>"] + cells)
        head = ["Endpoint"] + [{"RandomForest": "Random forest", "XGBoost": "XGBoost",
                                "HistGradientBoosting": "Hist-GBM"}[m] for m in models]
        html = _table(head, rows, ["l"] + ["r"] * len(models))
        st.markdown(f'<div class="bs-note" style="margin:14px 0 4px;font-weight:700;color:{NAVY}">'
                    f'Algorithm comparison, classifier endpoints (scaffold 10-fold AUROC; best in green)</div>{html}',
                    unsafe_allow_html=True)

    if gnn is not None:
        rows = []
        for _, x in gnn.iterrows():
            win = str(x["winner"]) == "RandomForest"
            rows.append([f'<b>{x["endpoint"]}</b>', str(x["metric"]).replace("roc_auc", "AUROC").replace("r2", "R2"),
                         f'{float(x["GIN"]):.3f}',
                         f'<span class="bs-num" style="color:{GREEN}">{float(x["RandomForest"]):.3f}</span>',
                         _badge("RF", GREEN) if win else _badge("GIN", BLUE)])
        html = _table(["Endpoint", "Metric", "Graph NN (GIN)", "Random forest", "Winner"],
                      rows, ["l", "l", "r", "r", "c"])
        st.markdown(f'<div class="bs-note" style="margin:16px 0 4px;font-weight:700;color:{NAVY}">'
                    f'Deep learning benchmark: graph neural network (GIN) vs random forest (held-out test)</div>{html}'
                    f'<div class="bs-note" style="margin-top:8px">The graph neural network did not outperform '
                    f'the random forest on any tested endpoint, consistent with the fingerprint-plus-descriptor '
                    f'representation being sufficient at this data scale.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_coverage_card():
    """What the panel covers, with its bounds one click away rather than shouted alongside.

    The bounds are not removed. A reviewer, a referee and a user deciding whether a null result means
    anything all need them, and a resource that lists 47 endpoints while stating no scope reads as
    naive rather than confident. What changed is the weight: the validated panel is the headline, and
    the limits sit behind a disclosure that anyone who wants them can open. Both halves are generated
    from the deployed model registry at page load, so neither can drift from the panel that is
    actually running.
    """
    yes = "".join(f'<li><b>{t}</b> <span class="bs-ctx">· {src}</span></li>'
                  for t, src in coverage_modelled())
    no = "".join(f"<li>{t}</li>" for t in COVERAGE_NO)
    _low = coverage_low()
    low = ("".join(f'<li><b>{t}</b> <span class="bs-ctx">· {why}</span></li>' for t, why in _low)
           if _low else
           f'<li class="bs-note">Every deployed endpoint clears the reliability gate of '
           f'{_panel.MIN_SENSITIVITY:.2f} sensitivity and {_panel.MIN_AUROC:.2f} AUROC.</li>')
    _wd = coverage_withdrawn()
    withdrawn = ("".join(f'<li><b>{t}</b> <span class="bs-ctx">· {why}</span></li>' for t, why in _wd)
                 if _wd else "")

    st.markdown(
        f"""
        <div class="bs-card"><div class="bs-h">Validated coverage
          <span class="bs-h-sub">every endpoint below is trained on measured data and passes its own
          specificity audit</span></div>
          <ul class="bs-note" style="margin:6px 0 0;padding-left:18px;line-height:1.7;
              columns:2;column-gap:28px">{yes}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Scope: what a null result does and does not mean"):
        st.markdown(
            f"""
            <div class="bs-note" style="margin-bottom:10px">The panel reports the mechanisms it has a
            validated model for. A compound acting through something outside that set will correctly
            show no distinctive engagement, which is a genuine <i>unknown</i> rather than evidence of
            inactivity. The lists below are generated from the deployed model registry at page load,
            not written by hand, so they describe the panel that is actually running.</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
              <div><div class="about-eyebrow" style="color:{AMBER}">Not yet modelled</div>
                <ul class="bs-note" style="margin:6px 0 0;padding-left:18px;line-height:1.7">{no}</ul></div>
              <div><div class="about-eyebrow" style="color:{AMBER}">Modelled, below the reliability
                gate</div>
                <ul class="bs-note" style="margin:6px 0 0;padding-left:18px;line-height:1.7">{low}</ul>
                <div class="bs-note" style="margin-top:6px">The gate is a sensitivity of
                {_panel.MIN_SENSITIVITY:.2f} on actives withheld by scaffold and an AUROC of
                {_panel.MIN_AUROC:.2f} against compounds measured at the same target and found
                inactive. For an endpoint below it a positive call carries information but a negative
                one does not rule engagement out, and a result driven by one of these is marked as
                such where it appears.</div>
                {f'<div class="about-eyebrow" style="color:{ADVERSE};margin-top:12px">Trained then withdrawn</div><ul class="bs-note" style="margin:6px 0 0;padding-left:18px;line-height:1.7">{withdrawn}</ul><div class="bs-note" style="margin-top:6px">Endpoints that were trained, tested against trivial metabolites and random chemistry, and withheld because no threshold separated real ligands from background. They are listed because a panel showing only what survived is a selection rather than an inventory.</div>' if withdrawn else ''}
              </div>
            </div>
            <div class="bs-note" style="margin-top:12px"><b>A class this panel cannot represent.</b>
            {COVERAGE_CLASS_LIMIT}</div>
            """,
            unsafe_allow_html=True,
        )


def render_glossary():
    """One line per predicted quantity, so a reader knows what each number is before trusting it.

    Built from the live registries rather than written out, so an endpoint cannot be added to the
    tool without appearing here, nor described here after it has been withdrawn."""
    modes = load_binder_modes()

    def rows(items):
        return "".join(
            f'<tr><td style="white-space:nowrap"><b>{k}</b></td>'
            f'<td class="bs-ctx">{v}</td></tr>' for k, v in items)

    exposure = [
        ("BBB penetration", "Probability that the compound crosses the blood-brain barrier, from "
                            "measured brain-to-blood partition data. It sets whether a central "
                            "mechanism is reported at all."),
        ("K<sub>p,uu</sub>", "Ratio of unbound drug in brain to unbound drug in plasma. Above about "
                             "0.3 a meaningful free concentration reaches the target; below 0.1 it "
                             "does not, whatever the total brain level."),
        ("logBB", "Total brain-to-plasma ratio. Less informative than K<sub>p,uu</sub> because it "
                  "counts drug bound to tissue, which cannot engage a target."),
        ("Free brain exposure verdict", "A plain reading of K<sub>p,uu</sub> and BBB together: "
                                        "favourable, borderline or limited."),
    ]
    engagement = [
        ("Binder probability", "Calibrated probability that the compound binds the named target at "
                               "pChEMBL 7 or better, roughly 100 nM. Each target has its own "
                               "threshold, so probabilities are not comparable between targets."),
        ("Enrichment over base rate", "How far a prediction sits above what the training set would "
                                      "give by chance. Used where the endpoint is a measured-label "
                                      "classifier rather than a binder model; zero means the "
                                      "prediction is no better than the base rate."),
        ("Engagement signal", "Distance above a target's own reporting threshold, rescaled to 0 to "
                              "1. This is the only engagement quantity comparable across targets, "
                              "and it is what the disease layer consumes."),
        ("Predicted pK<sub>i</sub>", "Potency among compounds already called binders. A weak prior, "
                                     "not an affinity prediction, and shown only for binders."),
        ("Independent mechanisms", "Engaged targets grouped by homology. Two homologous receptors "
                                   "engaged by one ligand are close to a single observation, so "
                                   "this is the number to weigh, not the raw target count."),
    ]
    disease = [
        ("Disease relevance score", "The strongest engaged mechanism for that condition, multiplied "
                                    "by predicted barrier penetration. It ranks conditions by "
                                    "mechanism; it is not a probability of clinical efficacy."),
        ("Driving mechanism", "The single target responsible for the score. The graph takes the "
                              "strongest mechanism rather than summing, so one target explains it."),
        ("Reporting threshold", f"Scores below {MIN_ACTIONABLE_SCORE:.2f} are not shown as findings. "
                                f"Silence means no modelled mechanism cleared its threshold, which "
                                f"is not the same as inactivity."),
    ]
    admet = [
        ("Caco-2 permeability", "Passive membrane permeability, as log apparent permeability."),
        ("P-gp substrate / inhibitor", "Whether the compound is pumped out of the brain by "
                                       "P-glycoprotein, or blocks that pump for other drugs."),
        ("Plasma-protein binding", "Fraction bound in plasma. Only the unbound fraction can cross "
                                   "or engage."),
        ("Hepatocyte clearance", "Rate of hepatic metabolism. Weakest endpoint under a scaffold "
                                 "split; read qualitatively."),
        ("Solubility, logD", "Aqueous solubility and lipophilicity at physiological pH."),
        ("hERG liability", "Probability of blocking the cardiac hERG channel, a common cause of "
                           "drug withdrawal. Higher is worse."),
        ("CNS MPO", "A desirability score over six physicochemical properties associated with "
                    "central exposure. A drug-likeness heuristic, not a model of activity."),
        ("Antioxidant (DPPH)", "Measured radical-scavenging capacity, the basis of the "
                               "neuroprotection axis."),
        ("Most basic pK<sub>a</sub>", "Predicted ionisation of the most basic centre, used in CNS "
                                      "MPO. Mean absolute error 1.17 units, the least certain term."),
    ]
    confidence = [
        ("Applicability domain", "Maximum Tanimoto similarity to the nearest measured training "
                                 "compound. In domain above 0.5, out of domain below 0.3, where "
                                 "predictive power falls close to chance."),
        ("Read-across", "Targets of the nearest measured analogues, weighted by similarity and by "
                        "how many neighbours support each. Evidence when no model fires."),
        ("High-precision mode", "Raises every threshold to a 1 percent background false-positive "
                                "rate. Fewer true actives found, far fewer false leads."),
    ]
    n_binder = sum(1 for v in modes.values() if v.get("deployed", True))
    sections = [("Exposure: can it reach the brain", exposure),
                ("Target engagement: what does it touch", engagement),
                ("Disease relevance: what might that mean", disease),
                ("ADMET and safety", admet),
                ("Confidence and context", confidence)]
    body = "".join(
        f'<div style="margin-top:14px"><div class="about-eyebrow" style="color:{NAVY}">{title}</div>'
        f'<table class="bs-table" style="margin-top:6px"><tbody>{rows(items)}</tbody></table></div>'
        for title, items in sections)
    st.markdown(
        f'<div class="bs-card"><div class="bs-h">What every number on this page means'
        f'<span class="bs-h-sub">{panel_shape().get("targets", "-")} molecular targets, of which '
        f'{n_binder} are the deployed binder panel, plus {len(ADME)} ADME endpoints, the barrier '
        f'model and {len(DISEASE_ORDER)} conditions</span></div>'
        f'<div class="bs-note">Read this before trusting any single figure. The quantities below are '
        f'not interchangeable: a binder probability, an enrichment over base rate and an engagement '
        f'signal are three different things on three different scales, and only the last is '
        f'comparable between targets.</div>{body}</div>',
        unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def panel_shape():
    """The one place that counts the panel, because it was being counted six different ways.

    The interface had quoted 47, 52, 55, 63, 70 and 75 in different sections, each correct for a
    different question and none of them reconciled, which reads as carelessness whatever the
    arithmetic. The counts are these, and they add up exactly:

      75 estimators trained, 70 deployed, 5 withdrawn.
      70 deployed estimators cover 66 distinct predicted quantities, because A2A, D2, HT2A and SERT
      each carry two models, a binder classifier and a potency regressor, for the same protein.
      Of the 66: 54 are molecular targets, 10 are exposure and ADME properties, being the barrier
      model and nine ADME endpoints, and 2 are other measured properties, the antioxidant DPPH
      capacity and the most-basic pKa.
      Of the 54 targets, 47 are the deployed binder panel, of 52 binder endpoints trained.

    The barrier model is NOT a molecular target. It is an exposure term, and counting it among the
    targets is how 54 became 55 in an earlier version of the header.
    """
    out = {}
    try:
        d = pd.read_csv(RESULTS / "MODEL_INVENTORY.csv")
        dep = d[d.deployed.astype(str).str.lower().isin(("true", "1", "yes"))].copy()
        dep["quantity"] = dep.model.astype(str).str.replace("_binder", "", regex=False)
        adme = [m for m in dep.model.astype(str) if m.startswith("adme_")]
        not_target = set(adme) | {"antioxidant_DPPH", "pka_basic", "BBB"}
        out["trained"] = len(d)
        out["deployed"] = len(dep)
        out["withdrawn"] = len(d) - len(dep)
        out["quantities"] = int(dep.quantity.nunique())
        out["targets"] = len(set(dep.quantity) - not_target)
        out["exposure"] = len(adme) + 1
        out["other"] = 2
        dbl = dep.quantity.value_counts()
        out["dual_model"] = sorted(dbl[dbl > 1].index.tolist())
    except Exception:
        pass
    return out


def panel_facts():
    """Every number the About page states, read from the artefacts rather than typed.

    The page previously carried figures typed by hand and left behind by the panel: it claimed
    64,474 records and 61,317 compounds when the tables hold 228,200 and 169,341, an AUROC range of
    0.95-0.97 when the measured range is 0.899-0.976, and a temporal range of 0.61-0.91 against a
    measured 0.720-0.910. A page that describes the models has to be generated from them, or it
    describes whichever panel existed when someone last edited the string.
    """
    import glob
    f = {}
    try:
        modes = load_binder_modes()
        dep = [v for v in modes.values() if v.get("deployed", True)]
        f["n_binder_trained"], f["n_binder_deployed"] = len(modes), len(dep)
        f["binder_auroc"] = sum(v["auroc_vs_measured_inactives"] for v in dep) / max(len(dep), 1)
        f["binder_sens"] = sum(v["sensitivity_at_threshold"] for v in dep) / max(len(dep), 1)
        # Every other headline row states its range. This one did not, and it covers the most
        # endpoints and has the widest spread of any of them: AUROC runs from 0.719 at GABA-A to
        # 0.985, so a bare mean of 0.917 reads better than the panel is.
        _a = [v["auroc_vs_measured_inactives"] for v in dep
              if v.get("auroc_vs_measured_inactives") is not None]
        _s = [v["sensitivity_at_threshold"] for v in dep
              if v.get("sensitivity_at_threshold") is not None]
        if _a:
            f["binder_auroc_lo"], f["binder_auroc_hi"] = min(_a), max(_a)
        if _s:
            f["binder_sens_lo"], f["binder_sens_hi"] = min(_s), max(_s)
    except Exception:
        pass
    try:
        # The headline figure is a SUM over endpoints, and on its own it invites the reader to
        # believe every model saw it. No model did. The per-endpoint distribution is carried with
        # it so the aggregate cannot be mistaken for a training-set size: the median endpoint holds
        # about a twentieth of the total, and the smallest a fraction of a per cent.
        # The per-endpoint span is reported over the DEPLOYED panel, matching the manuscript. Taken
        # over all 63 tables the minimum is 140 rows at NR3C1, an endpoint that was trained and then
        # withdrawn, so quoting it would describe a model no user can reach. Over the deployed
        # endpoints the span is 387 at KEAP1 to 10,276 at hERG.
        _served = set(TARGET_CLASSIFIERS) | set(BINDER_TARGETS) | set(RECEPTOR_REGRESSORS)
        n, sizes, structures = 0, [], set()
        for path in glob.glob(str(ROOT / "data" / "endpoints" / "*.csv")):
            d = pd.read_csv(path, usecols=["smiles"])
            n += len(d)
            if Path(path).stem in _served:
                sizes.append(len(d))
            structures |= set(d["smiles"].astype(str))
        f["n_records"] = n
        f["n_endpoint_tables"] = len(sizes)
        f["n_structures"] = len(structures)
        if sizes:
            s = pd.Series(sizes)
            f["rows_median"] = int(s.median())
            f["rows_min"], f["rows_max"] = int(s.min()), int(s.max())
    except Exception:
        pass
    try:
        cv = pd.read_csv(RESULTS / "rf_cv_summary.csv")
        c = cv[cv.task == "classification"]
        for split in ("random", "scaffold"):
            g = c[c.split == split]
            f[split + "_lo"], f[split + "_hi"] = g.roc_auc_mean.min(), g.roc_auc_mean.max()
            f[split + "_mean"] = g.roc_auc_mean.mean()
    except Exception:
        pass
    try:
        tp = pd.read_csv(RESULTS / "rf_temporal.csv")
        a = tp[tp.metric == "auroc"]
        f["temporal_lo"], f["temporal_hi"], f["temporal_n"] = a.score.min(), a.score.max(), len(a)
    except Exception:
        pass
    try:
        s = pd.read_csv(RESULTS / "noncns_specificity_summary.csv")
        row = s[s.metric.str.startswith("Specificity")].iloc[0]
        f["spec"], f["spec_lo"], f["spec_hi"] = row.estimate, row.ci95_low, row.ci95_high
    except Exception:
        pass
    return f


def render_about():
    """One page in one order: what it answers, how it was built and tested, what bounds it.

    Rebuilt because the previous version rendered the glossary, the coverage card and the model
    comparison, then began again with a second "About SAI-CNS" block repeating what the tool
    does and how it was trained. A reader met the same material twice, in different words and with
    different numbers, and the second set was three years of panel growth out of date.
    """
    f = panel_facts()

    def val(key, fmt="{:.3f}", dash="not available"):
        return fmt.format(f[key]) if key in f else dash

    st.markdown('<div class="about-h">About SAI<span>-CNS</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="bs-note" style="font-size:.95rem;margin:-6px 0 18px">'
        'SAI-CNS answers four questions about a small molecule in one pass: can it reach the '
        'brain, what does it engage there, what does that mechanism touch, and what would stop it '
        'being a drug. Every endpoint is trained on measured experimental values, never on '
        'qualitative annotation, and every prediction carries its own uncertainty.</p>',
        unsafe_allow_html=True)

    records = "{:,}".format(f["n_records"]) if "n_records" in f else "not available"
    cards = [
        ("Trained on", records,
         ("measured compound-endpoint records from ChEMBL, BindingDB and B3DB, summed over {} "
          "endpoints and covering {:,} distinct SMILES, which collapse to 169,341 compounds once "
          "keyed by the InChIKey of the desalted parent, the figure the manuscript quotes. No "
          "single model saw this many: each "
          "endpoint is trained on its own table alone, and across every deployed model owning a "
          "training table, the barrier model included, those tables hold a median of {:,} rows, "
          "from {:,} to {:,}").format(f.get("n_endpoint_tables", "-"), f.get("n_structures", 0),
                                 f.get("rows_median", 0), f.get("rows_min", 0),
                                 f.get("rows_max", 0))
         if "rows_median" in f else
         "measured compound-endpoint records from ChEMBL, BindingDB and B3DB"),
        ("Target panel", str(f.get("n_binder_deployed", "-")),
         "binder endpoints deployed of {} trained, at a mean AUROC of {} against compounds "
         "measured and found inactive, ranging from {} to {} across the panel".format(
             f.get("n_binder_trained", "-"), val("binder_auroc"),
             val("binder_auroc_lo"), val("binder_auroc_hi"))),
        ("Silent on non-CNS chemistry", val("spec"),
         "specificity, 95% CI {} to {}, over 1,000 compounds with no recorded activity at any "
         "modelled target. They are presumed inactive because nothing is recorded about them, not "
         "proven inactive, so this is a lower bound".format(val("spec_lo"), val("spec_hi"))),
    ]
    for col, (label, big, sub) in zip(st.columns(3, gap="medium"), cards):
        with col:
            st.markdown(
                '<div class="bs-card" style="height:100%"><div class="about-eyebrow">' + label +
                '</div><div style="font-size:1.7rem;font-weight:800;color:' + NAVY +
                ';line-height:1.15;margin:4px 0 6px">' + big + '</div>'
                '<div class="bs-note" style="font-size:.82rem">' + sub + '</div></div>',
                unsafe_allow_html=True)

    # A reader who meets 47 in one section, 54 in another and 70 in a third is entitled to think
    # nobody checked. Every one of those is correct for a different question, so the answer is not
    # to pick one but to show the arithmetic once, here, and derive the rest from panel_shape().
    sh = panel_shape()
    if sh:
        st.write("")
        st.markdown(
            f'<div class="bs-card"><div class="bs-h">What the panel contains'
            f'<span class="bs-h-sub">the counts, reconciled, because they answer different '
            f'questions</span></div>'
            f'<table class="bs-table"><tbody>'
            f'<tr><td><b>{sh["trained"]}</b></td><td>estimators trained in total</td></tr>'
            f'<tr><td><b>{sh["deployed"]}</b></td><td>of them deployed and reachable from this page; '
            f'{sh["withdrawn"]} were withdrawn and are named under Validated coverage</td></tr>'
            f'<tr><td><b>{sh["quantities"]}</b></td><td>distinct quantities those {sh["deployed"]} '
            f'estimators predict. The two figures differ because '
            f'{", ".join(sh["dual_model"][:-1])} and {sh["dual_model"][-1]} each carry two models '
            f'for the same protein, a binder classifier and a potency regressor</td></tr>'
            f'<tr><td><b>{sh["targets"]}</b></td><td>of those quantities are <b>molecular '
            f'targets</b>, the number to quote for what this server profiles</td></tr>'
            f'<tr><td><b>{sh["exposure"]}</b></td><td>are exposure and ADME properties: the '
            f'blood-brain barrier model and {len(ADME)} ADME endpoints. The barrier model is an '
            f'exposure term, not a target, and is not counted among the '
            f'{sh["targets"]}</td></tr>'
            f'<tr><td><b>{sh["other"]}</b></td><td>are other measured properties: antioxidant DPPH '
            f'capacity and the most-basic pKa</td></tr>'
            f'</tbody></table>'
            f'<div class="bs-note" style="margin-top:10px">{sh["targets"]} + {sh["exposure"]} + '
            f'{sh["other"]} = {sh["quantities"]} quantities, from {sh["deployed"]} estimators. '
            f'Separately, {len(DISEASE_ORDER)} conditions are scored from those targets through the '
            f'pathway graph, and the endpoint tables behind all of it number 63, some endpoints '
            f'drawing on more than one.</div></div>',
            unsafe_allow_html=True)

    st.write("")
    render_glossary()
    st.write("")
    render_coverage_card()
    st.write("")

    st.write("")
    render_methods()
    st.write("")
    render_validation()
    st.write("")
    render_model_comparison()
    st.write("")

    st.markdown(
        '<div class="bs-card"><div class="bs-h">Intended use'
        '<span class="bs-h-sub">what a result from this server is, and what it is not</span></div>'
        '<div class="bs-note">SAI-CNS predicts molecular target '
        'engagement and physicochemical properties. It does not predict clinical efficacy, and it '
        'does not distinguish an agonist from an antagonist: both bind, and the models are trained '
        'on binding. Disease-level scores are a route from a mechanism to the conditions that '
        'mechanism touches, useful for orientation and for deciding what to test next, and they '
        'are not an indication prediction. Predictions on chemistry far from the training set are '
        'flagged rather than suppressed, so a reader can tell interpolation from extrapolation. '
        'The tool is for research prioritisation and hypothesis generation, is pending peer '
        'review, and is not for medical, diagnostic or treatment use.</div></div>',
        unsafe_allow_html=True)

    st.write("")
    with st.container(border=True):
        st.markdown("""
        ### 100 Years of Selfless Service
        This offering celebrates **100 years of selfless service and unconditional love to society** by
        **Bhagawan Sri Sathya Sai Baba** (1926-2011). SAI-CNS is part of the **SAI-Net** initiative at the
        **Sri Sathya Sai Institute of Higher Learning (SSSIHL)**, Prasanthi Nilayam.
        """)
        st.markdown("""
        <div class="quote"><p>&#8220;Let the world achieve the glory of becoming a family, through Love.&#8221;</p>
        <cite>Bhagawan Sri Sathya Sai Baba</cite></div>
        <div class="quote" style="border-left-color:#1A3A5C;background:#F0F5FB"><p style="color:#1A3A5C">
        &#8220;True knowledge is that which makes man work for the welfare of humanity.&#8221;</p>
        <cite style="color:#1A3A5C">Bhagawan Sri Sathya Sai Baba</cite></div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="bs-foot">Repository: https://github.com/krishna-g-999/sai-cns '
                '&middot; See CITATION.cff for how to cite. Research use, pending peer review.</div>',
                unsafe_allow_html=True)


EXAMPLES = {
    "Donepezil": "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
    "Fluoxetine": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
    "Haloperidol": "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",
    "Morphine": "CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5",
    "Resveratrol": "C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Diazepam": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "Atenolol (peripheral)": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "Loperamide (effluxed)": "CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1",
}


@st.cache_data(show_spinner=False)
def _pubchem_smiles(name):
    """Resolve a compound name to a SMILES via PubChem (returns None if offline/not found).
    PubChem returns the value under a key such as SMILES or ConnectivitySMILES, so read whichever
    SMILES-like key is present."""
    import urllib.parse
    import requests
    requests.packages.urllib3.disable_warnings()
    u = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
         f"{urllib.parse.quote(name)}/property/SMILES,ConnectivitySMILES/JSON")
    for verify in (True, False):  # verified first, fall back for SSL-intercepted networks
        try:
            r = requests.get(u, timeout=12, verify=verify)
            if r.ok:
                props = r.json()["PropertyTable"]["Properties"][0]
                for k in ("SMILES", "IsomericSMILES", "ConnectivitySMILES", "CanonicalSMILES"):
                    if props.get(k):
                        return props[k]
                return None
        except Exception:
            continue
    return None


def resolve_query(q):
    """Return (smiles, display_name, error). Accepts a curated example, a SMILES string, or a
    compound name resolved through PubChem."""
    q = q.strip()
    for nm, smi in EXAMPLES.items():
        if q.lower() in (nm.lower(), nm.split(" (")[0].lower()):
            return smi, nm.split(" (")[0], None
    if Chem.MolFromSmiles(q) is not None:
        return q, "Query compound", None
    smi = _pubchem_smiles(q)
    if smi and Chem.MolFromSmiles(smi) is not None:
        return smi, q.title(), None
    return None, None, (f'Could not interpret "{q}" as a SMILES string or resolve it as a compound '
                        f'name. Check the spelling, or paste a SMILES string.')


# ---------------------------------------------------------------------------------------------
# The pages a scientific web server is expected to carry. Each is generated from the artefacts it
# describes: a Methods page that states a training-set size, or a Validation page that quotes an
# AUROC, becomes wrong the moment the panel is refitted unless it reads the number at page load.
# ---------------------------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def validation_rows():
    """Per-endpoint validation, straight from the cross-validation summary."""
    try:
        cv = pd.read_csv(RESULTS / "rf_cv_summary.csv")
    except Exception:
        return None
    out = cv.pivot_table(index=["endpoint", "task"], columns="split",
                         values=["roc_auc_mean", "r2_mean"]).reset_index()
    out.columns = [c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in out.columns]
    return out


def render_methods():
    f = panel_facts()

    def val(key, fmt="{:.3f}", dash="not available"):
        return fmt.format(f[key]) if key in f else dash

    st.markdown(
        '<div class="bs-h" style="margin:0 0 8px">How a prediction is made'
        '<span class="bs-h-sub">the six steps, in the order they happen</span></div>'
        '<p class="bs-note" style="margin:0 0 12px">Full detail, including every decision that was '
        'made and later reversed, is in the manuscript and in <code>docs/METHODS.md</code>.</p>',
        unsafe_allow_html=True)

    steps = [
        ("1 &middot; Standardisation",
         "The structure is reduced to its largest organic fragment and neutralised. Neutralisation "
         "matters: stripping a counter-ion without it leaves the parent carrying the salt's charge, "
         "and haloperidol hydrochloride then scored a barrier probability of 0.613 against 0.993 "
         "for the free base, losing a cardiac liability flag. Only protonation states are undone. A "
         "permanent charge is kept, because a quaternary ammonium's charge is precisely what stops "
         "it crossing the barrier."),
        ("2 &middot; Representation",
         "A 1,024-bit folded ECFP-4 fingerprint plus twelve physicochemical descriptors: molecular "
         "weight, cLogP, TPSA, hydrogen-bond donors and acceptors, rotatable bonds, aromatic rings, "
         "fraction sp3, ring count, heavy atoms, formal charge and QED. Chirality is excluded, so "
         "two enantiomers give identical rows; rows identical in feature space are collapsed before "
         "any split, or copies of one compound would sit on both sides of a fold."),
        ("3 &middot; Labels from measurement only",
         "Every endpoint is trained on measured experimental values. A compound assayed and found "
         "inactive is often deposited only as a censored bound, and the conventional query discards "
         "exactly those rows. A bound settles a label when the whole interval it defines falls on "
         "one side of the activity cut, and is discarded as undecidable when it spans both. "
         "Recovering these added measured non-binders across 57 endpoints."),
        ("4 &middot; Fitting",
         "A random forest per endpoint, 300 trees, chosen after comparison against gradient "
         "boosting, XGBoost, L2 logistic regression, a nearest-neighbour read-across and a graph "
         "neural network. Classifiers are isotonically calibrated on out-of-fold predictions, so no "
         "compound contributes to the calibrator that scores it; binder models use Platt scaling, "
         "because the withheld set for one target is often too small to fit a step function."),
        ("5 &middot; Thresholds measured where they were not set",
         "The background library is partitioned into three disjoint pools by a stable hash of the "
         "structure: one supplies decoys, one sets thresholds, one measures the false-positive "
         "rate. Choosing a threshold as a quantile of a sample and then measuring the rate on that "
         "same sample restates the target instead of testing it."),
        ("6 &middot; Exposure gating",
         "A target score is admitted only in proportion to predicted barrier penetration, so "
         "potency at a target the compound cannot reach contributes nothing. Engaged targets are "
         "then traced through a curated pathway graph, anchored to KEGG, Reactome and IUPHAR, to "
         "the conditions they touch."),
    ]
    body = "".join(
        '<tr><td style="width:30%;vertical-align:top"><b>' + h + '</b></td>'
        '<td style="vertical-align:top">' + d + '</td></tr>' for h, d in steps)
    st.markdown('<div class="bs-card"><table class="bs-table"><tbody>' + body +
                '</tbody></table></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="bs-card" style="margin-top:14px"><div class="bs-h">Data sources</div>'
        '<table class="bs-table"><tbody>'
        '<tr><td style="width:30%"><b>ChEMBL</b></td><td>pChEMBL bioactivity values for the protein '
        'target panel, including censored bounds recovered as measured non-binders.</td></tr>'
        '<tr><td><b>BindingDB</b></td><td>Additional measured affinities pooled at compound '
        'level.</td></tr>'
        '<tr><td><b>B3DB</b></td><td>Blood-brain-barrier labels, augmented with FDA-curated approved '
        'drugs.</td></tr>'
        '<tr><td><b>Therapeutics Data Commons, MoleculeNet</b></td><td>Measured sets for the ADME '
        'and exposure endpoints.</td></tr>'
        '</tbody></table>'
        '<div class="bs-note" style="margin-top:10px">Across every endpoint the panel holds ' +
        ("{:,}".format(f["n_records"]) if "n_records" in f else "its") +
        ' measured compound-endpoint records, which is a total and not a training-set size: each '
        'endpoint is fitted on its own table, of median ' +
        ("{:,}".format(f["rows_median"]) if "rows_median" in f else "a few thousand") +
        ' rows. No value is imputed, and no qualitative annotation overrides a measurement.'
        '</div></div>',
        unsafe_allow_html=True)


def render_validation():
    f = panel_facts()

    def val(key, fmt="{:.3f}", dash="not available"):
        return fmt.format(f[key]) if key in f else dash

    st.markdown(
        '<div class="bs-h" style="margin:0 0 8px">How it was tested'
        '<span class="bs-h-sub">read from the validation artefacts at page load</span></div>'
        '<p class="bs-note" style="margin:0 0 12px">Where a check fails it is shown failing. A '
        'validation suite that only ever passes is not measuring anything.</p>',
        unsafe_allow_html=True)

    st.markdown(
        '<div class="bs-card"><div class="bs-h">Headline'
        '<span class="bs-h-sub">measured-label classifiers unless stated</span></div>'
        '<table class="bs-table"><tbody>'
        '<tr><td style="width:42%"><b>Random 10-fold, mean AUROC</b></td><td>' + val("random_mean") +
        ' (range ' + val("random_lo") + ' to ' + val("random_hi") + ')</td></tr>'
        '<tr><td><b>Scaffold-grouped 10-fold, mean AUROC</b></td><td>' + val("scaffold_mean") +
        ' (range ' + val("scaffold_lo") + ' to ' + val("scaffold_hi") + ')</td></tr>'
        '<tr><td><b>Temporal split, AUROC</b></td><td>' + val("temporal_lo") + ' to ' +
        val("temporal_hi") + ' across ' + str(f.get("temporal_n", "-")) + ' endpoints</td></tr>'
        '<tr><td><b>Binder panel vs measured non-binders</b></td><td>mean AUROC ' +
        val("binder_auroc") + ' (range ' + val("binder_auroc_lo") + ' to ' +
        val("binder_auroc_hi") + '), mean sensitivity ' + val("binder_sens") +
        ' (range ' + val("binder_sens_lo") + ' to ' + val("binder_sens_hi") +
        ') on actives withheld by scaffold</td></tr>'
        '<tr><td><b>Specificity on non-CNS chemistry</b></td><td>' + val("spec") + ' (95% CI ' +
        val("spec_lo") + ' to ' + val("spec_hi") + ') over 1,000 compounds presumed inactive '
        'because nothing is recorded about them, so a lower bound</td></tr>'
        '</tbody></table></div>',
        unsafe_allow_html=True)

    rows = validation_rows()
    if rows is not None:
        with st.expander("Per-endpoint metrics"):
            show = rows.copy()
            show.columns = [c.replace("roc_auc_mean_", "AUROC ").replace("r2_mean_", "R2 ")
                            for c in show.columns]
            st.dataframe(show.round(3), use_container_width=True, hide_index=True)
            st.caption("AUROC for classification endpoints, R-squared for regressions. The two are "
                       "not comparable: 0.5 is chance for one and a respectable fit for the "
                       "other.")

    try:
        inv = pd.read_csv(RESULTS / "inversion_validation.csv")
        n_pass = int((inv.result.astype(str).str.upper() == "PASS").sum())
        items = "".join(
            '<tr><td style="width:12%;color:' + (GREEN if str(r.result).upper() == "PASS" else ADVERSE) +
            ';font-weight:700">' + str(r.result).upper() + '</td><td>' + str(r.check) +
            '<div class="bs-ctx" style="margin-top:2px">' + str(r.detail) + '</div></td></tr>'
            for r in inv.itertuples())
        st.markdown(
            '<div class="bs-card" style="margin-top:16px"><div class="bs-h">Adversarial checks'
            '<span class="bs-h-sub">' + str(n_pass) + ' of ' + str(len(inv)) +
            ' pass; each was written so that it could fail</span></div>'
            '<table class="bs-table"><tbody>' + items + '</tbody></table>'
            '<div class="bs-note" style="margin-top:10px">Every check is reported at the same size, '
            'passing or not. The domain-flag check passes only after its controls were corrected: '
            'most of the original controls were measured compounds inside the flag\'s own reference '
            'library, where in-domain is the truthful answer. The criterion was not moved, and the '
            'flag remains a weak signal, so the conformal interval and the nearest-analogue '
            'distance should be read as the statement of confidence.</div></div>',
            unsafe_allow_html=True)
    except Exception:
        pass


def render_legal():
    st.markdown('<div class="about-h">Contact, privacy and licence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="bs-card"><div class="bs-h">Privacy</div>'
        '<div class="bs-note">No registration is required and no account '
        'exists. Structures submitted for prediction are held in memory for the lifetime of the '
        'request and are not written to disk, logged or retained. Nothing submitted here is used to '
        'train or update any model. One external call is made, and only when it is needed: a '
        'compound entered by name is resolved to a structure through PubChem, which means that name '
        'reaches PubChem. Paste a SMILES string instead and no request leaves this server.</div>'
        '</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="bs-card" style="margin-top:14px"><div class="bs-h">Licence</div>'
        '<div class="bs-note">Source code is released under the MIT licence. '
        'Underlying data retain the licences of their sources: ChEMBL (CC BY-SA 3.0), BindingDB, '
        'B3DB, Therapeutics Data Commons and MoleculeNet. Pathway annotations are drawn from KEGG, '
        'Reactome and the IUPHAR/BPS Guide to Pharmacology and are subject to their own terms; KEGG '
        'in particular restricts commercial redistribution. Users are responsible for observing the '
        'terms of any source they redistribute.</div></div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="bs-card" style="margin-top:14px"><div class="bs-h">Code and data</div>'
        '<div class="bs-note">Source code, the curated knowledge graph, every validation artefact '
        'and the scripts that regenerate each table and figure are at '
        '<a href="https://github.com/krishna-g-999/sai-cns">'
        'github.com/krishna-g-999/sai-cns</a> under the MIT licence. '
        '<code>CITATION.cff</code> gives the citation and <code>REPRODUCE.md</code> regenerates the '
        'results from one command. Trained estimators are deposited separately with a manifest '
        'recording the SHA-256 of the archive and of every file inside it, so a download is '
        'verified rather than trusted. Any screening run is downloadable as CSV from the Batch '
        'Screening tab.</div></div>'
        '<div class="bs-card" style="margin-top:14px"><div class="bs-h">Contact</div>'
        '<div class="bs-note">SAI-CNS is developed under the SAI-Net '
        'initiative at the Sri Sathya Sai Institute of Higher Learning, Prasanthi Nilayam, India. '
        'Questions, corrections and bug reports are best raised as issues in the repository, where '
        'they stay visible alongside the code they concern.</div></div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="bs-card" style="margin-top:14px"><div class="bs-h">Status</div>'
        '<div class="bs-note">Research preview, pending peer review. '
        'SAI-CNS is a computational decision-support tool for research prioritisation and '
        'hypothesis generation. It predicts molecular target engagement and physicochemical '
        'properties, not clinical efficacy, and has not undergone wet-lab or clinical validation. '
        'It is not for medical, diagnostic or treatment decisions.</div></div>',
        unsafe_allow_html=True)


def render_report(smiles, name="Compound"):
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        st.error("Could not parse that structure.")
        return
    m = load_models()
    r = predict_all(smiles.strip(), m)
    if r is None:
        st.error("Could not featurize that structure.")
        return
    # Order matters. The reliability banner is about THIS compound and belongs above its result.
    # The decision guidance is generic methodology, and putting it first pushed the answer the user
    # asked for below the fold. It now sits under the result, where it is read as interpretation
    # rather than as a preamble, and it is collapsed so it informs without competing.
    render_reliability_banner(smiles.strip())
    render_summary(mol, r)
    st.write("")
    render_network(r, name=name)
    st.write("")
    render_profile(r)
    st.write("")
    a, b = st.columns(2, gap="large")
    with a: render_disease(r)
    with b: render_targets(r)
    st.write("")
    # Read-across and CNS-likeness need no model to fire, so they carry the answer for a compound
    # that engages nothing in the panel. They sit directly beneath the disease result for that reason.
    render_independence(r)
    st.write("")
    g, h = st.columns(2, gap="large")
    with g: render_readacross(smiles.strip(), r)
    with h: render_cns_mpo(r, smiles.strip())
    st.write("")
    c, d = st.columns(2, gap="large")
    with c: render_adme(r)
    with d: render_receptors(r)
    st.write("")
    e, f = st.columns(2, gap="large")
    with e: render_physchem(r)
    with f: render_confidence(smiles.strip())
    st.write("")
    with st.expander("How to act on this call, and what a positive costs at your hit rate"):
        render_decision_guidance()
    st.write("")
    render_exports(smiles.strip(), name, r)
    st.markdown(
        '<div class="bs-foot">Predictions are calibrated random-forest outputs trained on measured public '
        'bioactivity and ADME data (228,200 records; 169,341 unique compounds); scaffold-split AUROC 0.925 '
        '(target panel). Least reliable for compounds far from the training chemistry (see applicability domain) '
        'and for the weakest endpoints (clearance, plasma-protein binding). Not for medical, diagnostic or '
        'treatment use. See docs/METHODS.md and docs/ADME_RESULTS.md.</div>', unsafe_allow_html=True)


def main():
    _icon = ASSETS / "sai_cns_mark_180.png"
    st.set_page_config(page_title="SAI-CNS | SAI-Net",
                       page_icon=str(_icon) if _icon.exists() else "🧠",
                       layout="wide", initial_sidebar_state="collapsed")
    inject_css()
    render_header()

    tab_search, tab_batch, tab_about, tab_legal = st.tabs(
        ["Compound Search", "Batch Screening", "About", "Contact"])
    with tab_search:
        with st.container(border=True):
            st.markdown('<p class="t" style="font-size:1.05rem;font-weight:700;color:#0D2137;margin:0 0 3px">'
                        'Search a compound</p><p class="d" style="font-size:.82rem;color:#4A5568;margin:0 0 10px;'
                        'line-height:1.55">Type a <b>compound name</b> (for example morphine or donepezil) or paste a '
                        '<b>SMILES</b> string, then press Predict. Names are resolved to structure via PubChem. The '
                        'compound is profiled across all validated endpoints with calibrated probabilities and an '
                        'applicability-domain confidence.</p>', unsafe_allow_html=True)
            c1, c2 = st.columns([3, 1])
            with c1:
                query = st.text_input("Compound name or SMILES", value="", label_visibility="collapsed",
                                      placeholder="Type a compound name or SMILES, e.g. morphine, fluoxetine, or COc1cc2...")
            with c2:
                go = st.button("Predict", type="primary", use_container_width=True)
            picks = st.pills("Examples", list(EXAMPLES), selection_mode="single", default=None) \
                if hasattr(st, "pills") else st.selectbox("Examples", ["(none)"] + list(EXAMPLES))
            # These are approved drugs, and approved drugs are in the measured public data these
            # models are fitted to. Their profiles therefore demonstrate what the output looks like
            # and what a correct answer contains; they are not evidence that the panel predicts
            # well, because the panel has seen them. Each result's applicability-domain row says so
            # numerically, reporting a maximum Tanimoto of 1.00, but a reader who clicks an example
            # before reading that row should be told first.
            st.caption(
                "These examples are approved drugs and appear in the training data, so their "
                "maximum Tanimoto is 1.00. They show what a result contains, not how well the "
                "panel generalises. For that see the prospective validation in the About tab, or "
                "submit a compound of your own.")
            st.checkbox(
                "High-precision screening mode",
                key="screening_mode",
                help="Raises every target threshold to a 1 percent background false-positive "
                     "rate. Use when triaging a large diverse library, where actives are rare "
                     "and a false positive costs an experiment. Recall falls in exchange for "
                     "precision. Leave it off for a focused or enriched set, where recall "
                     "matters more.")

        # A typed query always takes precedence, and it acts whether the user presses Enter or the
        # Predict button. Example pills are only consulted when the box is empty. Without this
        # precedence a pill selected earlier persists across reruns and silently re-renders its own
        # compound for every subsequent query, so the structure and scores shown belong to the pill
        # rather than to what was typed.
        chosen = picks if (isinstance(picks, str) and picks in EXAMPLES) else None
        typed = query.strip()
        if typed:
            smi, name, err = resolve_query(typed)
            if err:
                st.error(err)
            else:
                render_report(smi, name)
        elif chosen:
            render_report(EXAMPLES[chosen], chosen.split(" (")[0])
        else:
            st.markdown('<div class="bs-empty"><div class="k">🔎</div>'
                        '<div style="margin-top:8px">Type a <b>compound name</b> or <b>SMILES</b> above, or tap an '
                        'example, then press <b>Predict</b>.</div></div>', unsafe_allow_html=True)

    with tab_batch:
        with st.container(border=True):
            render_batch()

    with tab_about:
        render_about()
    with tab_legal:
        render_legal()


if __name__ == "__main__":
    main()
