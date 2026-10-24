"""How much measured data exists for non-CNS targets the panel could expand into?

The deployed panel is CNS-focused. Expanding it into a genuinely multi-therapeutic-area tool is a
claim about data, not about ambition, so the candidate list has to be measured before it is
proposed. This script is the non-CNS counterpart of assess_unmodelled_targets.py and keeps the same
bar: 800 activities carrying a pChEMBL value, which is the threshold every deployed endpoint had to
clear. Candidates span oncology, inflammation and autoimmune disease, cardiovascular and metabolic
disease, infectious disease, and respiratory disease, because breadth is the point of the exercise.
Targets already carried by the panel are deliberately excluded so that nothing already modelled is
re-proposed.

Target identifiers are resolved by searching ChEMBL, never taken from memory, and the resolved
identifier and preferred name are printed with every count so the mapping can be checked rather than
trusted. That discipline earned its place in the CNS assessment twice over, and it earned it again
here. A search for "HIV-1 protease" returned a target holding 1,627 activities while the target
actually named "Human immunodeficiency virus type 1 protease" holds 6,421, so the obvious search
term would have understated the best available target roughly four-fold. Searches for ABL1, AKT1,
Aurora-A, p38-alpha, 5-lipoxygenase, COX-1 and S1P1 each returned a PROTEIN FAMILY or a neighbouring
protein ahead of the intended single protein, and a search for leukocyte elastase returned leukocyte
elastase inhibitor, which is a different molecule entirely. Read the chembl_pref_name column before
quoting any row.

Volume alone does not decide the question, so evidence diversity is reported beside it, exactly as
in the CNS assessment. A target with tens of thousands of potency values from one screening campaign
is one library measured once, and a scaffold-split model trained on it learns the library.

Two further measurements are taken here that the CNS assessment did not need, both prompted by the
deployment record of the existing panel. Of the five endpoints currently withheld by the deployment
gate, three are NFKB1, NR3C1 and NRF2: a transcription factor, a nuclear receptor and a pathway
node, whose public potency data comes largely from reporter and phenotypic assays that are indirect
and heterogeneous between laboratories. Their failure appears to track target class and assay
readout rather than therapeutic area, so a count alone would not have predicted it.

The first additional measurement is therefore assay directness, derived from fields actually
returned by the API rather than from assumption. assay_type distinguishes binding from functional
measurements, and bao_label gives the assay format, which separates an isolated protein from a
cell-based or whole-organism readout. Reporting the share of binding assays and the dominant format
makes it visible when a target clears the bar on volume supplied by cell-based reporter work, which
is the pattern that preceded gate failure before.

The second is the supply of inactive measurements. The default training mode for new endpoints is
hybrid_decoys_plus_measured_inactives, the honest headline metric is AUROC against measured
inactives rather than against decoys, and the median endpoint carries only about 240 measured
inactives. A target rich in actives but devoid of weak or censored measurements therefore cannot be
evaluated honestly even when it trains, so weak pChEMBL values and censored greater-than
measurements are counted directly. These are counts of what exists in ChEMBL, not of what would
survive curation, and they are reported as an indication of supply rather than as a yield.

Assessment only. No training data is fetched and no model is trained.

Read-only with respect to existing files. Writes results/noncns_target_data.csv
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings()
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
BASE = "https://www.ebi.ac.uk/chembl/api/data"
MIN_ACTIVITIES = 800   # the bar every deployed endpoint had to clear

# Words that mark a description as a reporter readout rather than a measurement on the protein.
# Used only to label the dominant assay, never to filter or to adjust a count.
REPORTER_WORDS = ("reporter", "luciferase", "luminescen", "transcription", "transactivation",
                  "gene expression", "mrna", "promoter", "nf-kb activation", "translocation")
PHENOTYPIC_WORDS = ("viability", "cytotoxic", "proliferation", "antiproliferative", "growth inhib",
                    "survival", "apoptosis", "colony", "antiviral activity", "cell growth")

# (target class, therapeutic area, label, search term, organism the match must belong to)
#
# Candidates are chosen for well-precedented medicinal chemistry, because a target with decades of
# published series is the one most likely to clear the bar, and they are weighted toward classes
# with direct binding or enzyme-inhibition readouts: kinases, proteases, other defined enzymes,
# GPCRs and transporters. Protein-protein interaction and nuclear-receptor candidates are included
# in small number and on purpose, as a test of whether the readout-driven failure seen in NFKB1,
# NR3C1 and NRF2 reproduces in targets of the same kind.
#
# Nothing already in the panel appears here. COX2, NLRP3, NFKB1, NR3C1, mTOR, HDAC1, HDAC6, CSF1R,
# DHODH, PDE4B, SIRT1, RIPK1, P2X7, Nav1.5, Nav1.7, Nav1.8, KEAP1, NRF2, GBA1 and MT1 already carry
# major non-CNS indications and are excluded as duplicates.
QUERIES = [
    # ---- Oncology -------------------------------------------------------------------------
    ("Receptor tyrosine kinase", "Oncology", "EGFR",
     "Epidermal growth factor receptor erbB1", "Homo sapiens"),
    ("Receptor tyrosine kinase", "Oncology", "VEGFR2 (KDR)",
     "Vascular endothelial growth factor receptor 2", "Homo sapiens"),
    ("Receptor tyrosine kinase", "Oncology", "ALK",
     "ALK tyrosine kinase receptor", "Homo sapiens"),
    ("Receptor tyrosine kinase", "Oncology", "MET",
     "Hepatocyte growth factor receptor", "Homo sapiens"),
    ("Receptor tyrosine kinase", "Oncology", "FLT3",
     "Receptor-type tyrosine-protein kinase FLT3", "Homo sapiens"),
    # resolved by gene symbol: a search for the family name returns the PROTEIN FAMILY entry first
    ("Cytoplasmic tyrosine kinase", "Oncology", "ABL1", "Tyrosine-protein kinase ABL1",
     "Homo sapiens"),
    ("Serine/threonine kinase", "Oncology", "BRAF",
     "Serine/threonine-protein kinase B-raf", "Homo sapiens"),
    ("Serine/threonine kinase", "Oncology", "CDK2", "Cyclin-dependent kinase 2", "Homo sapiens"),
    ("Serine/threonine kinase", "Oncology", "CDK9", "Cyclin-dependent kinase 9", "Homo sapiens"),
    ("Serine/threonine kinase", "Oncology", "Aurora-A", "Aurora kinase A", "Homo sapiens"),
    # ChEMBL names AKT1 by its protein name, not by the gene or the family
    ("Serine/threonine kinase", "Oncology", "AKT1", "RAC-alpha serine/threonine-protein kinase",
     "Homo sapiens"),
    ("Lipid kinase", "Oncology", "PI3K-alpha (PIK3CA)",
     "PI3-kinase p110-alpha subunit", "Homo sapiens"),
    ("Enzyme", "Oncology", "PARP1", "Poly [ADP-ribose] polymerase-1", "Homo sapiens"),
    ("Enzyme", "Oncology", "EZH2", "Histone-lysine N-methyltransferase EZH2", "Homo sapiens"),
    ("Chaperone", "Oncology", "HSP90-alpha", "Heat shock protein HSP 90-alpha", "Homo sapiens"),
    ("Epigenetic reader", "Oncology", "BRD4", "Bromodomain-containing protein 4", "Homo sapiens"),
    # the two protein-protein interaction candidates, included as a test of the class
    ("Protein-protein interaction", "Oncology", "BCL2", "Apoptosis regulator Bcl-2",
     "Homo sapiens"),
    ("Protein-protein interaction", "Oncology", "MDM2", "p53-binding protein Mdm-2",
     "Homo sapiens"),

    # ---- Inflammation and autoimmune ------------------------------------------------------
    ("Cytoplasmic tyrosine kinase", "Inflammation/autoimmune", "JAK1",
     "Tyrosine-protein kinase JAK1", "Homo sapiens"),
    ("Cytoplasmic tyrosine kinase", "Inflammation/autoimmune", "JAK2",
     "Tyrosine-protein kinase JAK2", "Homo sapiens"),
    ("Cytoplasmic tyrosine kinase", "Inflammation/autoimmune", "JAK3",
     "Tyrosine-protein kinase JAK3", "Homo sapiens"),
    ("Cytoplasmic tyrosine kinase", "Inflammation/autoimmune", "TYK2",
     "Non-receptor tyrosine-protein kinase TYK2", "Homo sapiens"),
    ("Cytoplasmic tyrosine kinase", "Inflammation/autoimmune", "SYK",
     "Tyrosine-protein kinase SYK", "Homo sapiens"),
    ("Cytoplasmic tyrosine kinase", "Inflammation/autoimmune", "BTK",
     "Tyrosine-protein kinase BTK", "Homo sapiens"),
    # MAPK14 resolved by its protein name; "p38 alpha" returns the family entry first
    ("Serine/threonine kinase", "Inflammation/autoimmune", "p38-alpha (MAPK14)",
     "Mitogen-activated protein kinase 14", "Homo sapiens"),
    ("Serine/threonine kinase", "Inflammation/autoimmune", "IRAK4",
     "Interleukin-1 receptor-associated kinase 4", "Homo sapiens"),
    ("Serine/threonine kinase", "Inflammation/autoimmune", "ROCK1",
     "Rho-associated protein kinase 1", "Homo sapiens"),
    # 5-LOX: the obvious search term returns 5-lipoxygenase-ACTIVATING protein, a different target
    ("Enzyme", "Inflammation/autoimmune", "5-LOX (ALOX5)",
     "Polyunsaturated fatty acid 5-lipoxygenase", "Homo sapiens"),
    # COX1 is distinct from the deployed COX2 endpoint; named in ChEMBL as a synthase
    ("Enzyme", "Inflammation/autoimmune", "COX1 (PTGS1)",
     "Prostaglandin G/H synthase 1", "Homo sapiens"),
    ("GPCR", "Inflammation/autoimmune", "S1P1 (S1PR1)",
     "Sphingosine 1-phosphate receptor 1", "Homo sapiens"),
    # the cytokine candidate, included as a test of the class rather than in expectation
    ("Protein-protein interaction", "Inflammation/autoimmune", "TNF-alpha",
     "Tumor necrosis factor", "Homo sapiens"),

    # ---- Cardiovascular and metabolic -----------------------------------------------------
    ("Protease", "Cardiovascular/metabolic", "Factor Xa", "Coagulation factor X", "Homo sapiens"),
    ("Protease", "Cardiovascular/metabolic", "Thrombin", "Thrombin", "Homo sapiens"),
    ("Protease", "Cardiovascular/metabolic", "Renin", "Renin", "Homo sapiens"),
    ("Protease", "Cardiovascular/metabolic", "ACE", "Angiotensin-converting enzyme",
     "Homo sapiens"),
    ("Protease", "Cardiovascular/metabolic", "DPP-4", "Dipeptidyl peptidase IV", "Homo sapiens"),
    ("Enzyme", "Cardiovascular/metabolic", "HMG-CoA reductase", "HMG-CoA reductase",
     "Homo sapiens"),
    ("Enzyme", "Cardiovascular/metabolic", "11beta-HSD1",
     "11-beta-hydroxysteroid dehydrogenase 1", "Homo sapiens"),
    ("Transporter", "Cardiovascular/metabolic", "SGLT2", "Sodium/glucose cotransporter 2",
     "Homo sapiens"),
    ("GPCR", "Cardiovascular/metabolic", "GLP-1R", "Glucagon-like peptide 1 receptor",
     "Homo sapiens"),
    ("GPCR", "Cardiovascular/metabolic", "AT1 (AGTR1)", "Type-1 angiotensin II receptor",
     "Homo sapiens"),
    ("GPCR", "Cardiovascular/metabolic", "ET-A (EDNRA)", "Endothelin receptor ET-A",
     "Homo sapiens"),
    # nuclear receptors, included knowing the class record and measured rather than assumed
    ("Nuclear receptor", "Cardiovascular/metabolic", "PPAR-gamma",
     "Peroxisome proliferator-activated receptor gamma", "Homo sapiens"),
    ("Nuclear receptor", "Cardiovascular/metabolic", "PPAR-alpha",
     "Peroxisome proliferator-activated receptor alpha", "Homo sapiens"),
    ("Nuclear receptor", "Cardiovascular/metabolic", "FXR (NR1H4)", "Farnesoid X receptor",
     "Homo sapiens"),

    # ---- Infectious disease ---------------------------------------------------------------
    # Pathogen targets are matched against the pathogen organism, not against Homo sapiens. The CNS
    # script could filter to human throughout; here that filter would discard the entire area.
    ("Viral enzyme", "Infectious disease", "HIV-1 reverse transcriptase",
     "Human immunodeficiency virus type 1 reverse transcriptase",
     "Human immunodeficiency virus 1"),
    # the full protein name, because "HIV-1 protease" resolves to a far smaller target
    ("Viral enzyme", "Infectious disease", "HIV-1 protease",
     "Human immunodeficiency virus type 1 protease", "Human immunodeficiency virus 1"),
    ("Viral enzyme", "Infectious disease", "HIV-1 integrase",
     "Human immunodeficiency virus type 1 integrase", "Human immunodeficiency virus 1"),
    ("Viral enzyme", "Infectious disease", "HCV NS5B polymerase",
     "Hepatitis C virus NS5B RNA-dependent RNA polymerase", "Hepatitis C virus"),
    ("Viral enzyme", "Infectious disease", "Influenza neuraminidase", "Neuraminidase",
     "Influenza A virus"),
    ("Viral enzyme", "Infectious disease", "SARS-CoV-2 Mpro", "3C-like proteinase",
     "Severe acute respiratory syndrome coronavirus 2"),
    ("Bacterial enzyme", "Infectious disease", "M. tuberculosis InhA",
     "Enoyl-[acyl-carrier-protein] reductase", "Mycobacterium tuberculosis"),
    ("Bacterial enzyme", "Infectious disease", "DNA gyrase B (E. coli)", "DNA gyrase subunit B",
     "Escherichia coli"),
    ("Parasite enzyme", "Infectious disease", "P. falciparum DHFR-TS",
     "Bifunctional dihydrofolate reductase-thymidylate synthase", "Plasmodium falciparum"),

    # ---- Respiratory ----------------------------------------------------------------------
    ("GPCR", "Respiratory", "beta2-adrenergic (ADRB2)", "Beta-2 adrenergic receptor",
     "Homo sapiens"),
    ("GPCR", "Respiratory", "M3 muscarinic (CHRM3)", "Muscarinic acetylcholine receptor M3",
     "Homo sapiens"),
    ("GPCR", "Respiratory", "CysLT1 (CYSLTR1)", "Cysteinyl leukotriene receptor 1",
     "Homo sapiens"),
    ("GPCR", "Respiratory", "H1 histamine (HRH1)", "Histamine H1 receptor", "Homo sapiens"),
    # CRTH2, named in ChEMBL as prostaglandin D2 receptor 2
    ("GPCR", "Respiratory", "CRTH2 (PTGDR2)", "Prostaglandin D2 receptor 2", "Homo sapiens"),
    # the obvious term returns leukocyte elastase INHIBITOR, which is not the enzyme
    ("Protease", "Respiratory", "Neutrophil elastase", "Neutrophil elastase", "Homo sapiens"),

    # two deployed non-CNS endpoints as positive controls, so the counts can be read against
    # something that is known to have trained and deployed in this pipeline
    ("Control (deployed)", "Control", "COX2 (deployed)", "Prostaglandin G/H synthase 2",
     "Homo sapiens"),
    ("Control (deployed)", "Control", "mTOR (deployed)",
     "Serine/threonine-protein kinase mTOR", "Homo sapiens"),
    # a deployed endpoint that the gate withheld, carried so the reporter-readout signature that
    # preceded its failure can be seen in the same columns as the candidates
    ("Control (withheld)", "Control", "NR3C1 (withheld by gate)", "Glucocorticoid receptor",
     "Homo sapiens"),
]


def get(url, params=None, tries=4):
    """Polite GET with backoff. Returns None rather than raising, so that a failure can be
    recorded as a failure instead of silently becoming a zero."""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60, verify=False)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return None


def resolve(term, organism):
    """Best-matching target of the requested organism.

    The organism is a parameter rather than a constant because the infectious-disease candidates do
    not live in Homo sapiens, and a human-only filter would report the whole area as absent. The
    match is a substring test, since ChEMBL records many pathogen targets against a strain or
    subtype name that contains the species name.
    """
    j = get(f"{BASE}/target/search.json", {"q": term, "limit": 40})
    if j is None:
        return "QUERY_FAILED"
    best = None
    for t in j.get("targets", []):
        if organism.lower() not in (t.get("organism") or "").lower():
            continue
        if t.get("target_type") not in ("SINGLE PROTEIN", "PROTEIN COMPLEX", "PROTEIN FAMILY"):
            continue
        name = (t.get("pref_name") or "").lower()
        score = 3 if term.lower() == name else (2 if term.lower() in name else
                                                (1 if name in term.lower() else 0))
        if t.get("target_type") == "SINGLE PROTEIN":
            score += 0.5          # prefer a defined protein over a family or complex
        if best is None or score > best[0]:
            best = (score, t)
    return best[1] if best else None


def count(params):
    """Total matching activities, or None if the request failed."""
    j = get(f"{BASE}/activity.json", dict(params, limit=1))
    return None if j is None else int(j["page_meta"]["total_count"])


def classify_directness(frac_binding, fmt_counts, descriptions):
    """Label the readout from fields the API actually returned.

    assay_type separates binding from functional measurement and bao_label gives the assay format,
    so the isolated-protein case can be told from the cell-based one without guessing. The
    description keywords only distinguish a reporter readout from a viability or growth readout
    once the format has already shown the assay to be cell based, because those two failure modes
    look identical in the format field and different in practice.
    """
    total = sum(fmt_counts.values()) or 1
    frac_cell = sum(v for k, v in fmt_counts.items() if "cell-based" in k) / total
    frac_org = sum(v for k, v in fmt_counts.items() if "organism-based" in k) / total
    frac_protein = sum(v for k, v in fmt_counts.items() if "single protein" in k) / total
    blob = " ".join(descriptions).lower()
    reporterish = sum(blob.count(w) for w in REPORTER_WORDS)
    phenoish = sum(blob.count(w) for w in PHENOTYPIC_WORDS)
    if frac_org >= 0.30:
        return "phenotypic", frac_cell
    if frac_cell >= 0.50:
        return ("cell_reporter" if reporterish >= phenoish else "phenotypic"), frac_cell
    if frac_binding >= 0.60 and frac_protein >= 0.40:
        return "direct_binding", frac_cell
    if frac_protein >= 0.40:
        return "enzymatic", frac_cell
    return "mixed", frac_cell


def profile_target(cid):
    """Activity count, evidence diversity, assay directness and inactive supply.

    A raw activity count is a poor measure of whether a target can be modelled, for the reason the
    CNS assessment set out: ninety thousand potency values from one high-throughput campaign are a
    single library screened once, and a model trained on them learns that library rather than the
    target. Diversity is measured on a 1,000-activity sample, which is the largest page the API
    serves, so the shares below describe the sample and not the whole target.

    Directness and inactive supply are measured alongside, because this pipeline has already shown
    that a target can clear the volume bar and still fail the deployment gate when its evidence is
    indirect, and that an endpoint cannot be honestly evaluated without measured inactives.
    """
    base = {"target_chembl_id": cid, "pchembl_value__isnull": "false"}
    total = count(base)
    if total is None:
        return None
    if total == 0:
        return {"n": 0, "n_docs": 0, "n_assays": 0, "top_assay": "", "top_share": 0.0,
                "frac_binding": 0.0, "directness": "no data", "fmt": "",
                "n_weak": 0, "n_censored": 0, "sampled": 0}
    s = get(f"{BASE}/activity.json", dict(base, limit=1000))
    if s is None:
        return None
    acts = s.get("activities", [])
    docs = {a.get("document_chembl_id") for a in acts if a.get("document_chembl_id")}
    assays = {a.get("assay_chembl_id") for a in acts if a.get("assay_chembl_id")}
    desc = Counter((a.get("assay_description") or "")[:70] for a in acts)
    top, cnt = (desc.most_common(1)[0] if desc else ("", 0))
    types = Counter(a.get("assay_type") for a in acts)
    frac_binding = types.get("B", 0) / len(acts) if acts else 0.0
    fmt_counts = Counter((a.get("bao_label") or "unknown").lower() for a in acts)
    directness, _ = classify_directness(
        frac_binding, fmt_counts, [a.get("assay_description") or "" for a in acts])

    # Supply of inactive and weak measurement, counted directly rather than inferred. The first is
    # the pool measured inactives are drawn from; the second is censored data, which carries real
    # information about inactivity even though it has no pChEMBL value.
    n_weak = count({"target_chembl_id": cid, "pchembl_value__lte": 5})
    n_censored = count({"target_chembl_id": cid, "standard_relation__exact": ">",
                        "standard_type__in": "IC50,Ki,Kd,EC50"})
    return {"n": total, "n_docs": len(docs), "n_assays": len(assays), "top_assay": top,
            "top_share": (cnt / len(acts)) if acts else 0.0, "frac_binding": frac_binding,
            "directness": directness,
            "fmt": fmt_counts.most_common(1)[0][0] if fmt_counts else "",
            "n_weak": n_weak, "n_censored": n_censored, "sampled": len(acts)}


def blank_row(group, area, label, cid, name, reason, trainable=None, zeros=False):
    z = 0 if zeros else None
    return {"group": group, "therapeutic_area": area, "mechanism": label, "chembl_id": cid,
            "chembl_pref_name": name, "activities_with_pchembl": z,
            "n_documents_in_sample": z, "n_assays_in_sample": z, "largest_assay_share": None,
            "dominant_assay": "", "assay_directness": "", "frac_binding_assay": None,
            "dominant_assay_format": "", "n_weak_or_inactive_pchembl": None,
            "n_censored_greater_than": None, "trainable": trainable, "reason": reason}


def main():
    rows = []
    for group, area, label, term, organism in QUERIES:
        t = resolve(term, organism)
        if t == "QUERY_FAILED":
            # A failed request must never be recorded as an absence of data. Reporting a network
            # error as "no ligands exist" is the one error that would make this whole exercise
            # actively misleading.
            rows.append(blank_row(group, area, label, None, None, "ChEMBL query failed; rerun"))
            print(f"  {label:28} QUERY FAILED, not a zero", flush=True)
            continue
        if t is None:
            rows.append(blank_row(group, area, label, None, None,
                                  f"no matching {organism} target in ChEMBL",
                                  trainable=False, zeros=True))
            print(f"  {label:28} no matching target for {organism}", flush=True)
            continue
        p = profile_target(t["target_chembl_id"])
        if p is None or p.get("n_weak") is None or p.get("n_censored") is None:
            rows.append(blank_row(group, area, label, t["target_chembl_id"], t.get("pref_name"),
                                  "ChEMBL query failed; rerun"))
            print(f"  {label:28} {t['target_chembl_id']:14} QUERY FAILED, not a zero", flush=True)
            continue
        # Volume and diversity decide trainability, on the same two conditions the CNS assessment
        # used. Directness is reported separately and does not override the count, because it is a
        # judgement about expected gate outcome rather than a measurement of whether data exists.
        enough = p["n"] >= MIN_ACTIVITIES
        diverse = p["n_docs"] >= 20 and p["top_share"] < 0.60
        reason = ("" if (enough and diverse) else
                  f"below the {MIN_ACTIVITIES}-activity bar" if not enough else
                  f"{p['n']:,} activities but {p['n_docs']} sources in a "
                  f"{p['sampled']}-activity sample and {p['top_share']:.0%} from one assay: "
                  f"a single campaign, not diverse chemistry")
        rows.append({"group": group, "therapeutic_area": area, "mechanism": label,
                     "chembl_id": t["target_chembl_id"],
                     "chembl_pref_name": t.get("pref_name"),
                     "activities_with_pchembl": p["n"],
                     "n_documents_in_sample": p["n_docs"],
                     "n_assays_in_sample": p["n_assays"],
                     "largest_assay_share": round(p["top_share"], 3),
                     "dominant_assay": p["top_assay"],
                     "assay_directness": p["directness"],
                     "frac_binding_assay": round(p["frac_binding"], 3),
                     "dominant_assay_format": p["fmt"],
                     "n_weak_or_inactive_pchembl": p["n_weak"],
                     "n_censored_greater_than": p["n_censored"],
                     "trainable": bool(enough and diverse), "reason": reason})
        flag = "TRAINABLE" if (enough and diverse) else ("thin evidence" if enough else "")
        print(f"  {label:28} {t['target_chembl_id']:14} {p['n']:7,} acts  "
              f"{p['n_docs']:4} src  {p['directness']:15} {flag}", flush=True)
        time.sleep(0.2)

    df = pd.DataFrame(rows).sort_values(
        ["therapeutic_area", "activities_with_pchembl"],
        ascending=[True, False], na_position="last")
    df.to_csv(OUT / "noncns_target_data.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 30)
    print()
    print(df.drop(columns=["dominant_assay", "reason", "group"]).to_string(index=False))

    failed = df[df.trainable.isna()]
    if len(failed):
        print(f"\n{len(failed)} queries failed and are NOT zeros: "
              f"{', '.join(failed.mechanism)}. Rerun before drawing any conclusion.")

    body = df[~df.group.str.startswith("Control")]
    can = body[body.trainable == True]  # noqa: E712  - explicit, since None must not count
    print(f"\n{len(can)} of {len(body)} candidates clear the {MIN_ACTIVITIES}-activity bar with "
          f"diverse evidence:")
    for area in sorted(can.therapeutic_area.unique()):
        print(f"  {area}")
        for _, r in can[can.therapeutic_area == area].iterrows():
            print(f"     {r.mechanism:28} {r.activities_with_pchembl:7,} acts  "
                  f"{r.n_documents_in_sample:4} src  {r.assay_directness:15} "
                  f"weak {r.n_weak_or_inactive_pchembl:6,}  censored {r.n_censored_greater_than:6,}")

    thin = body[(body.trainable == False) &  # noqa: E712
                (body.activities_with_pchembl >= MIN_ACTIVITIES)]
    if len(thin):
        print(f"\n{len(thin)} have the volume but not the diversity:")
        for _, r in thin.iterrows():
            print(f"   {r.mechanism:28} {r.reason}")

    indirect = can[~can.assay_directness.isin(["direct_binding", "enzymatic"])]
    if len(indirect):
        print(f"\n{len(indirect)} clear the bar on evidence that is not a direct readout, which is "
              f"the signature that preceded gate failure for NFKB1, NR3C1 and NRF2:")
        for _, r in indirect.iterrows():
            print(f"   {r.mechanism:28} {r.assay_directness:15} "
                  f"{r.frac_binding_assay:.0%} binding assays, dominant format "
                  f"{r.dominant_assay_format}")
    print("\nwrote", OUT / "noncns_target_data.csv")


if __name__ == "__main__":
    main()
