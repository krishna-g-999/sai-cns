"""How much measured data exists for the mechanisms the panel does not model?

The Coverage panel lists what BrainSafe does not cover. That list should rest on measured fact
rather than on recollection, because "there is not enough data" is exactly the kind of claim that
ages badly: ChEMBL grows, and a target that was untrainable two years ago may not be untrainable
now. This script asks ChEMBL directly and prints the count against the threshold actually used to
build the deployed panel, which was 800 activities carrying a pChEMBL value.

Target identifiers are resolved by searching ChEMBL, never taken from memory, and the resolved
identifier and preferred name are printed with every count so the mapping can be checked rather than
trusted. That check earned its place twice: an early version filtered to single proteins and so
reported the nicotinic subtypes, the kainate receptors and the NMDA receptor as absent when they are
annotated as complexes, and a name search for kainate 2 returned kainate 5. Read the pref_name
column before quoting any row.

Volume alone does not decide the question, so evidence diversity is reported beside it. A target with
tens of thousands of potency values from one screening campaign is one library measured once; a
scaffold-split model trained on it learns the library.

Read-only. Writes results/unmodelled_target_data.csv
"""
from __future__ import annotations

import sys
import time
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

# (group, label as written in the Coverage panel, search term)
QUERIES = [
    ("Kainate receptors", "GluK1 (GRIK1)", "Glutamate receptor ionotropic kainate 1"),
    ("Kainate receptors", "GluK2 (GRIK2)", "Glutamate receptor ionotropic, kainate 2"),
    ("Kainate receptors", "kainate receptor", "kainate receptor"),
    ("Nicotinic subtypes", "alpha4beta2 nAChR", "Neuronal acetylcholine receptor; alpha4/beta2"),
    ("Nicotinic subtypes", "alpha3beta4 nAChR", "Neuronal acetylcholine receptor; alpha3/beta4"),
    ("Nicotinic subtypes", "alpha6 nAChR", "Neuronal acetylcholine receptor subunit alpha-6"),
    ("NMDA channel site", "NMDA receptor", "Glutamate [NMDA] receptor"),
    ("NMDA channel site", "GluN1 subunit", "Glutamate receptor ionotropic NMDA 1"),
    ("Monoamine storage", "VMAT2 (SLC18A2)", "Synaptic vesicular amine transporter"),
    ("Calcium channels", "Cav2.2 N-type (CACNA1B)", "Voltage-gated N-type calcium channel alpha-1B"),
    ("Calcium channels", "Cav3.1 T-type (CACNA1G)", "Voltage-gated T-type calcium channel alpha-1G subunit"),
    ("Calcium channels", "T-type calcium channel", "T-type calcium channel"),
    ("Calcium channels", "alpha2delta-1 (CACNA2D1)", "Voltage-gated calcium channel alpha2/delta subunit 1"),
    ("Sodium subtypes", "Nav1.2 (SCN2A)", "Sodium channel protein type II alpha subunit"),
    ("Sodium subtypes", "Nav1.6 (SCN8A)", "Sodium channel protein type VIII alpha subunit"),
    ("Sodium subtypes", "Nav1.8 (SCN10A)", "Sodium channel protein type X alpha subunit"),
    ("Aggregation", "alpha-synuclein (SNCA)", "Alpha-synuclein"),
    ("Aggregation", "tau (MAPT)", "Microtubule-associated protein tau"),
    ("Aggregation", "huntingtin (HTT)", "Huntingtin"),
    ("ALS genetics", "SOD1", "Superoxide dismutase"),
    ("ALS genetics", "TDP-43 (TARDBP)", "TAR DNA-binding protein 43"),
    ("ALS genetics", "C9orf72", "C9orf72"),
    # GABA-A is deployed as one pooled endpoint and carries the thinnest evidence in the panel.
    # ChEMBL annotates the physiological subtype complexes separately, and they are better
    # populated than the pooled target, so they are measured here as a candidate improvement.
    ("GABA-A subtypes", "GABA-A alpha1/beta2/gamma2", "GABA-A receptor; alpha-1/beta-2/gamma-2"),
    ("GABA-A subtypes", "GABA-A alpha2/beta3/gamma2", "GABA-A receptor; alpha-2/beta-3/gamma-2"),
    ("GABA-A subtypes", "GABA-A alpha5/beta3/gamma2", "GABA-A receptor; alpha-5/beta-3/gamma-2"),
    # two deployed endpoints as positive controls, so the counts can be read against something known
    ("Control (deployed)", "SERT", "Serotonin transporter"),
    ("Control (deployed)", "Nav1.1 (SCN1A)", "Sodium channel protein type I alpha subunit"),
]


def get(url, params=None, tries=3):
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60, verify=False)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(2)
    return None


def resolve(term):
    """Best-matching human target. Protein complexes are included deliberately: the nicotinic
    subtypes, the kainate receptors and the NMDA receptor are annotated in ChEMBL as complexes, and
    filtering to single proteins alone silently reports them as absent when they are present."""
    j = get(f"{BASE}/target/search.json", {"q": term, "limit": 30})
    if j is None:
        return "QUERY_FAILED"
    best = None
    for t in j.get("targets", []):
        if t.get("organism") != "Homo sapiens":
            continue
        if t.get("target_type") not in ("SINGLE PROTEIN", "PROTEIN COMPLEX", "PROTEIN FAMILY"):
            continue
        name = (t.get("pref_name") or "").lower()
        score = 3 if term.lower() == name else (2 if term.lower() in name else
                                                (1 if name in term.lower() else 0))
        if t.get("target_type") == "SINGLE PROTEIN":
            score += 0.5          # prefer a defined protein over a complex when both match
        if best is None or score > best[0]:
            best = (score, t)
    return best[1] if best else None


def profile_target(cid):
    """Activity count plus the evidence diversity behind it.

    A raw activity count is a poor measure of whether a target can be modelled. Ninety thousand
    potency values from one high-throughput campaign are a single library screened once, and a model
    trained on them learns that library rather than the target. A few thousand values drawn from
    hundreds of medicinal-chemistry papers span many scaffolds and many laboratories, and that is
    what a scaffold-split model needs. The count and the diversity are therefore reported together.
    """
    j = get(f"{BASE}/activity.json",
            {"target_chembl_id": cid, "pchembl_value__isnull": "false", "limit": 1})
    if j is None:
        return None
    total = int(j["page_meta"]["total_count"])
    if total == 0:
        return {"n": 0, "n_docs": 0, "n_assays": 0, "top_assay": "", "top_share": 0.0}
    s = get(f"{BASE}/activity.json",
            {"target_chembl_id": cid, "pchembl_value__isnull": "false", "limit": 1000})
    acts = (s or {}).get("activities", [])
    docs = {a.get("document_chembl_id") for a in acts if a.get("document_chembl_id")}
    assays = {a.get("assay_chembl_id") for a in acts if a.get("assay_chembl_id")}
    from collections import Counter
    desc = Counter((a.get("assay_description") or "")[:70] for a in acts)
    top, cnt = (desc.most_common(1)[0] if desc else ("", 0))
    return {"n": total, "n_docs": len(docs), "n_assays": len(assays),
            "top_assay": top, "top_share": (cnt / len(acts)) if acts else 0.0,
            "sampled": len(acts)}


def main():
    rows = []
    for group, label, term in QUERIES:
        t = resolve(term)
        if t == "QUERY_FAILED":
            # A failed request must never be recorded as an absence of data. Reporting a network
            # error as "no ligands exist" is the one error that would make this whole exercise
            # actively misleading.
            rows.append({"group": group, "mechanism": label, "chembl_id": None,
                         "chembl_pref_name": None, "activities_with_pchembl": None,
                         "n_documents_in_sample": None, "n_assays_in_sample": None,
                         "largest_assay_share": None, "dominant_assay": "",
                         "trainable": None, "reason": "ChEMBL query failed; rerun"})
            print(f"  {label:26} QUERY FAILED, not a zero", flush=True)
            continue
        if t is None:
            rows.append({"group": group, "mechanism": label, "chembl_id": None,
                         "chembl_pref_name": None, "activities_with_pchembl": 0,
                         "n_documents_in_sample": 0, "n_assays_in_sample": 0,
                         "largest_assay_share": None, "dominant_assay": "",
                         "trainable": False, "reason": "no matching human target in ChEMBL"})
            print(f"  {label:26} no matching human target", flush=True)
            continue
        p = profile_target(t["target_chembl_id"])
        if p is None:
            rows.append({"group": group, "mechanism": label,
                         "chembl_id": t["target_chembl_id"],
                         "chembl_pref_name": t.get("pref_name"),
                         "activities_with_pchembl": None, "n_documents_in_sample": None,
                         "n_assays_in_sample": None, "largest_assay_share": None,
                         "dominant_assay": "", "trainable": None,
                         "reason": "ChEMBL query failed; rerun"})
            print(f"  {label:26} {t['target_chembl_id']:14} QUERY FAILED, not a zero", flush=True)
            continue
        # Two conditions, not one. Enough measurements, and enough independent sources for a
        # scaffold split to mean anything.
        enough = p["n"] >= MIN_ACTIVITIES
        diverse = p["n_docs"] >= 20 and p["top_share"] < 0.60
        reason = ("" if (enough and diverse) else
                  f"below the {MIN_ACTIVITIES}-activity bar" if not enough else
                  f"{p['n']:,} activities but {p['n_docs']} sources in a 1000-activity sample and "
                  f"{p['top_share']:.0%} from one assay: a single campaign, not diverse chemistry")
        rows.append({"group": group, "mechanism": label,
                     "chembl_id": t["target_chembl_id"],
                     "chembl_pref_name": t.get("pref_name"),
                     "activities_with_pchembl": p["n"],
                     "n_documents_in_sample": p["n_docs"],
                     "n_assays_in_sample": p["n_assays"],
                     "largest_assay_share": round(p["top_share"], 3),
                     "dominant_assay": p["top_assay"],
                     "trainable": bool(enough and diverse), "reason": reason})
        flag = "TRAINABLE" if (enough and diverse) else ("thin evidence" if enough else "")
        print(f"  {label:26} {t['target_chembl_id']:14} {p['n']:7,} acts  "
              f"{p['n_docs']:4} sources  {flag}", flush=True)
        time.sleep(0.15)

    df = pd.DataFrame(rows).sort_values(["group", "activities_with_pchembl"],
                                        ascending=[True, False], na_position="last")
    df.to_csv(OUT / "unmodelled_target_data.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 34)
    print()
    print(df.drop(columns=["dominant_assay"]).to_string(index=False))
    failed = df[df.trainable.isna()]
    if len(failed):
        print(f"\n{len(failed)} queries failed and are NOT zeros: "
              f"{', '.join(failed.mechanism)}. Rerun before drawing any conclusion.")
    body = df[df.group != "Control (deployed)"]
    can = body[body.trainable == True]  # noqa: E712  - explicit, since None must not count
    print(f"\n{len(can)} of {len(body)} unmodelled mechanisms are trainable on both counts:")
    for _, r in can.iterrows():
        print(f"   {r.mechanism:26} {r.activities_with_pchembl:,} activities from "
              f"{r.n_documents_in_sample} sources ({r.chembl_id})")
    thin = body[(body.trainable == False) & (body.activities_with_pchembl >= MIN_ACTIVITIES)]  # noqa: E712
    if len(thin):
        print(f"\n{len(thin)} have the volume but not the diversity:")
        for _, r in thin.iterrows():
            print(f"   {r.mechanism:26} {r.reason}")
    print("\nwrote", OUT / "unmodelled_target_data.csv")


if __name__ == "__main__":
    main()
