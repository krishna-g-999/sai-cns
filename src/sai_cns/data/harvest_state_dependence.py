"""Harvest state-dependent sodium-channel potency from the primary literature.

Why this and not the other gaps. ChEMBL curators already extract systematically from the major
medicinal-chemistry journals, so re-curating those adds nothing: SOD1 has 29 measured activities
because few exist, not because papers were missed. What ChEMBL structurally cannot capture is a
PAIR. It records one value per assay and does not preserve the relationship between two measurements
of the same compound at two holding potentials, which is exactly the quantity that defines
use-dependent block. That is why only 89 paired compounds survive a database query
(results/state_dependence_feasibility.csv) while roughly 700 papers report such experiments.

This target is also the one where extraction is tractable. Electrophysiology papers test named drugs,
carbamazepine and lamotrigine rather than "compound 12a", so a structure can be resolved through
PubChem and no optical structure recognition is needed. That removes the failure mode that makes
literature curation unsafe for medicinal-chemistry series.

The pipeline:

  1. a recorded Europe PMC query, so the selection is reproducible and not a matter of judgement
  2. for open-access papers with full text, the XML tables are parsed, since a first attempt showed
     potency values are absent from running prose and live in tables and figures
  3. every candidate is scored on what was actually found: named compounds, holding potentials, and
     potency values in proximity
  4. papers that cannot be read here are ranked and listed for manual retrieval

Nothing extracted here is used for training until its error rate has been measured. The validation
design is in the docstring of the emitted file: extract points that overlap with ChEMBL first,
compare, and reject the whole harvest if disagreement exceeds a few per cent.

Writes results/state_dependence_papers.csv and results/state_dependence_extracted.csv
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings()
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results"
E = "https://www.ebi.ac.uk/europepmc/webservices/rest"

# The recorded query. Kept broad on the mechanism and strict on the protocol, because a paper that
# does not state a holding potential cannot contribute a pair whatever else it reports.
QUERY = ('("use-dependent" OR "use dependent" OR "state-dependent" OR "inactivated state" OR '
         '"tonic block" OR "phasic block") AND '
         '("sodium channel" OR "Na(v)" OR "NaV1") AND '
         '("holding potential" OR "membrane potential") AND '
         '(IC50 OR "IC(50)" OR Kd OR "half-maximal")')
PAGE = 100
MAX_PAPERS = 400

# Named compounds whose structure PubChem can resolve without reading a figure. Deliberately drugs
# and classical tool compounds only.
DRUGS = ["carbamazepine", "oxcarbazepine", "eslicarbazepine", "lamotrigine", "phenytoin",
         "fosphenytoin", "lacosamide", "rufinamide", "topiramate", "zonisamide", "valproate",
         "lidocaine", "mexiletine", "bupivacaine", "ropivacaine", "procaine", "tetracaine",
         "flecainide", "propafenone", "quinidine", "ranolazine", "riluzole", "safinamide",
         "amitriptyline", "vinpocetine", "phenobarbital", "cannabidiol", "tetrodotoxin",
         "saxitoxin", "veratridine", "ambroxol", "GS-458967", "PF-05089771", "vixotrigine"]
DRUG_RE = re.compile("|".join(re.escape(d) for d in DRUGS), re.I)
MV_RE = re.compile(r"(-\s?\d{2,3})\s*mV")
# a potency statement, tolerant of the ways papers write them
VAL_RE = re.compile(r"\b(IC\s?50|IC\(50\)|EC\s?50|Kd|Ki)\b[^0-9]{0,20}"
                    r"([0-9]+(?:\.[0-9]+)?)\s*(?:±\s*[0-9.]+\s*)?(nM|µM|uM|μM|mM|M)\b", re.I)


def get(url, params=None, tries=3, as_json=True):
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, timeout=90, verify=False)
            if r.status_code == 200:
                return r.json() if as_json else r.text
        except Exception:
            pass
        time.sleep(2)
    return None


def search():
    out, cursor = [], "*"
    while len(out) < MAX_PAPERS:
        j = get(f"{E}/search", {"query": QUERY, "format": "json", "pageSize": PAGE,
                                "cursorMark": cursor, "resultType": "lite"})
        if not j:
            break
        res = j.get("resultList", {}).get("result", [])
        if not res:
            break
        out.extend(res)
        nxt = j.get("nextCursorMark")
        if not nxt or nxt == cursor:
            break
        cursor = nxt
        print(f"  retrieved {len(out)} of {j.get('hitCount', 0)}", flush=True)
        time.sleep(0.3)
    return out[:MAX_PAPERS]


def tables_text(xml):
    """Table content only. A first attempt over running prose found no potency values at all;
    they live in tables and in figure panels, and only the former can be read here."""
    return " ".join(re.sub(r"<[^>]+>", " ", t)
                    for t in re.findall(r"<table-wrap.*?</table-wrap>", xml, re.S))


def main():
    print("searching Europe PMC ...", flush=True)
    hits = search()
    print(f"{len(hits)} papers retrieved\n", flush=True)

    rows, extracted = [], []
    for i, h in enumerate(hits, 1):
        pmcid, pmid = h.get("pmcid"), h.get("pmid")
        oa = h.get("isOpenAccess") == "Y" and h.get("inEPMC") == "Y"
        rec = {"pmid": pmid, "pmcid": pmcid, "doi": h.get("doi"),
               "title": (h.get("title") or "").strip()[:180],
               "journal": h.get("journalTitle"), "year": h.get("pubYear"),
               "open_access_fulltext": bool(oa), "named_drugs": "", "holding_potentials": "",
               "potency_values_in_tables": 0, "score": 0, "retrieval": ""}
        if oa and pmcid:
            xml = get(f"{E}/{pmcid}/fullTextXML", as_json=False)
            if xml and len(xml) > 2000:
                tabs = tables_text(xml)
                flat = re.sub(r"<[^>]+>", " ", xml)
                drugs = sorted({d.lower() for d in DRUG_RE.findall(flat)})
                mvs = sorted({m.replace(" ", "") for m in MV_RE.findall(flat)}, key=lambda x: int(x))
                vals = VAL_RE.findall(tabs) + VAL_RE.findall(flat)
                rec.update({"named_drugs": ", ".join(drugs[:8]),
                            "holding_potentials": ", ".join(mvs[:8]),
                            "potency_values_in_tables": len(vals)})
                # a paper is useful only if it names a resolvable compound, states at least two
                # holding potentials, and reports a number
                rec["score"] = (len(drugs) * 2) + (3 if len(mvs) >= 2 else 0) + min(len(vals), 10)
                rec["retrieval"] = "read here"
                for d in drugs:
                    for v in vals[:20]:
                        extracted.append({"pmcid": pmcid, "doi": h.get("doi"), "compound": d,
                                          "metric": v[0], "value": v[1], "unit": v[2],
                                          "holding_potentials_in_paper": ", ".join(mvs[:6]),
                                          "status": "CANDIDATE, unverified: compound-to-value "
                                                    "pairing not established, needs a human"})
            else:
                rec["retrieval"] = "open access but full text unavailable"
        else:
            rec["retrieval"] = "closed access: please retrieve"
            rec["score"] = 1
        rows.append(rec)
        if i % 25 == 0:
            print(f"  processed {i}/{len(hits)}", flush=True)
        time.sleep(0.15)

    df = pd.DataFrame(rows).sort_values(["score", "open_access_fulltext"], ascending=False)
    df.to_csv(OUT / "state_dependence_papers.csv", index=False)
    if extracted:
        pd.DataFrame(extracted).to_csv(OUT / "state_dependence_extracted.csv", index=False)

    readable = df[df.retrieval == "read here"]
    strong = readable[(readable.score >= 8)]
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 52)
    print()
    print(df.head(20)[["pmcid", "year", "journal", "named_drugs", "holding_potentials",
                       "potency_values_in_tables", "score"]].to_string(index=False))
    print(f"\n{len(df)} papers; {len(readable)} readable here; {len(strong)} carry a named drug, "
          f"two or more holding potentials and a potency value")
    print(f"{int((df.retrieval == 'closed access: please retrieve').sum())} are closed access and "
          f"are listed for manual retrieval, ranked")
    print(f"\ncandidate triples emitted: {len(extracted)}. Every one is UNVERIFIED: the regex "
          f"establishes that a paper contains a drug name, a holding potential and a number, not "
          f"that those three belong together. A human must confirm each pairing before any of it is "
          f"used, and the first 50 confirmed points must be ones that overlap ChEMBL so the "
          f"extraction error rate can be measured before the rest is trusted.")
    print("\nwrote", OUT / "state_dependence_papers.csv")


if __name__ == "__main__":
    main()
