"""Can a state-dependence endpoint be built? Count the paired measurements that would be needed.

Use-dependent sodium-channel blockers, the classic antiepileptic class, bind the inactivated channel
far more tightly than the resting one. Their potency is therefore not a property of the molecule
alone but of the molecule and the voltage protocol: the same compound assayed at a holding potential
of -90 mV, where channels rest, and at -65 mV, where a fraction are inactivated, can differ by an
order of magnitude. Both values sit in ChEMBL under the same target, so a model trained on the pooled
values is fitted to a label that structure does not determine.

The quantity that IS a property of the molecule is the shift, the difference in potency between a
depolarised and a hyperpolarised holding potential. That is what "use-dependent" means, and it is in
principle trainable. It requires the same compound measured at two or more holding potentials, which
is the thing this script counts, because whether such pairs exist in useful numbers decides the
question and no amount of argument does.

Holding potentials are parsed from assay descriptions, which is imperfect: a description that omits
the protocol is discarded rather than assumed, so these counts are a lower bound on what a full
curation of the primary literature would find.

Read-only. Writes results/state_dependence_feasibility.csv and
results/state_dependence_pairs.csv
"""
from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3

urllib3.disable_warnings()
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
BASE = "https://www.ebi.ac.uk/chembl/api/data"

NAV = {"Nav1.1": "CHEMBL1845", "Nav1.2": "CHEMBL4187", "Nav1.3": "CHEMBL4296",
       "Nav1.6": "CHEMBL5202", "Nav1.7": "CHEMBL4296", "Nav1.8": "CHEMBL5451"}
# distinct targets only; Nav1.3 and Nav1.7 above share an identifier in one ChEMBL mapping, so the
# dictionary is reduced to unique identifiers before querying
TARGETS = {"Nav1.1": "CHEMBL1845", "Nav1.2": "CHEMBL4187", "Nav1.6": "CHEMBL5202",
           "Nav1.7": "CHEMBL4296", "Nav1.8": "CHEMBL5451"}

# "-90 mV", "at -65 mV", "-70mV". Sign is required: an unsigned number is not a holding potential.
HOLD = re.compile(r"(-\s?\d{2,3})\s?mV", re.I)
STATE = re.compile(r"use[- ]dependent|inactivat|frequency[- ]dependent|state[- ]dependent|"
                   r"tonic block|phasic|holding potential", re.I)


def get(url, params=None, tries=3):
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120, verify=False)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(3)
    return None


def fetch_all(cid):
    rows, url = [], (f"{BASE}/activity.json?target_chembl_id={cid}"
                     f"&pchembl_value__isnull=false&limit=1000")
    while url:
        j = get(url)
        if j is None:
            break
        for a in j["activities"]:
            smi, pv = a.get("canonical_smiles"), a.get("pchembl_value")
            if not smi or not pv:
                continue
            desc = str(a.get("assay_description") or "")
            rows.append({"smiles": smi, "pchembl": float(pv), "desc": desc,
                         "assay": a.get("assay_chembl_id"), "doc": a.get("document_chembl_id")})
        nxt = j["page_meta"]["next"]
        url = ("https://www.ebi.ac.uk" + nxt) if nxt else None
    return pd.DataFrame(rows)


def main():
    summary, pair_rows = [], []
    for lab, cid in TARGETS.items():
        print(f"[{lab}] fetching {cid} ...", flush=True)
        df = fetch_all(cid)
        if df.empty:
            print(f"[{lab}] no data", flush=True)
            continue
        df["holding"] = [int(m.group(1).replace(" ", "")) if (m := HOLD.search(d)) else None
                         for d in df.desc]
        df["state_wording"] = [bool(STATE.search(d)) for d in df.desc]
        with_hold = df.dropna(subset=["holding"])

        # compounds measured at two or more distinct holding potentials, which is what a shift needs
        pairs = 0
        deltas = []
        for smi, g in with_hold.groupby("smiles"):
            pots = sorted(g.holding.unique())
            if len(pots) < 2:
                continue
            pairs += 1
            lo, hi = min(pots), max(pots)          # lo is the more negative, hyperpolarised
            p_lo = g.loc[g.holding == lo, "pchembl"].median()
            p_hi = g.loc[g.holding == hi, "pchembl"].median()
            deltas.append(p_hi - p_lo)
            pair_rows.append({"target": lab, "smiles": smi, "hyperpolarised_mV": lo,
                              "depolarised_mV": hi, "pchembl_hyperpolarised": round(p_lo, 2),
                              "pchembl_depolarised": round(p_hi, 2),
                              "delta_pchembl": round(p_hi - p_lo, 2)})
        summary.append({
            "target": lab, "chembl_id": cid,
            "activities_with_pchembl": len(df),
            "with_state_wording": int(df.state_wording.sum()),
            "with_parsable_holding_potential": len(with_hold),
            "distinct_holding_potentials": int(with_hold.holding.nunique()),
            "compounds_at_two_or_more_potentials": pairs,
            "median_abs_shift": round(float(np.median(np.abs(deltas))), 2) if deltas else None,
            "max_abs_shift": round(float(np.max(np.abs(deltas))), 2) if deltas else None,
        })
        print(f"[{lab}] {len(df):,} activities, {len(with_hold):,} with a parsable holding "
              f"potential across {with_hold.holding.nunique()} distinct values, "
              f"{pairs} compounds measured at two or more", flush=True)
        time.sleep(0.2)

    s = pd.DataFrame(summary)
    s.to_csv(OUT / "state_dependence_feasibility.csv", index=False)
    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(OUT / "state_dependence_pairs.csv", index=False)
    pd.set_option("display.width", 220)
    print()
    print(s.to_string(index=False))

    total_pairs = int(s.compounds_at_two_or_more_potentials.sum())
    print(f"\ncompounds with a measurable shift, pooled across subtypes: {total_pairs}")
    if pair_rows:
        d = pd.DataFrame(pair_rows)
        print(f"shift magnitude: median |delta| {np.median(np.abs(d.delta_pchembl)):.2f} log units, "
              f"90th percentile {np.quantile(np.abs(d.delta_pchembl), 0.9):.2f}")
        print(f"distinct scaffolds are not counted here; {d.smiles.nunique()} distinct structures")
    # 800 is the bar every deployed endpoint had to clear
    print(f"\nVERDICT: {'trainable' if total_pairs >= 800 else 'NOT trainable'} as a separate "
          f"endpoint on this evidence ({total_pairs} paired compounds against a bar of 800)")
    print("\nwrote", OUT / "state_dependence_feasibility.csv")


if __name__ == "__main__":
    main()
