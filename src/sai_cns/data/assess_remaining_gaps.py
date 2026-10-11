"""One more attempt at every mechanism the panel still lacks, using angles not yet tried.

The first coverage audit asked a single question of each missing mechanism: does ChEMBL hold 800
activities for this target. That is the right first question and it is not the only one. Three angles
remain untried, and a gap that survives all of them is a real gap rather than an unexamined one:

  pooling        many kainate ligands are pan-kainate and many aggregation assays share a readout.
                 Subtypes that individually fall short may clear the bar together, and pooling is
                 defensible when the ligands do not discriminate between the members
  assay text     the NMDA channel-blocker site cannot be separated by target identifier because
                 ChEMBL annotates to the protein, but the assay descriptions name it. If enough
                 activities say "MK-801 site" or "phencyclidine site", a site-specific set can be
                 curated where a target-level query fails
  action type    ChEMBL records a direction for many activities and for drug mechanisms. The panel
                 currently predicts engagement without direction, which is a real limitation, and
                 whether it can be lifted depends on how many actives carry an agonist or antagonist
                 label

Each is counted here rather than argued. Read-only.
Writes results/remaining_gaps.csv and results/action_type_availability.csv
"""
from __future__ import annotations

import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings()
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results"
BASE = "https://www.ebi.ac.uk/chembl/api/data"
MIN_ACTIVITIES = 800


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


def count(cid, extra=None):
    p = {"target_chembl_id": cid, "pchembl_value__isnull": "false", "limit": 1}
    if extra:
        p.update(extra)
    j = get(f"{BASE}/activity.json", p)
    return int(j["page_meta"]["total_count"]) if j else None


def sample(cid, n=1000):
    j = get(f"{BASE}/activity.json",
            {"target_chembl_id": cid, "pchembl_value__isnull": "false", "limit": n})
    return j.get("activities", []) if j else []


# mechanism -> list of (label, ChEMBL target id). Identifiers carried over from the verified audit.
POOLS = {
    "Kainate receptors (pooled GluK1-K5)": [
        ("GluK1", "CHEMBL1918"), ("GluK2", "CHEMBL3683"), ("GluK3", "CHEMBL3684"),
        ("GluK5", "CHEMBL2675")],
    "Protein aggregation (pooled)": [
        ("alpha-synuclein", "CHEMBL6152"), ("tau", "CHEMBL1293224"), ("huntingtin", "CHEMBL5514")],
    "ALS genetics (pooled)": [
        ("SOD1", "CHEMBL2354"), ("TDP-43", "CHEMBL2362981")],
    "Calcium channel subtypes (pooled)": [
        ("Cav2.2", "CHEMBL4478"), ("Cav3.2", "CHEMBL1859"), ("alpha2delta-1", "CHEMBL1919")],
}
NMDA = ("NMDA receptor", "CHEMBL2015")
PCP_SITE = re.compile(r"MK[- ]?801|phencyclidine|\bPCP\b|dizocilpine|channel block|open[- ]channel|"
                      r"ketamine|memantine", re.I)

# targets whose actives would carry a direction, if any do
DIRECTION_TARGETS = [("D2", "CHEMBL217"), ("HT2A", "CHEMBL224"), ("OPRM1", "CHEMBL233"),
                     ("CB1", "CHEMBL218"), ("a7nAChR", "CHEMBL2492"), ("GABA_A", "CHEMBL2093872")]


def main():
    rows = []

    print("=== pooling subtypes that fall short individually ===", flush=True)
    for name, members in POOLS.items():
        tot, per, docs, shares = 0, [], set(), {}
        for lab, cid in members:
            n = count(cid)
            if n is None:
                per.append(f"{lab}: query failed")
                continue
            tot += n
            per.append(f"{lab} {n}")
            acts = sample(cid, 1000)
            for a in acts:
                if a.get("document_chembl_id"):
                    docs.add(a["document_chembl_id"])
            if acts:
                c = Counter((a.get("assay_description") or "")[:70] for a in acts)
                shares[lab] = c.most_common(1)[0][1] / len(acts)
            time.sleep(0.15)
        # Pooling can manufacture apparent diversity. Summing three sets that are each dominated by
        # a single screening campaign gives a healthy-looking document count while every member
        # remains one library measured once, so concentration is checked per member and the worst
        # member decides. Without this the aggregation pool passed on 114,943 activities that are
        # 86 and 98 per cent single-assay for tau and huntingtin.
        worst_share = max(shares.values()) if shares else 0.0
        worst_member = max(shares, key=shares.get) if shares else ""
        ok = tot >= MIN_ACTIVITIES and len(docs) >= 20 and worst_share < 0.60
        why = ""
        if tot < MIN_ACTIVITIES:
            why = f"pooled total {tot} still below {MIN_ACTIVITIES}"
        elif worst_share >= 0.60:
            why = (f"{worst_member} is {worst_share:.0%} one assay; pooling sets that are each one "
                   f"campaign does not create diverse chemistry")
        rows.append({"mechanism": name, "strategy": "pool subtypes", "activities": tot,
                     "distinct_documents_in_sample": len(docs),
                     "worst_member_assay_share": round(worst_share, 3),
                     "trainable": ok, "detail": "; ".join(per) + ("; " + why if why else "")})
        print(f"  {name:38} {tot:6,} activities, {len(docs):3} sources, worst member "
              f"{worst_share:.0%} one assay  {'TRAINABLE POOLED' if ok else 'rejected'}", flush=True)
        if why:
            print(f"       {why}", flush=True)

    print("\n=== curating the NMDA channel site from assay text ===", flush=True)
    lab, cid = NMDA
    acts = sample(cid, 1000)
    hits = [a for a in acts if PCP_SITE.search(str(a.get("assay_description") or ""))]
    tot = count(cid) or 0
    est = int(tot * len(hits) / len(acts)) if acts else 0
    rows.append({"mechanism": "NMDA channel-blocker (PCP) site", "strategy": "curate by assay text",
                 "activities": est, "distinct_documents_in_sample":
                     len({a.get("document_chembl_id") for a in hits}),
                 "trainable": est >= MIN_ACTIVITIES,
                 "detail": f"{len(hits)}/{len(acts)} sampled activities name the channel site; "
                           f"{tot} total activities at this target"})
    print(f"  {len(hits)}/{len(acts)} sampled activities name the channel-blocker site; "
          f"estimated {est} in total", flush=True)
    if hits:
        for d, c in Counter(str(a.get("assay_description"))[:88] for a in hits).most_common(3):
            print(f"     [{c}] {d}", flush=True)

    print("\n=== is an agonist / antagonist distinction learnable? ===", flush=True)
    drows = []
    for lab, cid in DIRECTION_TARGETS:
        acts = sample(cid, 1000)
        if not acts:
            print(f"  {lab:10} query failed", flush=True)
            continue
        # action_type is sometimes a nested object rather than a string
        def _act(a):
            v = a.get("action_type")
            return v.get("action_type") if isinstance(v, dict) else v
        at = Counter(_act(a) for a in acts)
        labelled = sum(v for k, v in at.items() if k)
        drows.append({"target": lab, "chembl_id": cid, "sampled": len(acts),
                      "with_action_type": labelled,
                      "fraction_labelled": round(labelled / len(acts), 4),
                      "breakdown": ", ".join(f"{k}:{v}" for k, v in at.most_common(5) if k) or "none"})
        print(f"  {lab:10} {labelled:4}/{len(acts)} activities carry an action type "
              f"({labelled/len(acts):.1%})  {drows[-1]['breakdown'][:60]}", flush=True)
        time.sleep(0.15)

    # the drug mechanism table is a separate, better-curated source of direction
    j = get(f"{BASE}/mechanism.json", {"limit": 1})
    n_mech = int(j["page_meta"]["total_count"]) if j else 0
    print(f"\n  ChEMBL drug mechanism records with a curated action type: {n_mech:,} in total, "
          f"covering approved and investigational drugs only", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "remaining_gaps.csv", index=False)
    if drows:
        pd.DataFrame(drows).to_csv(OUT / "action_type_availability.csv", index=False)
    pd.set_option("display.width", 230)
    pd.set_option("display.max_colwidth", 60)
    print()
    print(pd.DataFrame(rows).drop(columns=["detail"]).to_string(index=False))
    ok = [r["mechanism"] for r in rows if r["trainable"]]
    print(f"\n{len(ok)} of {len(rows)} previously unreachable mechanisms become trainable: "
          f"{ok or 'none'}")
    print("wrote", OUT / "remaining_gaps.csv")


if __name__ == "__main__":
    main()
