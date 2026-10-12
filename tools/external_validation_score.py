"""Put the externally reported compounds through the deployed server and record what it says.

Reads external_panel_resolved.csv, keeps only the rows proven absent from all 169,407 training
InChIKeys, queries the live server, and writes both answers side by side.

WHAT THIS CAN AND CANNOT SHOW.

The claims in the panel are potencies. The server's output is a calibrated probability that a
compound is ACTIVE at an endpoint, where "active" means pChEMBL >= 6.0, that is 1 uM or better.
The two are not the same question, and conflating them would manufacture agreement or
disagreement at will. So the comparison is made explicitly:

    reported IC50 -> pIC50 -> is the compound active BY THE SERVER'S OWN DEFINITION?

A compound reported at 27 uM has pIC50 4.57. Under the rule this project uses to label its own
training data it is INACTIVE. If the server declines to flag it, that is the correct answer, not
a miss, and this script records the expectation before the prediction so the reasoning cannot be
rearranged afterwards.

Tier C rows are docking scores from other groups. Comparing a SAI-CNS probability to a docking
score compares two predictions and measures nothing about the compound. They are scored because
the user asked for them and reported separately, never pooled with tier A.

Run:
    python tools/external_validation_score.py --url http://10.110.90.30:8702
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "results" / "tables" / "external_panel_resolved.csv"
OUT = ROOT / "results" / "tables" / "external_validation_results.csv"
ACTIVE_CUT = 6.0          # the project's own definition of active, see data/activity_labels.py


def get(url, timeout=180):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def potency_from_claim(claim: str):
    """Pull a single IC50/Ki in uM or mmol/L out of the claim text, if one is stated.

    Deliberately conservative: it returns a value only for an unambiguous single number with a
    unit it recognises. Anything else returns None and is reported as having no usable potency
    rather than being guessed at.
    """
    m = re.search(r"IC50\s*([\d.]+)\s*mmol/L", claim)
    if m:
        return float(m.group(1)) * 1000.0, "mmol/L"
    m = re.search(r"IC50\s*([\d.]+)(?:\s*\+/-\s*[\d.]+)?\s*uM", claim)
    if m:
        return float(m.group(1)), "uM"
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://10.110.90.30:8702")
    a = ap.parse_args()
    base = a.url.rstrip("/")

    with PANEL.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["status"] == "held_out"]
    print(f"held-out compounds to test: {len(rows)}\n")

    out = []
    for r in rows:
        q = r["smiles"]
        try:
            d = get(f"{base}/predict?q={urllib.parse.quote(q)}")
        except Exception as e:
            print(f"  FAILED {r['name']}: {type(e).__name__}")
            continue

        tgt = r["target"]
        prob = (d.get("targets") or {}).get(tgt)
        binder = (d.get("binders") or {}).get(tgt) or {}
        if prob is None:
            prob = binder.get("probability")
        thr = binder.get("threshold")
        engaged = binder.get("engaged")
        if engaged is None and prob is not None and thr is not None:
            engaged = prob >= thr

        uM, unit = potency_from_claim(r["claim"])
        pic50 = round(-math.log10(uM * 1e-6), 2) if uM else None
        expected = None
        if pic50 is not None:
            expected = "active" if pic50 >= ACTIVE_CUT else "inactive"

        ad = d.get("applicability_domain") or {}
        rec = {
            "name": r["name"], "tier": r["tier"], "claimed_target": tgt,
            "claim": r["claim"], "source": r["source"],
            "reported_uM": uM, "reported_pIC50": pic50,
            "expected_by_server_definition": expected or "n/a (no usable potency)",
            "predicted_probability": prob,
            "threshold": thr,
            "server_calls_engaged": engaged,
            "bbb_probability": (d.get("exposure") or {}).get("bbb_probability"),
            "ad_max_tanimoto": ad.get("max_tanimoto"),
            "ad_tier": ad.get("tier"),
        }
        if expected in ("active", "inactive") and engaged is not None:
            got = "active" if engaged else "inactive"
            rec["agrees_with_own_definition"] = int(got == expected)
        else:
            rec["agrees_with_own_definition"] = ""
        out.append(rec)

        p = f"{prob:.4f}" if isinstance(prob, (int, float)) else "n/a"
        t = f"{thr:.3f}" if isinstance(thr, (int, float)) else "n/a"
        print(f"  {r['tier']}  {r['name']:22s} {tgt:7s} p={p} thr={t} "
              f"engaged={str(engaged):5s} expected={expected or '-'} "
              f"AD={ad.get('max_tanimoto')}")

    cols = list(out[0].keys()) if out else []
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"\n  wrote {OUT.relative_to(ROOT)}  ({len(out)} rows)")

    graded = [r for r in out if r["agrees_with_own_definition"] != ""]
    if graded:
        n = sum(r["agrees_with_own_definition"] for r in graded)
        print(f"\n  Of {len(graded)} compounds with a usable reported potency, the server's "
              f"call matched\n  what its own active/inactive rule implies in {n}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
