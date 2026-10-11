"""Exercise a deployed SAI-CNS server over HTTP, the way a user does, and check it is correct.

A deployment can fail in ways a local test never sees: a stale image, a partially fetched model
set, a different scikit-learn, an environment variable that silently changes a threshold. All of
those produce a server that answers every request and returns wrong numbers. This script is the
check for that, and it is deliberately run against the URL rather than against the code.

Three phases, each answering a different question.

  reachable    does the server respond, and does it report the panel it is supposed to be serving:
               69 estimators, 51 graph targets, 16 conditions.

  faithful     does it return exactly what the validated tree returns. A fixed panel of compounds
               is scored and every number compared against a reference recorded from the validated
               local tree with --write-reference. Any difference above 1e-9 is a failure, because a
               correct deployment of a deterministic model has no reason to differ at all. This is
               the phase that catches a wrong image or a wrong model set.

  accurate     does it still achieve the published accuracy on data it has never seen. The 247
               compounds in trial_bbb_heldout_scored.csv are measured BBB outcomes whose InChIKeys
               are absent from every training set, and the panel's AUROC on them was 0.8010. The
               same compounds are sent through the server and the AUROC recomputed from what comes
               back over the wire.

Note on what "accuracy" can mean here. Sending famous drugs and observing that donepezil hits
AChE tests that the deployment works, not that the model generalises: donepezil is in the training
data. Only the held-out phase measures generalisation, and it is reported separately for that
reason.

Usage:
    python tools/test_live_server.py --write-reference          # once, on the validated tree
    python tools/test_live_server.py --url https://<host>       # against a deployment
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Versioned next to this script rather than under results/, which is gitignored: a reference
# the repository does not carry would make the faithful phase skip silently on a fresh clone.
REFERENCE = Path(__file__).resolve().parent / "live_server_reference.json"
HELDOUT = ROOT / "results" / "tables" / "trial_bbb_heldout_scored.csv"

# Deliberately spread: CNS-active, peripheral-by-design, a known BBB-excluded D2 antagonist, a
# bare SMILES and a trivial one, so resolution, gating and edge handling are all exercised.
PANEL = [
    "donepezil", "diazepam", "caffeine", "fluoxetine", "haloperidol", "memantine",
    "morphine", "loratadine", "domperidone", "levodopa", "risperidone", "quetiapine",
    "gabapentin", "ibuprofen", "metformin", "atenolol", "propranolol", "nicotine",
    "ketamine", "aspirin", "CCO", "c1ccccc1O",
]
TOL = 1e-9
EXPECTED_VERSION = {"n_models": 69, "n_graph_targets": 51, "n_conditions": 16}


def get(url, timeout=300):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def post(url, payload, timeout=900):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def flatten(d, prefix=""):
    """Every leaf of a prediction, as dotted key to value, so two responses can be compared."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


def phase_reachable(base):
    print("reachable")
    t0 = time.time()
    try:
        h = get(f"{base}/health")
    except Exception as e:
        print(f"  FAIL  /health unreachable: {type(e).__name__}: {e}")
        return False
    print(f"  /health   {h.get('status')}  ({time.time() - t0:.1f}s)")
    v = get(f"{base}/version")
    ok = True
    for k, want in EXPECTED_VERSION.items():
        got = v.get(k)
        if got != want:
            ok = False
        print(f"  {k:17s} {got}  (expected {want})  {'ok' if got == want else 'MISMATCH'}")
    print(f"  version          {v.get('version')}   mode {v.get('screening_mode')}")
    kg = (v.get("knowledge_graph") or {}).get("sha256", "")
    print(f"  graph sha256     {kg[:16]}...")
    return ok


def collect(base):
    """Score the panel through /predict, one request each, as an interactive user would."""
    out = {}
    for q in PANEL:
        try:
            out[q] = get(f"{base}/predict?q={urllib.parse.quote(q)}")
        except urllib.error.HTTPError as e:
            out[q] = {"_http": e.code, **json.loads(e.read() or b"{}")}
    return out


def phase_faithful(base):
    print("\nfaithful")
    if not REFERENCE.exists():
        print(f"  SKIP  no reference at {REFERENCE.relative_to(ROOT)}; run --write-reference first")
        return True
    ref = json.loads(REFERENCE.read_text())
    got = collect(base)
    worst, nvals, bad = 0.0, 0, []
    for q in PANEL:
        a, b = flatten(ref.get(q, {})), flatten(got.get(q, {}))
        if set(a) != set(b):
            bad.append((q, "shape", sorted(set(a) ^ set(b))[:4]))
            continue
        for k in a:
            x, y = a[k], b[k]
            numeric = (isinstance(x, (int, float)) and isinstance(y, (int, float))
                       and not isinstance(x, bool) and not isinstance(y, bool))
            if numeric:
                d = abs(float(x) - float(y))
                worst = max(worst, d)
                nvals += 1
                if d > TOL:
                    bad.append((q, k, f"{x} vs {y}"))
            elif x != y:
                bad.append((q, k, f"{x!r} vs {y!r}"))
    print(f"  compounds {len(PANEL)}   numeric values compared {nvals}")
    print(f"  largest absolute difference  {worst:.3g}   (tolerance {TOL:g})")
    if bad:
        print(f"  FAIL  {len(bad)} discrepancies:")
        for q, k, d in bad[:12]:
            print(f"    {q}  {k}  {d}")
        return False
    print("  every value returned by the server matches the validated tree exactly")
    return True


def auroc(y, p):
    """Rank-based AUROC with tie correction. Written out so the check needs no sklearn."""
    pairs = sorted(zip(p, y))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    pos = sum(1 for _, yy in pairs if yy == 1)
    neg = len(pairs) - pos
    if not pos or not neg:
        return float("nan")
    s = sum(r for r, (_, yy) in zip(ranks, pairs) if yy == 1)
    return (s - pos * (pos + 1) / 2.0) / (pos * neg)


def phase_accurate(base, published=0.8010):
    print("\naccurate  (247 compounds whose InChIKeys are in no training set)")
    if not HELDOUT.exists():
        print(f"  SKIP  {HELDOUT.relative_to(ROOT)} not present")
        return True
    import csv
    with HELDOUT.open() as fh:
        rows = list(csv.DictReader(fh))
    smiles = [r["smiles"] for r in rows]
    y = [int(r["y"]) for r in rows]
    t0 = time.time()
    got, CH = [], 150
    for i in range(0, len(smiles), CH):
        got.extend(post(f"{base}/batch", {"compounds": smiles[i:i + CH]})["results"])
    dt = time.time() - t0
    p, yy, nbad = [], [], 0
    for row, label in zip(got, y):
        v = row.get("bbb_probability")
        if row.get("status") != "ok" or v is None:
            nbad += 1
            continue
        p.append(float(v))
        yy.append(label)
    a = auroc(yy, p)
    print(f"  scored {len(p)} of {len(smiles)}   unscorable {nbad}   in {dt:.0f}s "
          f"({dt / max(1, len(smiles)) * 1000:.0f} ms per compound)")
    print(f"  AUROC over the wire  {a:.4f}        published  {published:.4f}")
    drift = abs(a - published)
    print(f"  difference {drift:.4f}  {'ok' if drift < 0.005 else 'INVESTIGATE'}")
    return drift < 0.005


def write_reference(base):
    ref = collect(base)
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(json.dumps(ref, indent=1, sort_keys=True))
    n = sum(len(flatten(v)) for v in ref.values())
    print(f"wrote {REFERENCE.relative_to(ROOT)}: {len(ref)} compounds, {n} values")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8899")
    ap.add_argument("--write-reference", action="store_true")
    ap.add_argument("--skip-heldout", action="store_true")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    print(f"SAI-CNS live server check against {base}\n")
    if a.write_reference:
        write_reference(base)
        return 0
    results = [("reachable", phase_reachable(base)), ("faithful", phase_faithful(base))]
    if not a.skip_heldout:
        results.append(("accurate", phase_accurate(base)))
    print("\n" + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results))
    return 0 if all(v for _, v in results) else 1


if __name__ == "__main__":
    sys.exit(main())
