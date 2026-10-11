"""Precompute per-endpoint training context so predictions can be reported HONESTLY.

For each classifier we store the base rate (prevalence of actives) — a calibrated probability
is only evidence of engagement when it EXCEEDS this. For each regression endpoint (receptors,
antioxidant) we store distribution quantiles so a prediction can be expressed as a percentile
against measured chemistry (and flagged when the model is non-discriminative).

Writes models_rf/endpoint_context.json.

Re-run after ANY retraining. This is not optional housekeeping: the interface reports
engagement as enrichment over the training base rate, so a base rate that predates the
models silently mis-scores every query. This docstring previously said "Run once", and
the file was three weeks behind the panel in production as a result, reporting a
compound at 0.70 against AChE as no engagement when the correct answer is 0.26.
tools/check_freshness.py now enforces it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models_rf" / "endpoint_context.json"

CLASSIFIERS = ["BBB", "AChE", "BChE", "BACE1", "GSK3B", "MAO_A", "MAO_B", "hERG"]
REGRESSORS = {"D2": "endpoints", "A2A": "endpoints", "HT2A": "endpoints", "SERT": "endpoints"}


def _quant(v):
    v = pd.to_numeric(v, errors="coerce").dropna().to_numpy()
    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    return {"n": int(v.size), "min": float(v.min()), "max": float(v.max()),
            "quantiles": {str(q): float(np.quantile(v, q)) for q in qs}}


def main():
    ctx = {"classifiers": {}, "regressors": {}}
    for ep in CLASSIFIERS:
        f = ROOT / "data" / "endpoints" / f"{ep}.csv"
        if f.exists():
            df = pd.read_csv(f)
            ctx["classifiers"][ep] = {"n": int(len(df)),
                                      "base_rate": float(df["label"].mean()) if "label" in df else None}
    for ep in REGRESSORS:
        f = ROOT / "data" / "endpoints" / f"{ep}.csv"
        if f.exists():
            df = pd.read_csv(f)
            col = "pchembl" if "pchembl" in df else df.columns[-1]
            ctx["regressors"][ep] = _quant(df[col])
    # antioxidant DPPH
    af = ROOT / "data" / "endpoints_reg" / "antioxidant_dpph.csv"
    if af.exists():
        df = pd.read_csv(af)
        col = "y" if "y" in df else ("pic50" if "pic50" in df else "value")
        ctx["regressors"]["antioxidant"] = _quant(df[col])
        ctx["regressors"]["antioxidant"]["col"] = col
    with OUT.open("w") as fh:
        json.dump(ctx, fh, indent=2)
    print("wrote", OUT)
    print(json.dumps(ctx, indent=2)[:1500])


if __name__ == "__main__":
    main()
