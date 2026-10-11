"""Measured natural-product activity from NPASS 3.0, joined to the panel's own targets.

The panel is short of one kind of chemistry, and the shortage is measurable rather than suspected.
The training library has a median fraction-sp3 of 0.36; withaferin A, a withanolide, has 0.79 and no
aromatic ring, and only 3.3 per cent of the library is both sp3-rich and non-aromatic. Submitted to
the server it returns a maximum Tanimoto of 0.31 with a bile acid as its nearest measured neighbour,
and no engagement call. That is honest behaviour on chemistry the panel does not cover.

The 214,740 COCONUT natural products already in the repository cannot close it. A decision recorded
on 2026-07-20 placed them as coverage and never as training labels, because they carry structures
and no measured endpoint values, and using them as labels would reintroduce the circularity the
project removed. That decision stands and this script does not touch them.

NPASS is different in the one way that matters: it carries measured activity, against named targets,
with the relation preserved. Rows from it enter under exactly the same rule as every other row.

Joining. NPASS keys targets by UniProt accession, so the panel is mapped to UniProt rather than
matched on name; `data/core_panel_target_ids.json` supplies the accessions already resolved, and the
rest are resolved from ChEMBL's target API and cached. Matching on a target name would silently
conflate isoforms and species.

Units. NPASS mixes nM, uM, ug/mL and per cent. Only concentration units convertible to molar are
used, and mass-per-volume is dropped rather than converted, because that conversion needs a
molecular weight per record and would introduce an error nobody could later trace. Percentages and
MIC values are not potencies and are excluded.

Nothing is merged. This writes a review table and a coverage report; adding any of it to training is
a separate, deliberate act that changes every model downstream.

Outputs:
  results/tables/npass_candidates.csv    one row per usable compound-target measurement
  results/tables/npass_coverage.csv      per-target summary of what it would add
  data/_chembl_cache/uniprot_map.json    resolved accessions, cached

Run:  python src/sai_cns/data/ingest_npass.py
      ... --npass-dir "NPASS data"      if the download sits elsewhere
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from activity_labels import ACTIVE_CUT, INACTIVE_CUT, label_from  # noqa: E402,F401

TAB = ROOT / "results" / "tables"
ENDPOINTS = ROOT / "data" / "endpoints"
CACHE = ROOT / "data" / "_chembl_cache"

# Potency types only. MIC is an antimicrobial growth endpoint and GI50 a cytotoxicity one; neither
# is target engagement, and folding them in would relabel the panel's meaning.
KEEP_TYPES = {"IC50", "Ki", "Kd", "EC50", "Potency"}
# Molar-convertible units only. ug/mL needs a per-record molecular weight to convert and is dropped.
TO_NM = {"nM": 1.0, "uM": 1e3, "mM": 1e6, "M": 1e9, "pM": 1e-3}
SP3_RICH, MAX_AROMATIC = 0.55, 1


def panel_uniprots() -> dict[str, str]:
    """Endpoint name to UniProt accession, resolved once and cached."""
    out: dict[str, str] = {}
    known = ROOT / "data" / "core_panel_target_ids.json"
    if known.exists():
        for ep, rec in json.loads(known.read_text(encoding="utf-8")).items():
            if rec.get("uniprot"):
                out[ep] = rec["uniprot"]

    cache = CACHE / "uniprot_map.json"
    if cache.exists():
        out.update({k: v for k, v in json.loads(cache.read_text()).items() if v})

    # anything still unresolved, from ChEMBL, using the ids the fetchers declare
    chembl_ids: dict[str, str] = {}
    for f in sorted((ROOT / "src" / "sai_cns" / "data").glob("fetch_*.py")):
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"[\"']([A-Za-z0-9_.]+)[\"']\s*:\s*\(?\s*[\"'](CHEMBL\d+)[\"']", text):
            chembl_ids.setdefault(m.group(1), m.group(2))
    missing = [e for e in chembl_ids if e not in out]
    if missing:
        try:
            import _tls
            s = _tls.session()
            for ep in missing:
                url = f"https://www.ebi.ac.uk/chembl/api/data/target/{chembl_ids[ep]}.json"
                try:
                    j = s.get(url, timeout=30).json()
                    for comp in j.get("target_components", []):
                        acc = comp.get("accession")
                        if acc:
                            out[ep] = acc
                            break
                except Exception:
                    continue
        except Exception as exc:
            print(f"  UniProt resolution unavailable ({exc}); using the cached map only")
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def describe(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    return {"fsp3": round(float(fsp3), 3), "aromatic_rings": int(arom),
            "mw": round(float(rdMolDescriptors.CalcExactMolWt(mol)), 1),
            "fills_the_gap": bool(fsp3 >= SP3_RICH and arom <= MAX_AROMATIC)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Join NPASS measured activity to the panel targets.")
    ap.add_argument("--npass-dir", default="NPASS data")
    args = ap.parse_args(argv)
    npass = ROOT / args.npass_dir
    if not npass.exists():
        raise SystemExit(f"NPASS directory not found: {npass}")

    print("resolving the panel to UniProt ...", flush=True)
    uni = panel_uniprots()
    want = {v.upper(): k for k, v in uni.items() if v}
    print(f"  {len(want)} panel targets carry an accession\n")

    print("reading NPASS ...", flush=True)
    tgt = pd.read_csv(npass / "NPASS3.0_target.txt", sep="\t", low_memory=False)
    tgt["acc"] = tgt.uniprot_id.astype(str).str.upper().str.strip()
    tgt = tgt[tgt.acc.isin(want)]
    tgt["endpoint"] = tgt.acc.map(want)
    print(f"  {len(tgt)} NPASS target records match {tgt.endpoint.nunique()} panel endpoints")

    acts = pd.read_csv(npass / "NPASS3.0_activities.txt", sep="\t", low_memory=False)
    acts = acts[acts.target_id.isin(set(tgt.target_id))]
    print(f"  {len(acts):,} activity records at those targets")

    acts = acts[acts.activity_type_grouped.isin(KEEP_TYPES)]
    acts = acts[acts.activity_units.isin(TO_NM)]
    acts["value"] = pd.to_numeric(acts.activity_value, errors="coerce")
    acts = acts[acts.value > 0]
    print(f"  {len(acts):,} with a potency type and a molar-convertible unit")

    struct = pd.read_csv(npass / "NPASS3.0_naturalproducts_structure.txt", sep="\t",
                         low_memory=False)[["np_id", "SMILES", "InChIKey"]]
    m = (acts.merge(tgt[["target_id", "endpoint"]], on="target_id", how="inner")
             .merge(struct, on="np_id", how="inner"))
    m = m[m.SMILES.notna()]
    print(f"  {len(m):,} joined to a structure\n")

    m["nm"] = m.value * m.activity_units.map(TO_NM)
    m["pchembl"] = 9.0 - m.nm.map(lambda v: math.log10(v) if v > 0 else float("nan"))
    m = m[m.pchembl.notna()]
    m["label"] = [label_from(p, r) for p, r in zip(m.pchembl, m.activity_relation)]
    m = m[m.label.notna()]
    print(f"{len(m):,} measurements settle a label under the project rule")

    seen_desc: dict[str, dict] = {}
    rows = []
    existing: dict[str, set] = {}
    for ep in m.endpoint.unique():
        p = ENDPOINTS / f"{ep}.csv"
        existing[ep] = (set(pd.read_csv(p, usecols=["smiles"]).smiles.astype(str))
                        if p.exists() else set())
    for r in m.itertuples():
        smi = str(r.SMILES)
        d = seen_desc.get(smi)
        if d is None:
            d = describe(smi)
            seen_desc[smi] = d if d else {}
        if not d:
            continue
        rows.append({"endpoint": r.endpoint, "np_id": r.np_id, "smiles": smi,
                     "inchikey": r.InChIKey, "pchembl": round(float(r.pchembl), 3),
                     "relation": r.activity_relation, "activity_type": r.activity_type_grouped,
                     "label": int(r.label),
                     "already_in_table": smi in existing.get(r.endpoint, set()), **d})

    cand = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    cand.to_csv(TAB / "npass_candidates.csv", index=False)

    summ = (cand.groupby("endpoint")
                .agg(measurements=("smiles", "size"),
                     distinct_compounds=("smiles", "nunique"),
                     new_to_endpoint=("already_in_table", lambda s: int((~s).sum())),
                     actives=("label", "sum"),
                     sp3_rich_non_aromatic=("fills_the_gap", "sum"),
                     median_fsp3=("fsp3", "median"))
                .reset_index().sort_values("new_to_endpoint", ascending=False))
    summ.to_csv(TAB / "npass_coverage.csv", index=False)

    print("\n=== what NPASS would add ===")
    print(f"  usable measurements        {len(cand):,}")
    print(f"  distinct compounds         {cand.smiles.nunique():,}")
    print(f"  new to their endpoint      {int((~cand.already_in_table).sum()):,}")
    gap = cand[cand.fills_the_gap]
    print(f"  sp3-rich, non-aromatic     {len(gap):,} "
          f"({100*len(gap)/max(len(cand),1):.1f}%), {gap.smiles.nunique():,} distinct")
    print(f"  median fsp3                {cand.fsp3.median():.2f}  "
          f"(library 0.36, withaferin A 0.79)")
    print(f"  endpoints touched          {cand.endpoint.nunique()}")
    print("\ntop endpoints by new compounds:")
    print(summ.head(12).to_string(index=False))
    print("\nwrote results/tables/npass_candidates.csv and npass_coverage.csv")
    print("Nothing was merged. Review before adding any of it to training.")


if __name__ == "__main__":
    main(sys.argv[1:])
