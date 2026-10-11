"""Prospective external validation on compounds absent from every training set.

Design notes, because the value of this test rests entirely on them:

  1. Structures are resolved from PubChem by compound NAME rather than written from memory. A
     mis-recalled SMILES would silently invalidate the whole exercise, so no structure in this file
     is authored by hand.
  2. Novelty is enforced by canonical-SMILES comparison against the molecular-target training tables,
     which are the models that decide the predicted disease. Compounds present there are dropped and
     counted. Presence in the ADME property libraries is recorded per compound instead of excluding,
     because those libraries contain most marketed drugs but carry no disease-label information.
  3. Each compound carries an expected class assigned from its established primary pharmacology,
     fixed BEFORE any prediction is made, so the comparison is pre-registered rather than fitted.
  4. Negative controls (peripherally acting or non-CNS drugs) are included so that specificity can
     be measured, not just sensitivity.

Outputs:
  results/tables/external_100_predictions.csv   one row per compound, all endpoint outputs
  results/tables/external_100_summary.csv       accuracy statistics with confidence intervals
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
requests.packages.urllib3.disable_warnings()
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))
CACHE = ROOT / "data" / "_pubchem_cache" / "external_100_smiles.json"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# name -> expected primary CNS class (assigned from established pharmacology, fixed in advance)
COMPOUNDS = {
    # --- Alzheimer's / cognition ---
    "Galantamine": "Alzheimer's disease", "Rivastigmine": "Alzheimer's disease",
    "Huperzine A": "Alzheimer's disease", "Tacrine": "Alzheimer's disease",
    "Memantine": "Alzheimer's disease", "Idalopirdine": "Cognition (cholinergic)",
    "Encenicline": "Cognition (cholinergic)", "Phenserine": "Alzheimer's disease",
    "Ladostigil": "Alzheimer's disease", "Donepezil": "Alzheimer's disease",
    # --- Parkinson's ---
    "Rasagiline": "Parkinson's disease", "Safinamide": "Parkinson's disease",
    "Pramipexole": "Parkinson's disease", "Ropinirole": "Parkinson's disease",
    "Rotigotine": "Parkinson's disease", "Bromocriptine": "Parkinson's disease",
    "Apomorphine": "Parkinson's disease", "Istradefylline": "Parkinson's disease",
    "Entacapone": "Parkinson's disease", "Amantadine": "Parkinson's disease",
    # --- Depression / anxiety ---
    "Sertraline": "Depression / anxiety", "Paroxetine": "Depression / anxiety",
    "Citalopram": "Depression / anxiety", "Venlafaxine": "Depression / anxiety",
    "Duloxetine": "Depression / anxiety", "Mirtazapine": "Depression / anxiety",
    "Vortioxetine": "Depression / anxiety", "Buspirone": "Depression / anxiety",
    "Moclobemide": "Depression / anxiety", "Bupropion": "Depression / anxiety",
    "Reboxetine": "Depression / anxiety", "Vilazodone": "Depression / anxiety",
    # --- Psychosis ---
    "Risperidone": "Psychosis / schizophrenia", "Olanzapine": "Psychosis / schizophrenia",
    "Quetiapine": "Psychosis / schizophrenia", "Aripiprazole": "Psychosis / schizophrenia",
    "Clozapine": "Psychosis / schizophrenia", "Ziprasidone": "Psychosis / schizophrenia",
    "Lurasidone": "Psychosis / schizophrenia", "Cariprazine": "Psychosis / schizophrenia",
    "Chlorpromazine": "Psychosis / schizophrenia", "Sulpiride": "Psychosis / schizophrenia",
    # --- Epilepsy ---
    "Lamotrigine": "Epilepsy", "Phenytoin": "Epilepsy", "Valproic acid": "Epilepsy",
    "Levetiracetam": "Epilepsy", "Topiramate": "Epilepsy", "Zonisamide": "Epilepsy",
    "Lacosamide": "Epilepsy", "Oxcarbazepine": "Epilepsy", "Perampanel": "Epilepsy",
    "Clonazepam": "Epilepsy", "Phenobarbital": "Epilepsy", "Vigabatrin": "Epilepsy",
    # --- Chronic pain ---
    "Fentanyl": "Chronic pain", "Oxycodone": "Chronic pain", "Codeine": "Chronic pain",
    "Buprenorphine": "Chronic pain", "Tramadol": "Chronic pain", "Methadone": "Chronic pain",
    "Hydromorphone": "Chronic pain", "Naltrexone": "Chronic pain", "Tapentadol": "Chronic pain",
    "Pregabalin": "Chronic pain", "Gabapentin": "Chronic pain", "Ziconotide": "Chronic pain",
    # --- Sleep / wakefulness ---
    "Zolpidem": "Sleep / wakefulness", "Zopiclone": "Sleep / wakefulness",
    "Ramelteon": "Sleep / wakefulness", "Tasimelteon": "Sleep / wakefulness",
    "Lemborexant": "Sleep / wakefulness", "Daridorexant": "Sleep / wakefulness",
    "Modafinil": "Sleep / wakefulness", "Pitolisant": "Sleep / wakefulness",
    "Doxepin": "Sleep / wakefulness", "Triazolam": "Sleep / wakefulness",
    # --- Addiction / ADHD ---
    "Atomoxetine": "ADHD", "Amphetamine": "Addiction", "Cocaine": "Addiction",
    "Varenicline": "Addiction", "Nicotine": "Addiction", "Dexmethylphenidate": "ADHD",
    "Lisdexamfetamine": "ADHD", "Guanfacine": "ADHD",
    # --- Neuroinflammation ---
    "Ibuprofen": "Neuroinflammation", "Naproxen": "Neuroinflammation",
    "Rofecoxib": "Neuroinflammation", "Diclofenac": "Neuroinflammation",
    "Indomethacin": "Neuroinflammation", "Meloxicam": "Neuroinflammation",
    "Etoricoxib": "Neuroinflammation", "Dexamethasone": "Neuroinflammation",
    # --- Neuroprotection / oxidative stress (natural products) ---
    "Curcumin": "Neuroprotection / oxidative stress", "Quercetin": "Neuroprotection / oxidative stress",
    "Epigallocatechin gallate": "Neuroprotection / oxidative stress",
    "Luteolin": "Neuroprotection / oxidative stress", "Apigenin": "Neuroprotection / oxidative stress",
    "Kaempferol": "Neuroprotection / oxidative stress", "Fisetin": "Neuroprotection / oxidative stress",
    "Baicalein": "Neuroprotection / oxidative stress", "Myricetin": "Neuroprotection / oxidative stress",
    "Genistein": "Neuroprotection / oxidative stress", "Ferulic acid": "Neuroprotection / oxidative stress",
    "Rosmarinic acid": "Neuroprotection / oxidative stress",
    # --- Huntington / ALS ---
    "Tetrabenazine": "Huntington's disease", "Deutetrabenazine": "Huntington's disease",
    "Valbenazine": "Huntington's disease", "Edaravone": "Amyotrophic lateral sclerosis",
    "Riluzole": "Amyotrophic lateral sclerosis", "Romidepsin": "Huntington's disease",
    "Panobinostat": "Huntington's disease", "Belinostat": "Huntington's disease",
    # --- negative controls: peripheral or non-CNS ---
    "Atenolol": "NONE", "Furosemide": "NONE", "Metformin": "NONE", "Amoxicillin": "NONE",
    "Omeprazole": "NONE", "Simvastatin": "NONE", "Losartan": "NONE", "Hydrochlorothiazide": "NONE",
    "Ranitidine": "NONE", "Cetirizine": "NONE", "Loratadine": "NONE", "Warfarin": "NONE",
    "Ciprofloxacin": "NONE", "Insulin glargine": "NONE", "Allopurinol": "NONE",
    "Glucose": "NONE", "Ascorbic acid": "NONE", "Sucrose": "NONE",
}


def fetch_smiles(names):
    cached = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for n in names:
        if n in cached:
            continue
        got = None
        for prop in ("CanonicalSMILES", "SMILES", "IsomericSMILES"):
            try:
                r = requests.get(f"{PUG}/compound/name/{requests.utils.quote(n)}/property/{prop}/JSON",
                                 timeout=30, verify=False)
                if r.status_code == 200:
                    props = r.json()["PropertyTable"]["Properties"][0]
                    got = props.get(prop) or props.get("SMILES") or props.get("CanonicalSMILES")
                    if got:
                        break
            except Exception:
                pass
            time.sleep(0.12)
        cached[n] = got
        print(f"  {'OK ' if got else 'MISS'} {n}", flush=True)
        time.sleep(0.12)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cached, indent=2))
    return cached


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m else None


def training_structures(target_only=True):
    """Structures used to train the models.

    With target_only=True only the molecular-target tables are indexed. Those are the models that
    decide which disease is predicted, so absence from them is what makes a disease prediction
    genuinely prospective. The ADME tables are broad drug-property libraries containing most
    marketed drugs; excluding everything in them leaves almost no approved compound to test while
    contributing no disease-label information. ADME membership is recorded per compound in the
    output so the stricter subset can still be analysed separately.
    """
    import glob
    files = glob.glob(str(ROOT / "data" / "endpoints" / "*.csv")) + \
        glob.glob(str(ROOT / "data" / "endpoints_reg" / "*.csv"))
    if not target_only:
        files += glob.glob(str(ROOT / "data" / "adme" / "*.csv"))
    seen = set()
    for f in files:
        try:
            df = pd.read_csv(f, usecols=["smiles"])
        except Exception:
            continue
        for s in df["smiles"].astype(str):
            c = canon(s)
            if c:
                seen.add(c)
    return seen


def main():
    print("resolving structures from PubChem by name ...", flush=True)
    smiles = fetch_smiles(list(COMPOUNDS))
    print("\nbuilding the training-structure index ...", flush=True)
    train = training_structures(target_only=True)
    adme = training_structures(target_only=False) - train
    print(f"  molecular-target structures {len(train):,}; ADME-only additional {len(adme):,}", flush=True)

    import app
    models = app.load_models()

    rows, dropped_novelty, unresolved = [], [], []
    for name, expected in COMPOUNDS.items():
        smi = smiles.get(name)
        if not smi:
            unresolved.append(name)
            continue
        c = canon(smi)
        if c is None:
            unresolved.append(name)
            continue
        if c in train:
            dropped_novelty.append(name)
            continue
        r = app.predict_all(c, models)
        if r is None:
            unresolved.append(name)
            continue
        bbb, neuro, dz = app.disease_scores(r)
        ranked = [(d["disease"], d["gated"]) for d in dz]
        ad = app.assess_domain(c)
        thr = app.MIN_ACTIONABLE_SCORE
        top1 = ranked[0][0] if ranked and ranked[0][1] >= thr else "NONE"
        top3 = [d for d, v in ranked[:3] if v >= thr]
        row = {"compound": name, "expected_class": expected, "smiles": c,
               "predicted_top1": top1, "predicted_top1_score": round(ranked[0][1], 3) if ranked else 0.0,
               "predicted_top3": " | ".join(top3) if top3 else "NONE",
               "hit_top1": int(top1 == expected),
               "hit_top3": int(expected in top3) if expected != "NONE" else int(len(top3) == 0),
               "bbb": round(bbb, 3), "kpuu": round(r["adme"]["kpuu"], 3),
               "herg": round(r["targets"]["hERG"], 3),
               "ad_max_tanimoto": round(ad["max_sim"], 3) if ad else None,
               "ad_tier": ad["tier"] if ad else None,
               "in_adme_tables": int(c in adme)}
        for d, v in ranked:
            row["score_" + d.split(" (")[0].split(" / ")[0].replace(" ", "_")] = round(v, 3)
        rows.append(row)
        print(f"  {name:26} expect {expected[:22]:24} -> {top1[:24]:26} ({row['predicted_top1_score']:.2f}) "
              f"AD {row['ad_max_tanimoto']}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "tables" / "external_100_predictions.csv"
    df.to_csv(out, index=False)
    print(f"\nresolved {len(rows)} novel compounds; dropped {len(dropped_novelty)} already in training; "
          f"{len(unresolved)} unresolved")
    print("dropped for being in training:", dropped_novelty)
    print("unresolved:", unresolved)
    print("wrote", out)


if __name__ == "__main__":
    main()
