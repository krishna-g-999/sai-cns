"""Verify every target-to-pathway assertion in the knowledge graph against KEGG and Reactome.

Pathway identifiers written from memory are unreliable: an identifier believed to denote the
KEAP1-NFE2L2 pathway proved to denote Interleukin-1 signalling, and a target believed to sit in a
microglial pathway was annotated to osteoclast differentiation. Every edge is therefore checked
against the source database's own gene-to-pathway membership before it is allowed into the graph.

Output: data/_chembl_cache/pathway_verification.json
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

requests.packages.urllib3.disable_warnings()
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "_chembl_cache" / "pathway_verification.json"

# target -> (KEGG gene id, UniProt accession, [candidate KEGG pathway ids to test])
TARGETS = {
    "AChE": ("hsa:43", "P22303", ["hsa04725"]),
    "BChE": ("hsa:590", "P06276", ["hsa04725"]),
    "BACE1": ("hsa:23621", "P56817", ["hsa05010"]),
    "GSK3B": ("hsa:2932", "P49841", ["hsa05010"]),
    "MAO_A": ("hsa:4128", "P21397", ["hsa04726", "hsa04728"]),
    "MAO_B": ("hsa:4129", "P27338", ["hsa04726", "hsa04728"]),
    "SERT": ("hsa:6532", "P31645", ["hsa04726"]),
    "D2": ("hsa:1813", "P14416", ["hsa04728"]),
    "D3": ("hsa:1814", "P35462", ["hsa04728"]),
    "HT2A": ("hsa:3356", "P28223", ["hsa04726"]),
    "HT1A": ("hsa:3350", "P08908", ["hsa04726"]),
    "HT6": ("hsa:3362", "P50406", ["hsa04726"]),
    "HT7": ("hsa:3363", "P34969", ["hsa04726"]),
    "DAT": ("hsa:6531", "Q01959", ["hsa04728"]),
    "NET": ("hsa:6530", "P23975", ["hsa04728", "hsa04080"]),
    "CB1": ("hsa:1268", "P21554", ["hsa04723"]),
    "OPRM1": ("hsa:4988", "P35372", ["hsa05032"]),
    "OPRK1": ("hsa:4986", "P41145", ["hsa05032"]),
    "H3": ("hsa:11255", "Q9Y5N1", ["hsa04080"]),
    "A1": ("hsa:134", "P30542", ["hsa04080"]),
    "A2A": ("hsa:135", "P29274", ["hsa04080"]),
    "a7nAChR": ("hsa:1139", "P36544", ["hsa04725"]),
    "LRRK2": ("hsa:120892", "Q5S007", ["hsa05012"]),
    "Sigma1": ("hsa:10280", "Q99720", ["hsa04080"]),
    "OX1": ("hsa:3061", "O43613", ["hsa04080"]),
    "OX2": ("hsa:3062", "O43614", ["hsa04080"]),
    "MT1": ("hsa:4543", "P48039", ["hsa04713"]),
    "GABA_A": ("hsa:2554", "P14867", ["hsa04727"]),
    "GluN2B": ("hsa:2904", "Q13224", ["hsa04724", "hsa05014"]),
    "mGluR5": ("hsa:2915", "P41594", ["hsa04724"]),
    "NLRP3": ("hsa:114548", "Q96P20", ["hsa04621"]),
    "P2X7": ("hsa:5027", "Q99572", ["hsa04621"]),
    "COX2": ("hsa:5743", "P35354", ["hsa00590"]),
    "CSF1R": ("hsa:1436", "P07333", ["hsa04060", "hsa04380"]),
    "PDE10A": ("hsa:10846", "Q9Y233", ["hsa04024"]),
    "PDE4B": ("hsa:5142", "Q07343", ["hsa04024"]),
    "HDAC1": ("hsa:3065", "Q13547", ["hsa05016"]),
    "HDAC6": ("hsa:10013", "Q9UBN7", ["hsa05014", "hsa05016"]),
    "mTOR": ("hsa:2475", "P42345", ["hsa04140", "hsa05014"]),
    "SIRT1": ("hsa:23411", "Q96EB6", ["hsa04213"]),
    "GBA1": ("hsa:2629", "P04062", ["hsa04142"]),
    "Nav1_5": ("hsa:6331", "Q14524", ["hsa04261"]),
    "KEAP1": ("hsa:9817", "Q14145", []),      # Reactome only
}
REACTOME_CHECK = {"KEAP1": "R-HSA-9755511"}


def kegg_pathways(gene):
    try:
        t = requests.get(f"https://rest.kegg.jp/get/{gene}", timeout=30).text
    except Exception:
        return []
    out, cap = [], False
    for line in t.split("\n"):
        if line.startswith("PATHWAY"):
            cap = True
            line = line[7:]
        elif cap and line[:1] not in (" ", "\t"):
            cap = False
        if cap and line.strip():
            out.append(line.strip())
    return out


def kegg_name(pid):
    try:
        t = requests.get(f"https://rest.kegg.jp/get/{pid}", timeout=30).text
        for line in t.split("\n"):
            if line.startswith("NAME"):
                return line[4:].strip().replace(" - Homo sapiens (human)", "")
    except Exception:
        pass
    return None


def main():
    res, names = {}, {}
    for tgt, (gene, uni, cands) in TARGETS.items():
        pl = kegg_pathways(gene)
        have = {p.split()[0] for p in pl}
        rec = {"kegg_gene": gene, "uniprot": uni, "verified": [], "rejected": []}
        for c in cands:
            if c in have:
                rec["verified"].append(c)
                if c not in names:
                    names[c] = kegg_name(c)
            else:
                rec["rejected"].append(c)
        res[tgt] = rec
        status = "OK " if rec["verified"] else "NONE"
        print(f"{status} {tgt:9} verified={rec['verified']} rejected={rec['rejected']}", flush=True)

    for tgt, rid in REACTOME_CHECK.items():
        try:
            j = requests.get(f"https://reactome.org/ContentService/data/mapping/UniProt/"
                             f"{TARGETS[tgt][1]}/pathways?species=9606", timeout=30).json()
            ok = any(p.get("stId") == rid for p in j)
            nm = next((p["displayName"] for p in j if p.get("stId") == rid), None)
            res[tgt]["reactome"] = {"id": rid, "verified": bool(ok), "name": nm}
            print(f"{'OK ' if ok else 'BAD'} {tgt:9} reactome {rid} -> {nm}", flush=True)
        except Exception as e:
            res[tgt]["reactome"] = {"id": rid, "verified": False, "error": str(e)[:60]}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"targets": res, "pathway_names": names}, indent=2))
    nv = [t for t, r in res.items() if not r["verified"] and "reactome" not in r]
    print(f"\ntargets with no verified KEGG pathway: {nv}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
