"""Turn a molecule into a graph for the GNN: atom nodes, bond edges, atom feature vectors.

Unlike the fingerprint pipeline, nothing is hashed. Each atom becomes a node with a feature vector
(element, degree, charge, hybridisation, aromaticity, hydrogen count, ring membership) and each bond
becomes an undirected edge. The GNN then learns its own representation by passing messages along the
bonds, which is the point of the method: the features are the raw graph, not a fixed fingerprint.

Pure RDKit + PyTorch tensors, no torch-geometric dependency.
"""
from __future__ import annotations

import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Elements common in drug-like molecules; anything else falls in the "other" bucket.
ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 11, 14, 34, 33]
HYBRIDS = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
           Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]


def _onehot(value, choices):
    v = [0.0] * (len(choices) + 1)
    v[choices.index(value) if value in choices else -1] = 1.0
    return v


def atom_features(atom) -> list[float]:
    return (_onehot(atom.GetAtomicNum(), ATOMS)
            + _onehot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
            + _onehot(atom.GetHybridization(), HYBRIDS)
            + _onehot(min(atom.GetTotalNumHs(), 4), [0, 1, 2, 3, 4])
            + [float(atom.GetFormalCharge()),
               float(atom.GetIsAromatic()),
               float(atom.IsInRing())])


NODE_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))


def _parent(smiles: str):
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
    return mol


def mol_to_graph(smiles: str):
    """Return (x, edge_index) tensors for one molecule, or None if it cannot be parsed.

    x: [n_atoms, NODE_DIM] float. edge_index: [2, n_edges] long, each bond added in both directions.
    """
    mol = _parent(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    src, dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return x, edge_index


def featurize_graphs(smiles_list):
    """Featurize many SMILES to graphs; returns (graphs, mask) aligned to the input."""
    graphs, mask = [], []
    for s in smiles_list:
        g = mol_to_graph(s)
        mask.append(g is not None)
        if g is not None:
            graphs.append(g)
    return graphs, torch.tensor(mask, dtype=torch.bool)
