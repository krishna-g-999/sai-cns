# Graph neural network module

A self-contained graph-neural-network track, kept **separate** from the fingerprint/tree pipeline in
`src/sai_cns/{features,models}` so the two never mix. Nothing here is imported by the deployed
random-forest tool; it is an independent line of investigation.

## Contents

- `graph_features.py` — turns a SMILES into a molecular graph (atom nodes with element / degree /
  charge / hybridisation / aromaticity / H-count / ring features, bonds as undirected edges). No
  hashing: the model sees the raw graph, not a fixed fingerprint.
- `gin_model.py` — a Graph Isomorphism Network (GIN) in **pure PyTorch** (no torch-geometric needed;
  message passing uses `index_add_`). Atom vectors are updated from neighbour sums through per-layer
  MLPs, then mean-pooled to a molecule vector and read out to the endpoint.
- `train_gnn.py` — trains the GIN and compares it, on the **identical scaffold hold-out**, to a random
  forest on ECFP features. Output: `results/gnn/gnn_vs_rf.csv`.

## Running the local demonstration (CPU)

```
python src/sai_cns/gnn/train_gnn.py
```

This runs four representative endpoints (BBB, BACE1, MAO-A, A2A) on a single scaffold hold-out. CPU is
adequate for this scale. It is a fair like-for-like GIN-vs-RF test, not the full 10-fold protocol.

## Running the full study (GPU cluster)

The full 13-endpoint, scaffold-10-fold GNN study should run on the GPU cluster, where it is fast. On
the cluster:

1. Create an environment with a CUDA build of PyTorch matching the node's GPU, e.g.
   `pip install torch --index-url https://download.pytorch.org/whl/cu121` (choose the CUDA version the
   cluster provides). The code here needs **only** `torch`, `rdkit`, `numpy`, `pandas`,
   `scikit-learn`; torch-geometric is **not** required.
2. Copy `data/endpoints/`, `data/endpoints_reg/` and `src/sai_cns/` across.
3. Extend `train_gnn.py` to loop all endpoints under `GroupKFold(10)` (the single-split logic is
   already here; wrap it in the fold loop) and set `device = "cuda"`.

## Honest expectation

On measured sets of 2,000–8,000 compounds with strong ECFP features, graph networks typically **tie or
slightly trail** gradient-boosted trees and random forests; their advantage appears at much larger data
scales and with pretraining. A result of "GIN ~ RF at this scale" is therefore an expected, meaningful
finding, not a failure: it says the representation is not yet the limiting factor at this data size.
The value of the GNN track is the future path (pretraining on millions of molecules, multi-task
learning), not an immediate accuracy jump. This module exists to test that claim with our own data
rather than assume it.
