# ADME / exposure module (ladder B)

The target-engagement panel answers "does this molecule bind the target and cross the barrier". It
does **not** answer how much free drug actually reaches the target in the brain at an achievable dose.
That is what this module adds: measured ADME / exposure endpoints that, together with the BBB model,
approximate **K_p,uu** — the unbound brain-to-plasma ratio, the quantity that connects molecular
binding to a real brain effect.

Kept separate from the target models (`data/adme/`, `models_rf/adme/`) so the two never mix.

## Endpoints and why each matters biologically

| Endpoint | Task | Biological role in brain exposure |
|---|---|---|
| Aqueous solubility (AqSolDB) | regression | a compound must dissolve to be absorbed at all |
| Lipophilicity, logD7.4 | regression | governs membrane partitioning and passive permeability |
| Caco-2 permeability | regression | passive permeability, a proxy for crossing the BBB by diffusion |
| P-glycoprotein inhibition | classification | P-gp actively pumps drug out of the brain. Note: this dataset (Broccatelli) is P-gp **inhibition**, a related liability flag, **not** the P-gp **substrate** call that most directly predicts efflux; a substrate dataset is a separate, planned endpoint. |
| Plasma-protein binding | regression | only the **unbound** fraction is free to cross and to act |
| Hepatocyte clearance | regression | how fast the body removes the drug, i.e. how long exposure lasts |

## Files

- `fetch_adme.py` — download and standardise the measured datasets (TDC / Harvard Dataverse and
  MoleculeNet). Output: `data/adme/*.csv`, `data/adme/SOURCE.md`.
- `train_adme.py` — random forest per endpoint, random and scaffold 10-fold, identical features and
  protocol to the target panel. Output: `results/tables/adme_cv_summary.csv`, `models_rf/adme/`.

## How this reaches K_p,uu

No single public dataset gives K_p,uu at scale, so it is approximated: high passive permeability (Caco-2),
not a P-gp substrate, and a workable free fraction (plasma-protein binding) together predict that a
BBB-positive compound will achieve meaningful **free** brain concentration. The combined readout is the
planned next step once the individual models are validated.
