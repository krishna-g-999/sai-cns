---
title: SAI-CNS
emoji: 🧠
colorFrom: yellow
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: BBB-gated prediction of small-molecule action in the brain
---

# SAI-CNS

Predicts, from a SMILES string or a compound name, whether a small molecule reaches the human brain,
what it engages there, what conditions that mechanism touches, and what would stop it being a drug.

Every endpoint is trained on measured experimental values, never on qualitative annotation. Every
prediction carries a calibrated probability, a conformal interval and the distance to the nearest
measured analogue, so a user can tell interpolation from extrapolation. Target scores are admitted
only in proportion to predicted blood-brain-barrier penetration, so potency at a target the compound
cannot reach contributes nothing.

The server reports where it does not work. Five endpoints were trained, tested and withdrawn, and
they are named. Four of nine falsification hypotheses were refuted and are published alongside the
five that survived. Compounds outside the applicability domain are flagged rather than guessed at,
and every prediction carries the recall the panel actually achieves at that distance from its
training chemistry, which falls from 0.86 for close analogues to 0.16 for a genuinely new scaffold.
A silent endpoint is therefore reported with the weight it deserves rather than as evidence of
inactivity.

- **Source, validation artefacts and reproduction:** https://github.com/krishna-g-999/brainsafe-ai
- **Licence:** MIT for the code. Underlying data retain their sources' licences (ChEMBL CC BY-SA 3.0,
  BindingDB, B3DB, Therapeutics Data Commons, MoleculeNet; pathway annotations from KEGG, Reactome
  and IUPHAR/BPS, whose terms restrict some redistribution).

Research preview, pending peer review. Not for medical, diagnostic or treatment use.
