"""The one definition of the applicability-domain fingerprint.

This module exists because the definition was previously written out in twelve places. Every copy
agreed, which is exactly why nobody noticed that all twelve were stereochemistry-blind: a Morgan
fingerprint built without `includeChirality` cannot tell a molecule from its own stereoisomer.

THE DEFECT. In external testing, schisandrin B and ajmaline each returned a maximum similarity of
1.00 and were reported "In domain, high confidence". Neither structure is in the reference set. A
different stereoisomer of each is. For ajmaline, a polycyclic indole alkaloid whose pharmacology is
stereochemistry-dependent, the server told the user it had seen that exact chemistry when it had
not.

THE FIX IS MEASURED AND READY, AND IS NOT YET SWITCHED ON. Setting INCLUDE_CHIRALITY = True and
rebuilding the reference corrects it: schisandrin B falls from 1.0000 to 0.7955 and ajmaline from
1.0000 to 0.8627, while coptisine, whose exact structure really is in the reference and which has
no stereocentres, correctly stays at 1.0000. Measured over 400 held-out queries against all 158,890
reference compounds, the median change in maximum similarity is 0.0000, 1.2 per cent of queries
move by more than 0.05 and 0.5 per cent change applicability-domain tier, so the measured
recall-versus-distance curve still applies.

WHY IT IS OFF. ad_reference.pkl is one of the 252 files covered by models_manifest.json, and
model_fetch verifies every one of them against the published archive on start-up. Changing the
reference without rebuilding and republishing that 0.87 GB archive puts the two out of step, and
the integrity layer then restores the published file, silently undoing the change. That is the
integrity layer working correctly. Switching the fix on therefore requires, in order:

    1. INCLUDE_CHIRALITY = True here
    2. python src/sai_cns/build_ad_reference.py          (rebuild over the same compound set)
    3. update models_manifest.json: the ad_reference.pkl entry, then archive_bytes/archive_sha256
    4. rebuild and re-upload the archive to the URL in models_manifest.json
    5. confirm model_fetch.ensure_models() no longer replaces the file

Step 4 needs a current Azure sign-in, which had expired when this was prepared.

A SEPARATE FINDING, deliberately not acted on. Rebuilding the reference from the current training
CSVs yields 185,789 compounds against the 158,890 it holds, so it is short of present training
chemistry by about 15 per cent. That is not a free improvement: enlarging the reference raises
median similarity from 0.7368 to 0.7550 and moves 14.5 per cent of queries by more than 0.05, while
expected_recall() maps similarity to recall using bands derived against the smaller set. The server
would then report MORE recall than the evidence supports, which is the wrong direction to err. The
shortfall makes the domain conservative, which is the safe direction. Correcting it properly means
re-deriving the recall bands and every published applicability-domain figure against the complete
set, and belongs in the follow-up rather than days before submission.
"""
from __future__ import annotations

from rdkit.Chem import rdFingerprintGenerator

RADIUS = 2
N_BITS = 2048

#: See the module docstring before changing this. It cannot be flipped on its own.
INCLUDE_CHIRALITY = False

#: Written into models_rf/ad_reference.pkl and checked on load.
SPEC = f"morgan-r{RADIUS}-{N_BITS}b-chirality{int(INCLUDE_CHIRALITY)}"

#: What a reference pickled before the spec existed was necessarily built with.
LEGACY_SPEC = f"morgan-r{RADIUS}-{N_BITS}b-chirality0"


def generator():
    """The applicability-domain fingerprint generator. Use this, never a local copy."""
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, fpSize=N_BITS, includeChirality=INCLUDE_CHIRALITY)


def fingerprint(mol):
    """Fingerprint one RDKit molecule, or None if it is None."""
    return None if mol is None else generator().GetFingerprint(mol)
