"""The one definition of the applicability-domain fingerprint.

This module exists because the definition was previously written out in twelve places. Every copy
agreed, which is exactly why nobody noticed that all twelve were stereochemistry-blind: a Morgan
fingerprint built without `includeChirality` cannot tell a molecule from its own stereoisomer.

THE DEFECT. In external testing, schisandrin B and ajmaline each returned a maximum similarity of
1.00 and were reported "In domain, high confidence". Neither structure is in the reference set. A
different stereoisomer of each is. For ajmaline, a polycyclic indole alkaloid whose pharmacology is
stereochemistry-dependent, the server told the user it had seen that exact chemistry when it had
not.

THE FIX IS APPLIED LOCALLY AND IS NOT YET PUBLISHED. INCLUDE_CHIRALITY is True and the rebuilt
reference is declared in models_manifest.json at version 1.3, which carries it as a per-file delta
rather than inside the base archive; the base archive is still v1.1. Setting it and rebuilding
corrected the reported similarities: schisandrin B falls from 1.0000 to 0.7955 and ajmaline from
1.0000 to 0.8627, while coptisine, whose exact structure is in the reference and which has no
stereocentres, correctly stays at 1.0000. All three were reverified against the rebuilt reference
on this machine.

UNTIL THE DELTA IS PUBLISHED A FRESH DEPLOYMENT HALTS RATHER THAN SERVING EITHER VALUE, and the
corrected ajmaline figure is therefore on no deployed server yet. ensure_models extracts the v1.1
base, finds ad_reference.pkl, ad_per_endpoint.pkl and readacross_index.pkl stale against the
manifest, cannot fetch them because their URLs are not live, and the final full verification
rejects the tree; load_models then calls st.error and st.stop. That is the integrity layer failing
safe rather than answering from a reference built with a different fingerprint. Publish with
tools/publish_model_archive.py --delta.

The reference was rebuilt over the SAME 158,890 compounds, with
build_ad_reference.py --same-set, so the only thing that changed is the fingerprint. Measured on
1,000 unseen background compounds the median change in maximum similarity is 0.0000, 5.9 per cent
move by more than 0.05 and 3.9 per cent change distance stratum. Two published figures moved with
it and were re-measured rather than left: the background's median similarity, 0.417 to 0.400, and
the fraction below Tanimoto 0.40, 43.3 to 49.4 per cent. The aggregate specificity is unaffected,
since whether the server fires does not depend on the reference.

WHY IT WAS OFF UNTIL NOW. ad_reference.pkl is one of the 252 files covered by
models_manifest.json, and model_fetch verifies every one of them against the published archive on
start-up. Changing the reference without rebuilding and republishing the archive puts the two out
of step, and the integrity layer then restores the published file, silently undoing the change.
That is the integrity layer working correctly. Switching it on therefore required, in order:

    1. INCLUDE_CHIRALITY = True here
    2. python src/sai_cns/build_ad_reference.py --same-set
    3. update models_manifest.json by hand: bump "version", and give every changed file its new
       sha256, bytes and a "url" under files/v<version>/. Do NOT run package_models.py for this.
       That script writes a complete new archive and names it in "archive", which would replace the
       published base and discard the per-file delta model_fetch now relies on. Nothing in the tree
       writes "base_archive_version" or files[].url; both are maintained by hand, which is a gap
       worth closing before anyone else changes a model file.
    4. python tools/publish_model_archive.py --delta       (ships only the changed files, in blocks)
    5. confirm model_fetch.ensure_models() no longer replaces the file

Anything that queries the reference must use generator() here rather than a private copy. Two
scripts still carried their own stereochemistry-blind generator when this was switched on,
evaluation/applicability_measures.py and evaluation/validate_inversion.py, and would have compared
a stereo-blind query against a chirality-aware reference, which returns similarities that are
wrong rather than missing. Both now import this module.

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
INCLUDE_CHIRALITY = True

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


def load_reference(path=None):
    """(smiles, fingerprints) from models_rf/ad_reference.pkl, with the spec checked.

    Every consumer of the reference must come through here. The reference is pickled as
    (smiles, fingerprints, spec); references written before the spec field existed are a
    two-element tuple and were necessarily built stereochemistry-blind, which is a known state
    rather than an unknown one and is read as LEGACY_SPEC.

    This exists because seven callers unpacked the pickle as a two-element tuple directly. That
    was harmless while the shipped reference predated the spec field, and became a ValueError the
    moment one was rebuilt, taking three of the four threshold steps and the binder trainers with
    it. A spec mismatch is raised rather than tolerated for the reason given at the top of this
    module: querying a reference with a different fingerprint returns similarities that are wrong
    rather than missing.
    """
    import pickle
    from pathlib import Path as _Path
    path = _Path(path) if path is not None else (
        _Path(__file__).resolve().parents[2] / "models_rf" / "ad_reference.pkl")
    with path.open("rb") as fh:
        obj = pickle.load(fh)
    smiles, fps = obj[0], obj[1]
    spec = obj[2] if len(obj) == 3 else LEGACY_SPEC
    if spec != SPEC:
        raise ValueError(
            f"{path.name} was built with fingerprint '{spec}' but this process computes '{SPEC}'. "
            f"Rebuild it with src/sai_cns/build_ad_reference.py and republish the model archive; "
            f"see the module docstring in src/sai_cns/ad_fingerprint.py.")
    return smiles, fps
