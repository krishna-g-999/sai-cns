"""Reversible integer encodings for the non-numeric metadata fields.

Model features are already numeric (see featurize.py). This module handles the *metadata* columns
that are text - endpoint name, modelling task, and data source - which the review asked to be
represented numerically "without conflicting". Each distinct value is mapped to a unique integer
code; the mapping is deterministic (sorted, so it is stable across runs) and reversible, and is
written to a JSON so the encoding is fully documented rather than implicit.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Fixed, documented code books. Sorted assignment guarantees no two values share a code.
ENDPOINT_CODES = {
    "BBB": 1, "AChE": 2, "BChE": 3, "BACE1": 4, "GSK3B": 5, "MAO_A": 6, "MAO_B": 7,
    "hERG": 8, "D2": 9, "A2A": 10, "HT2A": 11, "SERT": 12, "antioxidant_DPPH": 13,
}
TASK_CODES = {"classification": 0, "regression": 1}
SOURCE_CODES = {"ChEMBL_37": 1, "B3DB": 2, "ChEMBL_DPPH": 3, "DrugBank": 4, "COCONUT": 5,
                "PubChem_Natural": 6, "ChEMBL34": 7, "BBB_FDA_curated": 8}


def encode(value: str, codebook: dict[str, int]) -> int:
    """Return the integer code for a categorical value, raising if it is unknown (no silent 0)."""
    if value not in codebook:
        raise KeyError(f"'{value}' is not in the code book {sorted(codebook)}")
    return codebook[value]


def write_codebooks(path: Path | None = None) -> Path:
    """Persist all code books to JSON so the numeric encoding is auditable."""
    path = path or (ROOT / "docs" / "categorical_encodings.json")
    path.write_text(json.dumps(
        {"endpoint": ENDPOINT_CODES, "task": TASK_CODES, "source": SOURCE_CODES},
        indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print("wrote", write_codebooks())
