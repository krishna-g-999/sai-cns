"""Convert the supplementary technical documents to Word, in the same style as the manuscript.

These are the companion documents a PI or reviewer reads alongside the manuscript itself:
DATA_MANIFEST.md, ENDPOINT_JUSTIFICATION.md, METHODS.md, VALIDATION.md and decisions_log.md. They
previously existed only as Markdown; converting them with the same reference template used for the
manuscript (docs/nar_manuscript_reference.docx: Times New Roman, single-spaced, continuous line
numbers, no table of contents) means the whole packet sent out reads as one document set rather than
as a manuscript in one style and its evidence in another.

TECHNICAL_REPORT.md is not included here: it has its own converter
(src/sai_cns/analysis/technical_report_docx.py) because it embeds rendered flowcharts.

Run:  python tools/build_supplementary_docx.py
"""
from __future__ import annotations

from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DOCX = ROOT / "docs" / "nar_manuscript_reference.docx"

DOCS = [
    ROOT / "docs" / "DATA_MANIFEST.md",
    ROOT / "docs" / "ENDPOINT_JUSTIFICATION.md",
    ROOT / "docs" / "METHODS.md",
    ROOT / "docs" / "VALIDATION.md",
    ROOT / "docs" / "decisions_log.md",
    ROOT / "docs" / "ML_METHODS_AND_FORMULAS.md",
]


def main() -> None:
    if not REFERENCE_DOCX.exists():
        raise SystemExit(f"{REFERENCE_DOCX} not found")
    for md in DOCS:
        if not md.exists():
            print(f"  skip (not found): {md.relative_to(ROOT)}")
            continue
        out = md.with_suffix(".docx")
        pypandoc.convert_file(str(md), "docx", outputfile=str(out),
                              extra_args=[f"--resource-path={md.parent}",
                                          f"--reference-doc={REFERENCE_DOCX}"])
        print(f"  wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
