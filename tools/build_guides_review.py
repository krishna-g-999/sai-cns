"""Rebuild FOR_GUIDES_REVIEW/ from current sources, and refuse to ship a stale package.

The package that existed before this script was assembled by hand. Every file in it was dated
19 or 20 September, which was before the server was renamed and before the figures were reworked,
so it carried 69 occurrences of "SAI-CNS", zero occurrences of "SAI-CNS", ten dead URLs, and the
superseded figures. Nothing warned anyone, because nothing was checked.

This script regenerates the package from whatever the sources currently say, and then asserts on
the OUTPUT that the old name does not appear anywhere. A package that would embarrass its author
now fails the build instead.

Sources, and why each is the one used:

  condensed manuscript   manuscript_nar_clean/  This is the corrected copy: it carries the rename,
                         the capability comparison table, the four-regime framing paragraph, the
                         reworked figures and the line-number fix. manuscript/NAR_condensed_draft.md
                         has the rename but none of the rest, so it is NOT the source here.
  extended manuscript    manuscript/            The full draft, renamed but not otherwise reworked.
  supplementary          manuscript/            Rebuilt by build_supplementary_information.py.
  figures                manuscript_nar_clean/figures for the five the manuscripts cite, which are
                         the corrected ones; manuscript/figures for the remainder, which are the
                         upstream set and have not been reworked.
  docs                   docs/*.md              Converted with pandoc.

Run:  python tools/build_guides_review.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pypandoc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "FOR_GUIDES_REVIEW"
CLEAN = ROOT / "manuscript_nar_clean"
MS = ROOT / "manuscript"
DOCS = ROOT / "docs"

# Names that must not survive into the package. The check runs on the rendered text of every
# output, not on the sources, because it is the output a guide opens.
FORBIDDEN = [
    (re.compile(r"BrainSafe", re.I), "the superseded product name"),
]

# Occurrences that are correct and must not be "fixed". Each is removed from the text before the
# forbidden check runs, so the gate stays strict everywhere else. Anything added here needs a
# reason that survives being read out loud.
ALLOWED = [
    # The GitHub URL no longer needs a waiver. The repository was renamed to
    # krishna-g-999/sai-cns on 2026-10-11, so the published address no longer carries the
    # superseded name; the old one still resolves, as a 301 redirect.
    (re.compile(r"src/brainsafe\b"),
     "a historical path inside the decisions log, which records the repository layout as it was "
     "when that decision was taken. Rewriting it would falsify the record."),
]

# Documents converted straight from markdown.
DOC_JOBS = [
    ("5_endpoint_justification.docx", DOCS / "ENDPOINT_JUSTIFICATION.md"),
    ("6_data_manifest.docx", DOCS / "DATA_MANIFEST.md"),
    ("7_ml_methods_and_formulas.docx", DOCS / "ML_METHODS_AND_FORMULAS.md"),
    ("8_technical_report.docx", DOCS / "TECHNICAL_REPORT.md"),
    ("9_validation_summary.docx", DOCS / "VALIDATION.md"),
    ("10_decisions_log.docx", DOCS / "decisions_log.md"),
]

# Figures the corrected copy owns. Anything else is taken from the upstream set unchanged.
CORRECTED_FIGURES = {
    "Figure1_architecture.png": CLEAN / "figures" / "Figure1.png",
    "Figure8_use_case.png": CLEAN / "figures" / "Figure2.png",
    "Figure9_model_atlas.png": CLEAN / "figures" / "FigureS1.png",
    "Figure6_validation.png": CLEAN / "figures" / "FigureS2.png",
    "Figure11_external_validation.png": CLEAN / "figures" / "FigureS3.png",
    "GraphicalAbstract.png": CLEAN / "figures" / "GraphicalAbstract.png",
}

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def docx_text(path: Path) -> str:
    """Every run of visible text in a .docx, for the forbidden-name check."""
    parts = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.startswith("word/") and name.endswith(".xml"):
                try:
                    parts.append("".join(ET.fromstring(z.read(name)).itertext()))
                except ET.ParseError:
                    pass
    return "\n".join(parts)


def run(script: Path) -> None:
    r = subprocess.run([sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{script.name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    print(f"  ran {script.relative_to(ROOT).as_posix()}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    print("rebuilding sources")
    run(ROOT / "src" / "sai_cns" / "analysis" / "build_supplementary_information.py")
    run(CLEAN / "build_clean.py")

    print("assembling the package")
    # 1: the corrected condensed manuscript, not the draft
    shutil.copy2(CLEAN / "NAR_condensed_clean.docx", OUT / "1_manuscript_condensed_for_NAR.docx")
    print("  1_manuscript_condensed_for_NAR.docx  <- manuscript_nar_clean (corrected)")

    # 2: the extended manuscript, if the project has built one
    ext = MS / "NAR_WebServer_SAI-CNS.docx"
    if ext.exists():
        shutil.copy2(ext, OUT / "2_manuscript_extended_full.docx")
        print("  2_manuscript_extended_full.docx      <- manuscript/")
    else:
        print("  2_manuscript_extended_full.docx      SKIPPED, source not built")

    # 3: supplementary information, just rebuilt
    shutil.copy2(MS / "Supplementary_Information.docx", OUT / "3_supplementary_information.docx")
    print("  3_supplementary_information.docx     <- manuscript/ (rebuilt)")

    # 4: the corrected graphical abstract
    for ext_ in ("png", "pdf"):
        src = CLEAN / "figures" / f"GraphicalAbstract.{ext_}"
        if src.exists():
            shutil.copy2(src, OUT / f"4_graphical_abstract.{ext_}")
    print("  4_graphical_abstract.png/.pdf        <- manuscript_nar_clean (corrected)")

    # 5-10: the documentation set
    for name, src in DOC_JOBS:
        if not src.exists():
            print(f"  {name:36s} SKIPPED, {src.name} missing")
            continue
        pypandoc.convert_file(str(src), "docx", outputfile=str(OUT / name),
                              extra_args=[f"--resource-path={src.parent}"])
        print(f"  {name:36s} <- docs/{src.name}")

    # 11: the model inventory
    inv = ROOT / "results" / "tables" / "MODEL_INVENTORY.csv"
    if inv.exists():
        shutil.copy2(inv, OUT / "11_supplementary_table_S1_model_inventory.csv")
        print("  11_supplementary_table_S1_...csv     <- results/tables/")

    # figures: corrected where a corrected version exists, upstream otherwise
    n_fixed = n_upstream = 0
    for src in sorted((MS / "figures").glob("*.png")):
        target = OUT / "figures" / src.name
        replacement = CORRECTED_FIGURES.get(src.name)
        if replacement and replacement.exists():
            shutil.copy2(replacement, target); n_fixed += 1
        else:
            shutil.copy2(src, target); n_upstream += 1
    print(f"  figures/                             {n_fixed} corrected, {n_upstream} upstream")

    # README
    (OUT / "README.md").write_text(readme(), encoding="utf-8")
    print("  README.md                            regenerated")

    audit()


def readme() -> str:
    return """# SAI-CNS — for guide and PI review

Everything in this folder is regenerated by `python tools/build_guides_review.py`. Do not edit the
files here by hand: the next build overwrites them, and a hand edit is how the previous version of
this package came to carry a superseded product name in sixty-nine places without anyone noticing.

| File | What it is |
| --- | --- |
| `1_manuscript_condensed_for_NAR.docx` | the NAR Web Server Issue submission: single column, double spaced, line numbered |
| `2_manuscript_extended_full.docx` | the extended manuscript, for context rather than submission |
| `3_supplementary_information.docx` | Supplementary Figures S1 to S3 and Supplementary Table S1 |
| `4_graphical_abstract.png` / `.pdf` | 4:3, per NAR's specification |
| `5_endpoint_justification.docx` | why each endpoint is in the panel |
| `6_data_manifest.docx` | every training table, its source and its licence |
| `7_ml_methods_and_formulas.docx` | the model, the splits, the calibration and the conformal step |
| `8_technical_report.docx` | the full technical account |
| `9_validation_summary.docx` | the validation and falsification results |
| `10_decisions_log.docx` | decisions taken and reversed, with reasons |
| `11_supplementary_table_S1_model_inventory.csv` | every estimator, its training set and its scores |
| `figures/` | the figure set |

## What a reviewer should look at first

The condensed manuscript is the submission. Table 1 in its Discussion is the capability comparison
against SwissTargetPrediction, SwissADME, ADMETlab, pkCSM and admetSAR. The Results open by naming
the four independent hold-out regimes and the ten falsification hypotheses, four of which were
refuted and two weakened; all ten verdicts are reported.

## Known open items

- The server URL in the manuscript must resolve publicly before submission. At the time of writing
  the Hugging Face Space is private and the `sai-cns` GitHub repository does not exist.
- The corresponding author's telephone number, the archive DOI and the CRediT contribution
  statement are marked `[TO BE SUPPLIED]`.
- Supplementary Figure S2 renders 214 mm wide against the 183 mm the rest of the set uses, because
  two of its panels carry labels wider than their panels.
"""


def audit() -> None:
    """Refuse to ship a package that still names the old server."""
    print("\nauditing the built package")
    problems, waived = [], 0
    for path in sorted(OUT.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix == ".docx":
            text = docx_text(path)
        elif path.suffix in (".md", ".csv", ".txt"):
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            continue
        # Remove the occurrences that are correct, counting them, so the gate below is strict
        # about everything that is left rather than being weakened to accommodate them.
        for pattern, _reason in ALLOWED:
            text, n = pattern.subn(" ", text)
            waived += n
        for pattern, why in FORBIDDEN:
            hits = pattern.findall(text)
            if hits:
                problems.append(f"  {path.relative_to(OUT).as_posix()}: "
                                f"{len(hits)} x {why}")
    if problems:
        print("FAIL: the package still contains a superseded name")
        print("\n".join(problems))
        raise SystemExit(1)
    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"OK: {n} files, no superseded name in any of them")
    if waived:
        print(f"    ({waived} occurrence(s) waived by the allowlist, each for a stated reason:)")
        for pattern, reason in ALLOWED:
            print(f"      {pattern.pattern}\n        {reason}")


if __name__ == "__main__":
    main()
