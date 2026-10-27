"""Assemble everything the supervisors need into one folder, and verify it before writing it.

WHY A SCRIPT RATHER THAN A COPY. The last package sent for review was assembled by hand on
23 September and was still being treated as current a month later, by which time every calibration
figure in it was wrong, the screening sensitivities were the inflated ones, and the server URL it
quoted returned 503. Nothing warned anyone, because nothing checked. This rebuilds the folder from
the live sources every time and refuses to write it if the numbers do not reconcile.

The guard is the point. check_manuscript_numbers.py reconciles 51 quantities against 33 documents;
if any of them disagrees with its artefact, this script stops rather than producing a folder that
looks finished.

    python tools/build_supervisor_package.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "FOR_GUIDES_REVIEW" / f"TO_SEND_{date.today().isoformat()}"
REVIEW = ROOT / "FOR_GUIDES_REVIEW"
CLEAN = ROOT / "manuscript_nar_clean"

# (destination subfolder, source path, destination name). Order is the reading order.
ITEMS: list[tuple[str, Path, str]] = [
    ("01_read_first", REVIEW / "17_email_to_supervisors_2026-10-26.md", "email_to_send.md"),
    ("01_read_first", ROOT / "deploy" / "HOSTING_REQUEST.md", "hosting_request_for_IT.md"),

    ("02_for_the_journal", CLEAN / "NAR_proposal_clean.docx", "NAR_one_page_proposal.docx"),
    ("02_for_the_journal", CLEAN / "NAR_proposal_clean.pdf", "NAR_one_page_proposal.pdf"),
    ("02_for_the_journal", CLEAN / "NAR_cover_letter.docx", "NAR_cover_letter.docx"),
    ("02_for_the_journal", REVIEW / "16_NAR_submission_package_2026-10-26.md",
     "what_NAR_requires_and_our_status.md"),

    ("03_manuscripts", CLEAN / "NAR_final.docx", "manuscript_NAR.docx"),
    ("03_manuscripts", CLEAN / "JCheminf_final.docx", "manuscript_JCheminf.docx"),
    ("03_manuscripts", REVIEW / "3_supplementary_information.docx", "supplementary_information.docx"),
    ("03_manuscripts", CLEAN / "figures" / "GraphicalAbstract.png", "graphical_abstract.png"),

    ("04_how_it_was_built", REVIEW / "9_validation_summary.docx", "validation_summary.docx"),
    ("04_how_it_was_built", REVIEW / "6_data_manifest.docx", "data_manifest.docx"),
    ("04_how_it_was_built", REVIEW / "5_endpoint_justification.docx", "endpoint_justification.docx"),
    ("04_how_it_was_built", REVIEW / "7_ml_methods_and_formulas.docx", "ml_methods_and_formulas.docx"),
    ("04_how_it_was_built", REVIEW / "8_technical_report.docx", "technical_report.docx"),
    ("04_how_it_was_built", REVIEW / "12_technical_briefing_2026-10-25.md", "technical_briefing.md"),
    ("04_how_it_was_built", REVIEW / "13_decision_record_2026-10-25.md",
     "decision_record_and_benchmarks.md"),
]


def run(cmd: list[str], label: str) -> str:
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {label}: FAILED\n{r.stdout[-1500:]}\n{r.stderr[-800:]}")
        raise SystemExit(1)
    return r.stdout


def main() -> int:
    print("verifying before assembling")
    out = run(["src/sai_cns/analysis/check_manuscript_numbers.py"], "numeric check")
    if "DIFFERS, a document states a value the artefact does not support: 0" not in out:
        print("  numbers do not reconcile; refusing to build the package")
        for line in out.splitlines():
            if "DIFFERS" in line or "ABSENT" in line:
                print("   ", line)
        return 1
    checked = next((l for l in out.splitlines() if "quantities checked" in l), "").strip()
    print(f"  {checked}, no disagreements")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    missing, copied = [], 0
    for sub, src, name in ITEMS:
        dest = OUT / sub
        dest.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            missing.append(src.relative_to(ROOT).as_posix())
            continue
        shutil.copy2(src, dest / name)
        copied += 1

    # the thesis, every chapter, in reading order
    thesis_out = OUT / "05_thesis"
    thesis_out.mkdir(parents=True, exist_ok=True)
    chapters = sorted((ROOT / "thesis").glob("*.md"))
    for c in chapters:
        shutil.copy2(c, thesis_out / c.name)
    copied += len(chapters)

    # manuscript figures
    fig_out = OUT / "06_figures"
    fig_out.mkdir(parents=True, exist_ok=True)
    figs = sorted((CLEAN / "figures").glob("*.png"))
    for f in figs:
        shutil.copy2(f, fig_out / f.name)
    copied += len(figs)

    (OUT / "00_INDEX.md").write_text(index_text(len(chapters), len(figs), checked), encoding="utf-8")

    converted = to_word(OUT)
    print(f"  converted {converted} markdown file(s) to .docx")

    print(f"\nwrote {OUT.relative_to(ROOT).as_posix()}")
    print(f"  {copied} file(s) across 6 folders, plus the index")
    if missing:
        print(f"  {len(missing)} source(s) absent and skipped:")
        for m in missing:
            print(f"    {m}")
    return 0


def to_word(folder: Path) -> int:
    """Render every markdown file in the package as .docx and drop the markdown.

    The supervisors read these in Word, and a folder mixing .md and .docx makes the reader decide
    which copy is authoritative. Pandoc is already a dependency of the manuscript build, so this
    adds nothing new. The markdown stays in the repository as the editable source; only the copy
    inside the sent package is converted.
    """
    try:
        import pypandoc
    except ImportError:
        print("  pypandoc unavailable; markdown left as-is")
        return 0
    n = 0
    for md in sorted(folder.rglob("*.md")):
        docx = md.with_suffix(".docx")
        try:
            pypandoc.convert_file(str(md), "docx", outputfile=str(docx))
        except Exception as e:
            print(f"    {md.name}: conversion failed, markdown kept ({type(e).__name__})")
            continue
        md.unlink()
        n += 1
    return n


def index_text(n_chapters: int, n_figs: int, checked: str) -> str:
    return f"""# SAI-CNS: everything for review

Assembled {date.today().isoformat()} by `tools/build_supervisor_package.py`, which rebuilds this
folder from the live sources and refuses to write it unless the numbers reconcile. At the time of
writing: {checked}, with no disagreement between any document and the artefact behind it.

## Read in this order

**01_read_first.** `email_to_send.docx` is the covering note, written to be sent as it stands.
`hosting_request_for_IT.docx` is the request for an Institute subdomain, written so that it can be
forwarded to the IT department unchanged. These two carry the ask.

**02_for_the_journal.** The one-page proposal that goes to the Nucleic Acids Research Web Server
Issue, the cover letter that follows it at manuscript stage, and a note setting out exactly what the
journal requires and where we stand against each requirement.

**03_manuscripts.** The NAR manuscript, the longer Journal of Cheminformatics version, the
supplementary information and the graphical abstract.

**04_how_it_was_built.** The validation summary and data manifest are the two to read if time is
short. The endpoint justification explains why each of the 54 targets is in the panel; the methods
document writes out the mathematics; the technical report is the long form. The technical briefing
and decision record are written for a reader who wants to know what was tried, what failed, and why
each choice was made.

**05_thesis.** All {n_chapters} chapters.

**06_figures.** The {n_figs} manuscript figures.

## What is not settled

The server runs but has no permanent public address yet, which the journal requires before a
proposal can be sent. That is the subject of the hosting request.

Four items in the manuscripts are still open because they cannot be filled yet: the public server
URL, the archive deposit DOI, a CRediT author contribution statement, and a funding statement.
"""


if __name__ == "__main__":
    raise SystemExit(main())
