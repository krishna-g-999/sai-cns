"""Assemble the numbered submission package an institutional review needs.

The working repository is organised for building things: source beside caches, current artefacts
beside archives, and 135 scripts of which perhaps forty produced anything that is quoted. A reviewer
asked to approve an external submission needs the opposite arrangement, which is every claim placed
next to the file that supports it, in an order that can be walked from start to finish.

This copies rather than moves. Restructuring the repository itself would break the freshness graph,
the test suite and the deployed Space, all of which resolve paths that have been stable for months,
and would trade a working system for a tidier listing.

Numbering runs in the order a reviewer should read: what is being claimed, then the evidence, then
the code that produced the evidence, then the data underneath it. Directories that hold nothing are
not created, so an empty folder in the output means something failed rather than that nothing was
expected.

Run:  python tools/build_submission_package.py --out submission_package
      python tools/build_submission_package.py --out submission_package --with-models
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# section -> (source, destination subpath, glob or None for a single file)
LAYOUT: list[tuple[str, list[tuple[str, str, str | None]]]] = [
    # Exactly what goes to the journal: nothing here needs the reader to have opened any other
    # folder first. Everything else in this package is institutional/PI-review material the
    # journal itself does not ask for.
    ("00_SUBMIT_TO_NAR", [
        ("manuscript/NAR_WebServer_SAI-CNS_condensed.docx", "1_manuscript.docx", None),
        ("manuscript/figures/GraphicalAbstract.png", "2_graphical_abstract.png", None),
        ("manuscript/figures/GraphicalAbstract.pdf", "2_graphical_abstract.pdf", None),
        ("manuscript/Supplementary_Information.docx", "3_supplementary_information.docx", None),
        ("results/tables/MODEL_INVENTORY.csv", "4_supplementary_table_S1_model_inventory.csv",
         None),
    ]),
    ("01_PROPOSAL", [
        ("manuscript/NAR_proposal_onepage.md", "proposal_one_page.md", None),
        ("manuscript/NAR_proposal_onepage.docx", "proposal_one_page.docx", None),
    ]),
    ("02_MANUSCRIPT", [
        ("manuscript/NAR_WebServer_SAI-CNS_condensed.docx", "01_manuscript_for_NAR.docx", None),
        ("manuscript/NAR_condensed_draft.md", "01_manuscript_for_NAR.md", None),
        ("manuscript/Supplementary_Information.docx", "01b_supplementary_information.docx", None),
        ("manuscript/NAR_WebServer_SAI-CNS.docx", "02_manuscript_extended.docx", None),
        ("manuscript/NAR_WebServer_SAI-CNS_built.md", "manuscript.md", None),
        ("manuscript/NAR_WebServer_SAI-CNS_draft.md", "manuscript_source.md", None),
        ("manuscript/references.md", "references.md", None),
        ("manuscript/references_verified.json", "references_verification.json", None),
        ("manuscript/references_links.json", "references_pubmed_links.json", None),
        ("manuscript/tables_generated.md", "supplementary_tables.md", None),
    ]),
    ("03_FIGURES", [
        ("manuscript/figures", ".", "*.png"),
        ("manuscript/figures", ".", "*.pdf"),
    ]),
    ("04_TECHNICAL_REPORT", [
        ("docs/TECHNICAL_REPORT.docx", "technical_report.docx", None),
        ("docs/TECHNICAL_REPORT.md", "technical_report.md", None),
        ("docs/METHODS.docx", "methods.docx", None),
        ("docs/METHODS.md", "methods.md", None),
        ("docs/VALIDATION.docx", "validation_summary.docx", None),
        ("docs/VALIDATION.md", "validation_summary.md", None),
        ("docs/decisions_log.docx", "decisions_log.docx", None),
        ("docs/decisions_log.md", "decisions_log.md", None),
        ("docs/ENDPOINT_JUSTIFICATION.docx", "endpoint_justification.docx", None),
        ("docs/ENDPOINT_JUSTIFICATION.md", "endpoint_justification.md", None),
        ("docs/DATA_MANIFEST.docx", "data_manifest.docx", None),
        ("docs/DATA_MANIFEST.md", "data_manifest.md", None),
        ("docs/ML_METHODS_AND_FORMULAS.docx", "ml_methods_and_formulas.docx", None),
        ("docs/ML_METHODS_AND_FORMULAS.md", "ml_methods_and_formulas.md", None),
        ("results/tables/decision_tree_example_full.txt", "decision_tree_example_full.txt", None),
    ]),
    ("05_CODE/01_data_acquisition", [("src/sai_cns/data", ".", "*.py")]),
    ("05_CODE/02_featurisation", [("src/sai_cns/features", ".", "*.py")]),
    ("05_CODE/03_model_training", [("src/sai_cns/models", ".", "*.py")]),
    ("05_CODE/04_validation", [("src/sai_cns/evaluation", ".", "*.py")]),
    ("05_CODE/05_falsification", [("inversion", ".", "*.py")]),
    ("05_CODE/06_figures", [("src/sai_cns/figures", ".", "*.py")]),
    ("05_CODE/07_reporting", [("src/sai_cns/analysis", ".", "*.py")]),
    ("05_CODE/08_application", [
        ("app.py", "app.py", None),
        ("api.py", "api.py", None),
        ("model_fetch.py", "model_fetch.py", None),
        ("src/sai_cns/panel.py", "panel_registry.py", None),
    ]),
    ("05_CODE/09_tests", [("tests", ".", "*.py")]),
    ("05_CODE/10_integrity_tooling", [
        ("tools/check_freshness.py", "check_freshness.py", None),
        ("tools/compress_models.py", "compress_models.py", None),
    ]),
    ("06_TRAINING_DATA", [
        ("data/endpoints", "endpoints", "*.csv"),
        ("data/endpoints_reg", "endpoints_regression", "*.csv"),
        ("data/adme", "adme", "*.csv"),
        ("data/raw/measured_endpoints_SOURCE.md", "SOURCE_provenance.md", None),
        ("results/tables/master_training_usage.csv", "master_training_usage.csv", None),
    ]),
    ("07_MODELS", [
        ("models_manifest.json", "models_manifest_with_checksums.json", None),
        ("models_rf/binder_modes.json", "binder_panel_registry.json", None),
        ("models_rf/endpoint_context.json", "endpoint_base_rates.json", None),
        ("results/tables/MODEL_INVENTORY.csv", "model_inventory.csv", None),
        ("models_rf", "per_model_metadata", "*_meta.json"),
    ]),
    ("08_VALIDATION_RESULTS", [("results/tables", ".", "*.csv", {
        "master_feature_vectors.csv",   # gated behind --with-vectors, ~0.8 GB
        "master_training_usage.csv",    # already placed explicitly in 06_TRAINING_DATA
    })]),
    ("09_FALSIFICATION_SUITE", [
        ("inversion/results", "results", "*.csv"),
        ("inversion/REPORT.md", "falsification_report.md", None),
        ("inversion/results/GRAPH_FINGERPRINT.json", "graph_fingerprint.json", None),
    ]),
    ("10_REPRODUCIBILITY", [
        ("requirements.txt", "requirements_training.txt", None),
        ("deploy/huggingface/requirements.txt", "requirements_server.txt", None),
        ("deploy/huggingface/Dockerfile", "Dockerfile_server", None),
        ("deploy/huggingface/DEPLOY.md", "deployment_notes.md", None),
        ("Makefile", "Makefile", None),
        ("CITATION.cff", "CITATION.cff", None),
        ("LICENSE", "LICENSE", None),
    ]),
]


def copy(src: Path, dst: Path, pattern: str | None,
         exclude: set[str] | None = None) -> tuple[int, int]:
    """Copy one file, or every match of a pattern (skipping any name in `exclude`).

    Returns (files, bytes).
    """
    n = size = 0
    if pattern is None:
        if not src.exists():
            return 0, 0
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return 1, src.stat().st_size
    if not src.is_dir():
        return 0, 0
    for p in sorted(src.rglob(pattern)):
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        if exclude and p.name in exclude:
            continue
        rel = p.relative_to(src)
        out = (dst / rel) if dst.name != "." else (dst.parent / rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
        size += p.stat().st_size
    return n, size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-models", action="store_true",
                    help="include the fitted estimators, about 0.85 GB")
    ap.add_argument("--with-vectors", action="store_true",
                    help="include the full master feature-vector matrix, about 0.8 GB")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    print(f"assembling into {out}\n")

    summary, total_files, total_bytes, missing = [], 0, 0, []
    for section, items in LAYOUT:
        sec_files = sec_bytes = 0
        for item in items:
            rel_src, rel_dst, pattern, *rest = item
            exclude = rest[0] if rest else None
            src = ROOT / rel_src
            dst = out / section / rel_dst
            n, b = copy(src, dst, pattern, exclude)
            if n == 0:
                missing.append(f"{section}: {rel_src}")
            sec_files += n
            sec_bytes += b
        if sec_files:
            summary.append((section, sec_files, sec_bytes))
            total_files += sec_files
            total_bytes += sec_bytes
            print(f"  {sec_files:>4} files  {sec_bytes/1e6:>8.2f} MB  {section}")

    if args.with_models:
        n, b = copy(ROOT / "models_rf", out / "07_MODELS" / "fitted_estimators", "*.joblib")
        n2, b2 = copy(ROOT / "models_rf", out / "07_MODELS" / "fitted_estimators", "*.pkl")
        print(f"  {n + n2:>4} files  {(b + b2)/1e6:>8.2f} MB  07_MODELS/fitted_estimators")
        total_files += n + n2
        total_bytes += b + b2
        summary.append(("07_MODELS/fitted_estimators", n + n2, b + b2))

    if args.with_vectors:
        n, b = copy(ROOT / "results" / "tables" / "master_feature_vectors.csv",
                   out / "06_TRAINING_DATA" / "master_feature_vectors.csv", None)
        print(f"  {n:>4} files  {b/1e6:>8.2f} MB  06_TRAINING_DATA/master_feature_vectors.csv")
        total_files += n
        total_bytes += b
        summary.append(("06_TRAINING_DATA/master_feature_vectors.csv", n, b))

    (out / "PACKAGE_CONTENTS.json").write_text(json.dumps({
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_files": total_files,
        "total_megabytes": round(total_bytes / 1e6, 2),
        "fitted_estimators_included": bool(args.with_models),
        "master_feature_vectors_included": bool(args.with_vectors),
        "sections": [{"section": s, "files": f, "megabytes": round(b / 1e6, 2)}
                     for s, f, b in summary],
        "expected_but_absent": missing,
    }, indent=2), encoding="utf-8")

    (out / "README.md").write_text(f"""# SAI-CNS: submission package

Built {datetime.now().strftime("%Y-%m-%d %H:%M")} by `tools/build_submission_package.py`.
Everything here is a copy; the source of truth is the working repository
(https://github.com/krishna-g-999/sai-cns), and every file below regenerates from it with the
command named in the technical documents. Nothing in this package is hand-edited.

## Start here

**`00_SUBMIT_TO_NAR/`** is exactly what goes to the journal: the manuscript, the graphical abstract
as its own file, the Supplementary Information (the three validation figures cited in the manuscript
as Supplementary Figures S1 to S3), and Supplementary Table S1 (the full model inventory, as CSV).
Nothing else in this package is asked for by NAR; the remaining sections are the institutional and
PI-review record behind that manuscript.

## Everything else, for PI and institutional review

| Folder | What it is |
|---|---|
| `01_PROPOSAL/` | The original one-page project proposal. |
| `02_MANUSCRIPT/` | Every manuscript variant: the NAR-length submission copy, the extended full-length draft, the Supplementary Information, and the raw Markdown and reference data behind both. |
| `03_FIGURES/` | Every figure and the graphical abstract, PNG and PDF. |
| `04_TECHNICAL_REPORT/` | The full methods and results narrative, the endpoint justification (why these 75 estimators, why 5 were withdrawn, and the 1,688-target survey behind that), the data manifest, the ML methods and formulas document (every calculation the pipeline makes, with a live worked example), the validation summary, and the dated decisions log. Word and Markdown versions of each, plus the complete text export of one decision tree. |
| `05_CODE/` | Every script that produced a number or figure quoted anywhere above, organised by pipeline stage. |
| `06_TRAINING_DATA/` | The measured endpoint tables every model is trained and tested on, as CSV; `master_training_usage.csv` maps every (endpoint, compound) pair actually used to train or test a model, joinable to the numeric feature vectors in `master_feature_vectors.csv` (with `--with-vectors`, ~0.8 GB: the exact 1,036-column input every model receives, one row per distinct compound). |
| `07_MODELS/` | The model registry, per-model metadata, and (with `--with-models`) the fitted estimators themselves. |
| `08_VALIDATION_RESULTS/` | Every results table cited in the manuscript or technical report, as CSV. |
| `09_FALSIFICATION_SUITE/` | The ten-hypothesis falsification suite: evidence, verdicts, and the narrative report. |
| `10_REPRODUCIBILITY/` | Requirements, deployment configuration, licence, and citation metadata. |

`PACKAGE_CONTENTS.json` lists exactly what was copied and its size, section by section, and names
anything expected that was not found.
""", encoding="utf-8")

    print(f"\n  {total_files} files, {total_bytes/1e6:.1f} MB")
    if missing:
        print(f"\n  {len(missing)} expected item(s) not found:")
        for m in missing:
            print("   ", m)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
