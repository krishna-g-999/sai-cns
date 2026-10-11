"""Build the manuscript .docx from the markdown source, with the result figures embedded.

Single source of truth: the manuscript is written and maintained in
`manuscript/BS_MANUSCRIPT_FINAL.md`; this script converts it to .docx with pandoc and appends a
Figures section that embeds the generated result figures. Regenerate any time the markdown or the
figures change, so the .docx never drifts from the source again.

Run:  python src/sai_cns/build_manuscript_docx.py
"""
from __future__ import annotations

from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parents[2]
MD = ROOT / "manuscript" / "BS_MANUSCRIPT_FINAL.md"
FIG = ROOT / "results" / "figures"
OUT = ROOT / "manuscript" / "SAI-CNS_AI_Manuscript.docx"

# Figures to embed, in reading order, with captions.
FIGURES = [
    ("fig_rf_classification_auroc.png",
     "Random-forest classifiers, ten-fold cross-validation (random vs scaffold split)."),
    ("fig_rf_regression_r2.png",
     "Random-forest receptor and antioxidant regressors, ten-fold cross-validation."),
    ("fig_compound_counts.png", "Measured compounds per target endpoint."),
    ("fig_feature_block_ablation.png",
     "Feature-block ablation: fingerprint vs descriptors vs combined."),
    ("fig_descriptor_importance.png",
     "Permutation importance of the twelve descriptors per endpoint."),
    ("fig_calibration.png", "Probability calibration (expected calibration error, raw vs isotonic)."),
    ("fig_applicability_coverage.png",
     "Applicability-domain coverage of DrugBank per endpoint."),
    ("fig_learning_curve.png", "Scaffold-honest learning curves: where more data still helps."),
    ("fig_gnn_vs_rf.png", "Graph network (GIN) vs random forest on the same scaffold split."),
    ("fig_adme_performance.png", "ADME / exposure models, scaffold ten-fold performance."),
    ("fig_kpuu_exposure.png",
     "Predicted unbound brain exposure (K_p,uu) on known central and peripheral drugs."),
]


def main() -> None:
    md = MD.read_text(encoding="utf-8")
    lines = [md, "", "\\newpage", "", "# Figures", ""]
    for i, (name, cap) in enumerate(FIGURES, 1):
        p = FIG / name
        if p.exists():
            lines.append(f"**Figure {i}.** {cap}")
            lines.append("")
            lines.append(f"![Figure {i}]({p.as_posix()})")
            lines.append("")
    combined = "\n".join(lines)

    pypandoc.convert_text(
        combined, "docx", format="markdown",
        outputfile=str(OUT),
        extra_args=["--toc", "--toc-depth=2", f"--resource-path={ROOT.as_posix()}"],
    )
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
