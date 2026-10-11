"""Write the front page of the submission package: what it is and how to walk it.

An institutional reviewer opening 435 files needs to know, in the first minute, what is being
proposed, what has been done, what has NOT been done, and where to look to check any of it. That
last part matters most: a package that only shows what worked is a selection, not a submission.

Run:  python tools/build_start_here.py --package submission_package
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    args = ap.parse_args()
    pkg = Path(args.package).resolve()

    import pandas as pd
    import app as A

    sh = A.panel_shape()
    contents = json.loads((pkg / "PACKAGE_CONTENTS.json").read_text(encoding="utf-8"))
    ver = pd.read_csv(ROOT / "inversion/results/VERDICTS.csv")
    refuted = ver[ver.verdict.str.startswith("REFUTED")]

    sections = "\n".join(
        f"| `{s['section']}` | {s['files']} | {s['megabytes']:.1f} MB |"
        for s in contents["sections"])

    ref_list = "\n".join(f"- **{r.hypothesis}** — {r.headline}" for r in refuted.itertuples())

    text = f"""# SAI-CNS: submission package

Prepared {datetime.now().strftime('%d %B %Y')} for institutional review, ahead of a proposal to the
*Nucleic Acids Research* Web Server Issue.

**Live server:** https://huggingface.co/spaces/Krishnag999/brainsafe-ai
**Source:** https://github.com/krishna-g-999/brainsafe-ai
**Licence:** MIT for the code. Data retain their sources' licences.

---

## What is being proposed

A web server that takes one chemical structure and returns, in a few seconds, whether the compound
is likely to reach the human brain, which of {sh['targets']} molecular targets it is likely to
engage there, which conditions those mechanisms touch, and what would stop it being a drug. It is
built on {sh['deployed']} deployed estimators of {sh['trained']} trained, every one fitted on
measured experimental values rather than on qualitative annotation.

The submission to NAR is a two-step process. A one-page proposal is sent first, and only if it is
invited may a manuscript follow. **`01_PROPOSAL/` is the document that would be emailed.** Everything
else exists so that this claim can be checked before it leaves the institute.

## How to read this package

| Folder | Files | Size |
|---|---|---|
{sections}

Numbering follows the order a reviewer should walk. **`EVIDENCE_MAP.md` is the place to start**: it
lists every quantitative claim, the value the artefact currently holds, and the file that produced
it. It is generated from the artefacts rather than typed, so it cannot drift from them.

A reasonable path through:

1. `01_PROPOSAL/` — one page, the thing being sent.
2. `EVIDENCE_MAP.md` — every claim beside its source file.
3. `02_MANUSCRIPT/` — the full manuscript, with references and their PubMed links.
4. `03_FIGURES/` — every figure, all generated from the artefacts in `08_VALIDATION_RESULTS/`.
5. `04_TECHNICAL_REPORT/` — the long-form account, including a formal specification of the
   pipeline and a section answering the criticisms we expect a referee to raise.
6. `05_CODE/` — the scripts, numbered in the order the pipeline runs.
7. `06_TRAINING_DATA/`, `07_MODELS/`, `08_VALIDATION_RESULTS/` — the inputs, the fitted models
   and every result table.
8. `09_FALSIFICATION_SUITE/` — the attempts to break our own claims, including those that succeeded.

## What did not work, and is reported anyway

Nine hypotheses were written so that they could fail, each paired with a null model capable of
producing the same apparent success by accident. **Four were refuted**, and all four are published in
the manuscript and the technical report rather than removed:

{ref_list}

Five endpoints were trained, tested and then withheld because no decision threshold separated real
ligands from trivial metabolites. They are named in the interface and in the manuscript, because a
panel that shows only what survived is a selection rather than an inventory.

The applicability-domain flag is a weak signal, and is described as one. Recall on chemistry more
than a Tanimoto of 0.40 from anything the panel has measured is about 0.16, so a negative result on
a genuinely novel scaffold is close to uninformative; the server now reports the expected recall
beside every result rather than leaving that to be discovered.

## What is not in this package

The fitted estimators are 0.85 GB and are omitted unless the package was built with `--with-models`.
`07_MODELS/models_manifest_with_checksums.json` carries a SHA-256 for every one, so integrity can be
verified without shipping them, and both public mirrors above hold the complete set.

Raw API caches, archived earlier versions of the project, and the Python environment are omitted.
None is quoted anywhere.

## Outstanding before submission

- The proposal must be exported to PDF and emailed to `ds.narwbsrv@gmail.com`. NAR will not review a
  manuscript that arrives without a prior invitation.
- The manuscript runs longer than the 4 to 5 printed pages NAR asks for and needs trimming.
- Three Key Points, a graphical abstract, a cover letter listing comparable servers with their URLs,
  and six suggested referees are required at manuscript submission and are not yet written.
- A funding statement and a data-deposition DOI are needed.
- NAR expects a server to remain available for five years. The present deployment runs on a personal
  subscription, and a durable arrangement should be agreed before publication.
"""
    (pkg / "00_START_HERE.md").write_text(text, encoding="utf-8")
    print(f"wrote {pkg / '00_START_HERE.md'}")
    print(f"  sections: {len(contents['sections'])}, files: {contents['total_files']}, "
          f"{contents['total_megabytes']} MB")
    print(f"  refuted hypotheses listed: {len(refuted)}")


if __name__ == "__main__":
    main()
