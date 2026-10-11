"""Assemble a Hugging Face Space directory containing exactly what the server needs.

Copying the repository wholesale is the usual way this goes wrong. It carries the 155 MB of
scaffold-hold-out twins, the reviewer package, the manuscript figures and the validation artefacts,
none of which serve a prediction, and the upload is then several times larger than it has to be for
no benefit.

This copies the runtime set, writes the Space card and the git-LFS rules, and reports the size. It
does not push: pushing is a credentialled act and belongs to whoever owns the Space.

Run:  python deploy/huggingface/prepare_space.py --out ../sai-cns-space
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# What the server reads at runtime. Anything not named here is not copied, which is the point.
# requirements.txt is deliberately absent: the root one is the training environment and the Space
# gets the runtime subset from this directory instead. See deploy/huggingface/requirements.txt.
# models_manifest.json is absent here on purpose: it is rewritten below to describe the shipped
# subset rather than the repository, so the integrity check passes on what actually travelled.
FILES = ["app.py", "api.py", "serve.py", "model_fetch.py", "CITATION.cff", "LICENSE"]
DIRS = ["src", "assets", "results", "docs", ".streamlit"]

# inversion/results is 176 KB of falsification-suite output and was omitted until app.py began
# reading it. family_cofire() derives the co-firing badge from H8_family_correlation.csv, and a
# missing file there returns {} and silently drops the badge: the page would have looked correct
# while quietly losing a feature, which is the failure mode a deployment check is for. The interface
# also cites H2, H3, H7 and H8 by path in its own explanatory text, so shipping the directory rather
# than the one file means those citations point at something the reader can actually open.
EXTRA_DIRS = [Path("inversion") / "results"]

# data/ is 582 MB and the server reads 18 MB of it. The rest is the raw pulls, the API caches and
# the external sets that the training and validation scripts consume, none of which is opened to
# answer a query: app.py reads data/endpoints only, and does not import the pools module that
# reaches for the external drug list. Copying data/ wholesale made the upload three times larger
# than the thing being uploaded.
DATA_SUBDIRS = ["endpoints", "endpoints_reg", "adme", "readacross"]

# models_rf/ is copied separately so holdout/ can be left behind: it holds the scaffold-split twins
# used for validation and is never consulted to answer a query.
MODELS_SKIP = {"holdout"}

# The Hub rejects a push containing ANY binary file that is not in LFS, whatever its size, with
# "Your push was rejected because it contains binary files". Listing only the model formats is not
# enough: a 104 kB logo is refused on the same rule as a 58 MB forest. Every binary extension the
# assembled Space can contain is therefore named here, and the list is deliberately wider than what
# the current contents happen to include, so adding a figure or a PDF later does not fail the push.
_LFS_BINARY = ["joblib", "pkl", "tar.gz", "xlsx", "docx", "pdf", "png", "jpg", "jpeg", "gif",
               "ico", "svgz", "woff", "woff2", "ttf", "otf", "zip", "npz", "npy", "parquet", "h5"]
_LFS_RULE = "*.{} filter=lfs diff=lfs merge=lfs -text"

# Every file is stored and checked out byte for byte, with no line-ending translation. This is not
# tidiness. models_manifest.json records a size and a SHA-256 for each model file, and model_fetch
# refuses to serve predictions when a file does not match. Those figures are computed on the machine
# that assembles the Space, and Python writing JSON in text mode on Windows produces CRLF. Git then
# normalises the committed copy to LF, so binder_modes.json is 57,123 bytes on the build machine and
# 55,648 inside a Linux container, and 63 of the 64 metadata files fail the integrity check on a
# Space that is otherwise perfectly built. The models themselves are unaffected, being binary and in
# LFS, which is why the symptom looked like a partial upload rather than a text conversion.
_NO_EOL_CONVERSION = "* -text"
LFS = (_NO_EOL_CONVERSION + "\n"
       + "\n".join(_LFS_RULE.format(e) for e in _LFS_BINARY) + "\n")

# Compiled bytecode is build output, not source, and must never reach the Space. copy_tree already
# skips it, but anything that imports the app from inside the assembled directory recreates it, and
# verify_space.py does exactly that. Without this the first push fails on a .pyc file that was not
# there when the directory was assembled.
IGNORE = """__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
"""


def copy_tree(src: Path, dst: Path, skip: set[str] = frozenset()) -> int:
    total = 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        if rel.parts and rel.parts[0] in skip:
            continue
        if "__pycache__" in rel.parts or p.suffix == ".pyc":
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        total += p.stat().st_size
    return total


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Assemble a Space directory.")
    ap.add_argument("--out", required=True, help="target directory (the cloned Space)")
    ap.add_argument("--with-holdout", action="store_true",
                    help="include models_rf/holdout, which the server never reads")
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"assembling into {out}\n")

    total = 0
    for name in FILES:
        src = ROOT / name
        if not src.exists():
            print(f"  missing, skipped   {name}")
            continue
        shutil.copy2(src, out / name)
        total += src.stat().st_size
        print(f"  {src.stat().st_size/1e6:8.2f} MB  {name}")

    for name in DIRS:
        src = ROOT / name
        if not src.exists():
            print(f"  missing, skipped   {name}/")
            continue
        n = copy_tree(src, out / name)
        total += n
        print(f"  {n/1e6:8.2f} MB  {name}/")

    for rel in EXTRA_DIRS:
        src = ROOT / rel
        if not src.exists():
            print(f"  missing, skipped   {rel.as_posix()}/")
            continue
        n = copy_tree(src, out / rel)
        total += n
        print(f"  {n/1e6:8.2f} MB  {rel.as_posix()}/")

    for sub in DATA_SUBDIRS:
        src = ROOT / "data" / sub
        if not src.exists():
            print(f"  missing, skipped   data/{sub}/")
            continue
        n = copy_tree(src, out / "data" / sub)
        total += n
        print(f"  {n/1e6:8.2f} MB  data/{sub}/")

    skip = set() if args.with_holdout else MODELS_SKIP
    n = copy_tree(ROOT / "models_rf", out / "models_rf", skip=skip)
    total += n
    print(f"  {n/1e6:8.2f} MB  models_rf/" + ("" if args.with_holdout else "  (holdout excluded)"))

    # The manifest must describe what was shipped, not what exists in the repository. It lists 252
    # files including the 102 scaffold-hold-out twins that models_rf/holdout deliberately leaves
    # behind, and model_fetch.ensure_models() refuses to serve predictions when a manifest entry is
    # absent. That is the correct behaviour: a server that quietly loads whatever it finds returns
    # answers that are wrong in ways nothing reports. The wrong fix is to disable the check with
    # SAI_CNS_SKIP_MODEL_FETCH, which removes the integrity guarantee for every file including the
    # ones that did travel. The right fix is to describe the shipped set accurately, so the check
    # still verifies every model the Space actually serves.
    manifest = json.loads((ROOT / "models_manifest.json").read_text(encoding="utf-8"))
    if args.with_holdout:
        kept = manifest["files"]
    else:
        kept = {k: v for k, v in manifest["files"].items()
                if not any(f"/{s}/" in k for s in MODELS_SKIP)}
    dropped = len(manifest["files"]) - len(kept)
    manifest["files"] = kept
    manifest["n_files"] = len(kept)
    manifest["note"] = (manifest.get("note", "") +
                        f" Space build: {dropped} hold-out entries removed, as models_rf/holdout is "
                        f"not served.").strip()
    (out / "models_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  {'':8s}    models_manifest.json  ({len(kept)} entries, {dropped} hold-out removed)")

    shutil.copy2(HERE / "README.md", out / "README.md")       # the Space card, with its YAML block
    shutil.copy2(HERE / "requirements.txt", out / "requirements.txt")   # runtime subset, not root
    shutil.copy2(HERE / "Dockerfile", out / "Dockerfile")               # the SDK no longer builds it
    print(f"  {'':8s}    requirements.txt  (runtime subset)")
    print(f"  {'':8s}    Dockerfile")
    (out / ".gitattributes").write_text(LFS, encoding="utf-8")
    (out / ".gitignore").write_text(IGNORE, encoding="utf-8")
    print(f"  {'':8s}    README.md  (Space card)")
    print(f"  {'':8s}    .gitattributes  (git-LFS rules, {len(_LFS_BINARY)} binary patterns)")
    print(f"  {'':8s}    .gitignore")

    print(f"\n  total {total/1e9:.2f} GB")
    print("\nNext, from inside that directory:")
    print("    git lfs install")
    print("    git add .gitattributes && git commit -m 'Track model files with LFS'")
    print("    git add -A && git commit -m 'SAI-CNS'")
    print("    git push")
    print("\nTrack LFS before adding the models, or the push is rejected for file size.")


if __name__ == "__main__":
    main(sys.argv[1:])
