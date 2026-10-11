"""Package the deployed models as one archive, with a manifest that can verify it.

The model binaries are 0.78 GB and are excluded from the repository by .gitignore, so a clone cannot
run the server. Carrying them in git-lfs works on a host that provides the storage, but it ties the
deployment to that host and consumes the GitHub free quota outright. Publishing them once as a
citable archive and fetching at start-up instead makes the server deployable anywhere, and gives the
manuscript something to cite that is not a directory on a laptop.

The manifest is the point. It records the archive's SHA-256 and the size and checksum of every file
inside it, so a download can be verified rather than trusted: a truncated transfer or a silently
substituted archive is caught before a single model is loaded, instead of surfacing later as a
prediction that is subtly wrong. The manifest is small and is committed to the repository; the
archive is not.

Writes models_manifest.json and dist/sai_cns_models_v<version>.tar.gz
"""
from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
M = ROOT / "models_rf"
DIST = ROOT / "dist"
MANIFEST = ROOT / "models_manifest.json"
# Everything load_models() and the applicability-domain layer read. Directories are included whole.
INCLUDE_SUFFIX = {".joblib", ".pkl", ".json"}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while blk := fh.read(chunk):
            h.update(blk)
    return h.hexdigest()


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "1.0"
    if not M.exists():
        print("models_rf/ is absent; nothing to package")
        return 1
    DIST.mkdir(exist_ok=True)

    files = sorted(p for p in M.rglob("*")
                   if p.is_file() and p.suffix in INCLUDE_SUFFIX)
    total = sum(p.stat().st_size for p in files)
    print(f"packaging {len(files)} files, {total/1e9:.2f} GB", flush=True)

    archive = DIST / f"sai_cns_models_v{version}.tar.gz"
    t0 = time.time()
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for i, p in enumerate(files, 1):
            tar.add(p, arcname=p.relative_to(ROOT).as_posix())
            if i % 40 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    size = archive.stat().st_size
    print(f"wrote {archive.name}: {size/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)

    print("checksumming ...", flush=True)
    manifest = {
        "schema": 1,
        "version": version,
        "archive": archive.name,
        "archive_bytes": size,
        "archive_sha256": sha256(archive),
        "n_files": len(files),
        "unpacked_bytes": total,
        # Filled in by hand once the archive is published. Several may be listed and are tried in
        # order, so a mirror can be added without changing any code.
        "urls": [],
        "doi": "",
        "note": "Fetched at start-up by model_fetch.py when models_rf/ is absent. Every file is "
                "verified against its checksum before the server loads anything.",
        # POSIX separators always. Written with str() on Windows these become backslashes,
        # which are a legal filename character on Linux, so a manifest built on Windows made
        # every file appear missing on a Linux host and the server refused to start. The
        # archive itself was never affected: tarfile normalises member names.
        "files": {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size,
                                                   "sha256": sha256(p)}
                  for p in files},
    }
    # keep any urls or doi already recorded, so republishing does not wipe them
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        manifest["urls"] = old.get("urls", [])
        manifest["doi"] = old.get("doi", "")
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {MANIFEST.name}: {len(files)} files, archive sha256 "
          f"{manifest['archive_sha256'][:16]}...")
    print(f"\nNext: publish {archive} to Zenodo, then put the download URL in "
          f"{MANIFEST.name} under \"urls\" and the DOI under \"doi\".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
