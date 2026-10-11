"""Package the raw API caches as one citable archive, with a manifest that can verify them.

The caches under data/_chembl_cache, data/_bindingdb_cache and data/_pubchem_cache are the only
record of what ChEMBL, BindingDB and PubChem returned on the retrieval dates. They are committed to
the repository, so a clone can rebuild the endpoint tables without a network call. This script
publishes the same bytes a second way, as an archive to deposit alongside the model release, for two
reasons: the deposit is citable and versioned in a way a git directory is not, and a reader who wants
only the source data should not have to clone a repository to get it.

The manifest is the point, exactly as it is for the models. It records the archive's SHA-256 and the
size and checksum of every file inside, so a copy can be verified rather than trusted. Because the
caches are also in git, the manifest doubles as an integrity check on the working tree: --verify
compares what is on disk against what was published and reports any file that has drifted.

Writes caches_manifest.json (committed) and dist/brainsafe_source_caches_v<version>.tar.gz (not).

Run:  python src/sai_cns/data/package_caches.py
      python src/sai_cns/data/package_caches.py --verify   (check the tree against the manifest)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIST = ROOT / "dist"
MANIFEST = ROOT / "caches_manifest.json"
VERSION = "1.0"

CACHE_DIRS = [
    ROOT / "data" / "_chembl_cache",
    ROOT / "data" / "_bindingdb_cache",
    ROOT / "data" / "_pubchem_cache",
]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def cache_files() -> list[Path]:
    """Every cached response, in a stable order so the archive is reproducible."""
    files = []
    for d in CACHE_DIRS:
        if not d.is_dir():
            raise SystemExit(f"cache directory missing: {d.relative_to(ROOT).as_posix()}")
        files.extend(sorted(p for p in d.rglob("*") if p.is_file()))
    return files


def verify() -> int:
    """Compare the working tree against the published manifest. Returns a process exit code."""
    if not MANIFEST.exists():
        raise SystemExit(f"no manifest at {MANIFEST.name}; run without --verify to create one.")
    manifest = json.loads(MANIFEST.read_text())
    recorded = manifest.get("files", {})
    missing, changed = [], []
    for rel, meta in recorded.items():
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
        elif p.stat().st_size != meta["bytes"] or sha256(p) != meta["sha256"]:
            changed.append(rel)
    on_disk = {f.relative_to(ROOT).as_posix() for f in cache_files()}
    extra = sorted(on_disk - set(recorded))

    print(f"manifest records {len(recorded)} files from {manifest.get('created', 'an unknown date')}")
    for label, items in (("missing", missing), ("changed", changed), ("not in manifest", extra)):
        if items:
            print(f"  {label}: {len(items)}")
            for rel in items[:5]:
                print(f"    {rel}")
            if len(items) > 5:
                print(f"    ... and {len(items) - 5} more")
    if missing or changed:
        print("\nFAILED: the caches on disk are not the caches that were published.")
        return 1
    print("\nOK: every cached response matches its published checksum."
          + (f" {len(extra)} newer file(s) are not yet in the manifest." if extra else ""))
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Package or verify the raw API caches.")
    ap.add_argument("--verify", action="store_true",
                    help="check the working tree against caches_manifest.json instead of packaging")
    args = ap.parse_args(argv)

    if args.verify:
        raise SystemExit(verify())

    files = cache_files()
    DIST.mkdir(parents=True, exist_ok=True)
    archive = DIST / f"brainsafe_source_caches_v{VERSION}.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for p in files:
            # as_posix so an archive built on Windows carries the same member names as one built
            # on Linux; package_models.py records the same lesson.
            tar.add(p, arcname=p.relative_to(ROOT).as_posix())

    manifest = {
        "schema": 1,
        "version": VERSION,
        "created": time.strftime("%Y-%m-%d"),
        "describes": "raw ChEMBL, BindingDB and PubChem responses behind data/endpoints/",
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "n_files": len(files),
        "unpacked_bytes": sum(p.stat().st_size for p in files),
        "files": {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in files},
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        manifest["urls"] = old.get("urls", [])
        manifest["doi"] = old.get("doi", "")
    else:
        manifest["urls"] = []
        manifest["doi"] = ""
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(f"wrote {archive.relative_to(ROOT).as_posix()} "
          f"({manifest['archive_bytes'] / 1e6:.1f} MB, {len(files)} files)")
    print(f"wrote {MANIFEST.name}: archive sha256 {manifest['archive_sha256'][:16]}...")
    if not manifest["doi"]:
        print("\nNext: deposit the archive, then record its DOI and download URLs in "
              f"{MANIFEST.name} (the 'doi' and 'urls' fields).")


if __name__ == "__main__":
    main()
