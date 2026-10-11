"""Make sure the models are present before anything tries to load them.

The model binaries are 0.78 GB and are not in the repository, so a fresh clone or a container built
from it has the code and none of the science. This fetches them from a published archive on first
run, and does nothing at all when they are already there, which is the case on any developer machine
and on any host that bakes them into the image.

Verification is not optional here. A truncated download or a substituted archive would not announce
itself: the server would start, load whatever it found, and return predictions that are wrong in ways
no user could detect. Every file is therefore checked against the SHA-256 recorded in
models_manifest.json before the server is allowed to proceed, and a mismatch is fatal rather than a
warning.

Behaviour, in order:
  1. if models_manifest.json is absent, do nothing: this is a development tree that manages its own
     models, and inventing a fetch would be worse than leaving it alone
  2. if every file named in the manifest is present with the right size and checksum, do nothing
  3. otherwise download the archive from the first URL that works, verify it, extract, and verify
     every extracted file

Set SAI_CNS_SKIP_MODEL_FETCH=1 to bypass entirely, for an image that already carries the models.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "models_manifest.json"
CHUNK = 1 << 20


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while blk := fh.read(CHUNK):
            h.update(blk)
    return h.hexdigest()


def _missing(manifest, quick=True):
    """Files the manifest names that are absent, the wrong size, or the wrong content.

    quick=True checks existence and size only, which is what a start-up path can afford; the full
    checksum pass runs after a download, when correctness matters more than a second of latency.

    Separators are normalised because a manifest written on Windows records "models_rf\\X". A
    backslash is a legal filename character on Linux, so those keys resolve to single files that do
    not exist, every file appears to be missing, and the server refuses to start while downloading
    the archive over and over. That is precisely what happened on the first cloud deployment."""
    out = []
    for rel, meta in manifest["files"].items():
        p = ROOT.joinpath(*rel.replace("\\", "/").split("/"))
        if not p.exists() or p.stat().st_size != meta["bytes"]:
            out.append(rel)
        elif not quick and _sha256(p) != meta["sha256"]:
            out.append(rel)
    return out


def _download(url, dest, expect_bytes, log):
    log(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as fh:
        got = 0
        while blk := r.read(CHUNK):
            fh.write(blk)
            got += len(blk)
            if expect_bytes and got % (64 << 20) < CHUNK:
                log(f"  {got/1e9:.2f} / {expect_bytes/1e9:.2f} GB")
    return dest.stat().st_size


def _flag(name):
    """Read SAI_CNS_<name>, falling back to the former BRAINSAFE_<name>.

    The project was renamed, but a deployment configured before the rename sets the old variable.
    Dropping the fallback would not fail loudly: the flag would simply read as unset, and on a host
    that already carries the models that means re-verifying 150 checksums on every start-up, or
    worse, deciding they are missing and trying to download them. Accepting both names costs two
    lines and removes a class of silent misconfiguration.
    """
    for var in (f"SAI_CNS_{name}", f"BRAINSAFE_{name}"):
        v = os.environ.get(var, "").strip().lower()
        if v:
            return v in ("1", "true", "yes")
    return False


def ensure_models(verbose=True):
    """Returns True if the models are present and verified, False if they could not be obtained."""
    def log(m):
        if verbose:
            print(f"[model_fetch] {m}", flush=True)

    if _flag("SKIP_MODEL_FETCH"):
        return True
    if not MANIFEST.exists():
        return True                      # development tree; not ours to manage

    manifest = json.loads(MANIFEST.read_text())
    missing = _missing(manifest, quick=True)
    if not missing:
        return True

    # One download at a time. A Streamlit host starts a script run per session, and on the first
    # cloud deployment several runs each began fetching 0.79 GB simultaneously, competing for
    # bandwidth and disk. The first process through the gate downloads; the others wait and then
    # re-check, by which time the files are usually already there.
    lock = ROOT / ".model_fetch.lock"
    waited = 0
    while lock.exists() and waited < 900:
        if waited == 0:
            log("another process is already fetching; waiting")
        time.sleep(5)
        waited += 5
    if waited:
        if not _missing(manifest, quick=True):
            log("models were fetched by the other process")
            return True
    try:
        lock.touch()
    except OSError:
        pass

    log(f"{len(missing)} of {manifest['n_files']} model files missing or the wrong size")
    urls = [u for u in manifest.get("urls", []) if u]
    if not urls:
        lock.unlink(missing_ok=True)
        log("no download URL is recorded in models_manifest.json, so they cannot be fetched.")
        log("Publish the archive and add its URL under \"urls\", or place models_rf/ here yourself.")
        return False

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / manifest["archive"]
        for url in urls:
            try:
                size = _download(url, tmp, manifest.get("archive_bytes"), log)
            except Exception as e:
                log(f"  failed: {str(e)[:120]}")
                continue
            if manifest.get("archive_bytes") and size != manifest["archive_bytes"]:
                log(f"  wrong size: {size} against {manifest['archive_bytes']}; discarding")
                continue
            log("verifying the archive checksum ...")
            if _sha256(tmp) != manifest["archive_sha256"]:
                log("  CHECKSUM MISMATCH: the archive is not the one this build expects. Discarding.")
                continue
            log("extracting ...")
            with tarfile.open(tmp, "r:gz") as tar:
                # never let an archive write outside the project, whatever it claims its paths are
                for member in tar.getmembers():
                    target = (ROOT / member.name).resolve()
                    if not str(target).startswith(str(ROOT.resolve())):
                        log(f"  refusing path outside the project: {member.name}")
                        return False
                # filter="data" refuses absolute paths, parent traversal, links and device nodes,
                # and drops ownership and permission metadata. It is the default from Python 3.14
                # and is requested explicitly so the behaviour does not depend on the interpreter.
                try:
                    tar.extractall(ROOT, filter="data")
                except TypeError:
                    tar.extractall(ROOT)      # Python older than 3.11.4
            break
        else:
            lock.unlink(missing_ok=True)
            log("every download URL failed")
            return False

    log("verifying every extracted file ...")
    bad = _missing(manifest, quick=False)
    lock.unlink(missing_ok=True)
    if bad:
        log(f"FAILED: {len(bad)} files do not match their checksum, for example {bad[:3]}")
        return False
    log(f"all {manifest['n_files']} model files present and verified")
    return True


if __name__ == "__main__":
    ok = ensure_models()
    if not ok:
        print("\nModels are not available. The server cannot produce predictions without them.")
    sys.exit(0 if ok else 1)
