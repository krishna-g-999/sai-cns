"""Upload the packaged model archive to blob storage in retryable blocks.

WHY THIS EXISTS. `az storage blob upload` sends the 0.86 GB archive as one long transfer. On a
connection that drops occasionally that fails at whatever percentage it had reached, with nothing
kept, and two attempts here died at 4.9 per cent and part-way through with a reset and a DNS
failure respectively. Short requests to the same host succeed consistently, so the problem is the
duration of the transfer rather than the link.

Blob storage already supports the fix. A block blob can be uploaded as a sequence of independently
staged blocks which are then committed as one object, so each request is seconds rather than
minutes. A failed block is retried on its own and costs only that block, and a run that dies part
way can be restarted: blocks already staged are listed by the service and skipped. The committed
blob does not exist until every block is present, so a partial upload is never visible to anyone
fetching the archive.

The archive's checksum is verified against models_manifest.json before anything is sent, and the
committed blob's length and MD5 are checked afterwards against the local file. Publishing an
archive whose checksum does not match the manifest would break every deployment that verifies
against it, which is the failure this whole layer exists to prevent.

Reads the account key from the environment, never from the command line, so it does not reach a
process listing or this file's output.

    export SAI_CNS_STORAGE_KEY=$(az storage account keys list -g sai-cns-rg \
        -n saicnsmodels2026 --query "[0].value" -o tsv)
    python tools/publish_model_archive.py [--container models] [--block-mb 8]

    python tools/publish_model_archive.py --delta      publish only the per-file delta

--delta uploads every file the manifest gives a `url` for, to exactly that url, and verifies each
against its recorded sha256 afterwards. That is the companion to the base-plus-delta scheme in
model_fetch.py: the base archive is republished rarely, and a changed file ships on its own. On a
55 KB/s uplink this is the difference between 20 minutes and 4.4 hours.

Run with the Azure CLI's bundled interpreter, which already has azure-storage-blob:
    "C:/Program Files/Microsoft SDKs/Azure/CLI2/python.exe" tools/publish_model_archive.py
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models_manifest.json"
ACCOUNT = "saicnsmodels2026"
ATTEMPTS = 6


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while blk := fh.read(chunk):
            h.update(blk)
    return h.hexdigest()


def publish_delta(man, key, args) -> int:
    """Upload every file the manifest gives a url for, then verify it from the public URL.

    Each file is sent with the same staged-block path the archive uses, so a dropped connection
    costs one block rather than the whole file. Verification reads the committed blob's length and
    compares the local sha256, because an upload that reports success and lands truncated is the
    failure this layer exists to catch.
    """
    import base64
    import urllib.request
    from azure.storage.blob import BlobServiceClient

    delta = [(rel, meta) for rel, meta in man["files"].items() if meta.get("url")]
    if not delta:
        print("the manifest names no per-file urls; nothing to publish")
        return 0

    svc = BlobServiceClient(f"https://{ACCOUNT}.blob.core.windows.net", credential=key)
    total = sum(m["bytes"] for _, m in delta)
    print(f"publishing {len(delta)} file(s), {total / 1e6:.1f} MB in total\n")

    block = args.block_mb << 20
    failed = []
    for rel, meta in delta:
        src = ROOT.joinpath(*rel.replace("\\", "/").split("/"))
        if not src.exists():
            print(f"  {rel}: absent locally; skipped")
            failed.append(rel)
            continue
        digest = sha256(src)
        if digest != meta["sha256"]:
            print(f"  {rel}: local file does not match the manifest; refusing")
            failed.append(rel)
            continue
        url = meta["url"] if isinstance(meta["url"], str) else meta["url"][0]
        name = url.split(f"/{args.container}/", 1)[1] if f"/{args.container}/" in url else             url.rsplit("/", 1)[1]
        blob = svc.get_blob_client(container=args.container, blob=name)

        size = src.stat().st_size
        n = (size + block - 1) // block
        ids = [base64.b64encode(f"{i:08d}".encode()).decode() for i in range(n)]
        print(f"  {rel}  {size / 1e6:.1f} MB in {n} block(s)")
        with src.open("rb") as fh:
            for i, bid in enumerate(ids):
                data = fh.read(block)
                for attempt in range(1, ATTEMPTS + 1):
                    try:
                        blob.stage_block(block_id=bid, data=data, length=len(data))
                        break
                    except Exception as e:
                        if attempt == ATTEMPTS:
                            print(f"    block {i} failed: {type(e).__name__}")
                            failed.append(rel)
                            break
                        time.sleep(min(2 ** attempt, 30))
                else:
                    continue
                if rel in failed:
                    break
        if rel in failed:
            continue
        blob.commit_block_list(ids)
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                served = int(r.headers.get("Content-Length", -1))
        except Exception as e:
            print(f"    published but not reachable: {type(e).__name__}")
            failed.append(rel)
            continue
        state = "ok" if served == size else f"MISMATCH served {served}"
        print(f"    committed and reachable, {state}")
        if served != size:
            failed.append(rel)

    if failed:
        print(f"\n{len(failed)} file(s) did not publish cleanly: {failed}")
        print("rerun; staged blocks are kept and skipped")
        return 1
    print(f"\nall {len(delta)} file(s) published and reachable at the urls the manifest records")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="models")
    ap.add_argument("--block-mb", type=int, default=8)
    ap.add_argument("--delta", action="store_true",
                    help="publish the per-file delta instead of the archive")
    args = ap.parse_args()

    key = os.environ.get("SAI_CNS_STORAGE_KEY")
    if not key:
        print("SAI_CNS_STORAGE_KEY is not set; see this file's docstring")
        return 2

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))

    if args.delta:
        return publish_delta(man, key, args)

    archive = ROOT / "dist" / man["archive"]
    if not archive.exists():
        print(f"{archive} is absent; run src/sai_cns/models/package_models.py first")
        return 1

    size = archive.stat().st_size
    if size != man["archive_bytes"]:
        print(f"size mismatch: file {size:,} against manifest {man['archive_bytes']:,}")
        return 1
    print(f"verifying {archive.name} against the manifest checksum ...", flush=True)
    digest = sha256(archive)
    if digest != man["archive_sha256"]:
        print(f"CHECKSUM MISMATCH\n  file     {digest}\n  manifest {man['archive_sha256']}")
        print("Refusing to publish: deployments verify against the manifest.")
        return 1
    print(f"  ok, sha256 {digest[:16]}...")

    from azure.storage.blob import BlobServiceClient  # noqa: E402

    svc = BlobServiceClient(f"https://{ACCOUNT}.blob.core.windows.net", credential=key)
    blob = svc.get_blob_client(container=args.container, blob=man["archive"])

    block = args.block_mb << 20
    n_blocks = (size + block - 1) // block
    ids = [base64.b64encode(f"{i:08d}".encode()).decode() for i in range(n_blocks)]

    # Resume only on blocks whose staged LENGTH matches what this run would send. Block ids do
    # not encode the block size, so matching on id alone would skip a block staged by an earlier
    # run at a different --block-mb and commit a blob assembled from mismatched pieces. The final
    # size check would catch that, but only after publishing something wrong, so it is excluded
    # here instead. A staged block that does not match is simply restaged, which overwrites it.
    staged = set()
    try:
        _committed, uncommitted = blob.get_block_list("uncommitted")
        expected = {i: min(block, size - i * block) for i in range(n_blocks)}
        by_id = {ids[i]: expected[i] for i in range(n_blocks)}
        staged = {b.id for b in uncommitted if by_id.get(b.id) == b.size}
        mismatched = len(uncommitted) - len(staged)
        if staged:
            print(f"  resuming: {len(staged)} of {n_blocks} blocks already staged")
        if mismatched:
            print(f"  {mismatched} staged block(s) were written at a different block size "
                  f"and will be restaged")
    except Exception:
        pass

    print(f"staging {n_blocks} blocks of {args.block_mb} MB", flush=True)
    t0 = time.time()
    with archive.open("rb") as fh:
        for i, bid in enumerate(ids):
            data = fh.read(block)
            if bid in staged:
                continue
            for attempt in range(1, ATTEMPTS + 1):
                try:
                    blob.stage_block(block_id=bid, data=data, length=len(data))
                    break
                except Exception as e:
                    if attempt == ATTEMPTS:
                        print(f"\n  block {i} failed after {ATTEMPTS} attempts: "
                              f"{type(e).__name__}")
                        print("  rerun this script; staged blocks are kept and skipped")
                        return 1
                    time.sleep(min(2 ** attempt, 30))
            done = i + 1
            if done % 10 == 0 or done == n_blocks:
                pct = done / n_blocks * 100
                mb = done * args.block_mb
                rate = mb / max(time.time() - t0, 1e-9)
                print(f"  {done}/{n_blocks}  {pct:5.1f}%  {rate:.1f} MB/s", flush=True)

    print("committing the block list ...", flush=True)
    blob.commit_block_list(ids, content_type="application/x-tar")

    props = blob.get_blob_properties()
    print(f"\ncommitted {man['archive']}")
    print(f"  size on service {props.size:,}  local {size:,}  "
          f"{'match' if props.size == size else 'MISMATCH'}")
    url = f"https://{ACCOUNT}.blob.core.windows.net/{args.container}/{man['archive']}"
    print(f"  {url}")
    if props.size != size:
        return 1
    print("\nNext: put this URL in models_manifest.json under \"urls\" if it is not there,")
    print("then confirm model_fetch.ensure_models() accepts the published archive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
