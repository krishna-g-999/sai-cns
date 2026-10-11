"""Recompress the model files, verifying that predictions are unchanged.

The panel is 2.12 GB on disk, and most of that is not information. joblib writes an uncompressed
pickle unless asked otherwise: the binder models were dumped with compress=3 and average 3 MB, while
the eight calibrated core classifiers were dumped without and average 117 MB. hERG_calibrated alone
is 200 MB and holds five complete 300-tree forests, because CalibratedClassifierCV(cv=5) keeps one
per fold. Those five are not redundant, so they stay; what is redundant is storing them raw.

Size matters here for one reason. A 2 GB panel cannot be hosted on any free tier, and a server that
is not reachable is not a web server. Compression is the one lever that costs nothing scientifically.

Nothing is trusted. Every file is recompressed to a temporary path, reloaded, and scored on a fixed
pseudorandom matrix against the original; the file is only replaced if the predictions agree to
1e-12. That tolerance is not arbitrary and is not slack: a random forest with n_jobs=-1 sums tree
votes in completion order, so calling one estimator twice already differs by around 1e-16. The same
tolerance is pinned in tests/test_integration_smoke.py for the same reason.

A file that fails verification is left exactly as it was and reported, so a partial run degrades to
"fewer files compressed" rather than to a panel that predicts differently.

Run:  python tools/compress_models.py            # report what would be saved, change nothing
      python tools/compress_models.py --apply    # recompress in place, verifying each file
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models_rf"
LEVEL = 3                 # 6 buys about 2 per cent more and costs noticeably more time to write
TOL = 1e-12               # the tolerance the test suite already pins for this pipeline
N_PROBE = 32              # rows in the verification matrix
MIN_SAVING = 0.10         # skip files where compression would save less than this fraction


def probe(n_features: int, seed: int = 0) -> np.ndarray:
    """A fixed pseudorandom feature matrix. Same every run, so a report is comparable."""
    return np.random.default_rng(seed).random((N_PROBE, n_features)).astype(np.float32)


def score(est, X):
    """Whatever this estimator answers with, as a float vector, or None if it answers nothing."""
    for meth in ("predict_proba", "predict"):
        if hasattr(est, meth):
            try:
                out = np.asarray(getattr(est, meth)(X), dtype=float)
            except Exception:
                return None
            return out[:, 1] if out.ndim == 2 and out.shape[1] > 1 else out.ravel()
    return None


def n_features_of(est) -> int | None:
    n = getattr(est, "n_features_in_", None)
    if n:
        return int(n)
    for attr in ("calibrated_classifiers_", "estimators_"):
        inner = getattr(est, attr, None)
        if inner:
            got = n_features_of(inner[0])
            if got:
                return got
    for attr in ("estimator", "base_estimator"):
        inner = getattr(est, attr, None)
        if inner is not None:
            got = n_features_of(inner)
            if got:
                return got
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Recompress model files, verifying predictions.")
    ap.add_argument("--apply", action="store_true", help="rewrite the files (default: report only)")
    ap.add_argument("--level", type=int, default=LEVEL)
    args = ap.parse_args(argv)

    paths = sorted(p for p in MODELS.rglob("*.joblib") if p.is_file())
    print(f"{len(paths)} model files under {MODELS.relative_to(ROOT)}")
    print(f"{'':44s} {'before':>9s} {'after':>9s}  verdict")

    before_total = after_total = 0
    changed = skipped = failed = 0
    for path in paths:
        size_before = path.stat().st_size
        before_total += size_before
        try:
            est = joblib.load(path)
        except Exception as exc:
            after_total += size_before
            failed += 1
            print(f"  {str(path.relative_to(MODELS)):42s} {size_before/1e6:8.1f}M "
                  f"{'':9s}  unreadable: {type(exc).__name__}")
            continue

        tmp = path.with_suffix(".joblib.recompress")
        joblib.dump(est, tmp, compress=args.level)
        size_after = tmp.stat().st_size

        if size_after > size_before * (1 - MIN_SAVING):
            tmp.unlink()
            after_total += size_before
            skipped += 1
            print(f"  {str(path.relative_to(MODELS)):42s} {size_before/1e6:8.1f}M "
                  f"{size_after/1e6:8.1f}M  already compact, left alone")
            continue

        # Verification. An estimator that cannot be scored is not rewritten: unable to check is not
        # the same as checked and fine.
        nf = n_features_of(est)
        ok, detail = False, "no feature count, cannot verify"
        if nf:
            X = probe(nf)
            a = score(est, X)
            b = score(joblib.load(tmp), X)
            if a is None or b is None:
                detail = "estimator exposes no predict method, cannot verify"
            else:
                worst = float(np.abs(a - b).max())
                ok = worst <= TOL
                detail = f"max |diff| {worst:.1e}"

        if not ok:
            tmp.unlink()
            after_total += size_before
            failed += 1
            print(f"  {str(path.relative_to(MODELS)):42s} {size_before/1e6:8.1f}M "
                  f"{size_after/1e6:8.1f}M  NOT REPLACED, {detail}")
            continue

        after_total += size_after
        changed += 1
        if args.apply:
            tmp.replace(path)
        else:
            tmp.unlink()
        print(f"  {str(path.relative_to(MODELS)):42s} {size_before/1e6:8.1f}M "
              f"{size_after/1e6:8.1f}M  ok, {detail}")

    print()
    print(f"  files recompressed        {changed}")
    print(f"  already compact           {skipped}")
    print(f"  left alone (unverifiable) {failed}")
    print(f"  {before_total/1e9:.2f} GB -> {after_total/1e9:.2f} GB "
          f"({100*after_total/max(before_total,1):.0f} per cent of the original)")
    if not args.apply:
        print("\nreport only; nothing was written. Re-run with --apply to recompress in place.")
    else:
        print("\nRewritten. Regenerate the manifest: python src/sai_cns/models/package_models.py 1.1")


if __name__ == "__main__":
    main(sys.argv[1:])
