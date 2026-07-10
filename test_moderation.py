"""Evaluate moderate_ad() against the labelled test set in test_samples/.

- Runs every file in test_samples/clean/ (expected ACCEPT) and
  test_samples/nsfw/ (expected REFUSE).
- Prints a per-file results table, then a summary with accuracy,
  false accepts (NSFW that slipped through - the dangerous error) and
  false refuses (clean content wrongly blocked).
- Sweeps several thresholds so the best one can be picked with data.
  Scores are threshold-independent, so inference runs once and the sweep
  just re-applies each cutoff.
- Times every moderate_ad() call (model load excluded via a warm-up call)
  and reports average latency.

Exit code 0 (PASS) iff there are zero false accepts at the chosen threshold.
"""

import argparse
import statistics
import sys
from pathlib import Path

from PIL import Image

from moderation import (
    DEFAULT_THRESHOLD,
    IMAGE_EXTS,
    VIDEO_EXTS,
    get_classifier,
    moderate_ad,
)

SAMPLES = Path(__file__).parent / "test_samples"
SWEEP_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
SUPPORTED = IMAGE_EXTS | VIDEO_EXTS


def collect_files():
    cases = []  # (path, expected_decision)
    for folder, expected in [("clean", "ACCEPT"), ("nsfw", "REFUSE")]:
        d = SAMPLES / folder
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in SUPPORTED:
                cases.append((p, expected))
    return cases


def warm_up():
    """Load the model and run one dummy inference so timings reflect steady state."""
    backend, clf = get_classifier()
    clf(Image.new("RGB", (224, 224), (128, 128, 128)))
    return backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"decision threshold for the PASS/FAIL verdict (default {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    cases = collect_files()
    if not cases:
        print(f"No test files found under {SAMPLES}. Run make_test_samples.py first.")
        return 2

    print("Loading model (warm-up, not counted in latency)...")
    backend = warm_up()
    print(f"Backend: {backend}")
    if backend == "skin-heuristic":
        print("NOTE: real model not downloaded yet - using the offline skin-ratio")
        print("      fallback. Run download_model.py to get the real classifier.")

    results = []
    for path, expected in cases:
        try:
            r = moderate_ad(path, threshold=args.threshold)
        except Exception as exc:  # noqa: BLE001 - a broken file counts as a failure
            print(f"ERROR moderating {path.name}: {exc}")
            return 2
        r["expected"] = expected
        results.append(r)

    # ---- per-file table -------------------------------------------------
    print()
    header = f"{'filename':<28} {'type':<6} {'expected':<8} {'actual':<8} {'score':>6} {'ms':>7}  result"
    print(header)
    print("-" * len(header))
    for r in results:
        ok = r["decision"] == r["expected"]
        print(
            f"{r['file']:<28} {r['media_type']:<6} {r['expected']:<8} "
            f"{r['decision']:<8} {r['nsfw_score']:>6.3f} {r['latency_s'] * 1000:>7.0f}  "
            f"{'pass' if ok else 'FAIL'}"
        )

    # ---- summary at chosen threshold ------------------------------------
    def tally(threshold):
        fa = sum(1 for r in results if r["expected"] == "REFUSE" and r["nsfw_score"] < threshold)
        fr = sum(1 for r in results if r["expected"] == "ACCEPT" and r["nsfw_score"] >= threshold)
        correct = len(results) - fa - fr
        return correct, fa, fr

    correct, false_accepts, false_refuses = tally(args.threshold)
    accuracy = 100.0 * correct / len(results)

    print()
    print(f"Summary @ threshold {args.threshold}:")
    print(f"  files tested:   {len(results)}")
    print(f"  accuracy:       {accuracy:.1f}%  ({correct}/{len(results)})")
    print(f"  false accepts:  {false_accepts}  (NSFW that slipped through - dangerous)")
    print(f"  false refuses:  {false_refuses}  (clean content wrongly blocked)")

    # ---- threshold sweep -------------------------------------------------
    print()
    print("Threshold sweep (same scores, different cutoffs):")
    print(f"  {'threshold':>9} {'accuracy':>9} {'false_acc':>10} {'false_ref':>10}")
    for t in SWEEP_THRESHOLDS:
        c, fa, fr = tally(t)
        print(f"  {t:>9.2f} {100.0 * c / len(results):>8.1f}% {fa:>10} {fr:>10}")

    # ---- latency ----------------------------------------------------------
    lat = [r["latency_s"] for r in results]
    img_lat = [r["latency_s"] for r in results if r["media_type"] == "image"]
    vid_lat = [r["latency_s"] for r in results if r["media_type"] == "video"]
    print()
    print("Latency per moderate_ad() call (model already loaded):")
    print(f"  average: {statistics.mean(lat) * 1000:.0f} ms   "
          f"min: {min(lat) * 1000:.0f} ms   max: {max(lat) * 1000:.0f} ms")
    if img_lat:
        print(f"  images:  {statistics.mean(img_lat) * 1000:.0f} ms avg over {len(img_lat)}")
    if vid_lat:
        print(f"  videos:  {statistics.mean(vid_lat) * 1000:.0f} ms avg over {len(vid_lat)}")

    # ---- verdict -----------------------------------------------------------
    print()
    if false_accepts == 0:
        print(f"OVERALL: PASS - zero false accepts at threshold {args.threshold}")
        return 0
    print(f"OVERALL: FAIL - {false_accepts} false accept(s) at threshold {args.threshold}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
