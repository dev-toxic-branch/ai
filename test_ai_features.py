"""Tests for ad_features.py (Feature 1 mobile crop, 2 dayparting, 3 matching).

Exit code 0 only if every check passes.
"""

import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ad_features import (
    detect_main_subject,
    filter_by_daypart,
    generate_mobile_variant,
    get_current_daypart,
    pick_final_variant,
    precompute_variant_embeddings,
    select_best_matching_variant,
)

SAMPLES = Path(__file__).parent / "test_samples" / "agents"
PERSON_IMG = SAMPLES / "person.jpg"
TEST_VIDEO = SAMPLES / "person_landscape.mp4"
MOBILE_OUT = SAMPLES / "person_mobile.mp4"

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  {'pass' if ok else 'FAIL':<5} {name:<52} {detail}")


# ---------------------------------------------------------------------------
# Test asset builders (person photo from Wikipedia; synthetic landscape video)
# ---------------------------------------------------------------------------
def fetch_person_photo():
    if PERSON_IMG.exists():
        return True
    try:
        from make_test_samples import USER_AGENT, _wiki_lead_image_url

        url = _wiki_lead_image_url("Usain Bolt")
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            PERSON_IMG.write_bytes(resp.read())
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  note: could not fetch person photo ({exc}); center-crop path only")
        return False


def build_test_video(with_person):
    """1920x1080 landscape clip; person (if available) pasted RIGHT of center,
    so a correct crop must shift right to keep them in frame."""
    if TEST_VIDEO.exists():
        return
    w, h, fps, secs = 1920, 1080, 12, 3
    person = None
    if with_person:
        img = cv2.imread(str(PERSON_IMG))
        if img is not None:
            scale = (h * 0.8) / img.shape[0]
            person = cv2.resize(img, (int(img.shape[1] * scale), int(h * 0.8)))
    writer = cv2.VideoWriter(str(TEST_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(fps * secs):
        frame = np.full((h, w, 3), (200, 220, 235), dtype=np.uint8)
        cv2.rectangle(frame, (0, 900), (w, 1080), (90, 140, 90), -1)
        if person is not None:
            ph, pw = person.shape[:2]
            x0 = 1350 - pw // 2  # subject center at x=1350 (right of center)
            y0 = 950 - ph
            frame[y0:y0 + ph, x0:x0 + pw] = person
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
print("FEATURE 1 - auto-reformat (mobile 9:16 crop)")
have_person = fetch_person_photo()
build_test_video(have_person)

t0 = time.perf_counter()
out = generate_mobile_variant(str(TEST_VIDEO), str(MOBILE_OUT))
gen_time = time.perf_counter() - t0

cap = cv2.VideoCapture(out)
out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
ok_frame, mid_frame = cap.read()
cap.release()

check("output is 1080x1920", (out_w, out_h) == (1080, 1920), f"got {out_w}x{out_h}")
print(f"  info  generation time: {gen_time:.1f}s for a 3s clip")

if have_person:
    mid_path = SAMPLES / "_mobile_mid_frame.jpg"
    cv2.imwrite(str(mid_path), mid_frame)
    bbox = detect_main_subject(mid_path)
    check("subject still visible in mobile frame", bbox is not None, f"bbox={bbox}")
    mid_path.unlink(missing_ok=True)
else:
    check("center-crop fallback produced a file", Path(out).stat().st_size > 0, "")

# ---------------------------------------------------------------------------
print("\nFEATURE 2 - dayparting (rule-based)")
mock = [
    {"id": "m", "daypart": "morning"},
    {"id": "a", "daypart": "afternoon"},
    {"id": "e", "daypart": "evening"},
    {"id": "u", "daypart": "any"},
]
morning_ts = datetime(2026, 7, 10, 8, 30, tzinfo=timezone.utc)
dp = get_current_daypart(morning_ts)
check("8:30 UTC is morning", dp == "morning", f"got {dp}")

dp_tokyo = get_current_daypart(datetime(2026, 7, 10, 23, 30, tzinfo=timezone.utc), tz="Asia/Tokyo")
check("23:30 UTC is morning in Tokyo (+9)", dp_tokyo == "morning", f"got {dp_tokyo}")

dp_night = get_current_daypart(datetime(2026, 7, 10, 2, 0, tzinfo=timezone.utc))
check("2:00 UTC is night (midnight wrap)", dp_night == "night", f"got {dp_night}")

ids = sorted(v["id"] for v in filter_by_daypart(mock, "morning"))
check("morning filter keeps morning + any", ids == ["m", "u"], f"got {ids}")

no_match = [{"id": "e1", "daypart": "evening"}, {"id": "e2", "daypart": "evening"}]
kept = filter_by_daypart(no_match, "morning")
check("fail-safe returns full list on zero match", len(kept) == 2, f"kept {len(kept)}")

# ---------------------------------------------------------------------------
print("\nFEATURE 3 - content-context matching (MiniLM embeddings)")
variants = precompute_variant_embeddings([
    {"id": "budget", "description": "Budget-friendly prices, save on your trip"},
    {"id": "luxury", "description": "Luxury premium first-class experience"},
])
check("embeddings precomputed (384-dim)", all(len(v["embedding"]) == 384 for v in variants), "")

winner = select_best_matching_variant("cheap travel tips for students", variants)
from ad_features import compute_similarity, embed_text  # noqa: E402

query = embed_text("cheap travel tips for students")
scores = {v["id"]: compute_similarity(query, v["embedding"]) for v in variants}
check("budget variant wins for 'cheap travel tips'", winner["id"] == "budget",
      f"scores={ {k: round(s, 3) for k, s in scores.items()} }")

fallback = select_best_matching_variant("", variants)
check("empty text falls back to first variant", fallback["id"] == "budget" and fallback["match_score"] is None, "")

single = select_best_matching_variant("anything", [variants[1]])
check("single variant skips matching", single["id"] == "luxury" and single["match_score"] is None, "")

# ---------------------------------------------------------------------------
print("\nCOMBINED - pick_final_variant (daypart then context)")
full_set = precompute_variant_embeddings([
    {"id": "m-budget", "daypart": "morning", "description": "Cheap breakfast deals and low prices"},
    {"id": "m-luxury", "daypart": "morning", "description": "Exclusive gourmet morning experience"},
    {"id": "e-budget", "daypart": "evening", "description": "Discount dinner offers"},
    {"id": "any-generic", "daypart": "any", "description": "Great products all day"},
])

t0 = time.perf_counter()
final = pick_final_variant(full_set, "morning", "student saving money on food")
pick_time = time.perf_counter() - t0

check("evening variant eliminated, budget match wins", final["id"] == "m-budget",
      f"got {final['id']} score={final['match_score'] and round(final['match_score'], 3)}")
check("end-to-end under 1 second", pick_time < 1.0, f"{pick_time * 1000:.0f} ms")

single_after_filter = pick_final_variant(
    [{"id": "only-m", "daypart": "morning"}, {"id": "e", "daypart": "evening"}],
    "morning", "whatever",
)
check("single survivor skips matching (score None)",
      single_after_filter["id"] == "only-m" and single_after_filter["match_score"] is None, "")

# ---------------------------------------------------------------------------
print(f"\nOVERALL: {'PASS' if all(results) else 'FAIL'} ({sum(results)}/{len(results)})")
sys.exit(0 if all(results) else 1)
