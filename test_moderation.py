"""Test suite for the 5-agent ad moderation pipeline.

Part 1 - isolated unit test per agent (each agent callable on its own).
Part 2 - integration tests through moderate_ad(): every scenario from the spec,
         with a results table showing which agent flagged each file.
Part 3 - per-agent average latency + end-to-end time per file.

Exit code 0 only if every test passes.
"""

import statistics
import sys
import time
from pathlib import Path

from moderation import (
    check_nudity,
    check_speech_profanity,
    check_text_profanity,
    check_threat,
    extract_image_text,
    has_audio_track,
    load_all_models,
    moderate_ad_verbose,
)

SAMPLES = Path(__file__).parent / "test_samples"
CLEAN = SAMPLES / "clean"
NSFW = SAMPLES / "nsfw"
AGENTS = SAMPLES / "agents"

agent_latency = {}  # agent name -> list of seconds


def timed(agent, fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    agent_latency.setdefault(agent, []).append(time.perf_counter() - t0)
    return out


def nudity_sample():
    """A real NSFW-positive image from test_samples/nsfw/."""
    for p in sorted(NSFW.iterdir()):
        if p.suffix.lower() in {".webp", ".jpg", ".jpeg", ".png"} and "README" not in p.name:
            best = p
            if p.suffix.lower() == ".webp":  # the photographic sample, strongest positive
                return p
    return best


# ---------------------------------------------------------------------------
# Part 1: one isolated test per agent
# ---------------------------------------------------------------------------
def unit_tests():
    rows = []

    def check(name, ok, detail=""):
        rows.append((name, ok, detail))

    flagged, score = timed("1 nudity", check_nudity, CLEAN / "landscape.jpg")
    check("A1 nudity: clean image not flagged", not flagged, f"score={score:.3f}")
    flagged, score = timed("1 nudity", check_nudity, nudity_sample())
    check("A1 nudity: nsfw image flagged", flagged, f"score={score:.3f}")

    flagged, conf, classes = timed("2 threat", check_threat, CLEAN / "landscape.jpg")
    check("A2 threat: clean image not flagged", not flagged, f"classes={classes}")
    weapon = AGENTS / "weapon.jpg"
    if weapon.exists():
        flagged, conf, classes = timed("2 threat", check_threat, weapon)
        check("A2 threat: weapon photo flagged", flagged, f"conf={conf:.2f} classes={classes}")
    else:
        check("A2 threat: weapon photo flagged", False, "weapon.jpg missing - rerun make_test_samples.py")

    text = timed("3 ocr", extract_image_text, AGENTS / "profane_overlay.jpg")
    check("A3 ocr: reads overlay text", "shit" in text.lower(), f"read={text!r}")
    text = timed("3 ocr", extract_image_text, CLEAN / "landscape.jpg")
    check("A3 ocr: textless image handled", isinstance(text, str), f"read={text!r}")

    check("A4 profanity: catches swear", timed("4 text", check_text_profanity, "what the fuck"), "")
    check("A4 profanity: clean text passes", not timed("4 text", check_text_profanity, "family picnic day"), "")
    check("A4 profanity: None-safe", not check_text_profanity(None), "")

    check("A5 speech: detects no-audio track", not has_audio_track(CLEAN / "bouncing_logo.mp4"), "")
    check("A5 speech: detects audio track", has_audio_track(AGENTS / "profane_audio.mp4"), "")
    flagged, transcript = timed("5 speech", check_speech_profanity, AGENTS / "profane_audio.mp4")
    check("A5 speech: profane audio flagged", flagged, f"transcript={transcript!r}")

    return rows


# ---------------------------------------------------------------------------
# Part 2: integration through moderate_ad()
# ---------------------------------------------------------------------------
def integration_cases():
    return [
        # (test name, file, text, expected decision, agent expected to flag or None)
        ("clean image, no text", CLEAN / "product_ad.jpg", None, "accepted", None),
        ("clean video, no text", CLEAN / "color_fade_banner.mp4", None, "accepted", None),
        ("nudity image", nudity_sample(), None, "refused", "nudity"),
        ("weapon visible, no nudity", AGENTS / "weapon.jpg", None, "refused", "threat"),
        ("profane text overlaid on image", AGENTS / "profane_overlay.jpg", None, "refused", "ocr_profanity"),
        ("profane ad caption", CLEAN / "landscape.jpg", "get this fucking deal", "refused", "text_profanity"),
        ("profane spoken audio", AGENTS / "profane_audio.mp4", None, "refused", "speech_profanity"),
        ("corrupted file", AGENTS / "corrupted.jpg", None, "refused", "unreadable_file"),
    ]


def main():
    print("=" * 76)
    load_all_models()

    print()
    print("PART 1 - isolated agent tests")
    print("-" * 76)
    all_ok = True
    for name, ok, detail in unit_tests():
        all_ok &= ok
        print(f"  {'pass' if ok else 'FAIL':<5} {name:<44} {detail}")

    print()
    print("PART 2 - integration via moderate_ad()")
    header = f"{'test':<34} {'expected':<9} {'actual':<9} {'flagged by':<28} {'sec':>5}  result"
    print(header)
    print("-" * len(header))
    for name, path, text, expected, expect_agent in integration_cases():
        decision, details = moderate_ad_verbose(path, text=text)
        ok = decision == expected
        if ok and expect_agent is not None:
            ok = expect_agent in details["flagged_by"]
        all_ok &= ok
        flagged = ",".join(details["flagged_by"]) or "-"
        print(f"{name:<34} {expected:<9} {decision:<9} {flagged:<28} "
              f"{details['latency_s']:>5.1f}  {'pass' if ok else 'FAIL'}")

    print()
    print("PART 3 - per-agent average latency (isolated calls)")
    for agent in sorted(agent_latency):
        lat = agent_latency[agent]
        print(f"  agent {agent:<10} avg {statistics.mean(lat) * 1000:7.0f} ms over {len(lat)} call(s)")

    print()
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
