# Ad Content Filtration — 5 local open-source agents

Moderates video/image ads and returns exactly `"accepted"` or `"refused"`.
**Everything runs locally on CPU — no Claude/OpenAI/hosted APIs anywhere.**

| # | Check | Model | Size |
|---|-------|-------|------|
| 1 | Nudity | Falconsai/nsfw_image_detection (ViT) | ~330 MB |
| 2 | Violence / weapons | Subh775/Threat-Detection-YOLOv8n (nano) | ~6 MB |
| 3 | Text-in-image profanity | microsoft/trocr-base-printed (OCR) | ~1.3 GB |
| 4 | Caption/metadata profanity | better-profanity wordlist | 0 (no model) |
| 5 | Spoken profanity | openai-whisper **base** (hard-coded) | ~145 MB |

If ANY agent flags, the ad is refused. Fail-safe: corrupted files or agent
crashes also refuse (log the error, never fail open).

## Setup (once)

```powershell
pip install --user ultralytics better-profanity openai-whisper imageio-ffmpeg
python download_model.py     # Falconsai NSFW weights
python download_models.py    # TrOCR + YOLO + Whisper weights (resumable, flaky-network-proof)
python make_test_samples.py  # generate/fetch the test set
```

ffmpeg comes from `imageio-ffmpeg` (bundled static binary) — no system install.

## Use

```python
from moderation import moderate_ad, moderate_ad_verbose, load_all_models

load_all_models()                     # once at app startup, prints summary
moderate_ad("ad.mp4")                 # -> "accepted" | "refused"
moderate_ad("poster.jpg", text="50% off!")   # caption checked too
decision, details = moderate_ad_verbose("ad.mp4")  # details for admin dashboard:
# details["flagged_by"], details["checks"], details["errors"], details["latency_s"]
```

Videos: 3 frames (start/middle/end) are extracted once and shared by agents
1–3; audio is transcribed only if an audio track exists. Each agent is a lazy
singleton, so any single agent can be demoed in isolation:

```python
from moderation import check_threat
print(check_threat("frame.jpg"))      # loads only YOLO, nothing else
```

## Test

```powershell
python test_moderation.py   # unit test per agent + integration table + latency
```

Exit code 0 = every check passed. The table shows which agent flagged each file.

Test data: `test_samples/clean/` (synthetic ads), `test_samples/nsfw/`
(NSFW positives), `test_samples/agents/` (weapon photo, profane overlay,
TTS profane audio, corrupted file), `test_samples/borderline_art/`
(classical paintings — informational only, a policy call, not counted).
