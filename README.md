# Ad Moderation — NSFW detection for ad creatives

Moderates ad images and short video clips: returns **ACCEPT** or **REFUSE**
with an NSFW score (0–1) and per-call latency. Built for a live demo.

## Quick start

```powershell
# 1. (once, needs internet) fetch the real classifier — resumable, survives bad wifi
python download_model.py

# 2. run the full evaluation: results table, accuracy, threshold sweep, latency
python test_moderation.py

# 3. moderate any single file (demo command)
python moderation.py path\to\ad_image.jpg
python moderation.py path\to\ad_clip.mp4
```

## How it works

- `moderation.py` — `moderate_ad(path, threshold)` scores one image/video.
  Videos: up to 8 evenly-sampled frames, worst frame wins.
  Two backends, picked automatically:
  - **vit-model** — `Falconsai/nsfw_image_detection` ViT classifier, used when
    `models/nsfw_image_detection/model.safetensors` is complete. Accurate.
  - **skin-heuristic** — offline OpenCV skin-ratio fallback used until the
    model is downloaded. Stopgap only: use `--threshold 0.5` with it, and
    expect mistakes on skin-toned content (e.g. food close-ups, portraits).
- `test_moderation.py` — runs every file in `test_samples/clean/` (expected
  ACCEPT) and `test_samples/nsfw/` (expected REFUSE). Prints per-file results,
  accuracy, **false accepts** (NSFW that slipped through — the error that
  matters most), false refuses, a threshold sweep (0.5/0.6/0.7/0.8) and average
  latency. Exit code 0 only if false accepts == 0.
- `make_test_samples.py` — regenerates the test set: synthetic clean ads/clips,
  plus public-domain artistic nudes from Wikimedia Commons as safe NSFW
  stand-ins. Drop your own files into any folder; the test picks them up.
- Test folders: `test_samples/clean/` (expected ACCEPT), `test_samples/nsfw/`
  (expected REFUSE, counts toward false accepts), and
  `test_samples/borderline_art/` (classical paintings the ViT model
  intentionally treats as art, not porn — scored and printed for information,
  but pass/fail is a policy decision, so they are not counted).
- `download_model.py` — chunked, endlessly-retrying model downloader for
  unreliable networks. Rerunning always resumes, never restarts.

## Picking the threshold

Run `python test_moderation.py` and read the sweep table: choose the highest
threshold that still gives **0 false accepts**. Pass it in the demo via
`moderate_ad(file, threshold=...)` or `python test_moderation.py --threshold 0.6`.
