---
title: Ad AI Tools — Mobile Crop & Variant Picker
emoji: 📱
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 📱 Ad platform AI tools

- **Mobile variant** — YOLOv8-nano tracks the main subject across the video,
  one smooth 9:16 crop, ffmpeg-scaled to 1080×1920.
- **Pick ad variant** — filters variants by daypart (rule-based, timezone-aware)
  then picks the best semantic match for the surrounding page text
  (all-MiniLM-L6-v2 embeddings).

API via `gradio_client`: `api_name="/mobile_variant"` and `"/pick_variant"`.
