---
title: Ad Content Moderation
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 🛡️ Ad Content Moderation — 5 open-source agents

Moderates video/image ads and returns **accepted** or **refused**, with a
per-agent breakdown. Everything runs on this Space's free CPU with
open-source models — no hosted LLM/vision APIs:

1. **Nudity** — Falconsai/nsfw_image_detection (ViT)
2. **Weapons / violence** — Subh775/Threat-Detection-YOLOv8n
3. **Text-in-image profanity** — microsoft/trocr-base-printed + better-profanity
4. **Caption profanity** — better-profanity
5. **Spoken profanity** — openai-whisper "base"

Videos: 3 frames (start/middle/end) are shared by agents 1–3; audio is
transcribed only when a track exists. Any flag ⇒ refused; corrupt files and
agent errors also refuse (fail-safe). Built for a hackathon demo.
