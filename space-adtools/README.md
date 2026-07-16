---
title: Ad Variant Picker
emoji: 🎯
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 🎯 Ad variant picker

Filters ad variants by the viewer's time of day — pass the viewer's IANA
timezone (e.g. `Africa/Algiers`) and the daypart is computed automatically —
then picks the best semantic match for the surrounding page text
(all-MiniLM-L6-v2 embeddings, rule-based dayparting, open-source, CPU).

API via `gradio_client`: `api_name="/pick_variant"` with
(variants_json, timezone, daypart_override, surrounding_text).
