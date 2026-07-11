"""HF Space: ad-platform AI tools - mobile 9:16 auto-crop + variant picking.

CPU open-source models only (YOLOv8n + all-MiniLM-L6-v2).
"""

import spaces  # MUST be the first import on ZeroGPU

import json
import tempfile
import uuid
from pathlib import Path

import gradio as gr

from ad_features import (
    generate_mobile_variant,
    get_current_daypart,
    pick_final_variant,
)

EXAMPLE_VARIANTS = json.dumps(
    [
        {"id": "budget", "daypart": "any", "description": "Budget-friendly prices, save big"},
        {"id": "luxury", "daypart": "evening", "description": "Luxury premium experience"},
    ],
    indent=2,
)


def mobile_variant(video_path):
    if not video_path:
        raise gr.Error("Upload a landscape MP4 first.")
    out = Path(tempfile.gettempdir()) / f"mobile_{uuid.uuid4().hex[:10]}.mp4"
    return generate_mobile_variant(video_path, out)


@spaces.GPU(duration=60)
def pick_variant(variants_json, daypart, surrounding_text):
    try:
        variants = json.loads(variants_json)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"variants must be a JSON list: {exc}")
    if not variants:
        raise gr.Error("variants list is empty")
    daypart = daypart or get_current_daypart()
    chosen = pick_final_variant(variants, daypart, surrounding_text or "")
    chosen.pop("embedding", None)  # numpy vector - not JSON-friendly
    return chosen


with gr.Blocks(title="Ad AI Tools") as demo:
    gr.Markdown(
        "# 📱 Ad platform AI tools\n"
        "Mobile 9:16 auto-crop (YOLOv8n subject tracking) and daypart+context "
        "variant selection (MiniLM embeddings). Open-source models on CPU."
    )
    with gr.Tab("Mobile variant (9:16 crop)"):
        v_in = gr.Video(label="Landscape video (MP4)")
        v_btn = gr.Button("Generate mobile version", variant="primary")
        v_out = gr.Video(label="1080x1920 result")
        v_btn.click(mobile_variant, inputs=v_in, outputs=v_out, api_name="mobile_variant")
    with gr.Tab("Pick ad variant"):
        j_in = gr.Textbox(label="Variants (JSON list)", value=EXAMPLE_VARIANTS, lines=10)
        d_in = gr.Dropdown(
            ["", "morning", "afternoon", "evening", "night"],
            label="Daypart (empty = now, UTC)", value="",
        )
        t_in = gr.Textbox(label="Surrounding page text", value="cheap deals for students")
        p_btn = gr.Button("Pick best variant", variant="primary")
        p_out = gr.JSON(label="Chosen variant")
        p_btn.click(pick_variant, inputs=[j_in, d_in, t_in], outputs=p_out, api_name="pick_variant")

demo.launch(ssr_mode=False)
