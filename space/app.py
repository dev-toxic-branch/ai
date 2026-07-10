"""Hugging Face Space front-end for the 5-agent ad moderation pipeline.

Runs entirely on the Space's free CPU with open-source models - no external
LLM/vision APIs. Models download from the HF hub on first startup.
"""

import gradio as gr

from moderation import load_all_models, moderate_ad_verbose

print("Loading models (first startup downloads them - a few minutes)...")
load_all_models()


def moderate(file_path, caption):
    if not file_path:
        return "no file", {}
    decision, details = moderate_ad_verbose(file_path, text=caption or None)
    label = "✅ ACCEPTED" if decision == "accepted" else "🚫 REFUSED"
    return label, details


demo = gr.Interface(
    fn=moderate,
    inputs=[
        gr.File(label="Ad image or video", type="filepath"),
        gr.Textbox(label="Caption / ad text (optional)"),
    ],
    outputs=[
        gr.Textbox(label="Decision"),
        gr.JSON(label="Per-agent breakdown (admin view)"),
    ],
    title="🛡️ Ad Content Moderation — 5 open-source agents",
    description=(
        "Nudity (Falconsai ViT) · weapons/violence (YOLOv8n) · text-in-image "
        "profanity (TrOCR) · caption profanity (better-profanity) · spoken "
        "profanity (Whisper base). If ANY agent flags, the ad is refused. "
        "All models run locally on this Space's CPU."
    ),
    examples=[
        ["samples/product_ad.jpg", ""],
        ["samples/weapon.jpg", ""],
        ["samples/profane_overlay.jpg", ""],
        ["samples/profane_audio.mp4", ""],
        ["samples/landscape.jpg", "get this fucking deal"],  # caption-check demo
    ],
    cache_examples=False,
    flagging_mode="never",
)

demo.launch()
