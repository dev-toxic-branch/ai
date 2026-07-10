"""Hugging Face Space front-end for the 5-agent ad moderation pipeline.

Runs on ZeroGPU (or CPU) with open-source models only - no external APIs.
Models download from the HF hub on first startup.
"""

import spaces  # MUST be the first import on ZeroGPU (patches CUDA init)

from pathlib import Path

import gradio as gr

from moderation import load_all_models, moderate_ad_verbose

print("Loading models (first startup downloads them - a few minutes)...")
load_all_models()


@spaces.GPU(duration=120)
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
        "All models run locally on this Space."
    ),
    # only offer example rows whose file actually exists on the Space
    examples=[
        [f, cap]
        for f, cap in [
            ["samples/product_ad.jpg", ""],
            ["samples/weapon.jpg", ""],
            ["samples/profane_overlay.jpg", ""],
            ["samples/profane_audio.mp4", ""],
            ["samples/landscape.jpg", "get this fucking deal"],  # caption-check demo
        ]
        if Path(f).is_file()
    ]
    or None,
    cache_examples=False,
    flagging_mode="never",
)

demo.launch(ssr_mode=False)
