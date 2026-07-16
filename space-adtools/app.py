"""HF Space: ad-variant picker - daypart (timezone-aware) + context matching.

CPU open-source model only (all-MiniLM-L6-v2).
"""

import spaces  # MUST be the first import on ZeroGPU

import json

import gradio as gr

from ad_features import get_current_daypart, pick_final_variant

EXAMPLE_VARIANTS = json.dumps(
    [
        {"id": "budget", "daypart": "any", "description": "Budget-friendly prices, save big"},
        {"id": "luxury", "daypart": "evening", "description": "Luxury premium experience"},
        {"id": "breakfast", "daypart": "morning", "description": "Fresh morning coffee deals"},
    ],
    indent=2,
)


@spaces.GPU(duration=60)
def pick_variant(variants_json, timezone_name, daypart, surrounding_text):
    try:
        variants = json.loads(variants_json)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"variants must be a JSON list: {exc}")
    if not variants:
        raise gr.Error("variants list is empty")

    # explicit daypart wins; otherwise derive it from the viewer's timezone
    if not daypart:
        try:
            daypart = get_current_daypart(tz=timezone_name or "UTC")
        except Exception as exc:  # noqa: BLE001 - bad tz name
            raise gr.Error(f"unknown timezone '{timezone_name}': {exc}")

    chosen = pick_final_variant(variants, daypart, surrounding_text or "")
    chosen.pop("embedding", None)  # numpy vector - not JSON-friendly
    chosen["daypart_used"] = daypart
    return chosen


with gr.Blocks(title="Ad Variant Picker") as demo:
    gr.Markdown(
        "# 🎯 Ad variant picker\n"
        "Filters ad variants by the viewer's time of day (timezone-aware), then "
        "picks the best semantic match for the surrounding page text "
        "(all-MiniLM-L6-v2 embeddings, open-source, on-Space)."
    )
    j_in = gr.Textbox(label="Variants (JSON list)", value=EXAMPLE_VARIANTS, lines=12)
    tz_in = gr.Textbox(
        label="Viewer timezone (IANA name, e.g. Africa/Algiers, Europe/Paris)",
        value="UTC",
    )
    d_in = gr.Dropdown(
        ["", "morning", "afternoon", "evening", "night"],
        label="Daypart override (leave empty to compute from timezone)",
        value="",
    )
    t_in = gr.Textbox(label="Surrounding page text", value="cheap deals for students")
    p_btn = gr.Button("Pick best variant", variant="primary")
    p_out = gr.JSON(label="Chosen variant")
    p_btn.click(
        pick_variant,
        inputs=[j_in, tz_in, d_in, t_in],
        outputs=p_out,
        api_name="pick_variant",
    )

demo.launch(ssr_mode=False)
