"""Visual demo for the 5-agent ad moderation pipeline.

    python demo_app.py          then open  http://127.0.0.1:7860

Upload any image/video (plus optional caption), or click a bundled test
sample, and see the accepted/refused verdict with the per-agent breakdown.
"""

import base64
import tempfile
from pathlib import Path

from flask import Flask, abort, render_template_string, request, send_file

from moderation import IMAGE_EXTS, VIDEO_EXTS, load_all_models, moderate_ad_verbose

BASE = Path(__file__).parent
SAMPLES = BASE / "test_samples"
SAMPLE_FOLDERS = ["clean", "nsfw", "agents", "borderline_art"]
SUPPORTED = IMAGE_EXTS | VIDEO_EXTS

app = Flask(__name__)

PAGE = """
<!doctype html>
<title>Ad Moderation Demo</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 880px; margin: 2rem auto; padding: 0 1rem; background:#f7f7fa; color:#1a1a24; }
  h1 { font-size: 1.5rem; } h2 { font-size: 1.1rem; margin-top: 1.6rem; }
  .card { background:#fff; border:1px solid #e2e2ea; border-radius:10px; padding:1rem 1.2rem; margin:.8rem 0; }
  .verdict { font-size:2rem; font-weight:800; padding:.6rem 1.2rem; border-radius:10px; display:inline-block; }
  .accepted { background:#e5f7e9; color:#137a33; border:2px solid #35b95c; }
  .refused  { background:#fdeaea; color:#b01818; border:2px solid #e04545; }
  .chip { display:inline-block; background:#b01818; color:#fff; border-radius:999px; padding:.15rem .7rem; margin:.15rem; font-size:.85rem; }
  table { border-collapse: collapse; width:100%; } td, th { border-bottom:1px solid #eee; padding:.4rem .6rem; text-align:left; font-size:.9rem; }
  .samples a { display:inline-block; margin:.15rem; padding:.3rem .7rem; background:#eef; border:1px solid #ccd; border-radius:6px; text-decoration:none; color:#224; font-size:.85rem; }
  img.preview, video.preview { max-width:340px; max-height:260px; border-radius:8px; border:1px solid #ddd; }
  .lat { color:#666; font-size:.85rem; }
  input[type=text] { width:60%; padding:.4rem; }
  button { padding:.45rem 1.1rem; border-radius:6px; border:0; background:#3457d5; color:#fff; font-weight:600; cursor:pointer; }
</style>
<h1>🛡️ Ad Moderation — live demo</h1>
<div class="card">
  <form method="post" action="/moderate" enctype="multipart/form-data">
    <p><b>Upload an ad</b> (image or video): <input type="file" name="file" required></p>
    <p>Caption / ad text (optional): <input type="text" name="caption" placeholder="e.g. 50% off this week only"></p>
    <button type="submit">Moderate</button>
  </form>
</div>

{% if result %}
<div class="card">
  <p><span class="verdict {{ result.decision }}">{{ result.decision | upper }}</span></p>
  {% if media_tag %}{{ media_tag | safe }}{% endif %}
  <p><b>File:</b> {{ result.file }} <span class="lat">— processed in {{ result.latency_s }}s</span></p>
  {% if result.flagged_by %}
    <p><b>Flagged by:</b> {% for a in result.flagged_by %}<span class="chip">{{ a }}</span>{% endfor %}</p>
  {% else %}
    <p><b>All five checks passed.</b></p>
  {% endif %}
  <table>
    <tr><th>Check</th><th>Result</th></tr>
    {% for name, info in result.checks.items() %}
      <tr><td>{{ name }}</td><td><code>{{ info }}</code></td></tr>
    {% endfor %}
    {% for name, err in result.errors.items() %}
      <tr><td>{{ name }}</td><td style="color:#b01818"><code>error: {{ err }}</code></td></tr>
    {% endfor %}
  </table>
</div>
{% endif %}

<h2>Or click a bundled test sample</h2>
{% for folder, files in samples.items() %}
  <div class="card samples"><b>{{ folder }}/</b><br>
  {% for f in files %}<a href="/sample/{{ folder }}/{{ f }}">{{ f }}</a>{% endfor %}
  </div>
{% endfor %}
<p class="lat">Tip: the "profane caption" check — upload any clean image and type a swear word in the caption box.</p>
"""


def list_samples():
    out = {}
    for folder in SAMPLE_FOLDERS:
        d = SAMPLES / folder
        if d.is_dir():
            out[folder] = [p.name for p in sorted(d.iterdir()) if p.suffix.lower() in SUPPORTED]
    return out


def media_tag_for(path):
    """Inline preview: base64 <img> for images, streaming <video> for samples."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f'<img class="preview" src="data:image;base64,{b64}">'
    try:
        rel = path.relative_to(SAMPLES)
        return f'<video class="preview" src="/media/{rel.as_posix()}" controls muted></video>'
    except ValueError:
        return ""  # uploaded temp video - skip preview, verdict is what matters


@app.route("/")
def index():
    return render_template_string(PAGE, result=None, samples=list_samples(), media_tag=None)


@app.route("/moderate", methods=["POST"])
def moderate_upload():
    f = request.files["file"]
    caption = request.form.get("caption") or None
    suffix = Path(f.filename).suffix.lower() or ".bin"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / ("upload" + suffix)
        f.save(tmp)
        _, details = moderate_ad_verbose(tmp, text=caption)
        details["file"] = f.filename
        tag = media_tag_for(tmp) if suffix in IMAGE_EXTS else ""
    return render_template_string(PAGE, result=details, samples=list_samples(), media_tag=tag)


@app.route("/sample/<folder>/<name>")
def moderate_sample(folder, name):
    if folder not in SAMPLE_FOLDERS or "/" in name or "\\" in name or ".." in name:
        abort(404)
    path = SAMPLES / folder / name
    if not path.is_file():
        abort(404)
    _, details = moderate_ad_verbose(path)
    return render_template_string(
        PAGE, result=details, samples=list_samples(), media_tag=media_tag_for(path)
    )


@app.route("/media/<folder>/<name>")
def media(folder, name):
    if folder not in SAMPLE_FOLDERS or "/" in name or "\\" in name or ".." in name:
        abort(404)
    path = SAMPLES / folder / name
    if not path.is_file():
        abort(404)
    return send_file(path)


if __name__ == "__main__":
    load_all_models()
    print("\nOpen http://127.0.0.1:7860 in your browser.")
    app.run(host="127.0.0.1", port=7860, debug=False)
