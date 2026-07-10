"""Build the test_samples/ folder structure.

test_samples/clean/ - synthetic ad-like images and short clips generated locally.
test_samples/nsfw/  - public-domain artistic nudes fetched from Wikimedia Commons
                      (safe benchmark stand-ins for explicit content; real NSFW
                      detectors are routinely evaluated on artistic nudity).

Safe to re-run: existing files are kept, missing ones are recreated.
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).parent / "test_samples"
CLEAN = BASE / "clean"
NSFW = BASE / "nsfw"

USER_AGENT = "hackiwha-ai-moderation-test/1.0 (hackathon demo; test sample fetch)"

# Wikipedia articles whose lead image is a public-domain artistic nude.
# We pull each page's lead image via the MediaWiki pageimages API.
NSFW_WIKI_PAGES = [
    "La maja desnuda",
    "Venus of Urbino",
    "Olympia (Manet)",
    "Grande Odalisque",
    "The Birth of Venus",
    "Rokeby Venus",
]


def make_clean_images():
    CLEAN.mkdir(parents=True, exist_ok=True)

    # Landscape: sky gradient, sun, grass
    img = Image.new("RGB", (640, 480))
    draw = ImageDraw.Draw(img)
    for y in range(480):
        blue = int(235 - y * 0.2)
        draw.line([(0, y), (640, y)], fill=(120, 170, blue))
    draw.rectangle([0, 360, 640, 480], fill=(60, 140, 60))
    draw.ellipse([480, 40, 560, 120], fill=(255, 230, 100))
    img.save(CLEAN / "landscape.jpg")

    # Product-ad mockup: bottle-ish shape on a banner
    img = Image.new("RGB", (640, 480), (245, 245, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 640, 90], fill=(30, 60, 160))
    draw.text((20, 30), "SUMMER SALE  -50%", fill=(255, 255, 255))
    draw.rounded_rectangle([270, 160, 370, 420], radius=30, fill=(200, 40, 60))
    draw.rectangle([300, 120, 340, 170], fill=(150, 30, 45))
    draw.rectangle([285, 230, 355, 330], fill=(255, 255, 255))
    draw.text((295, 270), "BRAND", fill=(20, 20, 20))
    img.save(CLEAN / "product_ad.jpg")

    # Food ad: pizza-like disc with toppings
    img = Image.new("RGB", (640, 480), (250, 240, 220))
    draw = ImageDraw.Draw(img)
    draw.ellipse([140, 60, 500, 420], fill=(230, 180, 90))
    draw.ellipse([170, 90, 470, 390], fill=(200, 60, 40))
    rng = np.random.default_rng(7)
    for _ in range(14):
        x, y = rng.integers(210, 400), rng.integers(130, 320)
        draw.ellipse([x, y, x + 34, y + 34], fill=(240, 220, 160))
    draw.text((240, 440), "HOT & FRESH DELIVERY", fill=(80, 40, 20))
    img.save(CLEAN / "food_ad.jpg")

    # Tech ad: dark banner with phone silhouette
    img = Image.new("RGB", (640, 480), (18, 18, 24))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([250, 90, 390, 390], radius=24, fill=(40, 40, 55))
    draw.rounded_rectangle([262, 110, 378, 350], radius=10, fill=(70, 130, 220))
    draw.text((250, 420), "THE NEW PHONE X", fill=(230, 230, 230))
    img.save(CLEAN / "tech_ad.jpg")


def make_clean_videos():
    """Two short clips: a bouncing-logo ad and a color-fade banner."""
    specs = [
        ("bouncing_logo.mp4", _bouncing_logo_frames),
        ("color_fade_banner.mp4", _color_fade_frames),
    ]
    for name, frame_fn in specs:
        out = CLEAN / name
        if out.exists():
            continue
        w, h, fps = 320, 240, 12
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frame_fn(w, h, n_frames=fps * 3):
            writer.write(frame)
        writer.release()
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError(f"Failed to write video {out}")


def _bouncing_logo_frames(w, h, n_frames):
    x, y, dx, dy, r = 60, 60, 7, 5, 24
    for _ in range(n_frames):
        frame = np.full((h, w, 3), (240, 235, 225), dtype=np.uint8)
        cv2.circle(frame, (x, y), r, (60, 60, 200), -1)
        cv2.putText(frame, "AD", (x - 14, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        x, y = x + dx, y + dy
        if x - r <= 0 or x + r >= w:
            dx = -dx
        if y - r <= 0 or y + r >= h:
            dy = -dy
        yield frame


def _color_fade_frames(w, h, n_frames):
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        color = (int(200 * (1 - t) + 40 * t), int(80 + 120 * t), int(60 + 100 * (1 - t)))
        frame = np.full((h, w, 3), color, dtype=np.uint8)
        cv2.putText(frame, "BIG SALE", (60, 130), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 3)
        yield frame


def _wiki_lead_image_url(page_title):
    api = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json"
        "&prop=pageimages&piprop=thumbnail&pithumbsize=800&titles="
        + urllib.parse.quote(page_title)
    )
    req = urllib.request.Request(api, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        thumb = page.get("thumbnail", {})
        if "source" in thumb:
            return thumb["source"]
    return None


def fetch_nsfw_samples():
    NSFW.mkdir(parents=True, exist_ok=True)
    readme = NSFW / "README.md"
    if not readme.exists():
        readme.write_text(
            "# NSFW test samples\n\n"
            "These are public-domain artistic nudes (classic paintings) pulled from\n"
            "Wikimedia Commons - safe, legal stand-ins used to exercise the NSFW\n"
            "detector without storing explicit content in the repo.\n\n"
            "For a stricter evaluation before the demo, drop a few samples from a\n"
            "real NSFW benchmark set into this folder; the test script picks up\n"
            "every image/video here automatically.\n"
        )

    fetched, failed = [], []
    for title in NSFW_WIKI_PAGES:
        slug = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out = NSFW / f"{slug}.jpg"
        # don't refetch samples that were re-triaged into borderline_art/
        if out.exists() or (BASE / "borderline_art" / out.name).exists():
            fetched.append(out.name)
            continue
        try:
            url = _wiki_lead_image_url(title)
            if not url:
                raise ValueError("no lead image found")
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            out.write_bytes(data)
            # sanity check it decodes as an image
            Image.open(out).verify()
            fetched.append(out.name)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed.append((title, str(exc)))
            if out.exists():
                out.unlink()
    return fetched, failed


AGENTS = BASE / "agents"


def make_agent_samples():
    """Per-agent positives: profane overlay (OCR), profane speech (Whisper),
    weapon photo (YOLO), corrupted file (fail-safe path)."""
    import subprocess

    from PIL import ImageFont

    AGENTS.mkdir(parents=True, exist_ok=True)

    # Agent 3: big clear profane text on a plain banner (TrOCR-friendly)
    out = AGENTS / "profane_overlay.jpg"
    if not out.exists():
        img = Image.new("RGB", (640, 200), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arialbd.ttf", 72)
        except OSError:
            font = ImageFont.load_default()
        draw.text((30, 60), "SHIT HAPPENS", fill=(10, 10, 10), font=font)
        img.save(out, quality=95)

    # Fail-safe path: right extension, garbage bytes
    out = AGENTS / "corrupted.jpg"
    if not out.exists():
        out.write_bytes(b"this is definitely not a jpeg " * 100)

    # Agent 5: clean visuals + spoken profanity via Windows TTS, muxed with ffmpeg
    out = AGENTS / "profane_audio.mp4"
    if not out.exists():
        from imageio_ffmpeg import get_ffmpeg_exe

        wav = AGENTS / "_tts.wav"
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{wav}'); "
            "$s.Speak('Buy this shit right now, it is a damn good deal'); "
            "$s.Dispose()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", script],
                       check=True, timeout=120)
        subprocess.run(
            [get_ffmpeg_exe(), "-y", "-i", str(CLEAN / "bouncing_logo.mp4"),
             "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)],
            check=True, capture_output=True, timeout=120,
        )
        wav.unlink(missing_ok=True)

    # Agent 2: real handgun photo (clean product shot, no nudity) from Wikipedia
    out = AGENTS / "weapon.jpg"
    if not out.exists():
        try:
            url = _wiki_lead_image_url("Glock")
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                out.write_bytes(resp.read())
            Image.open(out).verify()
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED to fetch weapon photo: {exc}")
            if out.exists():
                out.unlink()


if __name__ == "__main__":
    make_clean_images()
    make_clean_videos()
    print(f"clean/: {sorted(p.name for p in CLEAN.iterdir())}")
    fetched, failed = fetch_nsfw_samples()
    print(f"nsfw/:  {fetched}")
    for title, err in failed:
        print(f"  FAILED to fetch '{title}': {err}")
    make_agent_samples()
    print(f"agents/: {sorted(p.name for p in AGENTS.iterdir())}")
