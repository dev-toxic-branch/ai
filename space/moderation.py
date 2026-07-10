"""Content filtration for a video/image ad platform - 5 local open-source agents.

    Agent 1  check_nudity           Falconsai/nsfw_image_detection (ViT)
    Agent 2  check_threat           Subh775/Threat-Detection-YOLOv8n (YOLOv8-nano)
    Agent 3  extract_image_text     microsoft/trocr-base-printed (OCR)
    Agent 4  check_text_profanity   better-profanity wordlist (no model)
    Agent 5  check_speech_profanity openai-whisper "base" + Agent 4

Everything runs locally on CPU. No hosted LLM/vision APIs anywhere.

Public entry point:
    moderate_ad(file_path, text=None) -> "accepted" | "refused"
Verbose variant (same decision plus per-agent details for the admin dashboard):
    moderate_ad_verbose(file_path, text=None) -> (decision, details_dict)

Fail-safe policy: a corrupted file, or any agent crashing, counts as a flag
(refuse) - moderation errors must never let an ad slip through unreviewed.

Run `python download_model.py` and `python download_models.py` once to fetch
all weights; both are resumable and safe on flaky networks.
"""

import logging
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("moderation")

BASE_DIR = Path(__file__).parent
NSFW_DIR = BASE_DIR / "models" / "nsfw_image_detection"
TROCR_DIR = BASE_DIR / "models" / "trocr_base_printed"
YOLO_WEIGHTS = BASE_DIR / "models" / "threat_yolov8n" / "best.pt"

# 0.7: on our test set clean ads score <= 0.01 and real NSFW >= 0.99, so 0.7
# sits in the empty middle with a wide margin against both error types.
NUDITY_THRESHOLD = 0.7
# YOLO detections below 0.5 confidence on a nano model are mostly noise.
THREAT_CONF_THRESHOLD = 0.5
N_VIDEO_FRAMES = 3  # start / middle / end

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ---------------------------------------------------------------------------
# Model loading - each agent has a lazy singleton so it can be demoed alone;
# load_all_models() eager-loads everything once at app startup with a summary.
# ---------------------------------------------------------------------------
_loaded = {}


def _load_nudity():
    # Falconsai ViT-base (~330MB): the reference open NSFW classifier; only size offered.
    from transformers import pipeline

    src = str(NSFW_DIR) if NSFW_DIR.is_dir() else "Falconsai/nsfw_image_detection"
    return pipeline("image-classification", model=src, device=-1)  # -1 = CPU


def _load_threat():
    # YOLOv8-NANO variant (~6MB): smallest YOLO size, real-time even on 4 CPU cores.
    from ultralytics import YOLO

    if YOLO_WEIGHTS.is_file():
        weights = str(YOLO_WEIGHTS)
    else:  # no local copy (e.g. running on HF Spaces) - fetch from the hub
        from huggingface_hub import hf_hub_download

        weights = hf_hub_download("Subh775/Threat-Detection-YOLOv8n", "weights/best.pt")
    return YOLO(weights)


def _load_ocr():
    # trocr-BASE-printed (~1.3GB) as specified; 'small' exists but base-printed is
    # the smallest variant with reliable accuracy on clean poster/overlay text.
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    src = (
        str(TROCR_DIR)
        if (TROCR_DIR / "model.safetensors").is_file()
        else "microsoft/trocr-base-printed"  # hub fallback for cloud hosting
    )
    processor = TrOCRProcessor.from_pretrained(src)
    model = VisionEncoderDecoderModel.from_pretrained(src)
    model.eval()
    return processor, model


def _load_profanity():
    # Pure-Python wordlist, zero download, microseconds per check.
    from better_profanity import profanity

    profanity.load_censor_words()
    return profanity


def _load_whisper():
    # Whisper "base" (~145MB, ~1GB RAM): hard-coded - small/medium/large are too
    # slow for a live demo on a 4-core CPU.
    import whisper

    return whisper.load_model("base", device="cpu")


_LOADERS = {
    "nudity (Falconsai ViT)": ("nudity", _load_nudity),
    "threat (YOLOv8n)": ("threat", _load_threat),
    "ocr (TrOCR base printed)": ("ocr", _load_ocr),
    "profanity (better-profanity)": ("profanity", _load_profanity),
    "speech (Whisper base)": ("whisper", _load_whisper),
}


def _get(key):
    """Lazy singleton per model, so one agent can be tested without the others."""
    if key not in _loaded:
        for label, (k, loader) in _LOADERS.items():
            if k == key:
                _loaded[key] = loader()
                break
    return _loaded[key]


def load_all_models():
    """Eager-load every model once at startup and print a summary table.

    Raises RuntimeError naming the model if any fails - a silently missing
    agent must not weaken moderation.
    """
    import torch

    if torch.cuda.is_available():
        log.warning("A GPU is available and may be used - expected CPU-only mode!")
    else:
        print("No GPU detected: all models run in CPU mode (as intended).")

    print("Loading moderation models:")
    for label, (key, loader) in _LOADERS.items():
        if key in _loaded:
            print(f"  {label:<32} already loaded")
            continue
        t0 = time.perf_counter()
        try:
            _loaded[key] = loader()
        except Exception as exc:
            raise RuntimeError(f"FAILED to load model '{label}': {exc}") from exc
        print(f"  {label:<32} loaded in {time.perf_counter() - t0:5.1f}s (cpu)")


# ---------------------------------------------------------------------------
# Agent 1 - nudity
# ---------------------------------------------------------------------------
def check_nudity(image_path):
    """Returns (flagged, nsfw_confidence 0-1)."""
    out = _get("nudity")(str(image_path), top_k=None)
    score = next((float(e["score"]) for e in out if e["label"].lower() == "nsfw"), 0.0)
    return score >= NUDITY_THRESHOLD, score


# ---------------------------------------------------------------------------
# Agent 2 - violence / weapons
# ---------------------------------------------------------------------------
def check_threat(image_path):
    """Returns (flagged, max_confidence, detected_class_names)."""
    results = _get("threat").predict(
        str(image_path), device="cpu", conf=THREAT_CONF_THRESHOLD, verbose=False
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return False, 0.0, []
    classes = [results[0].names[int(c)] for c in boxes.cls.tolist()]
    return True, float(boxes.conf.max()), classes


# ---------------------------------------------------------------------------
# Agent 3 - text-in-image (OCR) profanity
# ---------------------------------------------------------------------------
def extract_image_text(image_path):
    """OCR the image; returns "" cleanly when no text is found."""
    import torch

    processor, model = _get("ocr")
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = model.generate(pixel_values, max_new_tokens=32)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    # TrOCR emits lone punctuation on textless images - treat that as "no text"
    return text if any(ch.isalnum() for ch in text) else ""


# ---------------------------------------------------------------------------
# Agent 4 - metadata text profanity (also reused by Agents 3 and 5)
# ---------------------------------------------------------------------------
def check_text_profanity(text):
    """Returns True if text contains profanity. None/empty-safe."""
    if not text or not text.strip():
        return False
    return _get("profanity").contains_profanity(text)


# ---------------------------------------------------------------------------
# Agent 5 - speech profanity
# ---------------------------------------------------------------------------
def _ffmpeg_exe():
    # imageio-ffmpeg ships a static ffmpeg binary - no system install needed.
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()


def has_audio_track(video_path):
    """True if the container has an audio stream (checked before any decoding)."""
    proc = subprocess.run(
        [_ffmpeg_exe(), "-hide_banner", "-i", str(video_path)],
        capture_output=True, text=True, timeout=60,
    )
    return "Audio:" in proc.stderr  # ffmpeg lists streams on stderr


def transcribe_audio(video_path):
    """Extract mono 16kHz audio with ffmpeg, then transcribe with Whisper base."""
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.wav"
        subprocess.run(
            [_ffmpeg_exe(), "-y", "-i", str(video_path),
             "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)],
            capture_output=True, timeout=300, check=True,
        )
        # Hand Whisper a numpy array so it never needs its own ffmpeg on PATH
        with wave.open(str(wav), "rb") as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        audio = audio.astype(np.float32) / 32768.0
    result = _get("whisper").transcribe(audio, fp16=False)  # fp16 needs a GPU
    return result["text"].strip()


def check_speech_profanity(video_path):
    """Returns (flagged, transcript). Skips silently if there is no audio track."""
    if not has_audio_track(video_path):
        return False, ""
    transcript = transcribe_audio(video_path)
    return check_text_profanity(transcript), transcript


# ---------------------------------------------------------------------------
# Frame extraction (shared by Agents 1, 2, 3 - extracted once per video)
# ---------------------------------------------------------------------------
def extract_frames(video_path, tmpdir, n=N_VIDEO_FRAMES):
    """Save n representative frames (start/middle/end) as JPGs, return paths."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        indices = sorted({0, total // 2, total - 1})[:n]
        paths = []
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            p = Path(tmpdir) / f"frame_{i}.jpg"
            cv2.imwrite(str(p), frame)
            paths.append(p)
        if not paths:
            raise ValueError(f"no readable frames in video: {video_path}")
        return paths
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
last_moderation_details = {}


def moderate_ad(file_path, text=None):
    """Moderate one ad. Returns ONLY "accepted" or "refused"."""
    return moderate_ad_verbose(file_path, text)[0]


def moderate_ad_verbose(file_path, text=None):
    """Same decision as moderate_ad, plus a details dict for the admin dashboard."""
    global last_moderation_details
    path = Path(file_path)
    details = {"file": path.name, "checks": {}, "flagged_by": [], "errors": {}}
    t0 = time.perf_counter()

    def run(agent, fn):
        """Fail-safe agent runner: an agent crash counts as a flag, never a crash."""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            log.error("agent %s failed on %s: %s", agent, path.name, exc)
            details["errors"][agent] = str(exc)
            details["flagged_by"].append(agent)
            return None

    try:
        with tempfile.TemporaryDirectory() as td:
            is_video = path.suffix.lower() in VIDEO_EXTS
            if is_video:
                frames = extract_frames(path, td)
            elif path.suffix.lower() in IMAGE_EXTS:
                Image.open(path).convert("RGB")  # fail early on corrupt images
                frames = [path]
            else:
                raise ValueError(f"unsupported file type: {path.suffix}")

            # Agents 1-3 share the already-extracted frames
            def nudity():
                worst = max(check_nudity(f)[1] for f in frames)
                details["checks"]["nudity"] = {"score": round(worst, 4)}
                if worst >= NUDITY_THRESHOLD:
                    details["flagged_by"].append("nudity")

            def threat():
                hits, worst = [], 0.0
                for f in frames:
                    flagged, conf, classes = check_threat(f)
                    if flagged:
                        hits += classes
                        worst = max(worst, conf)
                details["checks"]["threat"] = {"score": round(worst, 4), "classes": sorted(set(hits))}
                if hits:
                    details["flagged_by"].append("threat")

            def ocr():
                texts = [extract_image_text(f) for f in frames]
                combined = " ".join(t for t in texts if t)
                details["checks"]["ocr_text"] = {"text": combined}
                if check_text_profanity(combined):
                    details["flagged_by"].append("ocr_profanity")

            run("nudity", nudity)
            run("threat", threat)
            run("ocr_profanity", ocr)

            if text is not None:
                def caption():
                    flagged = check_text_profanity(text)
                    details["checks"]["text_profanity"] = {"flagged": flagged}
                    if flagged:
                        details["flagged_by"].append("text_profanity")
                run("text_profanity", caption)

            if is_video:
                def speech():
                    flagged, transcript = check_speech_profanity(path)
                    details["checks"]["speech"] = {"transcript": transcript, "flagged": flagged}
                    if flagged:
                        details["flagged_by"].append("speech_profanity")
                run("speech_profanity", speech)

    except Exception as exc:  # noqa: BLE001 - unreadable/corrupt file: fail safe
        log.error("could not process %s: %s", path.name, exc)
        details["errors"]["file"] = str(exc)
        details["flagged_by"].append("unreadable_file")

    details["flagged_by"] = sorted(set(details["flagged_by"]))
    details["latency_s"] = round(time.perf_counter() - t0, 2)
    decision = "refused" if details["flagged_by"] else "accepted"
    details["decision"] = decision
    last_moderation_details = details
    return decision, details


if __name__ == "__main__":
    import json
    import sys

    load_all_models()
    for arg in sys.argv[1:]:
        decision, info = moderate_ad_verbose(arg)
        print(f"\n{arg}: {decision}")
        print(json.dumps(info, indent=2))
