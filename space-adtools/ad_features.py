"""AI features for the ad platform - standalone functions, no server code.

    Feature 1  Auto-reformat: landscape video -> 9:16 mobile crop (YOLOv8n)
    Feature 2  Dayparting: rule-based time-of-day variant filtering (no ML)
    Feature 3  Content-context matching: all-MiniLM-L6-v2 embeddings

Everything runs locally on CPU with open-source models only.
Run `python download_models.py` once to fetch weights (resumable).

Backend contract (import and call directly):
    generate_mobile_variant(video_path, output_path) -> output_path
    get_current_daypart(timestamp=None, tz="UTC") -> str
    filter_by_daypart(variants, current_daypart) -> list[dict]
    precompute_variant_embeddings(variants) -> list[dict]   (once, at upload)
    select_best_matching_variant(surrounding_text, variants) -> dict
    pick_final_variant(variants, current_daypart, surrounding_text) -> dict
"""

import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

BASE_DIR = Path(__file__).parent
YOLO_GENERAL_WEIGHTS = BASE_DIR / "models" / "yolov8n" / "yolov8n.pt"
MINILM_DIR = BASE_DIR / "models" / "minilm"

# ===========================================================================
# FEATURE 1 - AUTO-REFORMAT (MOBILE 9:16 CROP)
# ===========================================================================

DETECT_CONF_THRESHOLD = 0.30  # below this, YOLOv8n detections are mostly noise
_yolo_general = None


def _get_yolo():
    """General-purpose YOLOv8-NANO (COCO classes) - smallest variant, CPU-friendly."""
    global _yolo_general
    if _yolo_general is None:
        from ultralytics import YOLO

        # fall back to the auto-downloading name if the local copy is absent
        src = str(YOLO_GENERAL_WEIGHTS) if YOLO_GENERAL_WEIGHTS.is_file() else "yolov8n.pt"
        _yolo_general = YOLO(src)
    return _yolo_general


def detect_main_subject(frame_path):
    """Return the main subject's bounding box (x1, y1, x2, y2) or None.

    Persons win over other object classes; ties break on confidence.
    """
    results = _get_yolo().predict(
        str(frame_path), device="cpu", conf=DETECT_CONF_THRESHOLD, verbose=False
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best = None  # (is_person, confidence, bbox)
    for cls_id, conf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
        is_person = results[0].names[int(cls_id)] == "person"
        key = (is_person, conf)
        if best is None or key > best[0]:
            best = (key, tuple(round(v) for v in xyxy))
    return best[1]


def compute_crop_window(frame_size, subject_bbox, target_ratio=9 / 16):
    """Return (x, y, w, h) of a target_ratio crop keeping the subject centered.

    frame_size is (width, height). Falls back to a center crop when
    subject_bbox is None. The window is always fully inside the frame.
    """
    frame_w, frame_h = frame_size

    # Largest target_ratio window that fits inside the frame
    crop_h = frame_h
    crop_w = int(round(crop_h * target_ratio))
    if crop_w > frame_w:
        crop_w = frame_w
        crop_h = int(round(crop_w / target_ratio))

    if subject_bbox is not None:
        cx = (subject_bbox[0] + subject_bbox[2]) / 2
        cy = (subject_bbox[1] + subject_bbox[3]) / 2
    else:
        cx, cy = frame_w / 2, frame_h / 2

    x = int(round(cx - crop_w / 2))
    y = int(round(cy - crop_h / 2))
    x = max(0, min(x, frame_w - crop_w))
    y = max(0, min(y, frame_h - crop_h))
    return (x, y, crop_w, crop_h)


def _ffmpeg_exe():
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()


def generate_mobile_variant(video_path, output_path):
    """Create a 1080x1920 mobile version of a landscape video.

    Samples the main subject once per second, averages the detected centers,
    and applies ONE stable crop for the whole clip (no per-frame jitter).
    Returns output_path as str.
    """
    video_path, output_path = Path(video_path), Path(output_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # one sample per second of footage
        step = max(int(round(fps)), 1)
        centers = []
        with tempfile.TemporaryDirectory() as td:
            for idx in range(0, max(total, 1), step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                sample = Path(td) / "sample.jpg"
                cv2.imwrite(str(sample), frame)
                bbox = detect_main_subject(sample)
                if bbox is not None:
                    centers.append(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))
    finally:
        cap.release()

    # Average subject position across samples -> one smooth, consistent crop
    if centers:
        mean_cx = sum(c[0] for c in centers) / len(centers)
        mean_cy = sum(c[1] for c in centers) / len(centers)
        pseudo_bbox = (mean_cx, mean_cy, mean_cx, mean_cy)  # zero-size box at mean center
    else:
        pseudo_bbox = None
    x, y, w, h = compute_crop_window((frame_w, frame_h), pseudo_bbox)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg_exe(), "-y", "-i", str(video_path),
        "-vf", f"crop={w}:{h}:{x}:{y},scale=1080:1920",
        "-c:a", "copy",
        "-loglevel", "error",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg crop failed: {proc.stderr.strip()}")
    return str(output_path)


# ===========================================================================
# FEATURE 2 - TIME-OF-DAY VARIANT SELECTION (DAYPARTING, rule-based)
# ===========================================================================

# (start_hour_inclusive, end_hour_exclusive)
DAYPARTS = {
    "morning": (6, 11),
    "afternoon": (11, 17),
    "evening": (17, 22),
    "night": (22, 6),  # wraps midnight
}


def get_current_daypart(timestamp=None, tz="UTC"):
    """Return "morning" / "afternoon" / "evening" / "night" for the given time.

    timestamp: optional datetime (naive = assumed UTC) or None for "now".
    tz: IANA timezone name of the viewer (e.g. "Africa/Algiers"); default UTC.
    Pass the viewer's timezone whenever the backend knows it.
    """
    zone = ZoneInfo(tz)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    elif timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    hour = timestamp.astimezone(zone).hour

    for name, (start, end) in DAYPARTS.items():
        if start < end:
            if start <= hour < end:
                return name
        elif hour >= start or hour < end:  # night wraps midnight
            return name
    return "night"  # unreachable, but explicit


def filter_by_daypart(variants, current_daypart):
    """Keep variants tagged with current_daypart or "any".

    Fail-safe: if nothing matches, return the original list unchanged -
    a daypart mismatch must never leave the ad server with zero options.
    """
    matched = [
        v for v in variants
        if v.get("daypart", "any") in (current_daypart, "any")
    ]
    return matched if matched else list(variants)


# ===========================================================================
# FEATURE 3 - CONTENT-CONTEXT MATCHING (EMBEDDING SIMILARITY)
# ===========================================================================

_embedder = None  # (tokenizer, model), loaded once


def _get_embedder():
    """all-MiniLM-L6-v2 (~90MB): the standard small CPU sentence embedder.

    Loaded via plain transformers + mean pooling (identical output to the
    sentence-transformers wrapper, one less dependency).
    """
    global _embedder
    if _embedder is None:
        import torch  # noqa: F401 - ensures torch is importable before model load
        from transformers import AutoModel, AutoTokenizer

        src = (
            str(MINILM_DIR)
            if (MINILM_DIR / "model.safetensors").is_file()
            else "sentence-transformers/all-MiniLM-L6-v2"
        )
        tokenizer = AutoTokenizer.from_pretrained(src)
        model = AutoModel.from_pretrained(src)
        model.eval()
        _embedder = (tokenizer, model)
    return _embedder


def embed_text(text):
    """Return the 384-dim L2-normalized embedding of text as np.ndarray."""
    import torch

    tokenizer, model = _get_embedder()
    inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state  # (1, seq, 384)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    vec = torch.nn.functional.normalize(pooled, p=2, dim=1)[0].numpy()
    return vec.astype(np.float32)


def precompute_variant_embeddings(variants):
    """Add an "embedding" field to each variant (from its "description").

    Call ONCE at ad-upload time and persist the result; do not recompute
    per request. Store embedding as list(v["embedding"]) if serializing to JSON.
    """
    out = []
    for v in variants:
        v = dict(v)
        v["embedding"] = embed_text(v.get("description", "") or "")
        out.append(v)
    return out


def compute_similarity(vec_a, vec_b):
    """Cosine similarity between two vectors (accepts lists or arrays)."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def select_best_matching_variant(surrounding_text, variants):
    """Pick the variant whose description best matches surrounding_text.

    Variants must carry precomputed "embedding" fields. Returns a copy of the
    winning variant with "match_score" attached. Fallbacks: a single variant
    is returned as-is; empty surrounding_text returns the first variant.
    """
    if not variants:
        raise ValueError("variants list is empty")
    if len(variants) == 1:
        return dict(variants[0], match_score=None)
    if not surrounding_text or not surrounding_text.strip():
        return dict(variants[0], match_score=None)

    query = embed_text(surrounding_text)
    best, best_score = None, -2.0
    for v in variants:
        score = compute_similarity(query, v["embedding"])
        if score > best_score:
            best, best_score = v, score
    return dict(best, match_score=best_score)


# ===========================================================================
# COMBINED HELPER - dayparting then context matching
# ===========================================================================

def pick_final_variant(variants, current_daypart, surrounding_text):
    """Filter by daypart, then pick the best context match among survivors.

    Returns one variant dict; "match_score" is a float when embedding
    matching ran, or None when the choice fell out of dayparting alone.
    """
    candidates = filter_by_daypart(variants, current_daypart)
    if len(candidates) == 1:
        return dict(candidates[0], match_score=None)
    # ensure embeddings exist (upload-time precompute is the fast path)
    if any("embedding" not in v for v in candidates):
        candidates = precompute_variant_embeddings(candidates)
    return select_best_matching_variant(surrounding_text, candidates)
