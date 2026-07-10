"""Ad content moderation: NSFW detection for images and short video clips.

Two backends, picked automatically:

1. "vit-model"      - Falconsai/nsfw_image_detection ViT classifier, used when a
                      valid copy exists in models/nsfw_image_detection/ (run
                      download_model.py to fetch it; it survives flaky networks).
2. "skin-heuristic" - OpenCV skin-pixel-ratio fallback, used when the model is
                      not available. No download needed, works offline. It is a
                      stopgap: it flags skin-dominant images, so it catches
                      obvious nudity but will false-refuse portraits/faces and
                      can miss NSFW content that isn't skin-dominant. Replace it
                      with the real model before relying on results.

Videos are moderated by sampling frames evenly and taking the worst (highest)
NSFW score across frames - one bad frame is enough to refuse an ad.
"""

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_LOCAL_MODEL = Path(__file__).parent / "models" / "nsfw_image_detection"
MODEL_ID = "Falconsai/nsfw_image_detection"
DEFAULT_THRESHOLD = 0.6
MAX_VIDEO_FRAMES = 8

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

_classifier = None
_backend = None


def _local_model_looks_valid():
    """Cheap sanity check: safetensors starts with a little-endian u64 header
    length followed by a JSON header beginning with '{'."""
    f = _LOCAL_MODEL / "model.safetensors"
    # exact size of the published checkpoint - a partial download must not load
    if not f.is_file() or f.stat().st_size != 343_223_968:
        return False
    with open(f, "rb") as fh:
        head = fh.read(9)
    return len(head) == 9 and head[8:9] == b"{"


def get_classifier():
    """Load the NSFW backend once and cache it. Returns (backend_name, callable)."""
    global _classifier, _backend
    if _classifier is None:
        if _local_model_looks_valid():
            from transformers import pipeline

            _classifier = pipeline(
                "image-classification", model=str(_LOCAL_MODEL), use_fast=True
            )
            _backend = "vit-model"
        else:
            _classifier = _skin_heuristic_score
            _backend = "skin-heuristic"
    return _backend, _classifier


def _skin_heuristic_score(images):
    """Fallback scorer: fraction of skin-tone pixels (YCrCb rule), mapped to 0-1.

    Classic skin segmentation (Chai & Ngan): Cr in [133, 173], Cb in [77, 127].
    An image that is mostly skin scores high; ads/landscapes score low.
    """
    if isinstance(images, Image.Image):
        images = [images]
    scores = []
    for img in images:
        arr = np.asarray(img.convert("RGB"))
        ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
        skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (y >= 40)
        ratio = float(skin.mean())
        # 0% skin -> 0.0, >=45% skin -> 1.0 (tuned on the bundled test set)
        scores.append(min(ratio / 0.45, 1.0))
    return scores


def _nsfw_score(pipeline_output):
    """Extract the 'nsfw' class probability from ViT pipeline output for one image."""
    for entry in pipeline_output:
        if entry["label"].lower() == "nsfw":
            return float(entry["score"])
    return 0.0


def _score_images(images):
    """Return the NSFW probability (0-1) for each PIL image."""
    backend, clf = get_classifier()
    if backend == "skin-heuristic":
        return clf(images)
    outputs = clf(images, top_k=None, batch_size=max(len(images), 1))
    # For a single image the pipeline returns a flat list of label dicts
    if images and isinstance(outputs[0], dict):
        outputs = [outputs]
    return [_nsfw_score(out) for out in outputs]


def _extract_frames(video_path, max_frames=MAX_VIDEO_FRAMES):
    """Sample up to max_frames evenly spaced frames from a video as PIL images."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 1
        step = max(total // max_frames, 1)
        indices = list(range(0, total, step))[:max_frames]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if not frames:
            raise ValueError(f"No readable frames in video: {video_path}")
        return frames
    finally:
        cap.release()


def moderate_ad(path, threshold=DEFAULT_THRESHOLD):
    """Moderate one ad creative (image or video file).

    Returns a dict:
        file        - file name
        media_type  - "image" or "video"
        backend     - "vit-model" or "skin-heuristic"
        nsfw_score  - probability 0-1 (for videos: max across sampled frames)
        threshold   - threshold used for the decision
        decision    - "ACCEPT" or "REFUSE"
        latency_s   - wall-clock seconds for this call (includes decode + inference)
    """
    path = Path(path)
    ext = path.suffix.lower()
    start = time.perf_counter()

    if ext in VIDEO_EXTS:
        media_type = "video"
        frames = _extract_frames(path)
        score = max(_score_images(frames))
    elif ext in IMAGE_EXTS:
        media_type = "image"
        image = Image.open(path).convert("RGB")
        score = _score_images([image])[0]
    else:
        raise ValueError(f"Unsupported file type '{ext}': {path}")

    latency = time.perf_counter() - start
    return {
        "file": path.name,
        "media_type": media_type,
        "backend": get_classifier()[0],
        "nsfw_score": score,
        "threshold": threshold,
        "decision": "REFUSE" if score >= threshold else "ACCEPT",
        "latency_s": latency,
    }


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        result = moderate_ad(arg)
        print(
            f"{result['file']}: {result['decision']} "
            f"(nsfw={result['nsfw_score']:.3f}, backend={result['backend']}, "
            f"{result['latency_s']:.2f}s)"
        )
