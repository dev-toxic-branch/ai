"""Ad content moderation: NSFW detection for images and short video clips.

Uses the Falconsai/nsfw_image_detection ViT classifier (local inference via
transformers + torch). Videos are moderated by sampling frames evenly and
taking the worst (highest) NSFW score across frames — one bad frame is
enough to refuse an ad.
"""

import time
from pathlib import Path

import cv2
from PIL import Image

MODEL_ID = "Falconsai/nsfw_image_detection"
DEFAULT_THRESHOLD = 0.7
MAX_VIDEO_FRAMES = 8

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

_classifier = None


def get_classifier():
    """Load the NSFW classifier once and cache it (first call downloads the model)."""
    global _classifier
    if _classifier is None:
        from transformers import pipeline

        _classifier = pipeline("image-classification", model=MODEL_ID)
    return _classifier


def _nsfw_score(pipeline_output):
    """Extract the 'nsfw' class probability from pipeline output for one image."""
    for entry in pipeline_output:
        if entry["label"].lower() == "nsfw":
            return float(entry["score"])
    return 0.0


def _score_images(images):
    """Return the NSFW probability (0-1) for each PIL image."""
    clf = get_classifier()
    outputs = clf(images, top_k=None)
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
            f"(nsfw={result['nsfw_score']:.3f}, {result['latency_s']:.2f}s)"
        )
