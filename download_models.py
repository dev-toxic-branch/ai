"""Resilient downloader for ALL moderation models (flaky-network-proof).

Fetches, with unlimited resume-on-drop retries:
  - microsoft/trocr-base-printed  -> models/trocr_base_printed/   (OCR, ~1.3 GB)
  - Subh775/Threat-Detection-YOLOv8n weights/best.pt -> models/threat_yolov8n/ (~6 MB)
  - Whisper "base" checkpoint     -> ~/.cache/whisper/base.pt     (~145 MB)

(The Falconsai NSFW model is handled by the original download_model.py and is
already present.) Rerunning always resumes; nothing ever restarts from zero.

Usage:  python download_models.py
"""

import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent
CHUNK = 256 * 1024
USER_AGENT = "hackiwha-ai-moderation/1.0 (resumable model fetch)"

TROCR = "https://huggingface.co/microsoft/trocr-base-printed/resolve/main/"
YOLO = "https://huggingface.co/Subh775/Threat-Detection-YOLOv8n/resolve/main/"
# URL from openai-whisper's own _MODELS table; the hex segment is the sha256
# the library verifies, so a file fetched from here passes its integrity check.
WHISPER_BASE = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
)

# Ordered smallest-first so the quick wins land early; the 1.3GB OCR model last.
TARGETS = [
    (YOLO + "weights/best.pt", BASE_DIR / "models/threat_yolov8n/best.pt"),
    (WHISPER_BASE, Path.home() / ".cache/whisper/base.pt"),
    (TROCR + "config.json", BASE_DIR / "models/trocr_base_printed/config.json"),
    (TROCR + "generation_config.json", BASE_DIR / "models/trocr_base_printed/generation_config.json"),
    (TROCR + "preprocessor_config.json", BASE_DIR / "models/trocr_base_printed/preprocessor_config.json"),
    (TROCR + "tokenizer_config.json", BASE_DIR / "models/trocr_base_printed/tokenizer_config.json"),
    (TROCR + "special_tokens_map.json", BASE_DIR / "models/trocr_base_printed/special_tokens_map.json"),
    (TROCR + "vocab.json", BASE_DIR / "models/trocr_base_printed/vocab.json"),
    (TROCR + "merges.txt", BASE_DIR / "models/trocr_base_printed/merges.txt"),
    (TROCR + "model.safetensors", BASE_DIR / "models/trocr_base_printed/model.safetensors"),
]


def _remote_size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers.get("Content-Length", 0))


def _valid(path):
    """Cheap integrity checks on the first bytes of binary formats."""
    with open(path, "rb") as fh:
        head = fh.read(9)
    if path.suffix == ".safetensors":  # u64 header length then JSON '{'
        return len(head) == 9 and head[8:9] == b"{"
    if path.suffix == ".pt":  # torch checkpoints are zip files
        return head[:2] == b"PK"
    return True


def download(url, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    expected = None
    while expected is None:
        attempt += 1
        try:
            expected = _remote_size(url)
        except Exception as exc:  # noqa: BLE001
            print(f"  HEAD failed for {out.name} ({exc}); retry in 5s")
            time.sleep(5)

    if out.exists() and out.stat().st_size == expected and _valid(out):
        print(f"{out.name}: already complete ({expected:,} bytes)")
        return

    attempt = 0
    while True:
        have = out.stat().st_size if out.exists() else 0
        if have >= expected:
            break
        attempt += 1
        try:
            headers = {"User-Agent": USER_AGENT}
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp, open(out, "ab") as fh:
                last_report = time.monotonic()
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    have += len(chunk)
                    if time.monotonic() - last_report > 10:
                        print(f"  {out.name}: {have:,}/{expected:,} ({100.0 * have / expected:.1f}%)")
                        last_report = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - flaky network is the expected case
            wait = min(5 * attempt, 30)
            print(f"  {out.name}: dropped at {have:,} ({exc}); retry in {wait}s")
            time.sleep(wait)

    if not _valid(out):
        raise RuntimeError(f"{out} is full-size but corrupt - delete it and rerun.")
    print(f"{out.name}: DONE ({expected:,} bytes)")


if __name__ == "__main__":
    for url, dest in TARGETS:
        download(url, dest)
    print("\nALL MODELS DOWNLOADED.")
