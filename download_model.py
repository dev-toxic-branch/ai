"""Resilient downloader for the Falconsai/nsfw_image_detection model.

Built for bad networks: downloads in small resumable chunks with unlimited
retries, so connection resets just pause progress instead of losing it.
Leave it running in a spare terminal (or overnight); moderation.py picks up
the real model automatically once this finishes - no code change needed.

Usage:  python download_model.py
"""

import time
import urllib.request
from pathlib import Path

DEST = Path(__file__).parent / "models" / "nsfw_image_detection"
BASE = "https://huggingface.co/Falconsai/nsfw_image_detection/resolve/main/"
FILES = {
    "config.json": None,
    "preprocessor_config.json": None,
    "model.safetensors": 343_223_968,  # known size, lets us verify completion
}
CHUNK = 256 * 1024
USER_AGENT = "hackiwha-ai-moderation/1.0 (resumable model fetch)"


def _remote_size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers.get("Content-Length", 0))


def download(name, expected_size):
    url = BASE + name
    out = DEST / name
    if expected_size is None:
        expected_size = _remote_size(url)
    if out.exists() and out.stat().st_size == expected_size and _valid(out):
        print(f"{name}: already complete ({expected_size:,} bytes)")
        return

    attempt = 0
    while True:
        have = out.stat().st_size if out.exists() else 0
        if have >= expected_size:
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
                    if time.monotonic() - last_report > 5:
                        pct = 100.0 * have / expected_size
                        print(f"  {name}: {have:,}/{expected_size:,} bytes ({pct:.1f}%)")
                        last_report = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - flaky network is the expected case
            wait = min(5 * attempt, 30)
            print(f"  {name}: connection dropped at {have:,} bytes ({exc}); "
                  f"retrying in {wait}s (attempt {attempt})")
            time.sleep(wait)

    if not _valid(out):
        raise RuntimeError(
            f"{name} downloaded to full size but failed validation - delete it and rerun."
        )
    print(f"{name}: DONE ({expected_size:,} bytes)")
def _valid(path):
    if path.suffix != ".safetensors":
        return True
    with open(path, "rb") as fh:
        head = fh.read(9)
    return len(head) == 9 and head[8:9] == b"{"
if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    for fname, size in FILES.items():
        download(fname, size)
    print("\nAll files downloaded. moderation.py will now use the real ViT model.")
