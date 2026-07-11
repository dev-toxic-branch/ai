"""FastAPI server for ad_features — local-only, no external APIs.

Endpoints:
    GET  /                    health check + loaded models
    POST /mobile-variant      upload video → 9:16 mobile crop
    POST /pick-variant        JSON (variants, surrounding_text) → best match
    POST /daypart             JSON (tz, timestamp) → "morning"/"afternoon"/...
    POST /embed               JSON (text) → 384-dim embedding vector

Run:
    docker compose up --build
    # or directly:  uvicorn server:app --host 0.0.0.0 --port 8000
"""

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import ad_features

app = FastAPI(
    title="Ad Features API",
    description="Local ad feature engine — mobile crop, dayparting, embedding matching.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Startup: eager-load YOLO + MiniLM so first request isn't slow
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _warm_models():
    print("Warming YOLOv8n...")
    t0 = time.perf_counter()
    ad_features._get_yolo()
    print(f"  YOLOv8n ready in {time.perf_counter() - t0:.1f}s")

    print("Warming MiniLM-L6-v2...")
    t0 = time.perf_counter()
    ad_features._get_embedder()
    print(f"  MiniLM-L6-v2 ready in {time.perf_counter() - t0:.1f}s")


# ---------------------------------------------------------------------------
# GET / — health check
# ---------------------------------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "ok",
        "models": ["YOLOv8n", "all-MiniLM-L6-v2"],
        "features": ["auto-reformat", "dayparting", "content-context matching"],
    }


# ---------------------------------------------------------------------------
# POST /mobile-variant — upload landscape video → 9:16 mobile crop
# ---------------------------------------------------------------------------
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@app.post("/mobile-variant")
async def mobile_variant(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in ALLOWED_VIDEO:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {sorted(ALLOWED_VIDEO)}")

    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / f"input{suffix}"
        out = Path(td) / "mobile.mp4"

        with open(inp, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        try:
            result_path = ad_features.generate_mobile_variant(str(inp), str(out))
        except Exception as exc:
            raise HTTPException(500, f"Processing failed: {exc}")

        return FileResponse(
            result_path,
            media_type="video/mp4",
            filename=f"mobile_{file.filename}",
        )


# ---------------------------------------------------------------------------
# POST /daypart — current time-of-day for a timezone
# ---------------------------------------------------------------------------
class DaypartRequest(BaseModel):
    tz: str = "UTC"
    timestamp: str | None = None  # ISO format, optional


@app.post("/daypart")
def daypart(req: DaypartRequest):
    ts = None
    if req.timestamp:
        try:
            ts = datetime.fromisoformat(req.timestamp)
        except ValueError:
            raise HTTPException(400, f"Invalid ISO timestamp: {req.timestamp}")
    try:
        result = ad_features.get_current_daypart(timestamp=ts, tz=req.tz)
    except Exception as exc:
        raise HTTPException(400, f"Timezone error: {exc}")
    return {"daypart": result, "tz": req.tz}


# ---------------------------------------------------------------------------
# POST /pick-variant — filter by daypart + embedding match
# ---------------------------------------------------------------------------
class PickVariantRequest(BaseModel):
    variants: list[dict]
    surrounding_text: str = ""
    current_daypart: str | None = None  # auto-detect if omitted
    tz: str = "UTC"


@app.post("/pick-variant")
def pick_variant(req: PickVariantRequest):
    if not req.variants:
        raise HTTPException(400, "variants list is empty")

    daypart = req.current_daypart or ad_features.get_current_daypart(tz=req.tz)
    try:
        result = ad_features.pick_final_variant(req.variants, daypart, req.surrounding_text)
    except Exception as exc:
        raise HTTPException(500, f"Selection failed: {exc}")

    # Convert numpy embeddings to plain lists for JSON serialization
    if "embedding" in result and hasattr(result["embedding"], "tolist"):
        result["embedding"] = result["embedding"].tolist()

    return {"selected": result, "daypart": daypart}


# ---------------------------------------------------------------------------
# POST /embed — get text embedding vector
# ---------------------------------------------------------------------------
class EmbedRequest(BaseModel):
    text: str


@app.post("/embed")
def embed(req: EmbedRequest):
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty")
    vec = ad_features.embed_text(req.text)
    return {"embedding": vec.tolist(), "dim": len(vec)}


# ---------------------------------------------------------------------------
# POST /compute-similarity — cosine similarity between two texts
# ---------------------------------------------------------------------------
class SimilarityRequest(BaseModel):
    text_a: str
    text_b: str


@app.post("/compute-similarity")
def compute_similarity(req: SimilarityRequest):
    vec_a = ad_features.embed_text(req.text_a)
    vec_b = ad_features.embed_text(req.text_b)
    score = ad_features.compute_similarity(vec_a, vec_b)
    return {"similarity": score}
