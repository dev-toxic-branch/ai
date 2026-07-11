FROM python:3.11-slim

# ffmpeg for video crop/scaling
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY ad_features.py download_models.py download_model.py server.py ./
COPY models/ ./models/

# Download any missing model weights (resumable, idempotent)
RUN python download_models.py || true

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
