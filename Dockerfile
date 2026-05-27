# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage — install dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# System libraries required by docling (libGL via opencv-headless, lancedb, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-bake Docling layout/OCR models (~1 GB) to avoid cold-start at runtime
RUN PYTHONPATH=/install/lib/python3.12/site-packages \
    python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages + pre-baked Docling models from builder
COPY --from=builder /install /usr/local
COPY --from=builder /root/.cache/docling /root/.cache/docling

# Copy application source
COPY api.py models.py ragService.py document_loader.py ragas_runner.py __init__.py ./
COPY static/ ./static/
COPY ragdata/ ./ragdata/

# ragdata is mutable at runtime — override with a bind mount in production
VOLUME ["/app/ragdata"]

# Environment variable defaults (override at runtime via --env or .env mount)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHAT_MODEL=llama3.2:3b \
    EMBEDDING_MODEL=bge-m3:567m \
    VLM_MODEL=llava:7b \
    VLM_TIMEOUT=180

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
