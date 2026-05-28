# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage — install dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# System libraries required by docling (libGL via opencv-headless, lancedb, etc.)
# and spaCy (build tools for wheel compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Download the bundled spaCy model used by graph_engine extractors and retriever
RUN PYTHONPATH=/install/lib/python3.12/site-packages \
    python -m spacy download en_core_web_sm

# Pre-bake Docling layout/OCR models (~1 GB) to avoid cold-start at runtime
RUN PYTHONPATH=/install/lib/python3.12/site-packages \
    python -c "from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline; StandardPdfPipeline.download_models_hf(force=True)"

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

# Copy installed packages + pre-baked Docling models + spaCy model from builder
COPY --from=builder /install /usr/local
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application source
COPY api.py models.py ragService.py retriever.py document_loader.py ragas_runner.py __init__.py ./
COPY indexer/ ./indexer/
COPY graph_engine/ ./graph_engine/
COPY static/ ./static/

# data/ and ragdata/ are mutable at runtime — mount as volumes in production
VOLUME ["/app/data", "/app/ragdata"]

# Environment variable defaults (override at runtime via --env or compose env_file)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OLLAMA_INDEX_URL=http://host.docker.internal:11434 \
    OLLAMA_CHAT_URL=http://host.docker.internal:11434 \
    CHAT_MODEL=gemma4:e2b \
    EMBEDDING_MODEL=embeddinggemma:300m \
    VLM_MODEL=gemma4:e4b \
    VLM_TIMEOUT=180 \
    EXTRACT_MODEL=granite4.1:8b \
    GRAPH_EXTRACTOR=hybrid \
    GRAPH_CHUNK_SIZE=400 \
    GRAPH_CHUNK_OVERLAP=80 \
    RAGAS_MODE=gemma4:31b \
    TOP_K=5

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
