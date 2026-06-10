# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage — install dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# System libraries required by spaCy (build tools for wheel compilation) and lancedb
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Install spaCy model as a fallback when local bundled model is unavailable.
RUN PYTHONPATH=/install/lib/python3.12/site-packages \
  python -m spacy download en_core_web_sm || true

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Copy installed packages + spaCy model from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY api.py models.py rag_service.py retriever.py document_loader.py rewriter.py self_reflector.py __init__.py ./
COPY models/ ./models/
COPY indexer/ ./indexer/
COPY graph_engine/ ./graph_engine/
COPY static/ ./static/

# data/ and ragdata/ are mutable at runtime — mount as volumes in production
VOLUME ["/app/data", "/app/ragdata"]

# Environment variable defaults (override at runtime via --env or compose env_file)
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  DOCLING_URL=http://docling-serve:5001 \
  OLLAMA_INDEX_URL=http://host.docker.internal:11434 \
  OLLAMA_CHAT_URL=http://host.docker.internal:11434 \
  CHAT_MODEL=qwen3.6:35b \
  CHAT_MODEL_THINK=true \
  EMBEDDING_MODEL=embeddinggemma:300m \
  VLM_MODEL=llava:7b \
  VLM_TIMEOUT=180 \
  OLLAMA_TIMEOUT=1800 \
  EXTRACT_MODEL=granite4.1:8b \
  GRAPH_EXTRACTOR=hybrid \
  GRAPH_CHUNK_SIZE=400 \
  GRAPH_CHUNK_OVERLAP=80 \
  TOP_K=5

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
