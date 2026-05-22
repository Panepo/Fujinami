# Fujinami RAG Service

A hybrid **Retrieval-Augmented Generation (RAG)** system that combines [Microsoft GraphRAG](https://github.com/microsoft/graphrag), [Semantic Kernel](https://github.com/microsoft/semantic-kernel), and [LanceDB](https://lancedb.github.io/lancedb/) to answer questions over your document collections using locally-hosted [Ollama](https://ollama.com/) models.

---

## Features

- **Hybrid search** — blends dense vector search (LanceDB) with graph-based retrieval (GraphRAG knowledge graph) for richer answers
- **Three query modes** — `vector`, `hybrid`, and `global` (community-level summaries)
- **Multi-collection** — manage independent document collections via a REST API
- **Rich document ingestion** — powered by [Docling](https://github.com/DS4SD/docling); supports documents (`.pdf`, `.docx`, `.xlsx`, `.pptx`, `.md`, `.tex`, `.html`, `.csv`, and more), images (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`), audio (`.wav`, `.mp3`, `.m4a`, `.aac`, `.ogg`, `.flac`), and video (`.mp4`, `.avi`, `.mov`); embedded pictures are described inline by a VLM via Docling's built-in picture-description pipeline
- **Streaming responses** — optional token-by-token streaming on query endpoints
- **Built-in Web UI** — zero-configuration browser interface served at `/`
- **Fully local** — all LLM, embedding, and VLM calls go to Ollama; no cloud APIs required
- **RAGAS evaluation** — score RAG responses against 10 built-in metrics (Faithfulness, Context Recall, Context Precision, Response Relevancy, Factual Correctness, Noise Sensitivity, Semantic Similarity, BLEU, ROUGE) using a locally-hosted LLM

---

## Architecture Overview

```
Documents (.pdf .docx .xlsx .pptx .md .txt …)
Images   (.png .jpg .tiff .webp …)
Audio    (.wav .mp3 .m4a …)
Video    (.mp4 .avi .mov …)
        │
        ▼
  DocumentLoader  ──▶  Docling DocumentConverter
    (docling[asr])         ├─ OCR + table extraction
                           ├─ VLM picture description (llava:7b)
                           └─ export_to_markdown()
        │
        ▼
  ┌─────────────────────────────────┐
  │         Index Pipeline          │
  │                                 │
  │  GraphRAG CLI  ──▶  entities,   │
  │  (subprocess)       communities │
  │                     reports     │
  │                                 │
  │  SK Embeddings ──▶  LanceDB     │
  │  (bge-m3:567m)      chunks      │
  └─────────────────────────────────┘
        │
        ▼
  FastAPI server  ──▶  Web UI  /  REST API
        │
        ▼
  Query (vector | hybrid | global)
        │
        ▼
  llama3.2:3b  →  answer + source chunks
```

See [docs/dataflow-ragService.md](docs/dataflow-ragService.md) for full pipeline diagrams.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.12 (3.13+ not supported by onnxruntime) |
| [uv](https://github.com/astral-sh/uv) | latest |
| [Ollama](https://ollama.com/) | running locally on port `11434` |

### Required Ollama models

Pull these before first use:

```sh
# Chat and query-time (local)
ollama pull llama3.2:3b
ollama pull bge-m3:567m

# Index-time embeddings and VLM for picture description (can be on a remote GPU server)
ollama pull bge-m3:567m
ollama pull llava:7b   # used by Docling's picture-description pipeline
```

---

## Setup

### 1. Create a `.env` file

```env
# Remote Ollama server used during indexing (embeddings + VLM)
OLLAMA_INDEX_URL=

# Local Ollama server used at query time
OLLAMA_CHAT_URL=

# Model names
CHAT_MODEL=llama3.2:3b
EMBEDDING_MODEL=bge-m3:567m
VLM_MODEL=llava:7b

# Optional: VLM HTTP timeout in seconds (default 180)
VLM_TIMEOUT=180

# Model used for RAGAS evaluation (needs large context window, e.g. gemma4:e4b)
RAGAS_MODEL=gemma4:e4b

# Optional: Ollama request timeout for RAGAS evaluation in seconds (default 1800)
OLLAMA_TIMEOUT=1800
```

> If you only have one Ollama instance, set both `OLLAMA_INDEX_URL` and `OLLAMA_CHAT_URL` to the same URL.

### 2. Create the virtual environment and install dependencies

```sh
# Install uv (once)
pip install uv

# Create .venv and install dependencies
uv venv
uv pip install -r requirements.txt
```

### 3. Start the development server

```sh
uv run poe dev
# equivalent to: uvicorn api:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Usage

### Web UI

Navigate to [http://localhost:8000](http://localhost:8000) for the built-in interface. From there you can:

- Create and manage collections
- Upload documents
- Trigger indexing (with optional entity type selection)
- Run queries with `vector`, `hybrid`, or `global` mode

### REST API

Interactive docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

#### Collections

```http
GET    /collections                    # list all collections
POST   /collections                    # create a collection  { "name": "my-docs" }
PATCH  /collections/{name}             # rename               { "new_name": "new-name" }
DELETE /collections/{name}             # delete collection and all its data
```

#### Documents

```http
GET    /collections/{name}/documents              # list uploaded documents
POST   /collections/{name}/documents              # upload a file (multipart/form-data)
DELETE /collections/{name}/documents/{filename}   # delete a document
```

#### Indexing

```http
POST /collections/{name}/index          # trigger indexing (async, returns task_id)
                                        # body (optional): { "entity_types": ["person", "org"] }
GET  /collections/{name}/index/{task_id} # poll indexing status
GET  /tasks                              # list all pending/running tasks
```

#### Querying

```http
POST /collections/{name}/query
```

```json
{
  "query": "What are the main roles in the system?",
  "method": "hybrid",
  "top_k": 5,
  "stream": false
}
```

| Field | Values | Default |
|---|---|---|
| `method` | `vector` \| `hybrid` \| `global` | `hybrid` |
| `top_k` | integer | `5` |
| `stream` | `true` \| `false` | `false` |

Response includes `answer`, `sources` (chunk excerpts with doc references), and `graphrag_context`.

#### RAGAS Evaluation

```http
GET  /api/metrics                  # list available metrics and their required fields
POST /api/evaluate/single          # evaluate a single sample
POST /api/evaluate/batch           # evaluate a batch from a JSON or CSV file
```

**Single evaluation** (`POST /api/evaluate/single`):

```json
{
  "user_input": "What are the main roles in the system?",
  "response": "The main roles are Master, User, and Viewer.",
  "retrieved_contexts": ["Masters can manage …", "Viewers can only read …"],
  "reference": "The system has three roles: Master, User, and Viewer.",
  "metrics": ["faithfulness", "llm_context_recall", "response_relevancy"]
}
```

Returns `{ "scores": { "faithfulness": 0.95, "llm_context_recall": 0.88, … } }`.

**Batch evaluation** (`POST /api/evaluate/batch`):

Upload a `.json` (array of sample objects) or `.csv` file via `multipart/form-data` with a `metrics` form field (JSON-encoded list of metric IDs).

**Available metric IDs:**

| ID | Display Name | Required Fields | LLM | Embeddings |
|---|---|---|---|---|
| `faithfulness` | Faithfulness | `user_input`, `response`, `retrieved_contexts` | ✓ | |
| `llm_context_recall` | LLM Context Recall | `user_input`, `retrieved_contexts`, `reference` | ✓ | |
| `llm_context_precision` | LLM Context Precision | `user_input`, `retrieved_contexts`, `reference` | ✓ | |
| `context_precision_without_reference` | Context Precision (No Ref) | `user_input`, `response`, `retrieved_contexts` | ✓ | |
| `response_relevancy` | Response Relevancy | `user_input`, `response` | ✓ | ✓ |
| `factual_correctness` | Factual Correctness | `response`, `reference` | ✓ | |
| `noise_sensitivity` | Noise Sensitivity | `user_input`, `retrieved_contexts`, `response`, `reference` | ✓ | |
| `semantic_similarity` | Semantic Similarity | `response`, `reference` | | ✓ |
| `bleu_score` | BLEU Score | `response`, `reference` | | |
| `rouge_score` | ROUGE Score | `response`, `reference` | | |

---

## Project Structure

```
Fujinami/
├── .env                        # environment variables (create this)
├── python/
│   ├── api.py                  # FastAPI application and all HTTP endpoints
│   ├── ragService.py           # RagService: indexing + search logic
│   ├── document_loader.py      # Docling-based loader; converts all supported formats to markdown
│   ├── ragas_runner.py         # RAGAS metric registry and async evaluation runner
│   ├── models.py               # Pydantic request/response schemas
│   ├── install_dependency.py   # Dependency installer script
│   ├── pyproject.toml          # Project metadata and poe tasks
│   ├── static/
│   │   └── index.html          # Single-page Web UI
│   ├── data/                   # Uploaded source documents (per collection)
│   └── ragdata/                # GraphRAG artifacts + LanceDB vector store (per collection)
└── docs/
    └── dataflow-ragService.md  # Detailed pipeline and data-flow documentation
```

---

## Query Modes

| Mode | How it works | Best for |
|---|---|---|
| `vector` | Dense cosine similarity over LanceDB chunk embeddings | Precise factual lookups |
| `hybrid` | Vector search + GraphRAG local search combined | General question answering |
| `global` | GraphRAG community-level summary search | Broad thematic / cross-document questions |

---

## Entity Types

When triggering indexing you can pass a list of entity types to tune the GraphRAG knowledge graph extraction:

```
organization  person  geo  event  concept  technology  product  process  system
```

Omitting `entity_types` uses the GraphRAG defaults.

---

## Error Handling

| Condition | Behaviour |
|---|---|
| Docling models not downloaded | First call to `DocumentConverter` triggers automatic download (~1 GB layout/OCR models); bake into Docker image with `RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"`  |
| VLM picture description fails or times out | Warning logged by Docling; image rendered as placeholder; indexing continues |
| Unsupported file extension | File rejected at upload with HTTP 422 |
| `graphrag index` subprocess fails | Indexing task transitions to `error`; detail message returned |
| Ollama server unreachable | HTTP 500 propagated to API caller |
