# Fujinami

A hybrid **Retrieval-Augmented Generation (RAG)** system that combines [Microsoft GraphRAG](https://github.com/microsoft/graphrag), [Semantic Kernel](https://github.com/microsoft/semantic-kernel), and [LanceDB](https://lancedb.github.io/lancedb/) to answer questions over your document collections using locally-hosted [Ollama](https://ollama.com/) models.

---

## Features

- **Hybrid search** — blends dense vector search (LanceDB) with graph-based retrieval (GraphRAG knowledge graph) for richer answers
- **Three query modes** — `vector`, `hybrid`, and `global` (community-level summaries)
- **Multi-collection** — manage independent document collections via a REST API
- **Rich document ingestion** — supports `.txt`, `.md`, `.pdf`, `.docx`, and `.doc` files; embedded images are described inline by a VLM before indexing
- **Streaming responses** — optional token-by-token streaming on query endpoints
- **Built-in Web UI** — zero-configuration browser interface served at `/`
- **Fully local** — all LLM, embedding, and VLM calls go to Ollama; no cloud APIs required

---

## Architecture Overview

```
Documents (.txt .md .pdf .docx .doc)
        │
        ▼
  DocumentLoader  ──▶  VLM image descriptions (llava:7b)
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
# Chat and query-time embeddings (local)
ollama pull llama3.2:3b
ollama pull bge-m3:567m

# Index-time embeddings and VLM (can be on a remote GPU server)
ollama pull bge-m3:567m
ollama pull llava:7b
```

---

## Setup

### 1. Clone and enter the repo

```sh
git clone https://github.com/your-org/Fujinami.git
cd Fujinami
```

### 2. Create a `.env` file

```env
# Remote Ollama server used during indexing (embeddings + VLM)
OLLAMA_INDEX_URL=http://10.168.3.58:8088

# Local Ollama server used at query time
OLLAMA_CHAT_URL=http://localhost:11434

# Model names
CHAT_MODEL=llama3.2:3b
EMBEDDING_MODEL=bge-m3:567m
VLM_MODEL=llava:7b

# Optional: VLM HTTP timeout in seconds (default 180)
VLM_TIMEOUT=180
```

> If you only have one Ollama instance, set both `OLLAMA_INDEX_URL` and `OLLAMA_CHAT_URL` to the same URL.

### 3. Create the virtual environment and install dependencies

```powershell
cd python
poe sync      # creates .venv\fujinami_env312 and runs uv sync
poe install   # pip-installs all Python dependencies
```

Or manually:

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.venv\fujinami_env312'
uv sync
.venv\fujinami_env312\Scripts\Activate.ps1
python install_dependency.py
```

### 4. Start the development server

```powershell
poe dev
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

---

## Project Structure

```
Fujinami/
├── .env                        # environment variables (create this)
├── python/
│   ├── api.py                  # FastAPI application and all HTTP endpoints
│   ├── ragService.py           # RagService: indexing + search logic
│   ├── document_loader.py      # PDF/DOCX/DOC/TXT loader with VLM image descriptions
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
| VLM call fails or times out | Warning logged; image position left blank; indexing continues |
| `.doc` file on non-Windows | File skipped with warning |
| Unsupported file extension | File rejected at upload with HTTP 422 |
| `graphrag index` subprocess fails | Indexing task transitions to `error`; detail message returned |
| Ollama server unreachable | HTTP 500 propagated to API caller |

---

## Development

```powershell
# Activate the venv
.venv\fujinami_env312\Scripts\Activate.ps1

# Run with auto-reload
uvicorn api:app --reload --app-dir python

# Check API docs
start http://localhost:8000/docs
```
