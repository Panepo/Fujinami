# Fujinami RAG Service

A hybrid **Retrieval-Augmented Generation (RAG)** system that combines a local knowledge-graph engine, [Semantic Kernel](https://github.com/microsoft/semantic-kernel), and [LanceDB](https://lancedb.github.io/lancedb/) to answer questions over your document collections using locally-hosted [Ollama](https://ollama.com/) models.

---

## Features

- **Hybrid search** — blends dense vector search (LanceDB) with local knowledge-graph retrieval (`graph_engine`) for richer answers
- **Three query modes** — `vector`, `hybrid`, and `graph` (entity/relationship context)
- **Multi-collection** — manage independent document collections via a REST API
- **Rich document ingestion** — powered by [Docling](https://github.com/DS4SD/docling); supports documents (`.pdf`, `.docx`, `.xlsx`, `.pptx`, `.md`, `.tex`, `.html`, `.csv`, and more), images (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`), audio (`.wav`, `.mp3`, `.m4a`, `.aac`, `.ogg`, `.flac`), and video (`.mp4`, `.avi`, `.mov`); embedded pictures are described inline by a VLM via Docling's built-in picture-description pipeline
- **Incremental indexing** — SHA-256 content-hash delta detection; only new or modified files are reprocessed
- **Streaming responses** — optional SSE token-by-token streaming on query endpoints
- **Knowledge graph browser** — REST endpoints to inspect and filter extracted triples
- **Built-in Web UI** — zero-configuration browser interface served at `/`
- **Fully local** — all LLM, embedding, and VLM calls go to Ollama; no cloud APIs required

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
                           ├─ VLM picture description
                           └─ chunked output with metadata
        │
        ├──────────────────────────┐
        ▼                          ▼
  OllamaEmbedder             GraphPipeline
  (index-time)               (graph_engine/)
  POST /api/embed             spacy | llm | hybrid extractor
  L2-normalised float32       ── triples ──▶ LanceDB
        │                              graph_triples table
        ▼
  LanceDB "documents" table  (per-doc embedded.json cache)
        │
        ▼
  FastAPI server  ──▶  Web UI  /  REST API
        │
        ▼
  Query (vector | hybrid | graph)
  ├─ vector: LanceDB ANN search
  ├─ graph:  spaCy NER → LanceDB triple lookup
  └─ hybrid: both merged
        │
        ▼
  CHAT_MODEL  →  answer + source chunks
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

The exact model names are configured via `.env` variables (`CHAT_MODEL`, `EMBEDDING_MODEL`, `VLM_MODEL`, `EXTRACT_MODEL`). Pull whichever models you configure before first use:

```sh
# Chat and query-time (local, set as CHAT_MODEL / EMBEDDING_MODEL)
ollama pull llama3.2:3b
ollama pull bge-m3:567m

# Index-time VLM for picture description (can be on a remote GPU server, set as VLM_MODEL)
ollama pull llava:7b

# LLM for graph triple extraction (set as EXTRACT_MODEL; can be the same as CHAT_MODEL)
ollama pull llama3.2:3b
```

---

## Setup

### 1. Create a `.env` file

```env
# Remote Ollama server used during indexing (embeddings + VLM + graph extraction)
OLLAMA_INDEX_URL=

# Local Ollama server used at query time
OLLAMA_CHAT_URL=

# Model names
CHAT_MODEL=llama3.2:3b
EMBEDDING_MODEL=bge-m3:567m
VLM_MODEL=llava:7b
EXTRACT_MODEL=llama3.2:3b     # LLM used for graph triple extraction

# Knowledge-graph extraction
GRAPH_EXTRACTOR=hybrid         # spacy | llm | hybrid (default)
GRAPH_CHUNK_SIZE=400
GRAPH_CHUNK_OVERLAP=80

# Optional: VLM HTTP timeout in seconds (default 180)
VLM_TIMEOUT=180

# Optional: number of vector search results to return (default 5)
TOP_K=5

# Optional: number of vector search results to return (default 5)
TOP_K=5
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
GET    /collections/{name}/documents                           # list uploaded documents
POST   /collections/{name}/documents                           # upload a file (multipart/form-data)
DELETE /collections/{name}/documents/{filename}                # delete a document
GET    /collections/{name}/documents/{filename}/download       # download original file
GET    /collections/{name}/documents/{filename}/embedded       # download per-doc embedded.json
GET    /collections/{name}/documents/{filename}/chunks         # list all LanceDB chunks
```

#### Indexing

```http
POST /collections/{name}/index           # trigger indexing (async, returns task_id)
                                         # body (optional):
                                         # { "mode": "all", "force": false,
                                         #   "entity_types": ["person", "org"] }
GET  /collections/{name}/index/{task_id} # poll indexing status
```

`mode` values: `all` (default) · `vector` (LanceDB only) · `graph` (triple extraction only)
`force: true` ignores the file manifest and reprocesses all files.

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
| `method` | `vector` \| `hybrid` \| `graph` | `hybrid` |
| `top_k` | integer | `5` |
| `stream` | `true` \| `false` | `false` |

Response includes `answer`, `sources` (chunk excerpts with doc references), and `graphrag_context` (graph triples used).

#### Knowledge Graph

```http
GET /collections/{name}/graph/stats   # triple count in the graph store
GET /collections/{name}/graph         # browse triples
                                      # optional query params: source_doc, subject_type, predicate
```

---

## Project Structure

```
Fujinami/
├── .env                        # environment variables (create this)
├── api.py                      # FastAPI application and all HTTP endpoints
├── ragService.py               # RagService: thin facade over RagIndexer + RagRetriever
├── retriever.py                # RagRetriever: vector search, graph context, response generation
├── document_loader.py          # Docling-based loader; converts all formats to chunked output
├── models.py                   # Pydantic request/response schemas
├── pyproject.toml              # Project metadata and poe tasks
├── indexer/                    # RagIndexer package
│   ├── pipeline.py             #   orchestration: delta detection → load → embed → upsert
│   ├── delta.py                #   SHA-256 manifest helpers
│   ├── embedder.py             #   OllamaEmbedder (direct /api/embed, L2-normalised)
│   ├── graph.py                #   run_graph_extraction / remove_graph_triples
│   └── store.py                #   LanceDB open/upsert/remove helpers
├── graph_engine/               # Local knowledge-graph extraction package
│   ├── pipeline.py             #   GraphPipeline: chunk → extract → deduplicate → store
│   ├── store.py                #   LanceDBGraphStore (graph_triples table)
│   ├── models.py               #   Triple, Node, Edge dataclasses
│   ├── chunker.py              #   Overlapping text chunker
│   ├── deduplicator.py         #   Triple deduplication by triple_id
│   └── extractors/
│       ├── spacy_extractor.py  #   spaCy NER co-occurrence extractor
│       ├── llm_extractor.py    #   LLM-based structured triple extraction
│       └── hybrid_extractor.py #   SpacyExtractor + LLMExtractor merged
├── static/
│   └── index.html              # Single-page Web UI
├── data/                       # Uploaded source documents (per collection subfolder)
├── ragdata/                    # LanceDB vector + graph store (per collection subfolder)
│   └── {collection}/
│       ├── lancedb/            #   LanceDB DB (documents + graph_triples tables)
│       │   └── file_manifest.json
│       ├── embedded/           #   Per-document embedded.json cache
│       └── index_flags.json    #   {vector_indexed, graph_indexed}
└── docs/
    └── dataflow-ragService.md  # Detailed pipeline and data-flow documentation
```

---

## Query Modes

| Mode | How it works | Best for |
|---|---|---|
| `vector` | Dense cosine similarity over LanceDB chunk embeddings | Precise factual lookups |
| `hybrid` | Vector search + local graph triple lookup, merged context | General question answering |
| `graph` | spaCy NER entity extraction → `graph_triples` triple lookup | Entity/relationship questions |

---

## Graph Extraction

When triggering indexing you can control the extractor via the `GRAPH_EXTRACTOR` env variable (or override per-collection in future):

| Extractor | Description |
|---|---|
| `spacy` | spaCy NER co-occurrence: named entities become nodes; edges from sentence co-occurrence |
| `llm` | LLM-structured extraction: sends chunks to `EXTRACT_MODEL` and parses JSON triples |
| `hybrid` | Runs both `spacy` and `llm`, merges and deduplicates results (default) |

You can also pass `entity_types` in the index request body to hint the LLM extractor:

```
organization  person  geo  event  concept  technology  product  process  system
```

---

## Error Handling

| Condition | Behaviour |
|---|---|
| Docling models not downloaded | First call to `DocumentConverter` triggers automatic download (~1 GB layout/OCR models); bake into Docker image with `RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"` |
| VLM picture description fails or times out | Warning logged by Docling; image rendered as placeholder; indexing continues |
| Unsupported file extension | File rejected at upload with HTTP 422 |
| File fails to load (Docling error) | Warning logged; file excluded from manifest and retried on next index call |
| No changes detected (delta) | `index_documents` returns immediately without calling Ollama |
| `graph_engine` import fails | Graph extraction step skipped with a warning; vector indexing proceeds normally |
| Ollama server unreachable | HTTP 500 propagated to API caller; index task transitions to `"error"` |
| Query on collection with unindexed files | HTTP 409 returned; client must re-index before querying |
