# Data Flow: RAG Service — Semantic Kernel + Local graph_engine

---

## 1. System Context

```
┌────────────────────────────────────────────────────────────────┐
│                         Host Machine                           │
│                                                                │
│  ┌──────────────┐   ┌──────────────────────────────────────┐  │
│  │  ./data/     │   │  api.py (FastAPI REST server)        │  │
│  │  {coll}/     │──▶│  /collections CRUD                   │  │
│  │  .pdf .docx  │   │  /documents CRUD                     │  │
│  │  .xlsx .pptx │   │  /index  (async background task)     │  │
│  │  .md .html   │   │  /query  (vector | graph | hybrid)   │  │
│  │  .png .jpg   │   │  /graph  (triple browsing)           │  │
│  │  .wav .mp3   │   └──────────────┬───────────────────────┘  │
│  │  .mp4 …      │                 │                          │
│  └──────────────┘                  │                          │
│                          ┌─────────┴──────────┐               │
│                          │                    │               │
│                          ▼                    ▼               │
│               ┌──────────────────┐  ┌──────────────────────┐  │
│               │  RagIndexer      │  │  RagRetriever        │  │
│               │  (indexer/)      │  │  (retriever.py)      │  │
│               └────────┬─────────┘  └──────────┬───────────┘  │
│                        │                       │               │
│          ┌─────────────┴──────┐                │               │
│          │                    │                │               │
│          ▼                    ▼                ▼               │
│  ┌───────────────┐  ┌─────────────────┐  ┌──────────────────┐ │
│  │  LanceDB      │  │  graph_engine/  │  │  LanceDB         │ │
│  │  documents    │  │  (local KG      │  │  documents +     │ │
│  │  table        │  │  extraction)    │  │  graph_triples   │ │
│  │  ./ragdata/   │  │  graph_triples  │  │  (read)          │ │
│  │  {coll}/      │  │  table          │  └──────────────────┘ │
│  │  lancedb/     │  └─────────────────┘                       │
│  └───────────────┘                                            │
└────────────────────────────┬───────────────────────────────────┘
                             │ HTTP (Ollama API)
               ┌─────────────┴──────────────┐
               │                            │
               ▼                            ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│  OLLAMA_INDEX_URL        │  │  OLLAMA_CHAT_URL             │
│  (indexing only)         │  │  (querying + chat)           │
│                          │  │                              │
│  ┌────────────────────┐  │  │  ┌──────────────────────┐   │
│  │  EMBEDDING_MODEL   │  │  │  │  CHAT_MODEL          │   │
│  │  (index embeddings)│  │  │  │  (chat responses)    │   │
│  └────────────────────┘  │  │  └──────────────────────┘   │
│  ┌────────────────────┐  │  │  ┌──────────────────────┐   │
│  │  VLM_MODEL         │  │  │  │  EMBEDDING_MODEL     │   │
│  │  (VLM – images)    │  │  │  │  (query embeddings)  │   │
│  └────────────────────┘  │  │  └──────────────────────┘   │
│  ┌────────────────────┐  │  └──────────────────────────────┘
│  │  EXTRACT_MODEL     │  │
│  │  (LLM KG extract)  │  │
│  └────────────────────┘  │
└──────────────────────────┘
```

`RagService` (ragService.py) is a thin facade that owns one `RagIndexer` and one `RagRetriever`. The `api.py` FastAPI server manages a registry of `RagService` instances, one per collection. All model names and URLs are configured via environment variables.

---

## 2. Indexing Pipeline (Write Path)

### 2.1 High-Level Flow

```
User calls: await rag.index_documents(mode="all", force=False)
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 1 — Delta Detection                                    │
│  Load file_manifest.json  ({filename: sha256_hex})           │
│  Walk ./data/{collection}/ recursively                       │
│  Compute SHA-256 hash for each supported file                │
│  → new_files, modified_files, deleted_files                  │
│  If no changes detected → return immediately (no-op)         │
│  force=True → skip manifest, reprocess all files             │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 2 — Cleanup stale data                                 │
│  Remove LanceDB rows  where doc_id ∈ removed_sources         │
│  Remove graph_triples where source_doc ∈ removed_sources     │
└────────────────────┬─────────────────────────────────────────┘
                     │ changed_sources (new + modified)
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 3 — DocumentLoader.load_directory(files_filter=…)      │
│  Single Docling code path for all formats (see §2.2)         │
│                                                              │
│  any format ──▶  Docling DocumentConverter.convert(path)     │
│                      ├─ OCR, table extraction                │
│                      ├─ VLM picture description (VLM_MODEL)  │
│                      └─ export_to_markdown()                  │
│                                                              │
│  Returns: {filename: [chunk_dict, …]}                        │
│  (chunked output with section/page/language metadata)        │
└────────────────────┬─────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
┌─────────────────────┐  ┌───────────────────────────────────┐
│  STEP 4a            │  │  STEP 4b                          │
│  Graph Extraction   │  │  Vector Embedding (LanceDB)       │
│  (mode="graph"|     │  │  (mode="vector"|"all")            │
│  "all")             │  │                                   │
│                     │  │  OllamaEmbedder                   │
│  GraphPipeline.run()│  │  POST {OLLAMA_INDEX_URL}/api/embed│
│  graph_engine/      │  │  model: EMBEDDING_MODEL           │
│                     │  │  L2-normalised float32            │
│  Extractor choices: │  │                                   │
│  GRAPH_EXTRACTOR=   │  │  → Write per-doc embedded.json    │
│  "spacy"            │  │    ./ragdata/{coll}/embedded/     │
│  "llm"              │  │    {stem}.embedded.json           │
│  "hybrid" (default) │  │                                   │
│                     │  │  → Upsert into LanceDB            │
│  Triples stored in  │  │    ./ragdata/{coll}/lancedb/      │
│  graph_triples table│  │    table: "documents"             │
│  (same LanceDB DB)  │  └───────────────────────────────────┘
└─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 5 — Persist manifest + index flags                     │
│  Save file_manifest.json  (skip files that failed to load)   │
│  Save index_flags.json    {vector_indexed, graph_indexed}    │
└──────────────────────────────────────────────────────────────┘
```

Steps 4a and 4b are independent and run sequentially in the same async task. The `mode` parameter controls which steps execute:

| `mode` | Vector (LanceDB) | Graph extraction |
| :--- | :---: | :---: |
| `"all"` (default) | ✓ | ✓ |
| `"vector"` | ✓ | — |
| `"graph"` | — | ✓ |

---

### 2.2 Docling Load Flow (`DocumentLoader.load_directory`)

All input formats share a single code path through Docling's `DocumentConverter`.

```
any supported file
   │
   ▼
DocumentConverter.convert(str(path))
   │
   ├─ Format detection (by extension + magic bytes)
   │
   ├─ Layout model  (LayoutModel / EasyOCR)
   │      Segments pages into: text blocks, tables, figures
   │
   ├─ Table structure model
   │      Reconstructs rows/columns → rendered as Markdown table
   │
   ├─ Picture description pipeline  (PDF + image inputs)
   │      PdfPipelineOptions:
   │        do_picture_description = True
   │        enable_remote_services  = True
   │      PictureDescriptionApiOptions:
   │        url     = "{OLLAMA_INDEX_URL}/v1/chat/completions"
   │        timeout = VLM_TIMEOUT
   │        params  = {"model": VLM_MODEL}
   │        prompt  = "Describe this image in detail ..."
   │      └─ POST /v1/chat/completions  →  description text
   │           on failure: Docling logs warning, uses placeholder
   │
   ├─ ASR pipeline  (audio/video inputs; requires docling[asr] + ffmpeg)
   │      Transcribes audio → plain text
   │
   └─ result.document.export_to_markdown()
          Merges all blocks (text, tables, picture captions)
          into a single Markdown string
   │
   ▼
  chunked list of dicts per document
  [{"chunk_text": ..., "chunk_text_original": ..., "chunk_index": ...,
    "chunk_type": ..., "section_title": ..., "page_number": ...,
    "language": ...}, …]
  (returned to RagIndexer as {filename: [chunk_dict, …]})
```

### 2.3 Graph Extraction Flow (`graph_engine/`)

```
full_text (concatenated original chunk text)
   │
   ▼
GraphPipeline.run(text, source_doc)
   │
   ├─ Chunk text into overlapping windows
   │      chunk_size  = GRAPH_CHUNK_SIZE  (default 400)
   │      chunk_overlap = GRAPH_CHUNK_OVERLAP (default 80)
   │
   ├─ For each chunk, run extractor:
   │
   │  GRAPH_EXTRACTOR="spacy"   → SpacyExtractor
   │      spaCy NER on chunk → named entities as nodes
   │      co-occurrence within sentence → edge (predicate="co-occurs")
   │
   │  GRAPH_EXTRACTOR="llm"     → LLMExtractor
   │      POST {OLLAMA_INDEX_URL}/api/chat
   │      model: EXTRACT_MODEL
   │      Structured prompt → JSON list of triples
   │      {subject, subject_type, predicate, object, object_type, weight}
   │
   │  GRAPH_EXTRACTOR="hybrid"  → HybridExtractor (default)
   │      Run SpacyExtractor + LLMExtractor
   │      Merge and deduplicate results
   │
   ├─ Deduplication (by triple_id = sha256(method+subj+pred+obj))
   │
   └─ LanceDBGraphStore.add_triples(triples)
          table: "graph_triples"  (same LanceDB DB as "documents")
          Upserts on triple_id
```

---

## 3. Query Pipeline (Read Path)

### 3.1 `hybrid_search(query)`

```
User query string
       │
       ▼
  ┌────┴────────────────────────────────────────────────────┐
  │            Run concurrently (asyncio.gather)            │
  │                                                         │
  │  ┌───────────────────────┐  ┌─────────────────────────┐ │
  │  │  SK Vector Search     │  │  Graph Context Search   │ │
  │  │                       │  │                         │ │
  │  │  query                │  │  query                  │ │
  │  │    ──▶ embed (SK)     │  │    ──▶ spaCy NER        │ │
  │  │    EMBEDDING_MODEL    │  │         extract entities│ │
  │  │    OLLAMA_CHAT_URL    │  │    ──▶ LanceDBGraphStore│ │
  │  │    ──▶ cosine sim     │  │         .get_triples()  │ │
  │  │         LanceDB       │  │         by subject +    │ │
  │  │    ──▶ top-k chunks   │  │         object lookup   │ │
  │  │                       │  │    ──▶ formatted lines  │ │
  │  └──────────┬────────────┘  └────────────┬────────────┘ │
  └─────────────┼──────────────────────────── ┼─────────────┘
                │  vector_context             │  graph_context
                └──────────────┬──────────────┘
                               │
                               ▼
                  Merge context strings
                  "Vector Search Results:\n{vector_context}"
                  "Graph Search Results:\n{graph_context}"
                               │
                               ▼
                  SK ChatHistory prompt:
                  system: "Answer using only the provided context."
                  user:   "Context:\n{merged}\n\nQuestion: {query}"
                               │
                               ▼
                  OllamaChatCompletion
                  model: CHAT_MODEL
                  POST {OLLAMA_CHAT_URL}/v1/chat/completions
                               │
                               ▼
                        final response str
```

---

### 3.2 `vector_search(query)` — Pure Semantic Path

```
User query string
       │
       ▼
OllamaTextEmbedding  (EMBEDDING_MODEL)
  POST {OLLAMA_CHAT_URL}/v1/embeddings
       │
       ▼
  query_vector
       │
       ▼
LanceDB  ./ragdata/{coll}/lancedb/  table: "documents"
  ANN search (cosine) against stored chunk vectors
  lancedb.connect(path).open_table("documents")
    .search(query_vector).limit(TOP_K).to_list()
       │
       ▼
  top-k rows  {id, doc_id, text, vector, metadata}
       │
       ▼
  assembled context string
       │
       ▼
SK ChatHistory prompt (system + user with context)
OllamaChatCompletion  (CHAT_MODEL)
  POST {OLLAMA_CHAT_URL}/v1/chat/completions
       │
       ▼
  final response str
```

---

### 3.3 `global_search(query)` — Pure Graph Path

```
User query string
       │
       ▼
_graph_context(query)
   spaCy NER → entity names
   LanceDBGraphStore.get_triples(subject_name=entity)
                                + get_triples(object_name=entity)
   Format: "{subj} [{type}] —{pred}→ {obj} [{type}] (weight=…)"
       │
       ▼
  graph_context string
       │
       ▼
SK ChatHistory prompt (system + user with context)
OllamaChatCompletion  (CHAT_MODEL)
  POST {OLLAMA_CHAT_URL}/v1/chat/completions
       │
       ▼
  broad entity/relationship response str
```

---

### 3.4 Streaming (SSE)

When the client sets `stream: true` in the query request, `api.py` returns a `StreamingResponse` with `text/event-stream` media type. Events are emitted in order:

| SSE event | Payload |
| :--- | :--- |
| `chunks` | JSON array of retrieved `SourceChunk` objects (sent first) |
| `graphrag` | Graph context string (hybrid/graph only) |
| `token` | Individual LLM response tokens (streamed from Ollama) |
| `sources` | Final JSON array of `SourceChunk` objects |
| `done` | Empty string — signals end of stream |

---

## 4. REST API (api.py)

The FastAPI server exposes the following endpoints. All collection-scoped routes take `{name}` as a path parameter matching `^[a-zA-Z0-9_-]+$`.

### Collections

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/collections` | List all collections with `doc_count`, `index_status`, `vector_indexed`, `graph_indexed` |
| `POST` | `/collections` | Create a collection; creates `./data/{name}/` |
| `PATCH` | `/collections/{name}` | Rename a collection (renames both `data/` and `ragdata/` dirs) |
| `DELETE` | `/collections/{name}` | Delete collection and all its data/ragdata |

`index_status` values: `not_indexed` · `indexed` · `new_docs` (unindexed files present).

### Documents

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/collections/{name}/documents` | List documents |
| `POST` | `/collections/{name}/documents` | Upload a document file |
| `DELETE` | `/collections/{name}/documents/{filename}` | Delete a document |
| `GET` | `/collections/{name}/documents/{filename}/download` | Download original file |
| `GET` | `/collections/{name}/documents/{filename}/embedded` | Download per-doc `embedded.json` |
| `GET` | `/collections/{name}/documents/{filename}/chunks` | List all LanceDB chunks for a document |

### Indexing

| Method | Path | Body | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/collections/{name}/index` | `{entity_types?, mode?, force?}` | Trigger async indexing; returns `task_id` immediately (HTTP 202) |
| `GET` | `/collections/{name}/index/{task_id}` | — | Poll index task status: `pending` · `running` · `done` · `error` |

### Query

| Method | Path | Body | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/collections/{name}/query` | `{query, method?, top_k?, stream?}` | Query collection. Returns `QueryResponse` or SSE stream |

`method` values: `vector` · `graph` · `hybrid` (default).
Returns `409` if `index_status == "new_docs"`.

### Knowledge Graph

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/collections/{name}/graph/stats` | Triple count in `graph_triples` table |
| `GET` | `/collections/{name}/graph` | Browse triples; optional query params: `source_doc`, `subject_type`, `predicate` |

---

## 5. Data Stores and Their Contents

| Store | Location | Populated by | Read by |
| :--- | :--- | :--- | :--- |
| Raw input | `./data/{coll}/` | `POST /documents` upload | `RagIndexer` (DocumentLoader) |
| File manifest | `./ragdata/{coll}/lancedb/file_manifest.json` | `RagIndexer` (after each index run) | `RagIndexer` (delta detection) |
| Index flags | `./ragdata/{coll}/index_flags.json` | `RagIndexer` | `api.py` (`_get_index_flags`) |
| Per-doc embedded cache | `./ragdata/{coll}/embedded/{stem}.embedded.json` | `RagIndexer` (embed step) | `RagIndexer` (LanceDB upsert); `GET /embedded` endpoint |
| LanceDB vector chunks | `./ragdata/{coll}/lancedb/` table: `documents` | `RagIndexer` (upsert) | `RagRetriever` (vector search) |
| LanceDB graph triples | `./ragdata/{coll}/lancedb/` table: `graph_triples` | `graph_engine.store.LanceDBGraphStore` | `RagRetriever._graph_context()` |

### LanceDB Table: `documents`

| Field | Arrow Type | Notes |
| :--- | :--- | :--- |
| `id` | `string` | `{filename}#{chunk_index}` — unique chunk key |
| `doc_id` | `string` | Source filename — upsert/delete key |
| `text` | `string` | Raw chunk text returned in search results |
| `vector` | `fixed_size_list<float32>[dim]` | Dimension inferred from first embed call |
| `metadata` | `string` | JSON: `{"chunk_index": …, "chunk_type": …, "section_title": …, "page_number": …, "language": …}` |

### LanceDB Table: `graph_triples`

| Field | Arrow Type | Notes |
| :--- | :--- | :--- |
| `triple_id` | `string` | SHA-256 of `method+subject_id+predicate+object_id` |
| `source_doc` | `string` | Source filename |
| `method` | `string` | Extraction method: `spacy`, `llm`, `hybrid` |
| `subject_id` | `string` | Normalised subject identifier |
| `subject_name` | `string` | Human-readable subject name |
| `subject_type` | `string` | NER/LLM entity type |
| `predicate` | `string` | Relation label |
| `object_id` | `string` | Normalised object identifier |
| `object_name` | `string` | Human-readable object name |
| `object_type` | `string` | NER/LLM entity type |
| `weight` | `float32` | Confidence/frequency score |
| `subject_specs` | `string` | JSON extra attributes for subject node |
| `object_specs` | `string` | JSON extra attributes for object node |

---

## 6. External Service Interactions

| Call | Triggered by | Server | Endpoint | Env var |
| :--- | :--- | :--- | :--- | :--- |
| VLM picture description | `DocumentLoader` (`PictureDescriptionApiOptions`) | `OLLAMA_INDEX_URL` | `POST /v1/chat/completions` | `VLM_MODEL` |
| Chunk embedding (index) | `OllamaEmbedder.embed()` | `OLLAMA_INDEX_URL` | `POST /api/embed` | `EMBEDDING_MODEL` |
| LLM triple extraction | `LLMExtractor` / `HybridExtractor` | `OLLAMA_INDEX_URL` | `POST /api/chat` | `EXTRACT_MODEL` |
| Query embedding (retrieval) | `OllamaTextEmbedding` (SK) | `OLLAMA_CHAT_URL` | `POST /v1/embeddings` | `EMBEDDING_MODEL` |
| Chat response (non-streaming) | `OllamaChatCompletion` (SK) | `OLLAMA_CHAT_URL` | `POST /v1/chat/completions` | `CHAT_MODEL` |
| Chat response (streaming) | `OllamaChatCompletion.get_streaming_…` | `OLLAMA_CHAT_URL` | `POST /v1/chat/completions` | `CHAT_MODEL` |

Key environment variables:

| Variable | Purpose |
| :--- | :--- |
| `OLLAMA_INDEX_URL` | Ollama server for indexing (embeddings, VLM, LLM extraction) |
| `OLLAMA_CHAT_URL` | Ollama server for query-time embeddings and chat |
| `EMBEDDING_MODEL` | Model used for both index-time and query-time embeddings |
| `VLM_MODEL` | Vision-language model for picture descriptions |
| `CHAT_MODEL` | LLM for final answer generation |
| `EXTRACT_MODEL` | LLM for graph triple extraction (LLM/hybrid extractor) |
| `GRAPH_EXTRACTOR` | Extractor type: `spacy`, `llm`, or `hybrid` (default) |
| `GRAPH_CHUNK_SIZE` | Characters per chunk for graph extraction (default 400) |
| `GRAPH_CHUNK_OVERLAP` | Overlap between chunks (default 80) |
| `TOP_K` | Default number of vector search results (default 5) |
| `VLM_TIMEOUT` | HTTP timeout in seconds for VLM calls (default 180) |

---

## 7. File Lifecycle

```
SOURCE FILE                  PIPELINE                     OUTPUT / INDEX
──────────────────           ──────────────────────       ──────────────────────────────
./data/{coll}/foo.pdf
./data/{coll}/bar.xlsx  ──▶  DocumentLoader               ragdata/{coll}/embedded/
./data/{coll}/img.png        (Docling convert              {stem}.embedded.json
./data/{coll}/audio.mp3       → markdown chunks)
                                   │
                          ┌────────┴────────┐
                          ▼                 ▼
                   OllamaEmbedder     GraphPipeline
                   (index-time)       (graph extraction)
                          │                 │
                          ▼                 ▼
                   LanceDB                LanceDB
                   "documents"            "graph_triples"
                   table                  table
                   (persisted on disk;    (persisted on disk;
                    survives restart)      survives restart)
                          │
                          ▼
                   file_manifest.json
                   index_flags.json
```

Docling converts every input format to Markdown before chunking. Tables are rendered as pipe-delimited Markdown; embedded images are replaced by VLM-generated description text inline in the chunk.

---

## 8. Error and Skip Behaviour

| Condition | Handler | Effect on pipeline |
| :--- | :--- | :--- |
| VLM picture description fails / times out | Docling logs warning, inserts placeholder text | Indexing continues with partial content |
| Docling layout/OCR models not cached | Auto-downloaded on first `DocumentConverter()` call (~1 GB) | One-time cold start; bake into Docker with `RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"` |
| Audio/video on system without ffmpeg | Docling raises `ConversionError` | `load_directory()` logs warning, skips file; file excluded from manifest (retried next run) |
| Unsupported file extension | `upload_document` returns HTTP 422 | File rejected at API boundary |
| File fails to load (Docling error) | `RagIndexer` logs warning, excludes from manifest | File retried on next index call |
| No changes detected (delta) | `RagIndexer.index_documents()` returns immediately | No-op; manifest unchanged |
| `LanceDBGraphStore` not available (import error) | `run_graph_extraction()` logs warning, returns | Vector indexing proceeds; graph skipped |
| Ollama server unreachable | Raises connection error | Propagates to caller; task status set to `"error"` |
| Query on collection with `new_docs` | `api.py` returns HTTP 409 | Client must re-index before querying |
