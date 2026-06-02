# Data Flow: RAG Service — langchain-ollama + Local graph_engine

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
│                        │            ┌──────────┴────────────┐  │
│                        │            │  SelfReflector        │  │
│                        │            │  (self_reflector.py)  │  │
│                        │            │  self_rag=true path   │  │
│                        │            └──────────┬────────────┘  │
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
  │  │  Vector Search        │  │  Graph Context Search   │ │
  │  │                       │  │  (_graph_context)       │ │
  │  │  query                │  │                         │ │
  │  │    ──▶ OllamaEmbed-   │  │  query                  │ │
  │  │       dings.embed_    │  │    ──▶ Strategy 1:      │ │
  │  │       query()         │  │         spaCy NER +     │ │
  │  │    EMBEDDING_MODEL    │  │         noun-chunks     │ │
  │  │    OLLAMA_CHAT_URL    │  │    ──▶ Strategy 2:      │ │
  │  │    ──▶ ANN cosine     │  │         raw tokens      │ │
  │  │         LanceDB       │  │         (fallback)      │ │
  │  │    ──▶ top-k chunks   │  │    ──▶ Strategy 3:      │ │
  │  │    +  title-keyword   │  │         embedding sim   │ │
  │  │       match (merged)  │  │         (fallback)      │ │
  │  │                       │  │    ──▶ LanceDBGraphStore│ │
  │  │                       │  │         .get_triples()  │ │
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
                  langchain-core messages:
                  SystemMessage: "Answer using only the provided context."
                  HumanMessage:  "Context:\n{merged}\n\nQuestion: {query}"
                               │
                               ▼
                  ChatOllama.ainvoke()
                  model: CHAT_MODEL
                  base_url: OLLAMA_CHAT_URL
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
OllamaEmbeddings.embed_query()  (langchain-ollama)
  model: EMBEDDING_MODEL
  base_url: OLLAMA_CHAT_URL
       │
       ▼
  query_vector
       │
       ├─── ANN search (cosine)
       │    LanceDB ./ragdata/{coll}/lancedb/  table: "documents"
       │    .search(query_vector).limit(TOP_K).to_list()
       │           │
       │           ▼
       │    top-k vector rows  {id, doc_id, text, vector, metadata}
       │
       └─── Title-keyword match  (_title_search_results)
            Scan all rows; keep those whose section_title contains
            any keyword from the query (len > 2, case-insensitive)
            Merged (deduped by id) with vector results
       │
       ▼
  merged rows → assembled context string
       │
       ▼
langchain-core messages [SystemMessage, HumanMessage]
ChatOllama.ainvoke()  (CHAT_MODEL, OLLAMA_CHAT_URL)
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
_graph_context(query)   — three cascading strategies
   │
   ├─ Strategy 1: spaCy NER + noun-chunk extraction
   │    spaCy en_core_web_sm → doc.ents + doc.noun_chunks
   │    Normalise entity names (normalize_name)
   │    Also append raw query tokens (len > 3) — no duplicates
   │    LanceDBGraphStore.get_triples(subject_name=entity)
   │                     + get_triples(object_name=entity)
   │    (LIKE-based, case-insensitive substring match)
   │
   ├─ Strategy 2 (fallback — only when Strategy 1 yields nothing):
   │    Already merged via raw token append above
   │    If no lines produced: proceed to Strategy 3
   │
   └─ Strategy 3 (fallback — only when Strategies 1 & 2 yield nothing):
        store.get_all_entity_names() → list of all stored names
        OllamaEmbeddings.embed_query(query)
        OllamaEmbeddings.embed_documents(all_names)
        Cosine similarity → top-5 names (threshold ≥ 0.5)
        Fetch triples for each matched name
   │
   ▼
  graph_context string
  Format per line: "{subj} [{type}] —{pred}→ {obj} [{type}] (weight=…)"
       │
       ▼
langchain-core messages [SystemMessage, HumanMessage]
ChatOllama.ainvoke()  (CHAT_MODEL, OLLAMA_CHAT_URL)
       │
       ▼
  broad entity/relationship response str
```

---

### 3.4 Self-RAG Path (`self_rag=true`)

When the client sets `self_rag: true`, the `query_collection` endpoint bypasses the standard retrieval path entirely and delegates to `SelfReflector`. This mode is mutually exclusive with streaming (`stream: true` is ignored when `self_rag: true`).

`SelfReflector` is backed by a **LangGraph `QueryGraph`** (`graph_engine/query_graph.py`) — a `StateGraph` that routes adaptively based on context quality.

```
POST /collections/{name}/query  { self_rag: true, … }
       │
       ▼
 api.py — query_collection()
   self_rag=True branch (takes precedence over streaming)
       │
       ▼
 SelfReflector(rag).query(query, method, top_k)
       │
       ▼
 QueryGraph.ainvoke(QueryState)
   │
   ├─ method="vector" or "hybrid":
   │    vector_retrieve_node
   │      retriever_fn(question, top_k) → context_str, sources
   │      Appends to node_trace
   │    evaluate_context_node
   │      ChatOllama: "Is the context sufficient? YES/NO"
   │      needs_graph = reply.startswith("NO") or context empty
   │      Appends to node_trace
   │    ── needs_graph=True  ──► graph_retrieve_node
   │    ── needs_graph=False ──► generate_answer_node
   │
   └─ method="graph":
        graph_retrieve_node
          graph_context_fn(question) → _graph_context() (3 strategies)
          Appends to node_trace
        generate_answer_node
   │
   ▼
 generate_answer_node
   Merges vector context + graph context
   ChatOllama.ainvoke()
   Returns final answer, iterations count
       │
       ▼
 node_trace → translated to list[SelfRagStep]
   (node name → step key/label, duration_ms → result)
       │
       ▼
 (answer, sources, graphrag_context, SelfRagMeta)
       │
       ▼
 QueryResponse {
   collection, method, answer,
   sources, graphrag_context,
   self_rag_meta: SelfRagMeta   ← populated only on self-RAG path
 }
```

`SelfRagMeta` carries the full process log (`process_log: list[SelfRagStep]`) so clients can render a step-by-step transparency panel.

---

### 3.5 Streaming (SSE)

When the client sets `stream: true` in the query request, `api.py` returns a `StreamingResponse` with `text/event-stream` media type. Events are emitted as processing progresses:

| SSE event | Payload | Notes |
| :--- | :--- | :--- |
| `chunks` | JSON array of retrieved `SourceChunk` objects | Emitted first, before tokens |
| `graphrag` | Graph context string | Hybrid/graph only |
| `node_enter` | `{node, timestamp}` | Emitted when a processing node starts |
| `node_complete` | `{node, duration_ms}` | Emitted when a node finishes |
| `routing_decision` | `{needs_graph: bool}` | After evaluate_context (hybrid path) |
| `token` | Individual LLM response token string | Streamed via `ChatOllama.astream()` |
| `sources` | Final JSON array of `SourceChunk` objects | Emitted after all tokens |
| `done` | Empty string | Signals end of stream |
| `error` | `{detail, type, location}` | Emitted on streaming error |

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
| `POST` | `/collections/{name}/index` | `{mode?, force?}` | Trigger async indexing; returns `task_id` immediately (HTTP 202) |
| `GET` | `/collections/{name}/index/{task_id}` | — | Poll index task status: `pending` · `running` · `done` · `error` |
| `GET` | `/tasks` | — | List all pending/running tasks across all collections |
| `POST` | `/collections/{name}/rebuild` | — | Rebuild LanceDB table from cached `embedded.json` files (no re-embedding); returns `task_id` (HTTP 202) |
| `GET` | `/collections/{name}/debug/table` | — | Diagnostic: return total LanceDB row count and distinct `doc_id` list |

### Query

| Method | Path | Body | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/collections/{name}/query` | `{query, method?, top_k?, stream?, self_rag?}` | Query collection. Returns `QueryResponse` or SSE stream |

`method` values: `vector` · `graph` · `hybrid` (default).
`self_rag: true` activates the Self-RAG reflection loop; response includes `self_rag_meta`. Takes precedence over `stream`.
Returns `409` if `index_status == "new_docs"`.

### Knowledge Graph

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/collections/{name}/graph/stats` | Triple count in `graph_triples` table |
| `GET` | `/collections/{name}/graph` | Browse triples; optional query params: `source_doc`, `subject_type`, `predicate` |

### Response Models

**`QueryRequest`**

| Field | Type | Default | Notes |
| :--- | :--- | :--- | :--- |
| `query` | `string` | — | The user query |
| `method` | `"vector"` \| `"graph"` \| `"hybrid"` | `"hybrid"` | Retrieval strategy |
| `top_k` | `int` | `5` | Max vector chunks to retrieve |
| `stream` | `bool` | `false` | Return SSE stream (ignored when `self_rag=true`) |
| `self_rag` | `bool` | `false` | Activate Self-RAG reflection loop |

**`QueryResponse`**

| Field | Type | Notes |
| :--- | :--- | :--- |
| `collection` | `string` | Collection name |
| `method` | `string` | Retrieval method used |
| `answer` | `string` | Generated answer text |
| `sources` | `list[SourceChunk]` \| `null` | Retrieved chunks (null for graph-only) |
| `graphrag_context` | `string` \| `null` | Graph context string (hybrid/graph only) |
| `self_rag_meta` | `SelfRagMeta` \| `null` | Populated only when `self_rag=true` |

**`SelfRagMeta`**

| Field | Type | Notes |
| :--- | :--- | :--- |
| `needed` | `bool` | Whether retrieval was performed |
| `relevant_chunks` | `int` | Chunks that survived relevance filtering |
| `grounded` | `bool` | Whether the final answer passed grounding check |
| `iterations` | `int` | Number of retrieval/refinement iterations |
| `process_log` | `list[SelfRagStep]` | Full step-by-step trace |

**`SelfRagStep`**

| Field | Type | Notes |
| :--- | :--- | :--- |
| `step` | `string` | Machine key derived from `QueryGraph` node names: `vector_retrieve`, `evaluate_context`, `graph_retrieve`, `generate_answer`. Also `error` on failure. |
| `label` | `string` | Human-readable label (node name title-cased, underscores replaced by spaces) |
| `detail` | `string` \| `null` | Extra context (chunk counts, graph context size, evaluation decision) |
| `result` | `string` \| `null` | Node duration in ms (e.g. `"142 ms"`) |
| `ok` | `bool` \| `null` | `true` for completed nodes; `false` on error; `null` = neutral |

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
| Query embedding (retrieval) | `OllamaEmbeddings.embed_query()` (langchain-ollama) | `OLLAMA_CHAT_URL` | `POST /api/embeddings` | `EMBEDDING_MODEL` |
| Entity embedding similarity (graph fallback) | `OllamaEmbeddings.embed_documents()` (langchain-ollama) | `OLLAMA_CHAT_URL` | `POST /api/embeddings` | `EMBEDDING_MODEL` |
| Chat response (non-streaming) | `ChatOllama.ainvoke()` (langchain-ollama) | `OLLAMA_CHAT_URL` | `POST /api/chat` | `CHAT_MODEL` |
| Chat response (streaming) | `ChatOllama.astream()` (langchain-ollama) | `OLLAMA_CHAT_URL` | `POST /api/chat` | `CHAT_MODEL` |
| Context evaluation (Self-RAG) | `ChatOllama.ainvoke()` (QueryGraph evaluate_context_node) | `OLLAMA_CHAT_URL` | `POST /api/chat` | `CHAT_MODEL` |

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
