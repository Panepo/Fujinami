# Data Flow: `ragService.py` — Semantic Kernel + Microsoft GraphRAG

---

## 1. System Context

```
┌─────────────────────────────────────────────────────────────┐
│                        Host Machine                          │
│                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │  ./data/    │   │  RagService  │   │  LanceDB        │  │
│  │  (raw docs) │──▶│  ragService  │──▶│  ./ragdata/     │  │
│  │  .pdf .docx │   │  .py         │   │  lancedb/       │  │
│  │  .xlsx .pptx│   └──────┬───────┘   │  (persistent)   │  │
│  │  .md .html  │          │           └─────────────────┘  │
│  │  .png .jpg  │          │           ┌─────────────────┐  │
│  │  .wav .mp3  │          │           │  ./ragdata/     │  │
│  │  .mp4 …     │          │──────────▶│  GraphRAG index │  │
│  └─────────────┘          │           │  (KG artifacts) │  │
│                            │           └─────────────────┘  │
└────────────────────────────┼────────────────────────────────┘
                             │ HTTP (OpenAI-compatible API)
               ┌─────────────┴──────────────┐
               │                            │
               ▼                            ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│  OLLAMA_INDEX_URL        │  │  OLLAMA_CHAT_URL             │
│  http://10.168.3.58:8088 │  │  http://localhost:11434      │
│  (indexing only)         │  │  (querying + chat)           │
│                          │  │                              │
│  ┌────────────────────┐  │  │  ┌──────────────────────┐   │
│  │  bge-m3:567m       │  │  │  │  llama3.2:3b (Chat)  │   │
│  │  (index embeddings)│  │  │  └──────────────────────┘   │
│  └────────────────────┘  │  │  ┌──────────────────────┐   │
│  ┌────────────────────┐  │  │  │  bge-m3:567m         │   │
│  │  llava:7b          │  │  │  │  (query embeddings)  │   │
│  │  (VLM – images)    │  │  │  └──────────────────────┘   │
│  └────────────────────┘  │  └──────────────────────────────┘
└──────────────────────────┘
```

---

## 2. Indexing Pipeline (Write Path)

### 2.1 High-Level Flow

```
User calls: await rag.index_documents("./documents")
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 1 — File Discovery                                     │
│  Walk documents_dir recursively                              │
│  Filter: SUPPORTED_EXTENSIONS (~30 types)                    │
│  docs: .pdf .docx .xlsx .pptx .md .tex .html .csv …         │
│  imgs: .png .jpg .jpeg .tiff .bmp .webp                      │
│  audio: .wav .mp3 .m4a .aac .ogg .flac                       │
│  video: .mp4 .avi .mov                                       │
└────────────────────┬─────────────────────────────────────────┘
                     │ file paths
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 2 — DocumentLoader.load(file_path) → str               │
│  Single path for all formats (see §2.2)                      │
│                                                              │
│  any format ──▶  Docling DocumentConverter.convert(path)     │
│                      ├─ OCR, table extraction                │
│                      ├─ VLM picture description (llava:7b)   │
│                      └─ export_to_markdown()                  │
└────────────────────┬─────────────────────────────────────────┘
                     │ markdown text (tables, headings, picture descriptions)
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 3 — Write markdown files to ./data/                    │
│  (GraphRAG input directory)                                  │
└────────────────────┬─────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
┌─────────────────┐   ┌──────────────────────────────────────┐
│  STEP 4a        │   │  STEP 4b                             │
│  GraphRAG CLI   │   │  SK Embedding                        │
│  (subprocess)   │   │                                      │
│                 │   │  Chunk text → OllamaTextEmbedding    │
│  graphrag index │   │  GenerationService                   │
│  --root         │   │  bge-m3:567m @ OLLAMA_INDEX_URL      │
│  ./ragdata      │   │                                      │
│                 │   │  → Upsert chunks into LanceDB        │
│  Produces:      │   │    ./ragdata/lancedb/ (on disk)      │
│  entities       │   │    delete rows where doc_id=file     │
│  communities    │   │    then insert new rows              │
│  covariates     │   └──────────────────────────────────────┘
│  reports        │
└─────────────────┘
```

Both Step 4a and 4b run against the same markdown text, enabling dual retrieval at query time.

---

### 2.2 Docling Load Flow (`DocumentLoader.load`)

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
   │        params  = {"model": "llava:7b"}
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
  markdown str  (returned to RagService for indexing)
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
  │  │  SK Vector Search     │  │  GraphRAG Local Search  │ │
  │  │                       │  │                         │ │
  │  │  query                │  │  query                  │ │
  │  │    ──▶ embed (Ollama) │  │    ──▶ graphrag query   │ │
  │  │    bge-m3:567m        │  │         --method local  │ │
  │  │    OLLAMA_CHAT_URL    │  │         --root ./ragdata│ │
  │  │    ──▶ cosine sim     │  │  (query_embedding_model │ │
  │  │         LanceDB       │  │   @ OLLAMA_CHAT_URL)    │ │
  │  │    ──▶ top-k chunks   │  │    ──▶ entity/community │ │
  │  │                       │  │         context strings │ │
  │  └──────────┬────────────┘  └────────────┬────────────┘ │
  └─────────────┼──────────────────────────── ┼─────────────┘
                │  vector_context             │  graph_context
                └──────────────┬──────────────┘
                               │
                               ▼
                  Merge context strings
                  (vector_context + graph_context)
                               │
                               ▼
                  SK Kernel prompt template:
                  "Using the following context, answer: {query}
                   Context: {merged_context}"
                               │
                               ▼
                  OllamaChatCompletionService
                  model: llama3.2:3b  (CHAT_MODEL)
                  POST http://localhost:11434/v1/chat/completions
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
OllamaTextEmbeddingGenerationService  (bge-m3:567m)
  POST http://localhost:11434/v1/embeddings  (OLLAMA_CHAT_URL)
       │
       ▼
  query_vector
       │
       ▼
LanceDB  ./ragdata/lancedb/  table: "documents"
  ANN search (cosine) against stored chunk vectors
  lancedb.connect(path).open_table("documents")
    .search(query_vector).limit(5).to_list()
       │
       ▼
  top-k rows  {id, doc_id, text, vector, metadata}
       │
       ▼
  assembled context string
       │
       ▼
OllamaChatCompletionService  (llama3.2:3b)
  POST http://localhost:11434/v1/chat/completions  (OLLAMA_CHAT_URL)
       │
       ▼
  final response str
```

---

### 3.3 `global_search(query)` — Community Summary Path

```
User query string
       │
       ▼
graphrag query subprocess
  --method global
  --root ./ragdata
  (reads community_reports.parquet / JSON artifacts
   produced during indexing)
       │
       ▼
  community-level summary context
       │
       ▼
OllamaChatCompletionService  (llama3.2:3b)
  POST http://localhost:11434/v1/chat/completions  (OLLAMA_CHAT_URL)
       │
       ▼
  broad thematic response str
```

---

## 4. Data Stores and Their Contents

| Store | Location | Populated by | Read by |
| :--- | :--- | :--- | :--- |
| Raw input | `./data/` | `index_documents()` copy + `DocumentLoader` | GraphRAG CLI |
| GraphRAG artifacts | `./ragdata/output/` | `graphrag index` CLI | `local_search`, `global_search` |
| LanceDB vector chunks | `./ragdata/lancedb/` (on disk, persistent) | `index_documents()` embedding + upsert loop | `vector_search`, `hybrid_search` |
| GraphRAG config | `./ragdata/settings.yaml` | Static file (created once) | GraphRAG CLI at every index/query run |

### LanceDB Table Schema (`documents`)

| Field | Arrow Type | Notes |
| :--- | :--- | :--- |
| `id` | `string` | `{filename}#{chunk_index}` — unique chunk key |
| `doc_id` | `string` | Source filename — upsert/delete key |
| `text` | `string` | Raw chunk text returned in search results |
| `vector` | `fixed_size_list<float32>[384]` | Embedding dimension matches `bge-m3:567m` |
| `metadata` | `string` | JSON: `{"page": ..., "source": ..., "chunk_index": ...}` |

---

## 5. External Service Interactions

| Call | Triggered by | Server | Endpoint | Model |
| :--- | :--- | :--- | :--- | :--- |
| VLM picture description | Docling `_build_converter()` (via `PictureDescriptionApiOptions`) | `OLLAMA_INDEX_URL` | `POST /v1/chat/completions` | `llava:7b` |
| GraphRAG index completion (entity extraction, summarisation) | `graphrag index` CLI | `OLLAMA_CHAT_URL` | `/v1/chat/completions` | `llama3.2:3b` |
| GraphRAG index embedding (`indexing_embedding_model`) | `graphrag index` CLI | `OLLAMA_INDEX_URL` | `/v1/embeddings` | `bge-m3:567m` |
| SK chunk embedding | `index_documents()` | `OLLAMA_INDEX_URL` | `/v1/embeddings` | `bge-m3:567m` |
| GraphRAG query completion | `graphrag query` CLI | `OLLAMA_CHAT_URL` | `/v1/chat/completions` | `llama3.2:3b` |
| GraphRAG query embedding (`query_embedding_model`) | `graphrag query` CLI | `OLLAMA_CHAT_URL` | `/v1/embeddings` | `bge-m3:567m` |
| SK query embedding | `vector_search()` / `hybrid_search()` | `OLLAMA_CHAT_URL` | `/v1/embeddings` | `bge-m3:567m` |
| SK final LLM response | all search methods | `OLLAMA_CHAT_URL` | `/v1/chat/completions` | `llama3.2:3b` |

- `OLLAMA_INDEX_URL` = `http://10.168.3.58:8088` — remote GPU server (indexing, VLM, index-time embeddings)
- `OLLAMA_CHAT_URL` = `http://localhost:11434` — local server (all query-time calls)

---

## 6. File Lifecycle

```
SOURCE FILE                    INTERMEDIATE                   OUTPUT / INDEX
──────────────────             ──────────────────────         ──────────────────────────
./documents/foo.pdf   ──▶    ./data/foo.md            ──▶   ragdata/output/entities
./documents/bar.xlsx  ──▶    ./data/bar.md            ──▶   ragdata/output/communities
./documents/img.png   ──▶    ./data/img.md            ──▶   ragdata/output/covariates
./documents/audio.mp3 ──▶    ./data/audio.md          ──▶   ragdata/output/reports
                                        │
                                        └──▶ SK chunker + embed
                                                │
                                                ▼
                                LanceDB  ./ragdata/lancedb/
                                table: "documents"
                                (persisted on disk;
                                 survives process restart)
```

Docling converts every input format to Markdown before writing to `./data/`. Tables are rendered as pipe-delimited Markdown, and embedded images/pictures are replaced by Docling-generated description text inline in the document.

---

## 7. Error and Skip Behaviour

| Condition | Handler | Effect on pipeline |
| :--- | :--- | :--- |
| VLM picture description fails or times out | Docling logs warning, inserts placeholder text | Indexing continues with partial content |
| Docling layout/OCR models not cached | Auto-downloaded on first `DocumentConverter()` call (~1 GB) | One-time cold start; bake into Docker with `RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"` |
| Audio/video on system without ffmpeg | Docling raises `ConversionError` | `load_directory()` logs warning, skips file |
| Unsupported file extension | `DocumentLoader.load()` raises `ValueError` | File excluded from `./data/`; not indexed |
| `graphrag index` subprocess fails | `RagService.index_documents()` raises `RuntimeError` | Indexing halted; LanceDB table may be partially populated |
| Ollama server unreachable | Ollama connector raises connection error | Propagates to caller |
