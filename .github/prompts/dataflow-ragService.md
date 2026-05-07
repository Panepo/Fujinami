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
│  │  .txt .md   │   │  .py         │   │  lancedb/       │  │
│  │  .pdf .docx │   └──────┬───────┘   │  (persistent)   │  │
│  │  .doc       │          │           └─────────────────┘  │
│  └─────────────┘          │           ┌─────────────────┐  │
│                            │           │  ./ragdata/     │  │
│                            │──────────▶│  GraphRAG index │  │
│                            │           │  (KG artifacts) │  │
│                            │           └─────────────────┘  │
└────────────────────────────┼────────────────────────────────┘
                             │ HTTP (OpenAI-compatible API)
                             ▼
              ┌──────────────────────────────┐
              │  Ollama Server               │
              │  http://10.68.129.51:8088    │
              │                              │
              │  ┌──────────────────────┐    │
              │  │  qwen3.6:9b (LLM)    │    │
              │  └──────────────────────┘    │
              │  ┌──────────────────────┐    │
              │  │  locusai/all-minilm  │    │
              │  │  -l6-v2 (Embeddings) │    │
              │  └──────────────────────┘    │
              │  ┌──────────────────────┐    │
              │  │  llava:7b / minicpm-v│    │
              │  │  (VLM – images)      │    │
              │  └──────────────────────┘    │
              └──────────────────────────────┘
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
│  Walk documents_dir                                          │
│  Filter: .txt  .md  .pdf  .docx  .doc                        │
└────────────────────┬─────────────────────────────────────────┘
                     │ file paths
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 2 — DocumentLoader.load(file_path) → str               │
│  (per file, dispatched by extension)                         │
│                                                              │
│  .txt / .md  ──▶  read as-is                                 │
│  .pdf        ──▶  _load_pdf()   (see §2.2)                   │
│  .docx       ──▶  _load_docx()  (see §2.3)                   │
│  .doc        ──▶  _load_doc()   (see §2.4)                   │
└────────────────────┬─────────────────────────────────────────┘
                     │ plain text (with [IMAGE/DIAGRAM] blocks)
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 3 — Write enriched .txt files to ./data/               │
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
│  --root         │   │  (locusai/all-minilm-l6-v2)          │
│  ./ragdata      │   │                                      │
│                 │   │  → Upsert chunks into LanceDB        │
│  Produces:      │   │    ./ragdata/lancedb/ (on disk)      │
│  entities       │   │    delete rows where doc_id=file     │
│  communities    │   │    then insert new rows              │
│  covariates     │   └──────────────────────────────────────┘
│  reports        │
└─────────────────┘
```

Both Step 4a and 4b run against the same enriched text, enabling dual retrieval at query time.

---

### 2.2 PDF Load Flow (`_load_pdf`)

```
PDF file
   │
   ├──▶ pypdf.PdfReader
   │        │
   │        └── page.extract_text()  ──▶  page_text (str)
   │
   └──▶ pymupdf (fitz.open)
            │
            └── page.get_images(full=True)
                    │ per image block (xref, bbox, position)
                    ▼
            image_bytes  +  surrounding_text
                    │
                    ▼
            _describe_image(image_bytes, surrounding_text)
                    │
                    ▼
            "[IMAGE DESCRIPTION: ...]"
            or "[DIAGRAM: Node A → ... ]"
                    │
                    ▼
   splice at image's bbox position within page_text
            │
            ▼
   full enriched page text
```

---

### 2.3 DOCX Load Flow (`_load_docx`)

```
.docx file
   │
   └──▶ python-docx Document
            │
            ├── paragraph.runs  ──▶  text collected in order
            │
            └── paragraph.InlineShapes
                    │ per shape
                    ▼
            shape.image.blob  (image_bytes)
            +  paragraph.text  (surrounding_text)
                    │
                    ▼
            _describe_image(image_bytes, surrounding_text)
                    │
                    ▼
            "[IMAGE DESCRIPTION: ...]"
            or "[DIAGRAM: ...]"
                    │
                    ▼
   injected at paragraph position in assembled text
```

---

### 2.4 Legacy DOC Load Flow (`_load_doc`)

```
.doc file  (Windows only)
   │
   └──▶ win32com.client.Dispatch("Word.Application")
            │
            └── Document.SaveAs(tmp_path, FileFormat=wdFormatDocx)
                    │
                    ▼
            tmp .docx file
                    │
                    ▼
            _load_docx()  (delegates, then returns text)
```

Non-Windows: raises `NotImplementedError` → caller logs warning, skips file.

---

### 2.5 Image Description Sub-Flow (`_describe_image`)

```
image_bytes  +  surrounding_text
       │
       ▼
_detect_diagram_type(image_bytes)
       │
       │  One VLM call:
       │  POST /api/chat  (llava:7b / minicpm-v)
       │  Prompt: "Is this a photo, chart, flowchart, UML
       │           diagram, table, or other diagram?
       │           Reply with one word."
       │
       ▼
  type_tag: photo | chart | flowchart | uml | table | diagram
       │
       ▼
_build_vlm_prompt(type_tag, surrounding_text)
       │
       │  Template selection:
       │  flowchart/uml  → "List every node and every labeled
       │                    edge as: Node A → [label] → Node B"
       │  chart          → "Describe chart type, axes, series,
       │                    and key values"
       │  table          → "Extract the table as pipe-delimited
       │                    rows"
       │  photo/other    → generic description prompt
       │
       │  surrounding_text injected as grounding context
       │
       ▼
  prompt_str
       │
       ▼
POST http://10.68.129.51:8088/api/chat
  body: { model: "llava:7b",
          messages: [{ role: "user",
                       content: [
                         { type: "image_url",
                           image_url: { url: "data:image/...;base64,<b64>" }},
                         { type: "text", text: prompt_str }
                       ]}]}
       │
       ├── success ──▶ wrap result:
       │               "[DIAGRAM: ...]"  (flowchart/uml)
       │               "[IMAGE DESCRIPTION: ...]"  (others)
       │
       └── failure ──▶ log warning, return ""  (indexing continues)
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
  │  │         all-minilm    │  │         --method local  │ │
  │  │    ──▶ cosine sim     │  │         --root ./ragdata│ │
  │  │         LanceDB       │  │                         │ │
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
                  model: qwen3.6:9b
                  POST http://10.68.129.51:8088/v1/chat/completions
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
OllamaTextEmbeddingGenerationService  (all-minilm-l6-v2)
  POST http://10.68.129.51:8088/v1/embeddings
       │
       ▼
  query_vector  [384-dim float32[384] array]
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
OllamaChatCompletionService  (qwen3.6:9b)
  POST http://10.68.129.51:8088/v1/chat/completions
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
OllamaChatCompletionService  (qwen3.6:9b)
  POST http://10.68.129.51:8088/v1/chat/completions
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
| `vector` | `fixed_size_list<float32>[384]` | Embedding dimension matches `all-minilm-l6-v2` |
| `metadata` | `string` | JSON: `{"page": ..., "source": ..., "chunk_index": ...}` |

---

## 5. External Service Interactions

| Call | Triggered by | Endpoint | Model |
| :--- | :--- | :--- | :--- |
| Diagram type detection | `_detect_diagram_type()` | `POST /api/chat` | `llava:7b` (VLM) |
| Image description | `_describe_image()` | `POST /api/chat` | `llava:7b` (VLM) |
| GraphRAG LLM (entity extraction, summarisation) | `graphrag index` CLI | `/v1/chat/completions` | `qwen3.6:9b` |
| GraphRAG embeddings | `graphrag index` CLI | `/v1/embeddings` | `locusai/all-minilm-l6-v2` |
| SK chunk embedding | `index_documents()` | `/v1/embeddings` | `locusai/all-minilm-l6-v2` |
| SK query embedding | `vector_search()` / `hybrid_search()` | `/v1/embeddings` | `locusai/all-minilm-l6-v2` |
| Final LLM response | all search methods | `/v1/chat/completions` | `qwen3.6:9b` |

All calls target `http://10.68.129.51:8088` using Ollama's OpenAI-compatible REST API.

---

## 6. File Lifecycle

```
SOURCE FILE                 INTERMEDIATE                OUTPUT / INDEX
───────────────             ────────────────────        ──────────────────────────
./documents/foo.pdf  ──▶   ./data/foo.txt         ──▶  ragdata/output/entities
./documents/bar.docx ──▶   ./data/bar.txt         ──▶  ragdata/output/communities
./documents/baz.md   ──▶   ./data/baz.md (copy)   ──▶  ragdata/output/covariates
                                                        ragdata/output/reports
                                │
                                └──▶ SK chunker + embed
                                        │
                                        ▼
                                LanceDB  ./ragdata/lancedb/
                                table: "documents"
                                (persisted on disk;
                                 survives process restart)
```

Image-bearing pages produce enriched `.txt` where raw image bytes are replaced by inline `[DIAGRAM: ...]` or `[IMAGE DESCRIPTION: ...]` text blocks before being written to `./data/`.

---

## 7. Error and Skip Behaviour

| Condition | Handler | Effect on pipeline |
| :--- | :--- | :--- |
| VLM call times out or fails | `_describe_image()` logs warning, returns `""` | Image position left blank; indexing continues |
| `.doc` on non-Windows | `_load_doc()` raises `NotImplementedError` | `load_directory()` logs warning, skips file |
| Unsupported file extension | `DocumentLoader.load()` returns `""` | File excluded from `./data/`; not indexed |
| `graphrag index` subprocess fails | `RagService.index_documents()` raises `RuntimeError` | Indexing halted; LanceDB table may be partially populated |
| Ollama server unreachable | Ollama connector raises connection error | Propagates to caller |
