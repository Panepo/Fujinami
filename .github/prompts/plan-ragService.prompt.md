# Plan: Build `ragService.py` with Semantic Kernel + Microsoft GraphRAG

## Stack
- **LLM:** Ollama (`qwen3.6:9b`) at `http://10.68.129.51:8088`
- **Vector Store:** LanceDB (persistent, file-based) via `lancedb` + Semantic Kernel `LanceDBVectorStore` adapter
- **Graph Engine:** Microsoft GraphRAG (full pipeline — index + search)
- **Data Source:** Local text/markdown/PDF/Word files
- **Document Parsers:** `pypdf` (PDF text), `pymupdf` (PDF image extraction), `python-docx` (`.docx`), `pywin32` (`.doc` legacy on Windows)
- **Vision Model:** Ollama VLM (e.g., `llava:7b`) at the same endpoint — describes embedded images and their context
- **Embedding Model:** `locusai/all-minilm-l6-v2:latest` via Ollama (run `ollama pull locusai/all-minilm-l6-v2:latest` on the host)

---

## Phase 1 — Dependencies

Update `install_dependency.py` to install:
- `semantic-kernel[ollama]` — SK Python SDK with Ollama connector
- `graphrag` — Microsoft GraphRAG engine (Python-native)
- `lancedb` — persistent, embedded vector database (stores data as Arrow/Lance files on disk)
- `pyarrow` — required by LanceDB for schema definition and data serialization
- `pypdf` — PDF text extraction
- `pymupdf` — PDF image extraction (per-page, with position metadata)
- `python-docx` — `.docx` text extraction and inline image access
- `pywin32` — `.doc` (legacy Word) extraction via COM automation (Windows only)

---

## Phase 2 — GraphRAG Configuration

1. Create `python/ragdata/settings.yaml` — configures GraphRAG to use Ollama's OpenAI-compatible endpoint (`http://10.68.129.51:8088/v1`) for both LLM (`qwen3.6:9b`) and embeddings (`locusai/all-minilm-l6-v2:latest`)
2. Create `data/` — placeholder directory where local `.txt`, `.md`, `.pdf`, `.doc`, and `.docx` files go before indexing
3. Add `python/document_loader.py` — helper module that converts PDF/Word documents to plain-text `.txt` files in `./data/` before GraphRAG indexing runs

---

## Phase 3 — `ragService.py` Implementation

Create `python/ragService.py` with a `RagService` class:

| Method | Responsibility |
| :--- | :--- |
| `__init__(root_dir, lance_db_path)` | Build SK kernel with `OllamaChatCompletionService` + `OllamaTextEmbeddingGenerationService`; open (or create) a LanceDB database at `lance_db_path` (default `./ragdata/lancedb`); wrap it with SK's `LanceDBVectorStore` |
| `async index_documents(documents_dir)` | Copy docs to `./data/` → **convert PDF/doc/docx to `.txt` via `DocumentLoader`** → run GraphRAG CLI indexer (subprocess) → chunk + embed docs → **upsert vectors into LanceDB** (persisted on disk; re-indexing the same file overwrites its rows by `doc_id`) |
| `async hybrid_search(query)` | Run SK vector search **and** GraphRAG local search in parallel → merge context → generate response via SK |
| `async global_search(query)` | GraphRAG global search only — broad community-level summaries |
| `async vector_search(query)` | Pure SK semantic similarity search — no graph |

---

## Phase 4 — `DocumentLoader` Helper

Create `python/document_loader.py` with a `DocumentLoader` class:

| Method | Responsibility |
| :--- | :--- |
| `load(file_path) -> str` | Dispatch to the correct parser by extension and return extracted plain text (with inline image descriptions spliced in) |
| `_load_pdf(file_path) -> str` | Use `pypdf.PdfReader` for text per page; use `pymupdf` to extract images per page; call `_describe_image()` for each image and splice description at the image's position in the page text |
| `_load_docx(file_path) -> str` | Use `python-docx` to walk paragraphs and `InlineShape` objects; call `_describe_image()` for each shape and inject description at the run's paragraph position |
| `_load_doc(file_path) -> str` | Use `pywin32` COM automation (`win32com.client`) to open `.doc` and save as `.docx`, then delegate to `_load_docx`; raises `NotImplementedError` on non-Windows |
| `_describe_image(image_bytes, surrounding_text) -> str` | Detect diagram type via `_detect_diagram_type()`; build a tailored prompt via `_build_vlm_prompt()`; base64-encode image and POST to Ollama `/api/chat` with both the image and surrounding paragraph text as context; return result wrapped as `[IMAGE DESCRIPTION: ...]` or `[DIAGRAM: ...]` |
| `_detect_diagram_type(image_bytes) -> str` | Send image to VLM with a classifier prompt (*"Is this a photo, chart, flowchart, UML diagram, table, or other diagram? Reply with one word."*); return a type tag (`photo`, `chart`, `flowchart`, `uml`, `table`, `diagram`) |
| `_build_vlm_prompt(diagram_type, surrounding_text) -> str` | Select a structured prompt template based on diagram type; inject `surrounding_text` as grounding context. Templates:<br>• `flowchart/uml` → *"List every node and every labeled edge as: Node A → [edge label] → Node B"*<br>• `chart` → *"Describe the chart type, axes, data series, and key values"*<br>• `table` → *"Extract the table as pipe-delimited rows"*<br>• `photo/other` → generic description prompt |
| `load_directory(directory) -> dict[str, str]` | Walk a directory, call `load()` on each supported file, return `{filename: text}` |

Supported extensions: `.txt`, `.md`, `.pdf`, `.docx`, `.doc`

**Image processing flow:**
```
document page/paragraph
  ├── text            → extracted as-is
  └── image           → _detect_diagram_type() → type tag
                                ↓
                     _build_vlm_prompt(type, surrounding_text)
                                ↓
              Ollama VLM (image + context prompt)
                                ↓
        "[DIAGRAM: Node A → depends on → Node B ...]"  (structured)
        "[IMAGE DESCRIPTION: bar chart showing ...]"   (photo/chart)
                                ↓
          spliced inline at image's original position
                                ↓
    GraphRAG sees image content + relationships as part of surrounding text context
```

---

## Phase 5 — LanceDB Schema

Define a `PyArrow` schema for the LanceDB table used by the vector store:

| Field | Type | Notes |
| :--- | :--- | :--- |
| `id` | `string` | Unique chunk identifier (`{filename}#{chunk_index}`) |
| `doc_id` | `string` | Source filename — used as upsert key to overwrite on re-index |
| `text` | `string` | Raw chunk text stored for retrieval |
| `vector` | `fixed_size_list<float32>[384]` | Embedding vector (dimension matches `all-minilm-l6-v2`) |
| `metadata` | `string` | JSON-serialized dict (`page`, `source`, `chunk_index`) |

Table name: `documents` (created on first `index_documents()` call; re-opened on subsequent calls).

---

## Phase 6 — Wire Dependencies

Add new packages to the `install_dep()` function in `install_dependency.py` alongside the existing `agent-framework` install.

---

## Files

| File | Action |
| :--- | :--- |
| `python/ragService.py` | **Create** (main deliverable) |
| `python/document_loader.py` | **Create** — PDF/Word-to-text + VLM image description helper |
| `python/install_dependency.py` | **Update** — add pip installs for `semantic-kernel[ollama]`, `graphrag`, `lancedb`, `pyarrow`, `pypdf`, `pymupdf`, `python-docx`, `pywin32` |
| `python/ragdata/settings.yaml` | **Create** — GraphRAG config pointing to Ollama endpoint |
| `python/ragdata/lancedb/` | **Auto-created** — LanceDB stores its Lance/Arrow table files here at runtime |
| `data/` | **Create** — placeholder input directory |

---

## Verification

1. Run `install_dependency.py` and confirm `semantic-kernel` and `graphrag` install without conflict
2. Drop a sample `.md` file into `./data/` and call `index_documents()` — verify `ragdata/output/` is populated with GraphRAG entity/community artifacts
3. Drop a sample `.pdf` into `./data/` and call `index_documents()` — confirm `DocumentLoader` extracts text and the file is indexed correctly
4. Drop a sample `.docx` and `.doc` into `./data/` and call `index_documents()` — confirm both Word formats are extracted and indexed
5. Use a PDF or `.docx` containing an embedded image and call `index_documents()` — confirm the extracted text contains an `[IMAGE DESCRIPTION: ...]` block at the correct position and that GraphRAG indexes it alongside the surrounding text
6. Use a document containing a flowchart or UML diagram — confirm the extracted text contains a `[DIAGRAM: ...]` block with structured `Node → edge → Node` content; query `hybrid_search()` about a relationship shown only in the diagram and confirm the answer reflects it
7. Call `hybrid_search("test query")` and confirm a response is returned with context from both vector search and the graph
8. Call `global_search("test query")` to validate the community-summary path works independently
9. Restart the Python process and call `vector_search("test query")` **without** re-indexing — confirm results are still returned, proving LanceDB persistence survived the restart
10. Re-index the same document and confirm the LanceDB table has the same row count (upsert/overwrite, not duplicate append)

---

## Decisions & Constraints

- Vector store uses **LanceDB** — data is persisted as Lance/Arrow files under `./ragdata/lancedb/`; survives process restarts with no external server required
- GraphRAG indexing uses subprocess CLI (`graphrag index --root ./ragdata`) rather than the internal Python API for stability
- `global_search` is provided as an optional method, not the primary entry point
- `.env` pattern follows `ollamaService.py` convention (`env_file_path=".env"`)
- `DocumentLoader._load_doc` requires Windows + Word installed; on non-Windows environments, `.doc` files are skipped with a warning
- PDF text extraction uses `pypdf`; image extraction uses `pymupdf` — both operate on the same file in sequence
- `_describe_image()` now accepts `surrounding_text` — the paragraph(s) adjacent to the image — and passes it to the VLM as grounding context so diagrams are interpreted relative to the document's subject matter
- Diagram type detection uses a separate lightweight VLM call (classifier prompt); adds one extra round-trip per image but improves prompt targeting
- Structured diagram output (node/edge lists) gives GraphRAG's entity extractor explicit relationship triples to work with, rather than prose
- `_build_vlm_prompt()` prompt templates are defined as constants in `document_loader.py` and can be overridden at instantiation
- VLM image description is a **best-effort** enrichment: if Ollama VLM call fails (timeout, model not pulled), `_describe_image()` logs a warning and returns an empty string so indexing continues
- VLM model must be pulled separately on the Ollama host: `ollama pull minicpm-v` (or `llava:13b`); model name is configurable
- Image descriptions are injected **inline** at the image's position in the document, preserving text–image semantic proximity for GraphRAG entity extraction
- VLM inference adds latency during indexing (one call per image); this is a one-time cost per index build, not per query
- Scanned/image-only PDFs: each page rendered as image via `pymupdf` and described by VLM — effectively provides OCR-equivalent output as a side effect
- LanceDB is opened in **embedded mode** (`lancedb.connect(path)`) — no server process needed; safe for single-process use; concurrent multi-process writes are not supported
- Embedding dimension is fixed at **384** (matching `all-minilm-l6-v2`); changing the model requires dropping and recreating the LanceDB table
- Upsert strategy: delete existing rows where `doc_id == filename` then insert new rows — ensures re-indexing a file does not create duplicates
- Scope **excludes**: custom chunking strategies, auth, multi-tenancy, LanceDB cloud/remote deployment
