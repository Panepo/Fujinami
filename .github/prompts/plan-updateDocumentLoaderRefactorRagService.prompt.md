# Plan: Update document_loader + Refactor ragService

## TL;DR
Update `document_loader.py` to adopt the 5-stage structured pipeline from `.github/reference/document_loader/`, then split `ragService.py` into `indexer.py` (indexing) and `retriever.py` (search/query), keeping a thin `ragService.py` façade for backward-compat with `api.py`.

---

## Phase 1 — Update document_loader.py

**Goal:** Replace the simple Docling→Markdown conversion with a structured 5-stage pipeline, producing rich chunk dicts with metadata. Preserve existing behavior for formats not in the reference.

### Steps

1. **Stage 1 (parse) — structured element extraction**
   - Keep Docling's `DocumentConverter` as the engine
   - Replace `export_to_markdown()` with recursive element walk (from `01_parse.py`): headings, paragraphs, list_items, tables, pictures
   - Track section hierarchy via stack, record page numbers
   - For XLSX: use openpyxl-based element extraction (from `01_parse.py._parse_excel`)
   - For PPTX: detect heading shapes via bold/font-size/color signals (from `01_parse.py._pptx_heading_texts`)
   - Extract embedded images to temp dir as PNGs

2. **Stage 2 (tables) — table classification + LLM narration (always on)**
   - For each table element: detect ASCII box-drawing tables
   - Classify as `faq | spec | general` (from `02_table.py._classify_table_type`)
   - FAQ tables: expand each row to its own chunk (`02_table.py._faq_row_to_text`)
   - LLM narration via Ollama HTTP (reuse `OLLAMA_INDEX_URL` from ragService env)
   - Output: list of table chunk dicts with `ocr_difficulty`, `rows`, `cols`, `table_type`

3. **Stage 3 (vision) — VLM image summarization**
   - Replace current single-pass VLM description with Ollama 3-pass pipeline (from `03_vision.py.OllamaVisionSummarizer`):
     - Pass 1: image type classification (flowchart/architecture/chart/table/etc.)
     - Pass 2: type-adaptive structured extraction (`_STRUCTURED_PROMPTS`)
     - Pass 3: synthesis
   - Apply `_size_guard` to avoid hallucination on small images (<50k px²)
   - Upscale images <400×150px before sending to VLM
   - Reuse `vlm_model` and `ollama_base_url` constructor parameters (no API break)

4. **Stage 4 (text chunking) — smart RCTS chunking**
   - Replace current simple split with `RecursiveCharacterTextSplitter` (RCTS) from langchain (from `04_text_chunk.py`)
   - CJK-aware separators (。、；：)
   - Build contextual prefix per chunk: `[Document: {stem} | Section: {title} | Page: {n}]`
   - Store both `chunk_text_original` and `chunk_text_embedded` (prefix + text)
   - Apply `_merge_short_text_chunks` and `_merge_warning_headers` post-processing
   - Inject vision text at correct reading position (respects element order)
   - Graceful fallback if langchain not installed (simple character split)
   - Read `CHUNK_SIZE` and `CHUNK_OVERLAP` from env (defaults: `800` / `80`)

5. **Stage 5 (metadata) — enrichment**
   - Language detection via `langdetect` (ZH-TW, EN, fallback EN)
   - SHA-256 hash per chunk (`chunk_hash`)
   - Final `chunk_type` assignment: `picture | parts_table | procedure_step | text`
   - Merge all chunk sources (text, table, vision) into unified list

6. **Preserve unchanged behavior for unsupported formats**
   - Audio/video (`.wav`, `.mp3`, `.m4a`, `.aac`, `.ogg`, `.flac`, `.mp4`, `.avi`, `.mov`) — keep existing Docling→Markdown path
   - VTT (`.vtt`) — keep as-is
   - HTML/HTM/XHTML — keep as-is
   - MD/Markdown, ADOC/AsciiDoc, TeX — keep as-is
   - For these formats: wrap output as single text chunk with minimal metadata

7. **Update public API**
   - `load(file_path)` → returns `list[dict]` (chunk list) instead of `str`
   - `load_directory(directory, files_filter)` → returns `dict[str, list[dict]]` (filename → chunks)
   - `DocumentLoader.__init__` keeps same parameters (no break to RagService constructor)

---

## Phase 2 — Refactor ragService.py

**Goal:** Split the monolithic class into indexing and retrieval concerns. Preserve `api.py` compatibility via a thin `RagService` façade.

### Files

| File | Class | Responsibility |
|------|-------|---------------|
| `indexer.py` (new) | `RagIndexer` | delta detection, document loading, graph extraction, LanceDB upserting |
| `retriever.py` (new) | `RagRetriever` | vector search, graph triple search, hybrid search, response generation |
| `ragService.py` (updated) | `RagService` | thin façade composing `RagIndexer` + `RagRetriever`; same public API as today |
| `graph_engine/` (new) | — | copied from reference; `pipeline`, `models`, `store`, `extractors`, `chunker`, `deduplicator` |

### Steps

8. **Create `indexer.py` with `RagIndexer`**
   - Constructor: same params as current `RagService.__init__` (`collection_name`, `root_dir`, `lance_db_path`)
   - Methods moved from `RagService`:
     - `index_documents(documents_dir, entity_types, mode)`
     - `_load_index_flags()`, `_save_index_flags()`
     - `_compute_delta()` — **updated**: manifest stores `{filename: sha256_of_file_bytes}` instead of `{mtime, size}`
     - `_upsert_to_lancedb()` — updated to accept pre-chunked dicts from new loader
     - `_remove_from_lancedb()` — also calls `GraphStore.delete_by_source(source_doc)` to purge triples
     - `_chunk_text()` — **remove** (replaced by loader's Stage 4 chunking)
     - `_ensure_settings_yaml()` — **remove** (GraphRAG CLI config no longer needed)
     - `_run_graphrag_index()` — **remove** (replaced by `_run_graph_extraction()`)
   - **Add** `_run_graph_extraction(source_doc, full_text)` using `GraphPipeline`:
     - Instantiate `LanceDBGraphStore(lance_db_path)` (separate `graph_triples` table)
     - Instantiate extractor from `GRAPH_EXTRACTOR` env var: `"spacy"`, `"llm"`, or `"hybrid"` (default `"hybrid"`)
     - Call `pipeline.run(text=full_text, source_doc=filename)`
   - Update `index_documents` to consume `dict[str, list[dict]]` from loader
   - Full text for graph extraction: join `chunk_text_original` fields per document
   - LanceDB upsert: embed `chunk_text_embedded`, store `chunk_text_original` in `text` field
   - **No more `.txt` file writing**, no subprocess calls

9. **Create `retriever.py` with `RagRetriever`**
   - Constructor: same params
   - Methods moved from `RagService`:
     - `vector_search(query, top_k)`
     - `global_search(query)`
     - `hybrid_search(query, top_k)`
     - `get_document_chunks(filename)`
     - `_raw_vector_context()`, `_raw_vector_results()`
     - `_graphrag_search()` — **replace** with `_graph_context(query)` using `GraphStore`
     - `_generate_response()`
   - Shared state (LanceDB path, SK services, collection config) passed via constructor
   - Read `TOP_K` from env (default: `5`) as the default for `top_k` parameters
   - **`_graph_context(query)`** implementation:
     - Extract candidate entity names from query using lightweight spaCy NER
     - For each entity: `GraphStore.get_triples(subject_name=entity)` + `get_triples(object_name=entity)`
     - Format triples as: `"{subject} [{subject_type}] —{relation}→ {object} [{object_type}] (weight={w:.2f})"`
     - Return formatted string as graph context (replaces GraphRAG community summaries)
   - **`global_search(query)`**: uses `_graph_context()` only (no subprocess)
   - **`hybrid_search(query)`**: parallel `_raw_vector_context()` + `_graph_context()`, merged

10. **Update `ragService.py` as thin façade**
    - `RagService.__init__` instantiates both `RagIndexer` and `RagRetriever`
    - Delegates all method calls to the appropriate sub-object
    - No logic changes — purely delegation
    - `api.py` requires zero changes

---

## Phase 3 — Integrate graph_engine module

**Goal:** Copy and wire the reference `graph_engine/` package into the project, replacing the Microsoft GraphRAG subprocess dependency entirely.

### Steps

11. **Copy `graph_engine/` into project root**
    - Copy `.github/reference/graph_engine/` → `graph_engine/` (next to `indexer.py`, `retriever.py`)
    - Modules to copy: `__init__.py`, `base.py`, `models.py`, `chunker.py`, `deduplicator.py`, `store.py`, `pipeline.py`, `extractors/__init__.py`, `extractors/llm_extractor.py`, `extractors/spacy_extractor.py`, `extractors/hybrid_extractor.py`
    - `store.py` — already uses LanceDB (`graph_triples` table); no changes needed
    - `LanceDBGraphStore` requires same `lance_db_path` used by `RagIndexer`

12. **Update `requirements.txt`**
    - **Add**: `spacy`, `langdetect`
    - **Add**: `spacy` model download step in README/Dockerfile: `python -m spacy download en_core_web_sm`
    - **Remove**: `graphrag` package

13. **Add new env vars to `.env` / `.env.example`**
    - `EXTRACT_MODEL` — LLM model for `LLMExtractor` / `HybridExtractor` (e.g. `gemma3:4b`); can share `INDEX_MODEL`
    - `GRAPH_EXTRACTOR` — `"spacy"` / `"llm"` / `"hybrid"` (default `"hybrid"`)

14. **Remove GraphRAG config artifacts**
    - Delete `ragdata/settings.yaml` template (no longer used)
    - `_ensure_settings_yaml()` removed from `indexer.py` (Step 8)
    - GraphRAG output dirs (`ragdata/{collection}/output/`, `ragdata/{collection}/cache/`) no longer created
    - `index_flags.json` still retained (tracks `vector_indexed` / `graph_indexed`)

---

## Relevant Files

- `document_loader.py` — Full rewrite of `DocumentLoader` class
- `ragService.py` — Replace class body with façade delegation
- `indexer.py` — **New file** with `RagIndexer`
- `retriever.py` — **New file** with `RagRetriever`
- `graph_engine/` — **New directory** copied from `.github/reference/graph_engine/`
- `requirements.txt` — Add `langdetect`, `spacy`; remove `graphrag`
- `.env` / `.env.example` — Add `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `EXTRACT_MODEL`, `GRAPH_EXTRACTOR`
- `ragdata/settings.yaml` — **Delete** (GraphRAG CLI config no longer needed)
- `.github/reference/document_loader/01_parse.py` — Stage 1 template
- `.github/reference/document_loader/02_table.py` — Stage 2 template
- `.github/reference/document_loader/03_vision.py` — Stage 3 template
- `.github/reference/document_loader/04_text_chunk.py` — Stage 4 template
- `.github/reference/document_loader/05_metadata.py` — Stage 5 template
- `.github/reference/graph_engine/` — graph_engine template source
- `api.py` — **No changes needed** (depends only on RagService public API)

---

## Verification

1. Start service and create a collection — confirm no import errors, no GraphRAG imports
2. Upload a PDF and trigger indexing — verify chunks appear in LanceDB `documents` table with `chunk_type`, `language`, `chunk_hash`
3. Upload a PPTX/XLSX — verify table chunks with `table_type` and `ocr_difficulty`
4. Upload an image-heavy PDF — verify picture chunks with VLM summaries
5. After indexing — verify `graph_triples` table in LanceDB contains extracted triples with `source_doc`, `method`, `subject_name`, `predicate`, `object_name`
6. Query with `method=global` — verify response uses triple-based graph context (not community summaries)
7. Query with `method=hybrid` — verify answer merges vector chunks + graph triples; sources returned
8. Re-index same files — manifest hash comparison skips unchanged files
9. Delete a document, re-index — verify both `documents` rows and `graph_triples` rows for that `source_doc` are removed
10. Check no `graphrag` subprocess is spawned during indexing or querying

---

## Decisions

- `ragService.py` kept as façade → `api.py` unchanged
- Audio/video/VTT/HTML/ADOC/TEX/MD formats: wrap as single text chunk, minimal metadata
- langchain RCTS: optional dependency with simple-split fallback
- `load()` return type changes from `str` to `list[dict]` — breaking change within the service only (`api.py` never calls loader directly)
- **Table LLM narration: always on** — every table calls Ollama to generate natural language text
- **langdetect: required** — add to `requirements.txt`
- **Delta detection: SHA-256 content hash replaces mtime/size** — manifest stores `{filename: hash}`, recomputed from raw file bytes; removes reliance on filesystem metadata
- **`CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K` moved to `.env`** — defaults `800` / `80` / `5`; loader reads chunk params, retriever reads top_k; hardcoded constants in `ragService.py` removed
- **Microsoft GraphRAG replaced by `graph_engine/`** — no subprocess CLI, no `settings.yaml`, no community summaries; triples stored in LanceDB `graph_triples` table alongside the existing `documents` vector table
- **`GRAPH_EXTRACTOR` env var** — selects extractor: `"hybrid"` (default, spaCy NER + LLM relation), `"llm"` (full LLM extraction), `"spacy"` (fast local, no Ollama)
- **`EXTRACT_MODEL` env var** — model used by LLM/Hybrid extractor; can be same as `INDEX_MODEL`
- **`global_search()` now uses graph triples** — entities extracted from query via spaCy, triples retrieved from `graph_triples`, formatted as context for LLM; no more community summary dependency
