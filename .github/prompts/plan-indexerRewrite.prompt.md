# Plan: Rewrite indexer.py as indexer/ package

## Context
- Current `indexer.py`: single-file `RagIndexer` class. Uses SK `OllamaTextEmbedding`, DocumentLoader (stages 1-5), graph extraction, LanceDB upsert.
- Reference pipeline: `.github/reference/document-parsing/` — 6 stages; `06_embed.py` uses torch+transformers.
- Goal: convert to `indexer/` package, add per-document `embedded.json` cache, replace torch/transformers with Ollama HTTP calls.
- `ragService.py` imports `from indexer import RagIndexer, SUPPORTED_EXTENSIONS` — must keep this contract.

## New folder structure
```
indexer/
  __init__.py       — re-exports RagIndexer, SUPPORTED_EXTENSIONS
  pipeline.py       — RagIndexer class (orchestrates all stages)
  delta.py          — SHA-256 delta detection + manifest helpers
  embedder.py       — OllamaEmbedder (direct HTTP to /api/embed, L2-normalize)
  store.py          — LanceDB upsert/delete helpers
  graph.py          — graph extraction logic (moved from current indexer.py)
```

## Per-document embedded.json
- Path: `ragdata/{collection}/embedded/{doc_stem}.embedded.json`
- Schema: matches reference `06_embedded.json`:
  `{ model, dimension, device:"ollama", doc_stem, chunks:[{...05_metadata fields..., embedding:[...]}] }`
- Used as a cache: if file exists and chunk_hash matches, skip re-embedding

## Steps

### Phase 1 — Package scaffold
1. Create `indexer/` directory
2. Create `indexer/__init__.py` re-exporting `RagIndexer`, `SUPPORTED_EXTENSIONS`

### Phase 2 — delta.py
3. Move `_load_manifest`, `_save_manifest`, `_compute_delta`, `_load_index_flags`, `_save_index_flags` from current `indexer.py` into `indexer/delta.py` as standalone functions. Accept `ragdata_dir`, `lance_path`, `data_dir` as parameters.

### Phase 3 — embedder.py
4. Create `indexer/embedder.py` with `OllamaEmbedder` class:
   - `__init__(model, ollama_base_url, dimension)`
   - `embed(texts: list[str]) -> np.ndarray` — POST to `{ollama_base_url}/api/embed`, batch, L2-normalize float32
   - Properties: `model_name`, `dimension`
   - No torch, no transformers

### Phase 4 — store.py
5. Create `indexer/store.py` with:
   - `_LANCEDB_SCHEMA`, `_TABLE_NAME` constants
   - `open_or_create_table(db, schema)` helper
   - `remove_from_lancedb(table, doc_ids)`
   - `upsert_from_embedded_json(db, table, path)` — reads embedded.json, builds PA table, adds to LanceDB

### Phase 5 — graph.py
6. Create `indexer/graph.py` with:
   - `run_graph_extraction(source_doc, full_text, lance_path, ollama_url, extractor_type, extract_model, chunk_size, chunk_overlap)`
   - `remove_graph_triples(lance_path, doc_ids)`
   - Extracted directly from current `indexer.py`

### Phase 6 — pipeline.py
7. Create `indexer/pipeline.py` with new `RagIndexer` class:
   - Same `__init__` signature as current
   - `index_documents()` orchestrates:
     1. Delta detection (via delta.py)
     2. Load via DocumentLoader (stages 1–5)
     3. Embed via OllamaEmbedder, save `embedded.json` per doc
     4. Graph extraction (via graph.py)
     5. Upsert from `embedded.json` to LanceDB (via store.py)
     6. Save manifest
   - Internal helper: `_embed_and_save(filename, chunks) -> Path` writes embedded.json atomically

### Phase 7 — download endpoint (api.py)
8. Add `GET /collections/{name}/documents/{filename}/embedded` to `api.py`:
   - Resolves `ragdata/{name}/embedded/{doc_stem}.embedded.json` (where `doc_stem = Path(filename).stem`)
   - Returns `FileResponse` with `media_type="application/json"` and `Content-Disposition: attachment; filename="{doc_stem}.embedded.json"`
   - Returns `404` if the embedded.json does not yet exist (document not yet indexed)
   - Mirrors the existing `GET /collections/{name}/documents/{filename}/download` pattern

### Phase 8 — cleanup
9. Delete old `indexer.py` (after verifying no import breaks)
10. Verify `ragService.py` `from indexer import RagIndexer, SUPPORTED_EXTENSIONS` still works

## Embedded.json path
`ragdata/{collection_name}/embedded/{doc_stem}.embedded.json`
where `doc_stem = Path(filename).stem`

## Relevant files
- `indexer.py` — source to decompose
- `ragService.py` — imports `from indexer import RagIndexer, SUPPORTED_EXTENSIONS`
- `retriever.py` — imports from indexer indirectly (via ragService)
- `document_loader.py` — stages 1–5, unchanged
- `api.py` — add download endpoint (mirrors existing `/documents/{filename}/download` pattern)
- `.github/reference/document-parsing/06_embed.py` — reference for embedder interface
- `.github/reference/document-parsing-result/06_embedded.json` — target output schema

## Verification
1. `python -c "from indexer import RagIndexer, SUPPORTED_EXTENSIONS"` succeeds
2. Full import chain (`ragService.py`) loads without errors
3. Run `index_documents()` on `data/test/` — confirm `ragdata/test/embedded/*.embedded.json` files appear
4. Confirm LanceDB `documents` table row count matches chunk count across all embedded JSONs
5. Re-run with no file changes → no-op confirmed via manifest delta
6. `GET /collections/test/documents/{filename}/embedded` returns the `.embedded.json` as a download
7. `GET /collections/test/documents/nonexistent.txt/embedded` returns `404`

## Decisions
- `SUPPORTED_EXTENSIONS` stays defined in `document_loader.py`; `indexer/__init__.py` re-exports it from there
- Ollama embed call: direct `httpx` (sync, batched) — no SK dependency in embedder
- `embedded.json` dimension field inferred from first Ollama response, not hardcoded
- Per-document embedded.json written atomically (temp file + rename) to avoid partial writes
