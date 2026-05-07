# Plan: Incremental RAG Indexing (Per-File Delta)

## TL;DR
Add a per-file manifest (`file_manifest.json`) to track each source document's mtime/size. On `index_documents()` calls, only new/modified files go through VLM processing, `.txt` generation, and LanceDB embedding. Deleted files have their LanceDB chunks removed and `.txt` files deleted. GraphRAG is invoked only when any file changes — its own internal cache handles unchanged documents. If nothing changed, the entire pipeline is skipped immediately.

> **Key divergence from the reference:** The reference plan builds custom entity/relationship/community sidecars with Louvain detection in TypeScript. This project delegates all of that to the **GraphRAG CLI**, so those phases are dropped. The plan adapts only the vector store delta and file manifest concepts.

---

## Phase 1 — Manifest Infrastructure (`ragService.py`)

**Step 1.1 — Add `_manifest_path` attribute in `__init__()`**
- Value: `self._ragdata_dir / "lancedb" / "file_manifest.json"`
- Co-located with LanceDB data directory

**Step 1.2 — Add `_load_manifest()` private method**
- Read `file_manifest.json` → parse as `dict[str, dict]` with `{ mtime: float, size: int }` entries
- Return `{}` if file doesn't exist or JSON is malformed

**Step 1.3 — Add `_compute_delta(documents_dir, stored_manifest)` private method**
- Walk `documents_dir` with the same `rglob + SUPPORTED_EXTENSIONS` logic as `DocumentLoader`
- Keys: `file_path.name` (basename only — matches existing `doc_id` convention in `_upsert_to_lancedb()`)
- Return: `(new_files, modified_files, deleted_files, unchanged_files)` as `set[str]`

**Step 1.4 — Add `_save_manifest(documents_dir)` private method**
- Walk `documents_dir`, collect fresh `{ mtime, size }` for all currently-on-disk files
- Write as JSON to `_manifest_path`

---

## Phase 2 — File Filter in `document_loader.py`

**Step 2.1 — Add `files_filter` parameter to `load_directory()`**
- New signature: `load_directory(directory, files_filter: set[str] | None = None)`
- If `files_filter` is provided, skip `file_path.name` not in the set
- Unchanged call sites (no filter argument) continue to load all files — no breaking change

---

## Phase 3 — Incremental LanceDB Removal (`ragService.py`)

**Step 3.1 — Add `_remove_from_lancedb(doc_ids: list[str])` private method**
- Iterate `doc_ids`, call `self._table.delete(f"doc_id = '{safe_id}'")`
- Skip gracefully if `self._table is None`

**Step 3.2 — Decouple `_upsert_to_lancedb()` from deletion logic**
- Remove the per-doc delete block inside `_upsert_to_lancedb()` (current lines ~215–225)
- `_upsert_to_lancedb()` becomes pure-add only
- Deletions are now the caller's responsibility (handled in the new `index_documents()` flow)

---

## Phase 4 — Selective `.txt` Management (`ragService.py`)

**Step 4.1 — Delete `.txt` for removed sources**
- For each `doc_id` in `deleted_files | modified_files`: unlink `_data_dir / (Path(doc_id).stem + ".txt")` if it exists

**Step 4.2 — Write `.txt` only for changed sources**
- Only write `.txt` files for `new_files | modified_files`
- Unchanged sources retain their existing `.txt` files in `_data_dir`

**Step 4.3 — Conditional GraphRAG invocation**
- Call `_run_graphrag_index()` only when `new_files | modified_files | deleted_files` is non-empty

---

## Phase 5 — Refactor `index_documents()` Orchestration

**New flow:**
1. `stored_manifest = _load_manifest()`
2. `new_files, modified_files, deleted_files, unchanged_files = _compute_delta(documents_dir, stored_manifest)`
3. If all sets empty → `logger.info("No changes detected, skipping indexing")` → return
4. `removed_sources = deleted_files | modified_files`
5. `changed_sources = new_files | modified_files`
6. Delete `.txt` files for `removed_sources` (Step 4.1)
7. `_remove_from_lancedb(list(removed_sources))` (Step 3.1)
8. Load only `changed_sources` via `loader.load_directory(documents_dir, files_filter=changed_sources)` (Step 2.1)
9. Write `.txt` files for `changed_sources` (Step 4.2)
10. `_run_graphrag_index()` (Step 4.3)
11. `_upsert_to_lancedb(doc_texts)` — pure add for changed docs only (Step 3.2)
12. `_save_manifest(documents_dir)` — write fresh stats for all on-disk files

---

## Relevant Files
- `python/ragService.py` — primary modification (phases 1, 3, 4, 5)
- `python/document_loader.py` — add `files_filter` param to `load_directory()` (phase 2)
- `python/ragdata/lancedb/file_manifest.json` — new runtime artifact created at first run (not modified manually)

---

## Verification
1. First run (no manifest) → all N files indexed, `file_manifest.json` created with N entries
2. Re-run with no changes → "No changes detected" log, zero VLM/embed/GraphRAG calls
3. Add 1 new file → only that file loaded (VLM + embed), GraphRAG re-runs with cache hits for others
4. Modify 1 file → old LanceDB rows deleted, new chunks added; other files untouched
5. Delete 1 file → `.txt` removed from `python/data/`, `doc_id` rows deleted from LanceDB; manifest updated
6. Inspect `file_manifest.json` after each scenario to confirm accurate state

---

## Decisions
- **Manifest keys**: basename only — matches existing `doc_id = file_path.name` convention in `_upsert_to_lancedb()`
- **Manifest location**: `python/ragdata/lancedb/file_manifest.json` (co-located with LanceDB)
- **GraphRAG**: always re-invoked when any file changes; its internal cache skips unchanged documents
- **Entity/community phases from reference**: dropped — GraphRAG CLI manages all of this internally; no custom sidecars
- **`_upsert_to_lancedb()` refactor**: deletion extracted out to keep the method single-responsibility
- **Scope**: only `ragService.py` and `document_loader.py` modified; no new files or modules
- **Breaking change**: none — manifest is additive; first run auto-creates it
