## Plan: Multi-Collection RAG Service

**Goal:** Organize documents into named folders (A, B, C …), process each independently, and call `RagService("A")` / `RagService("B")` to query the right one.

---

### Target Folder Structure

```
d:\Github\Fujinami\
  data/
    A/        ← you place docs here
    B/
    C/
  ragdata/
    settings.yaml          ← existing template (unchanged)
    A/                     ← auto-created on first index
      settings.yaml        ← generated from template, input path → data/A
      lancedb/             ← isolated vector store for A
      output/              ← graphrag artifacts for A
    B/  …
    C/  …
  python/
    ragService.py          ← only file to modify
```

Each collection gets its own: LanceDB database, GraphRAG knowledge graph, file manifest, and `settings.yaml`.

---

### Steps

**Phase 1 — Add `collection_name` parameter to `RagService.__init__`**

1. Add `collection_name: str | None = None` to `__init__` signature
2. When set, override path construction:
   - `data_dir = root_dir / "data" / collection_name`
   - `ragdata_dir = root_dir / "ragdata" / collection_name`
   - LanceDB and manifest paths derive from the namespaced `ragdata_dir` (no other changes needed — `_TABLE_NAME = "documents"` is fine since each collection has an isolated LanceDB directory)

**Phase 2 — Auto-generate per-collection `settings.yaml`**

3. Add `_ensure_settings_yaml()` method:
   - Reads the root template `root_dir/ragdata/settings.yaml`
   - Loads it with `pyyaml`, replaces `input.base_dir` with the **absolute** path to `data_dir`
   - Writes to `ragdata_dir/settings.yaml` (overwrites if `base_dir` differs, skip if identical)
4. Call it in `__init__` when `collection_name` is set (creates `ragdata_dir` first)

**Phase 3 — Convenience factory**

5. Add module-level `get_rag_service(collection_name: str, root_dir=None) -> RagService` factory for cleaner call sites
6. Optionally default `index_documents()` argument to `self._data_dir` when `collection_name` is set

**Backward compatibility:** `collection_name=None` leaves all existing behavior unchanged.

---

### Usage After Implementation

```python
svc_a = RagService(collection_name="A")
svc_b = RagService(collection_name="B")

await svc_a.index_documents()   # indexes data/A/
await svc_b.index_documents()   # indexes data/B/

answer_a = await svc_a.hybrid_search("question about A")
answer_b = await svc_b.global_search("question about B")
```

---

### Relevant Files

- `python/ragService.py` — only file to modify; key symbols: `RagService.__init__`, `_run_graphrag_index`, `index_documents`
- `python/ragdata/settings.yaml` — becomes the template; `input.base_dir` is the field to override per collection

---

### Verification

1. Create `data/A/` and `data/B/` with different documents
2. Instantiate both services; confirm `ragdata/A/` and `ragdata/B/` are created with correct `settings.yaml`
3. Index each collection; confirm separate `lancedb/` dirs and `file_manifest.json` files
4. Query A → answer is scoped to A's docs; query B → scoped to B's docs

---

### Further Consideration

Should `index_documents()` default to `self._data_dir` automatically (no argument needed), or always require an explicit path? Recommend defaulting to `self._data_dir` when `collection_name` is set for cleaner usage.
