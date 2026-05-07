# Plan: FastAPI HTTP Server for Fujinami RAG

## TL;DR
Add `python/api.py` and `python/models.py` to expose `RagService` via a FastAPI HTTP server. Collections are managed as isolated `data/{name}` + `ragdata/{name}` directory pairs. A module-level registry caches live `RagService` instances. Endpoints cover the full lifecycle: create/rename/delete collection, add/remove documents, trigger indexing, and run inference.

---

## Phase 1 — Schemas (`models.py`)

1. Create `python/models.py` with Pydantic request/response models:
   - `CollectionCreateRequest` — `name: str`
   - `CollectionRenameRequest` — `new_name: str`
   - `CollectionInfo` — `name: str, doc_count: int`
   - `DocumentInfo` — `filename: str, size_bytes: int`
   - `IndexResponse` — `collection: str, status: str, task_id: str`
   - `IndexStatusResponse` — `task_id: str, status: Literal["pending","running","done","error"], detail: str | None`
   - `QueryRequest` — `query: str, method: Literal["vector","global","hybrid"] = "hybrid", top_k: int = 5, stream: bool = False`
   - `SourceChunk` — `doc_id: str, chunk_index: int, excerpt: str`
   - `QueryResponse` — `collection: str, method: str, answer: str, sources: list[SourceChunk] | None` (`None` for `global` — GraphRAG returns unstructured text only)

---

## Phase 2 — App & Registry (`api.py`)

2. Create `python/api.py` with:
   - Module-level `_registry: dict[str, RagService]`
   - `@asynccontextmanager` lifespan: scan existing `ragdata/` sub-dirs on startup and populate registry
   - `FastAPI(lifespan=lifespan)` app instance
   - Helper `_get_service(name)` → raises `404` if not in registry

---

## Phase 3 — Collection Endpoints

3. `GET /collections` — scan registry, return list of `CollectionInfo` (name + doc count)
4. `POST /collections` — validate name (alphanumeric + `-_`), instantiate `RagService(collection_name=name)`, add to registry
5. `PATCH /collections/{name}` — rename `data/{name}` → `data/{new_name}`, rename `ragdata/{name}` → `ragdata/{new_name}`, re-create `RagService` with new name, swap registry keys
6. `DELETE /collections/{name}` — remove from registry, `shutil.rmtree` on both `data/{name}` and `ragdata/{name}`

---

## Phase 4 — Document Endpoints

7. `GET /collections/{name}/documents` — list files in `data/{name}/` matching `SUPPORTED_EXTENSIONS`, return `list[DocumentInfo]`
8. `POST /collections/{name}/documents` — accept `UploadFile`, validate extension against `SUPPORTED_EXTENSIONS`, save to `data/{name}/{filename}`
9. `DELETE /collections/{name}/documents/{filename}` — unlink `data/{name}/{filename}` (no auto-index; caller must trigger `/index` separately)

---

## Phase 5 — Index & Inference Endpoints

10. `POST /collections/{name}/index` — enqueue a background task (`asyncio.create_task` wrapped in `BackgroundTasks`) that calls `await rag.index_documents()`; return `IndexResponse` with a `task_id` immediately so the caller is not blocked
11. `GET /collections/{name}/index/{task_id}` — return `IndexStatusResponse` (poll for `pending` / `running` / `done` / `error`); task state stored in a module-level `_tasks: dict[str, IndexStatusResponse]`
12. `POST /collections/{name}/query` — before generating the answer, call `_raw_vector_results()` to retrieve LanceDB rows and build `list[SourceChunk]`; if `stream=False`, return `QueryResponse` including `sources`; if `stream=True`, return a `StreamingResponse` (`text/event-stream`) with the SSE protocol below:
    ```
    event: token    data: "partial answer text…"
    event: token    data: "…continued…"
    event: sources  data: [{"doc_id":"…","chunk_index":0,"excerpt":"…"}, …]  (omitted for global)
    event: done     data: ""
    ```

---

## Phase 6 — HTML Frontend (`static/index.html`)

13. Create `python/static/index.html` — single-file vanilla HTML/JS (no build step) with four panels:
    - **Collections** — `GET /collections` to list; form to create (`POST`); rename (`PATCH`) and delete (`DELETE`) buttons per row
    - **Documents** — select active collection from dropdown; `GET /collections/{name}/documents` to list with file sizes; file input + upload button (`POST`); delete button per file
    - **Index** — button triggers `POST /collections/{name}/index`; displays returned `task_id`; polls `GET /collections/{name}/index/{task_id}` every 2 s with a status badge (`pending` / `running` / `done` / `error`)
    - **Query** — text area for query; `method` radio (`vector` / `global` / `hybrid`); `stream` checkbox; submit calls `POST /collections/{name}/query`; non-stream renders answer + collapsible sources table; stream mode reads `EventSource`-style fetch and appends `token` events incrementally, then renders sources from the final `sources` event
14. Mount in `api.py`: `app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")` and add `GET /` → `FileResponse("static/index.html")`

---
- `python/static/index.html` — **new**; vanilla HTML/JS UI served at `GET /`
- `python/ragService.py` — `RagService`, `get_rag_service()`, `SUPPORTED_EXTENSIONS` (imported from `document_loader`); **needs new internal method** `_raw_vector_results(query, top_k) → list[dict]` that returns raw LanceDB rows (keeping `doc_id`, `text`, `metadata`) instead of joining them into a string — used by the query endpoint to build `sources`
- `python/document_loader.py` — `SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".doc"}`
- `python/ragdata/settings.yaml` — template used per-collection by `_ensure_settings_yaml()`
- `python/models.py` — **new**
- `python/api.py` — **new**

---

## Verification
1. `uvicorn python.api:app --reload` starts without errors
2. `POST /collections` with `{"name": "test"}` → 200, directory `python/ragdata/test/` and `python/data/test/` created
3. `POST /collections/test/documents` with a `.pdf` file → 200, file appears in `python/data/test/`
4. `POST /collections/test/index` → 202 immediately with a `task_id`; poll `GET /collections/test/index/{task_id}` until `status: "done"`
5. `POST /collections/test/query` with `{"query": "…", "method": "hybrid", "stream": false}` → 200 with answer and `sources` list populated from LanceDB rows
6. `POST /collections/test/query` with `{"query": "…", "method": "global", "stream": false}` → 200 with answer and `sources: null`
7. `POST /collections/test/query` with `{"query": "…", "method": "vector", "stream": true}` → `text/event-stream` with `token` events followed by a `sources` event then `done`
8. `GET /` in browser → HTML UI loads; all four panels functional
8. `PATCH /collections/test` with `{"new_name": "test2"}` → dirs renamed, registry updated
9. `DELETE /collections/test2` → dirs removed, 404 on subsequent GET
10. Upload unsupported extension (`.exe`) → 422

---

## Decisions
- **HTML served by FastAPI** — `StaticFiles` mount at `/static`; `GET /` returns `FileResponse`; no separate web server needed
- **No JS framework** — vanilla `fetch` + DOM manipulation keeps the frontend a single file with zero build tooling
- **Source snapshots per method** — `vector` and `hybrid` return structured `sources` (from LanceDB rows); `global` returns `sources: null` because GraphRAG subprocess output is unstructured text with no chunk references
- **No auto-index on doc add/remove** — matches `RagService` design (incremental delta detection)
- **Async endpoints throughout** — `index_documents` and all search methods are already async
- **UploadFile** (multipart/form-data) for doc uploads, not base64 JSON
- **Registry in memory only** — no DB; registry rebuilt from disk on startup via lifespan
- **Rename atomicity** — rename `data/` first, then `ragdata/`, then swap registry; partial failure leaves dirs in new name state (acceptable given local filesystem scope)
- **Run command**: `uvicorn python.api:app --host 0.0.0.0 --port 8000 --reload` from repo root

---

## Excluded
- Authentication / API keys
