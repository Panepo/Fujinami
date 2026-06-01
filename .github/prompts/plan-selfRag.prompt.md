# Plan: Self-RAG Implementation

Self-RAG adds an opt-in self-reflection loop around the existing retrieval pipeline. The LLM makes 3–4 extra calls to decide: (1) is retrieval needed? (2) which chunks are relevant? (3) is the answer grounded? If grounding fails, it retries with a refined query. Enabled via a new checkbox in the UI.

---

## Phase 1 — Models (`models.py`)

1. Add `self_rag: bool = False` to `QueryRequest`
2. Add `SelfRagMeta` model with fields: `needed: bool`, `relevant_chunks: int`, `grounded: bool`, `iterations: int`
3. Add `self_rag_meta: SelfRagMeta | None` to `QueryResponse`

---

## Phase 2 — `self_reflector.py` (new file)

4. Create `SelfReflector(retriever, kernel)` class with:
   - `async def query(query, method, top_k)` — main entry point, returns `(answer, sources, graphrag_context, SelfRagMeta)`
   - `_should_retrieve(query) -> bool` — LLM call: "Does this question require document retrieval?"
   - `_filter_relevant(query, chunks) -> list[SourceChunk]` — per-chunk LLM relevance check, returns only relevant ones
   - `_check_grounding(query, answer, context) -> bool` — LLM call: "Is this answer supported by the context?"
   - Retry loop: up to 2 iterations — if grounding fails, refine query and re-retrieve

---

## Phase 3 — `api.py`

5. In `POST /collections/{name}/query`, branch on `request.self_rag`:
   - `True` → use `SelfReflector(retriever, kernel).query(...)`
   - `False` → existing flow unchanged
6. Pass `self_rag_meta` into `QueryResponse`

---

## Phase 4 — `static/index.html`

7. Add `<input type="checkbox" id="chk-self-rag" />` labeled "self-RAG" next to the existing "stream" checkbox
8. In `submitQuery()`, include `"self_rag": chk-self-rag.checked` in the POST body
9. After query completes, show a small self-RAG metadata panel if `self_rag_meta` is present in the response (iterations, chunks kept, grounded)

---

## Relevant Files

- `models.py` — `QueryRequest`, `QueryResponse`, new `SelfRagMeta`
- `api.py` — query handler ~line 573, kernel instantiation pattern to reuse
- `retriever.py` — `_raw_vector_results`, `_graph_context`, `_generate_response` (read-only, reused)
- `ragService.py` — how Semantic Kernel kernel is constructed (read-only reference)
- `static/index.html` — checkbox near `chk-stream`, `submitQuery()` JS, results area
- `self_reflector.py` — **new file**

---

## Verification

1. `uvicorn api:app --reload` → query with self-RAG unchecked → confirm existing behavior intact
2. Check self-RAG box → verify `self_rag_meta` in JSON response
3. Ask a factual question not needing docs (e.g., "What is 2+2?") → expect `needed: false`
4. Check `relevant_chunks` is ≤ `top_k` (filtering happened)
5. Force a weak-retrieval scenario → confirm `iterations` can reach 2

---

## Decisions

- Self-RAG reuses the same Ollama LLM — no new model required
- Max iterations = 2 (trade-off between quality and latency)
- If both `stream` and `self_rag` are checked, `self_rag` takes precedence (streaming is incompatible with multi-step reflection)

---

## Further Considerations

1. **Kernel access in `api.py`**: The Semantic Kernel `kernel` instance needs to be accessible when constructing `SelfReflector`. Need to confirm how it's instantiated in `api.py` vs. `ragService.py` — may need to expose it as a property on `RagRetriever` or `RagService`. This is the most likely blocker.
2. **Chunk-level relevance cost**: Filtering each chunk with a separate LLM call (top_k=5 → 5 calls) could be slow. Alternative: batch all chunks in one prompt. Recommend batching unless you prefer per-chunk precision.
