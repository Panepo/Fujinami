# Plan: Query Rewriting + HyDE (rewriter.py)

## TL;DR
Add a `QueryRewriter` class in `rewriter.py` that supports three query pre-processing modes — **HyDE** (generate hypothetical document, embed it, use that vector for search), **multi_query** (LLM generates N reformulations, parallel searches, dedup merge), and **step_back** (broaden the query to a higher-level question). Integrate as an optional pre-step in the existing query flow controlled by a new `rewrite` field on `QueryRequest`.

---

## Phase 1 — Data Model Changes (models.py)

1. Add `RewriteMeta` Pydantic model (fields: `mode`, `original_query`, `rewritten_queries: list[str]`, `hypothetical_document: str | None`)
2. Add `rewrite: Literal["hyde", "multi_query", "step_back"] | None = None` to `QueryRequest`
3. Add `rewrite_meta: RewriteMeta | None = None` to `QueryResponse`

## Phase 2 — Create rewriter.py

4. Create `rewriter.py` at project root with class `QueryRewriter`:
   - Constructor: `__init__(self, llm: ChatOllama, embedding_service: OllamaEmbeddings)` — reuse the same LLM + embedder instances as `RagRetriever` (same model, no extra connections)
   - `async hyde(query: str) -> tuple[list[float], RewriteMeta]` — LLM generates hypothetical doc (SystemMessage: "Write a short factual passage answering..."), embed with `embedding_service.embed_query(hypothetical_doc)`, return embedding + meta
   - `async multi_query(query: str, n: int = 3) -> tuple[list[str], RewriteMeta]` — LLM generates N alternative phrasings (one per line), returns `[original_query] + alternatives` + meta
   - `async step_back(query: str) -> tuple[list[str], RewriteMeta]` — LLM generalizes query, returns `[original_query, step_back_query]` + meta
   - `async rewrite(query: str, mode: str) -> tuple[list[str] | None, list[float] | None, RewriteMeta]` — dispatcher returning `(queries_or_None, hyde_embedding_or_None, meta)`

## Phase 3 — Retriever Extension (retriever.py)

5. Add `async _raw_vector_results_from_embedding(embedding: list[float], top_k: int) -> list[dict]` to `RagRetriever` — same body as `_raw_vector_results` but skips the `embed_query()` call and uses the pre-computed `embedding` directly with `self._table.search(embedding).limit(top_k).to_list()`

## Phase 4 — Service Delegation (rag_service.py)

6. Expose `_raw_vector_results_from_embedding` on `RagService` (thin delegation to `self._retriever._raw_vector_results_from_embedding(...)`) — same pattern as existing `_raw_vector_results`

## Phase 5 — API Integration (api.py)

7. In `query_collection` endpoint, after validation and before the existing vector fetch:
   - If `body.rewrite` is set and `body.method != "graph"`:
     - Instantiate `QueryRewriter(rag._chat_service, rag._retriever._query_embedding_service)`
     - Call `rewriter.rewrite(body.query, body.rewrite)` → `(queries, hyde_embedding, rewrite_meta)`
     - **HyDE branch**: use `_raw_vector_results_from_embedding(hyde_embedding, top_k)` instead of `_raw_vector_results(query, top_k)`
     - **multi_query / step_back branch**: run `asyncio.gather(*[rag._raw_vector_results(q, top_k) for q in queries])`, flatten + deduplicate by `(doc_id, chunk_index)` preserving first-seen order
   - Include `rewrite_meta` in the `QueryResponse` return
8. Streaming path: same pre-processing for rewrite before entering `_stream_answer()`, pass rewritten `vector_context` + `rewrite_meta` through (emit as new SSE `rewrite_meta` event before `chunks`)

## Phase 6 — UI Update (static/index.html)

9. In the Query Tab form, add a "Rewrite" select/dropdown: `None | HyDE | Multi-Query | Step-Back` (default None)
10. Map values to the `rewrite` field in the JSON body (`null | "hyde" | "multi_query" | "step_back"`)
11. In the results area, if `rewrite_meta` is present in the response, show a collapsible "Rewrite Info" section showing mode, rewritten queries, and (for HyDE) the hypothetical document

---

## Relevant Files
- `models.py` — add `RewriteMeta`, extend `QueryRequest` + `QueryResponse`
- `rewriter.py` — create new file (QueryRewriter class)
- `retriever.py` — add `_raw_vector_results_from_embedding()` method (~10 lines)
- `rag_service.py` — delegate `_raw_vector_results_from_embedding`
- `api.py` — integrate rewrite pre-step in `query_collection`, streaming path
- `static/index.html` — rewrite mode selector + display

---

## Verification
1. Manual: POST `/collections/{name}/query` with `{"query":"...","rewrite":"hyde","method":"vector"}` → confirm `rewrite_meta.hypothetical_document` is populated and answer is coherent
2. Manual: POST with `"rewrite":"multi_query"` → confirm `rewrite_meta.rewritten_queries` has 3 items
3. Manual: POST with `"rewrite":"step_back"` → confirm broader query in `rewrite_meta`
4. Manual: POST with `"rewrite":null` → confirm original behavior unchanged (backward compat)
5. Manual: POST with `"stream":true,"rewrite":"hyde"` → confirm SSE token stream still works

---

## Decisions
- `QueryRewriter` does NOT persist state; instantiated per-request in `api.py` (cheap constructor, reuses existing LLM/embedder connections)
- HyDE uses only the hypothetical doc embedding (not a blend of query + hyp doc) — simpler and the standard approach
- multi_query deduplication key: `(doc_id, chunk_index)` — first result wins (highest-ranking source preserved)
- `step_back` also retains the original query's results alongside the broadened query (union, deduped)
- Rewrite modes are skipped silently when `method == "graph"` (graph retrieval is entity-based, not embedding-based)
- `self_rag=true` takes precedence; rewrite can be combined but self_rag path ignores `rewrite` in Phase 1 (future enhancement)

## Further Considerations
1. **n for multi_query**: Default n=3 rewrites (hardcoded). Could be exposed as `top_k`-style param later.
2. **Prompt language**: HyDE and multi_query prompts assume English documents. If docs are multilingual, prompts need `lang` hint.
