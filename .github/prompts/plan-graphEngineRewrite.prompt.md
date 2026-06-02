# Plan: Rewrite graph_engine with LangGraph + UI Update

**Rewrite `graph_engine/` using LangGraph StateGraphs for both the extraction (indexing) pipeline and the adaptive query flow (matching the reference's `vector_retrieve → evaluate_context → [conditional] → generate_answer` pattern). Replace Semantic Kernel with `langchain-ollama` across all LLM call sites. The UI gains a real-time LangGraph flow trace panel driven by new node-level SSE events.**

---

## Phase 1 — Dependencies

**`requirements.txt`**
- Remove `semantic-kernel[ollama]`
- Add `langgraph`, `langchain-core`, `langchain-ollama`

---

## Phase 2 — graph_engine/ Rewrite

**`graph_engine/state.py` (NEW)**
Define two `TypedDict` states:
- `ExtractionState`: `raw_text`, `source_doc`, `method`, `chunks`, `triples`, `deduped_triples`, `stored_count`, `error`
- `QueryState`: `question`, `method`, `top_k`, `context`, `sources`, `graphrag_context`, `needs_graph`, `answer`, `iterations`, `node_trace` (list of `{node, started_at, duration_ms, detail}`)

**`graph_engine/pipeline.py` (FULL REWRITE)**
`GraphPipeline` → `ExtractionGraph` as a `StateGraph(ExtractionState)`:
- Node `chunk_node` → calls existing `chunk_text()` from `graph_engine/chunker.py`
- Node `extract_node` → instantiates spacy/llm/hybrid extractor based on `state.method`
- Node `deduplicate_node` → calls existing `deduplicate_triples()` from `graph_engine/deduplicator.py`
- Node `store_node` → calls `LanceDBGraphStore.add_triples()` from `graph_engine/store.py`
- Linear edges: `chunk → extract → deduplicate → store → END`
- Progress via `.stream()` iteration — emits node-enter events to caller
- Factory `build_pipeline(method, store, ...) → CompiledGraph` keeps same external call signature as today

**`graph_engine/query_graph.py` (NEW)**
`QueryGraph` wrapping a `StateGraph(QueryState)` — directly mirrors the reference architecture:
- `vector_retrieve_node`: calls retriever's vector search → fills `state.context` + `state.sources`
- `evaluate_context_node`: `ChatOllama` evaluates with a YES/NO prompt (identical pattern to reference) → sets `state.needs_graph`
- `graph_retrieve_node`: queries `LanceDBGraphStore` with spaCy NER on question → fills `state.graphrag_context`
- `generate_answer_node`: `ChatOllama` generates final answer → sets `state.answer`
- Conditional routing after `evaluate_context_node`: `needs_graph=True` → `graph_retrieve_node → generate_answer_node`; else → `generate_answer_node` directly
- `node_trace` appended in each node with timing for UI display
- For `method="graph"`: bypass vector_retrieve_node, force `needs_graph=True` at entry

**`graph_engine/__init__.py` (UPDATE)**
Export: `ExtractionGraph`, `QueryGraph`, `ExtractionState`, `QueryState` alongside existing `Node`, `Edge`, `Triple`

**UNCHANGED** (no edits): `graph_engine/models.py`, `graph_engine/store.py`, `graph_engine/chunker.py`, `graph_engine/deduplicator.py`, `graph_engine/base.py`, all of `graph_engine/extractors/`

---

## Phase 3 — LLM Layer Migration (*parallel with Phase 2*)

**`retriever.py` (REWRITE LLM section)**
- Replace `Kernel` + `OllamaChatCompletion` + `OllamaTextEmbedding` (SK) with `ChatOllama` + `OllamaEmbeddings` from `langchain-ollama`
- `_generate_response()`: `ChatPromptTemplate | ChatOllama` chain invoke
- `_raw_vector_context()`: `OllamaEmbeddings.embed_query()` for query embedding (returns `list[float]` directly, compatible with LanceDB)
- Keep `_chat_service` attribute pointing to the `ChatOllama` instance (used by `api.py` streaming)

**`self_reflector.py` (FULL REWRITE)**
Becomes a thin wrapper over `QueryGraph`:
- `async query(query, method, top_k)` → invokes `QueryGraph` → translates `state.node_trace` → `list[SelfRagStep]` → returns `(answer, sources, graphrag_context, SelfRagMeta)`
- Identical return type — no API change to callers
- `max_iterations` enforced via `state.iterations` counter + LangGraph conditional loop

**`api.py` (UPDATE `_stream_answer_inner`)**
- Replace `ChatHistory` + `OllamaChatPromptExecutionSettings` SK streaming with `ChatOllama.astream(messages)`
- Add 3 new SSE events to the streaming response:
  - `event: node_enter\ndata: {"node": "...", "timestamp": ...}`
  - `event: routing_decision\ndata: {"needs_graph": true|false}`
  - `event: node_complete\ndata: {"node": "...", "duration_ms": ...}`

---

## Phase 4 — Indexer Integration (*depends on Phase 2*)

**`indexer/graph.py` (MINOR UPDATE)**
- `run_graph_extraction()`: call `build_pipeline(...).invoke(ExtractionState(...))` instead of `GraphPipeline.run()`
- Feed node-enter events as on-progress callbacks to caller via `.stream()` iteration

---

## Phase 5 — UI Modifications (*parallel with Phase 2–4*)

**`static/index.html`** — Query Tab:
- Replace "Self-RAG Process" section with **"LangGraph Flow"** panel
- Add inline SVG/div mini flow diagram with 4 boxes: `vector_retrieve → evaluate_context → [graph_retrieve?] → generate_answer`
- SSE `node_enter` event → highlight active node box + start timer
- SSE `routing_decision` event → show "Graph Augmented" or "Vector Only" badge; dim `graph_retrieve` box if skipped
- SSE `node_complete` event → show duration on the completed node box
- Token streaming + sources/chunks display unchanged

---

## Phase 6 — Test Updates (*depends on Phase 2*)

**`graph_engine/tests/test_pipeline.py` (REWRITE)**
- Test `ExtractionGraph.invoke(ExtractionState(...))` end-to-end with mock extractor and mock store
- Assert state transitions: `chunks` populated → `triples` populated → `deduped_triples` → `stored_count > 0`

**`graph_engine/tests/test_types.py` (UPDATE)**
- Add `ExtractionState` and `QueryState` field validation tests

Other test files (`test_chunker`, `test_deduplicator`, `test_extractors`, `test_models`, `test_store`) — minimal or no changes since those modules are unchanged.

---

## Relevant Files

| File | Change |
|---|---|
| `requirements.txt` | Dep swap |
| `graph_engine/__init__.py` | Update exports |
| `graph_engine/pipeline.py` | Full rewrite → `ExtractionGraph` |
| `graph_engine/state.py` | NEW |
| `graph_engine/query_graph.py` | NEW |
| `retriever.py` | SK → `langchain-ollama` |
| `self_reflector.py` | Full rewrite → `QueryGraph` wrapper |
| `api.py` | Streaming + new node SSE events |
| `indexer/graph.py` | Pipeline API update |
| `static/index.html` | LangGraph trace panel |
| `graph_engine/tests/test_pipeline.py` | Rewrite for new API |
| `graph_engine/tests/test_types.py` | Add state tests |

**Out of scope**: `graph_engine/extractors/`, `graph_engine/models.py`, `graph_engine/store.py`, `graph_engine/chunker.py`, `graph_engine/deduplicator.py`, `models.py` (root), `document_loader.py`, `indexer/pipeline.py`, all other indexer files, Graph Tab vis.js visualization

---

## Further Considerations

1. **Iteration guard in QueryGraph**: `max_iterations=2` enforced via `state.iterations` counter + conditional edge back to `vector_retrieve_node` with refined query (query rewrite prompt identical to current `self_reflector.py`).
2. **`method="graph"` routing**: START → `graph_retrieve_node` directly (bypasses `vector_retrieve_node` and `evaluate_context_node`).
3. **langchain-ollama embed_query compatibility**: `OllamaEmbeddings.embed_query()` returns `list[float]` directly — no adapter needed for LanceDB.

---

## Verification Checklist

- [ ] `pip install langgraph langchain-core langchain-ollama` — no conflicts with existing deps
- [ ] `python -m pytest graph_engine/tests/` — all tests pass with new `ExtractionGraph` API
- [ ] `python -m pytest tests/` — FastAPI endpoint tests pass
- [ ] Index a test document → confirm triples stored in LanceDB
- [ ] Query with `method=hybrid`, `stream=true` → SSE stream contains `node_enter`, `routing_decision`, `node_complete` events before token stream
- [ ] UI: submit a query → LangGraph Flow panel animates nodes in real time → routing badge shows correct path
