# Data Flow: Self-RAG Reflection Loop (`self_reflector.py`)

---

## 1. Overview

`SelfReflector` wraps the existing `RagService` retrieval pipeline with four
LLM-gated decisions that mirror the Self-RAG academic framework:

| # | Decision | LLM prompt type | Fallback on parse error |
| --- | --- | --- | --- |
| 1 | Is retrieval needed? | JSON `{"needed": bool}` | `true` (always retrieve) |
| 2 | Which chunks are relevant? | JSON `[index, …]` array | return all chunks |
| 3 | Is the answer grounded in context? | JSON `{"grounded": bool}` | `true` (assume grounded) |
| 4 | Refine query for next iteration | Plain text rewrite | original query unchanged |

The loop repeats steps 2–4 up to `max_iterations` times (default **2**).
It is activated by setting `self_rag: true` in a `POST /collections/{name}/query` request.

---

## 2. Integration with `api.py`

```
POST /collections/{name}/query
  body: { query, method, top_k, self_rag: true }
       │
       ▼
api.py — query_collection()
  ├─ Guard: collection must not have new_docs (HTTP 409 otherwise)
  │
  └─ self_rag=True branch  (takes precedence over stream=True)
       │
       ▼
  SelfReflector(rag).query(query, method, top_k)
  ├─ rag  = RagService instance for the collection
  └─ max_iterations = 2  (hardcoded default)
       │
       ▼
  (answer, sources, graphrag_context, SelfRagMeta)
       │
       ▼
  QueryResponse {
    collection, method, answer,
    sources:         list[SourceChunk] | None,
    graphrag_context: str | None,
    self_rag_meta:   SelfRagMeta        ← populated on self-RAG path
  }
```

`SelfReflector` is lazily imported inside the route handler to avoid circular
import issues. It holds a reference to the `RagService` facade and delegates
all retrieval and generation calls through it.

---

## 3. Complete Data Flow

```
SelfReflector.query(query, method, top_k)
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1 — Retrieval Gate                                         │
│                                                                  │
│  Prompt: "Does the following question require looking up         │
│           specific documents? Answer {\"needed\": true/false}."  │
│                                                                  │
│  _llm_call(prompt)  →  ChatHistory(user=prompt)                  │
│  rag._chat_service.get_chat_message_contents()                   │
│  POST {OLLAMA_CHAT_URL}/v1/chat/completions  (CHAT_MODEL)        │
│                                                                  │
│  Parse JSON  →  needed: bool                                     │
│  On parse error  →  needed = True                                │
└─────────────────────────┬────────────────────────────────────────┘
                          │
              ┌───────────┴────────────┐
           needed=False            needed=True
              │                        │
              ▼                        ▼
   ┌────────────────────┐    ┌─────────────────────────────────┐
   │  Direct generation │    │  RETRIEVAL + REFLECTION LOOP    │
   │  (no retrieval)    │    │  (up to max_iterations=2)       │
   │                    │    └───────────────┬─────────────────┘
   │  _generate_response│                    │
   │  (query, "")       │                    ▼ (see §3.1)
   │                    │
   │  SelfRagMeta {     │
   │    needed=False,   │
   │    relevant_chunks=│
   │    grounded=True,  │
   │    iterations=0    │
   │  }                 │
   └────────────────────┘
```

### 3.1 Retrieval + Reflection Loop (one iteration)

```
current_query = query   (refined on subsequent iterations)
iterations    = 0

LOOP while iterations < max_iterations:
   iterations += 1
   │
   ├─ [if method != "graph"]
   │     VECTOR SEARCH
   │     rag._raw_vector_results(current_query, top_k)
   │       → RagRetriever._raw_vector_results()
   │       → SK OllamaTextEmbedding  (OLLAMA_CHAT_URL / EMBEDDING_MODEL)
   │       → LanceDB ANN cosine search  table: "documents"
   │       → raw_rows: list[dict]  (doc_id, text, metadata, vector)
   │
   ├─ [if method in ("graph", "hybrid")]
   │     GRAPH SEARCH
   │     rag._graphrag_search(current_query, method="local")
   │       → RagRetriever._graphrag_search()
   │       → spaCy NER on query  →  entity names
   │       → LanceDBGraphStore.get_triples(subject_name / object_name)
   │       → graphrag_context: str
   │
   ├─ BUILD SourceChunk list from raw_rows
   │     doc_id    = row["doc_id"]
   │     chunk_index = metadata["chunk_index"]
   │     excerpt   = row["text"][:200]
   │     full_text = row["text"]
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 2 — Relevance Filter (batched)                             │
│                                                                  │
│  If no chunks → skip, relevant_sources = []                      │
│                                                                  │
│  Prompt: "Given the question, identify which of the following    │
│           numbered excerpts are relevant. Reply ONLY with a JSON │
│           array of indices, e.g. [0, 2]. Return [] if none."     │
│                                                                  │
│  Items: "[0] {chunk0.excerpt}\n[1] {chunk1.excerpt}\n…"          │
│                                                                  │
│  _llm_call(prompt) → POST OLLAMA_CHAT_URL (CHAT_MODEL)           │
│  Parse JSON array  → indices: list[int]                          │
│  On parse error    → return ALL chunks unchanged                 │
│                                                                  │
│  relevant_sources = [chunks[i] for i in indices]                 │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  CONTEXT ASSEMBLY                                                │
│                                                                  │
│  parts = []                                                      │
│  if relevant_sources:                                            │
│      parts.append("\n\n".join(s.full_text for s in …))          │
│  elif vector_context:                                            │
│      parts.append(vector_context)      ← fallback               │
│  if graphrag_context and method != "vector":                     │
│      parts.append("Graph context:\n" + graphrag_context)        │
│                                                                  │
│  merged_context = "\n\n".join(parts)                             │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 3 — Answer Generation                                      │
│                                                                  │
│  rag._generate_response(current_query, merged_context)           │
│    → RagRetriever._generate_response()                           │
│    → SK ChatHistory (system + user with context)                 │
│    → OllamaChatCompletion  (OLLAMA_CHAT_URL / CHAT_MODEL)        │
│    → answer: str                                                 │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 4 — Grounding Check                                        │
│                                                                  │
│  Prompt: "Given the context and the answer, decide whether the   │
│           answer is fully supported by the context without       │
│           adding unsupported information.                        │
│           Reply ONLY {\"grounded\": true/false}."                │
│                                                                  │
│  context[:3000]  +  answer[:1500]                                │
│  _llm_call(prompt) → POST OLLAMA_CHAT_URL (CHAT_MODEL)           │
│  Parse JSON  → grounded: bool                                    │
│  On parse error → grounded = True                                │
└─────────────────────────┬────────────────────────────────────────┘
                          │
              ┌───────────┴────────────┐
          grounded=True           grounded=False
              │                        │
              ▼                        ▼
   Return (answer,           [if iterations < max_iterations]
    sources, graphrag,        STEP 5 — Query Refinement
    SelfRagMeta{              │
     grounded=True})          │  Prompt: "The question was not answered
                              │  satisfactorily. Rewrite it to be more
                              │  specific. Reply with only the rewritten
                              │  query as plain text."
                              │
                              │  _llm_call(prompt) → CHAT_MODEL
                              │  On error → use original query
                              │
                              │  current_query = refined_query
                              │
                              └─ LOOP (next iteration)

[if iterations == max_iterations and grounded=False]
   Return (answer, sources, graphrag,
           SelfRagMeta{ grounded=False, iterations=max_iterations })
```

---

## 4. LLM Call Path (`_llm_call`)

All four LLM-gated decisions share the same helper:

```
SelfReflector._llm_call(prompt: str) → str
   │
   ▼
ChatHistory()
  .add_user_message(prompt)
   │
   ▼
rag._chat_service                     ← property on RagService
  = rag._retriever._chat_service      ← OllamaChatCompletion (SK)
   │
   ▼
get_chat_message_contents(
  history,
  settings=PromptExecutionSettings()  ← default settings, no temperature override
)
   │
   ▼
POST {OLLAMA_CHAT_URL}/v1/chat/completions
  model: CHAT_MODEL
   │
   ▼
str(responses[0]).strip()             ← raw text reply
```

Note: `_llm_call` uses `PromptExecutionSettings()` with no custom temperature,
so the model's default temperature applies. All four prompts require structured
JSON output; the callers extract JSON by scanning for `{…}` or `[…]` substrings
before parsing, making them tolerant of surrounding prose in the model's reply.

---

## 5. Process Log (`SelfRagMeta.process_log`)

Every decision and retrieval step appends a `SelfRagStep` to an in-memory log.
The complete log is returned in `SelfRagMeta.process_log` so API clients can
render a transparency timeline.

### Step keys emitted per path

| Condition | Steps logged (in order) |
| --- | --- |
| `needed=False` | `retrieval_check`, `generation` |
| `needed=True`, grounded on iteration 1 | `retrieval_check`, `vector_search`?, `graph_search`?, `chunk_filter`, `generation`, `grounding_check` |
| `needed=True`, not grounded, refined | + `query_refinement`, then repeat from `vector_search` |
| `needed=True`, exhausted iterations | same as above, final `grounding_check` has `ok=False` |

`?` = step only emitted when the retrieval method includes that search type.

### `SelfRagStep` field values per step

| `step` key | `label` | `result` | `ok` |
| --- | --- | --- | --- |
| `retrieval_check` | "Retrieval needed?" | "Yes" / "No" | `true` / `false` |
| `vector_search` | "Iteration N: Vector search" | "{n} chunk(s) retrieved" | `len > 0` |
| `graph_search` | "Iteration N: Graph search" | "Context retrieved" / "No graph context" | `bool(context)` |
| `chunk_filter` | "Iteration N: Relevance filter" | "{m} of {n} relevant" | `relevant > 0` |
| `generation` | "Iteration N: Answer generation" | *(none)* | `None` |
| `grounding_check` | "Iteration N: Grounding check" | "Grounded ✓" / "Not grounded ✗" | `true` / `false` |
| `query_refinement` | "Iteration N: Query refinement" | "Refined: \"{new_query}\"" | `None` |

---

## 6. Decision-Point Prompts

### 6.1 Retrieval Gate
```
Does the following question require looking up specific documents or data to
answer accurately? Answer with a single JSON object: {"needed": true} or
{"needed": false}.

Question: {query}
```

### 6.2 Relevance Filter (batched)
```
Given the question below, identify which of the following numbered excerpts
are relevant to answering it. Reply ONLY with a JSON array of the relevant
indices, e.g. [0, 2]. Return [] if none are relevant.

Question: {query}

Excerpts:
[0] {chunk0.excerpt}
[1] {chunk1.excerpt}
…
```

### 6.3 Grounding Check
```
Given the context and the answer below, decide whether the answer is fully
supported by the context without adding unsupported information.
Reply ONLY with a JSON object: {"grounded": true} or {"grounded": false}.

Context:
{context[:3000]}

Answer:
{answer[:1500]}
```

### 6.4 Query Refinement
```
The following question was not answered satisfactorily from the retrieved
documents. Rewrite it to be more specific so that a document search would
return better results. Reply with only the rewritten query as plain text.

Original question: {query}
```

---

## 7. Method × Retrieval Matrix

The `method` field from `QueryRequest` controls which searches run inside the
loop. Graph context is excluded from `merged_context` when `method="vector"`.

| `method` | Vector search | Graph search | Context merged |
| :--- | :---: | :---: | :--- |
| `"vector"` | ✓ | — | relevant chunks only (or all chunks as fallback) |
| `"graph"` | — | ✓ | graph context only |
| `"hybrid"` | ✓ | ✓ | relevant chunks + graph context |

---

## 8. `SelfRagMeta` Response Schema

```
SelfRagMeta {
  needed:          bool          — was retrieval attempted?
  relevant_chunks: int           — chunks surviving relevance filter (last iteration)
  grounded:        bool          — did grounding check pass?
  iterations:      int           — number of completed iterations (0 if no retrieval)
  process_log:     SelfRagStep[] — ordered step trace
}

SelfRagStep {
  step:   string        — machine key (see §5)
  label:  string        — human-readable label
  detail: string | null — extra context (query text, counts, sizes)
  result: string | null — outcome description
  ok:     bool | null   — pass/fail (null = neutral / no decision)
}
```

---

## 9. Error Handling

| Failure | Recovery |
| --- | --- |
| `_should_retrieve` parse error | Default `needed=True` (safe — always retrieves) |
| `_filter_relevant` parse error | Return **all** chunks (no filtering; no data lost) |
| `_check_grounding` parse error | Default `grounded=True` (exit loop early) |
| `_refine_query` error | Use original query unchanged |
| LLM / Ollama unreachable | Exception propagates; `api.py` global handler returns HTTP 500 |
| Zero raw chunks retrieved | `relevant_sources=[]`, context uses `vector_context` fallback |

---

## 10. Relationship to Standard Query Path

```
POST /query  { self_rag: false }          POST /query  { self_rag: true }
       │                                          │
       ▼                                          ▼
api.py standard path                    api.py self-RAG branch
  ├─ raw vector results                   SelfReflector.query()
  ├─ graphrag search                        ├─ STEP 1: retrieval gate
  ├─ merge context                          ├─ LOOP:
  └─ _generate_response()                  │    vector + graph search
                                           │    relevance filter
                                           │    answer generation
  QueryResponse {                          │    grounding check
    self_rag_meta: null                    │    query refinement (if needed)
  }                                        └─ (answer, sources, meta)

                                         QueryResponse {
                                           self_rag_meta: SelfRagMeta
                                         }
```

Key differences vs. the standard path:
- No streaming support (`stream: true` is silently ignored).
- Chunks are filtered by an LLM relevance judge before context assembly.
- The answer is validated for grounding; the query is re-tried if it fails.
- Full process transparency is returned in `self_rag_meta.process_log`.
