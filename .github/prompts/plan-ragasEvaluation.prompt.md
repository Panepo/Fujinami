## Plan: RAGAS Evaluation — Dev-Only Dependency

**TL;DR**: Add RAGAS only to a separate `requirements-eval.txt` that the Dockerfile never touches. The test file uses `pytest.importorskip("ragas")` so it silently skips in Docker/CI. Locally, it instantiates `RagService` directly (no HTTP server needed), runs queries, and feeds results into RAGAS for scoring.

---

### Phase 1 — Dependency Isolation *(no Docker changes)*

1. Create `requirements-eval.txt` with `ragas>=0.2` and `datasets`. All other deps already in `requirements.txt`.
2. In `pyproject.toml`, add two `[tool.poe.tasks]` entries:
   - `eval-install`: `uv pip install -r requirements-eval.txt`
   - `eval`: `pytest tests/test_ragas_eval.py -v -s`
3. **`Dockerfile` — zero changes.** It only installs `requirements.txt`.

---

### Phase 2 — Question Loader (`tests/ragas_helpers.py`)

4. Write `load_questions(collection: str, questions_dir: Path) -> list[dict]`
   - Glob `{collection}_*.json` in `tests/questions/`
   - Flatten all files into one list: `{question, ground_truth, category}`
   - Fields mapped from JSON: `user_input → question`, `reference → ground_truth`, `子分類 → category`

---

### Phase 3 — RAG Output Collector (`tests/ragas_helpers.py`, same file)

5. Write `async collect_rag_samples(questions, collection) -> list[SingleTurnSample]`
   - Instantiate `RagService(collection_name=collection)` — reads env vars from `.env` automatically
   - For each question: `await svc.hybrid_search(q)` → answer string
   - For each question: `await svc._raw_vector_results(q)` → `[r["text"] for r in results]` as contexts list
   - Build `ragas.SingleTurnSample(user_input, response, retrieved_contexts, reference)`

> **Note**: `_raw_vector_results` is a private method. Since this is test/eval code, direct access is intentional and acceptable.

---

### Phase 4 — RAGAS Pytest Test (`tests/test_ragas_eval.py`)

6. Module-level guard: `ragas = pytest.importorskip("ragas")` → entire file skips if ragas absent
7. Add `pytest_addoption` with `--collection` (default: `"S510AD"`) so collection is configurable at run time
8. `test_ragas_evaluation(collection_name)`:
   - Load questions → collect samples (via `asyncio.run(...)`)
   - Build `EvaluationDataset(samples)`
   - Configure RAGAS LLM from env vars: `LangchainLLMWrapper(ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_CHAT_URL))`
   - Configure RAGAS embeddings: `LangchainEmbeddingsWrapper(OllamaEmbeddings(...))`
   - Instantiate metrics: `AnswerRelevancy`, `Faithfulness`, `LLMContextPrecisionWithoutReference` — each receiving the wrapped LLM
   - Call `evaluate(dataset, metrics=[...])`
   - `print(result.to_pandas())` — no `assert`, scores are report-only

---

**Relevant files**

- `requirements-eval.txt` — new, `ragas>=0.2`, `datasets`
- `pyproject.toml` — add `poe eval` + `poe eval-install` tasks
- `tests/ragas_helpers.py` — new, question loader + sample collector
- `tests/test_ragas_eval.py` — new, pytest RAGAS evaluation
- `Dockerfile` — **no changes**

---

**Verification**

1. `docker build .` completes without ragas in the image (check with `docker run ... pip show ragas` → not found)
2. Locally: `uv pip install -r requirements-eval.txt` then `pytest tests/test_ragas_eval.py -v -s --collection S510AD`
3. Without ragas installed: `pytest tests/test_ragas_eval.py` → shows `SKIPPED` (not error)
4. Score table printed to stdout after evaluation run

---

**Decisions**
- Direct Python calls (not HTTP) — faster, no server required, Ollama must still be running
- `_raw_vector_results` used for context extraction — internal but stable enough for eval
- RAGAS 0.2+ API (`SingleTurnSample` / `EvaluationDataset`) — newer stable API
- `context_precision_without_reference` = `LLMContextPrecisionWithoutReference` in RAGAS 0.2+
