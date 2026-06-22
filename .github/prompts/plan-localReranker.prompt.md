## Plan: Local Reranker Integration for Fujinami

Add an optional local reranker in the vector retrieval path so top-k context chunks are relevance-sorted before answer generation, while keeping current behavior as default-off and fully rollbackable by environment flag. The safest approach is to integrate reranking inside RagRetriever raw vector retrieval, overfetch candidates, rerank locally, then trim to requested top-k.

**Steps**
1. Phase 1: Architecture and model decision.
2. Create reranker.py as the dedicated local reranker module and define a stable LocalReranker interface with score(query, passages) and rerank(query, passages, top_k) methods.
3. Confirm retriever integration contract: RagRetriever consumes the reranker.py interface through composition instead of embedding model-loading logic inside retriever.py. This isolates model backend from API and graph flow.
4. Select default local model using your constraints (Traditional Chinese + English mixed, GPU if available with CPU fallback, <=150 ms target): evaluate at least two multilingual local candidates and lock one default with one fallback. Candidate set: BAAI/bge-reranker-v2-m3 (quality-first) and a lighter multilingual fallback for CPU latency.
5. Define feature flags and tunables with backward-compatible defaults: ENABLE_RERANKER=false, RERANKER_MODEL, RERANKER_OVERFETCH_FACTOR, RERANKER_MAX_CANDIDATES, RERANKER_DEVICE=auto, RERANKER_BATCH_SIZE.
6. Materialize local model artifact in repository for containerized/offline runtime: download `BAAI/bge-reranker-v2-m3` into `models/reranker/BAAI__bge-reranker-v2-m3` and pin `RERANKER_MODEL` to that local path in runtime config.
5. Phase 2: Retrieval-path integration.
7. Wire retriever.py to initialize LocalReranker lazily and call it from RagRetriever._raw_vector_results.
8. Update vector retrieval flow in RagRetriever._raw_vector_results to overfetch ANN candidates, run local reranker scores via reranker.py, reorder by reranker score descending, and return final top_k. Mark this as depending on Step 4.
9. Preserve no-reranker path exactly when disabled or model load fails (with warning log and deterministic fallback to ANN order). This is required for safe rollout.
10. Ensure async safety by moving reranker inference to thread executor where needed so API event loop remains responsive. This can run in parallel with Step 8 implementation details.
11. Keep self-RAG and hybrid flows unchanged at call sites so reranking applies transitively anywhere _raw_vector_results is used.
10. Phase 3: API/UI/config surfacing.
12. Expose new env keys in config diagnostics endpoint so operators can inspect active reranker settings.
13. Optionally add query-level override in QueryRequest (reranker on or off) only if needed for A/B testing. Keep this out of scope for initial minimal change unless explicitly requested.
14. Update README operational docs with local model setup, cache path, first-run warmup behavior, and rollback procedure.
15. Update Dockerfile to copy `reranker.py`, include `models/reranker/BAAI__bge-reranker-v2-m3`, and set `RERANKER_MODEL=/app/models/reranker/BAAI__bge-reranker-v2-m3` as container default.
14. Phase 4: Validation and rollout.
16. Add unit tests for reranker.py behavior: model lazy-load, scoring, ordering correctness, top_k trimming, and fallback behavior.
17. Add integration tests for retriever and API query path to verify source ordering changes only when reranker is enabled and responses remain schema-compatible.
18. Add evaluation run on existing question sets to compare baseline vs reranker for context precision and answer relevancy; include latency tracking and acceptance thresholds.
19. Roll out behind flag: run staging with ENABLE_RERANKER=true, then production enablement per collection after metric and latency pass.

**Relevant files**
- /d/Github/Fujinami/reranker.py — new dedicated local reranker module (LocalReranker, model init, scoring, rerank ordering, fallback behavior).
- /d/Github/Fujinami/retriever.py — primary insertion point for consuming reranker.py, overfetch-rerank-trim flow in RagRetriever._raw_vector_results.
- /d/Github/Fujinami/api.py — env config exposure and optional request-level override wiring if A/B control is added.
- /d/Github/Fujinami/models.py — optional QueryRequest extension for per-query reranker control (only if Step 12 approved).
- /d/Github/Fujinami/models/reranker/BAAI__bge-reranker-v2-m3/ — checked-in local reranker model artifacts used by container/runtime default.
- /d/Github/Fujinami/requirements.txt — add local reranker dependency set.
- /d/Github/Fujinami/pyproject.toml — keep dependency constraints aligned when needed.
- /d/Github/Fujinami/Dockerfile — copy reranker runtime module and pin container default reranker path to local model directory.
- /d/Github/Fujinami/tests/ — unit tests for reranker.py and integration tests for query behavior and response/source ordering with reranker enabled.
- /d/Github/Fujinami/README.md — operator documentation for model download, env flags, and rollback.

**Verification**
1. Unit: reranker-enabled ordering test proves returned top_k differs from raw ANN order when scores dictate; disabled mode returns raw ANN order unchanged.
2. Unit: failure-injection test for model load/inference confirms graceful fallback path with no request failure.
3. Integration: query endpoint with same input under reranker off vs on confirms stable response schema and changed source order where appropriate.
4. Performance: benchmark p50 and p95 added latency on representative collections for CPU and GPU-auto modes; pass if within <=150 ms added latency target in preferred runtime profile.
5. Quality: run existing evaluation dataset and compare context precision and answer relevancy; require non-regression and target measurable gain before default enablement.

**Decisions**
- Included scope: vector retrieval reranking only (affects vector and hybrid paths through shared retrieval function).
- Excluded scope: graph triple reranking, UI controls for reranker selection, and major query schema changes in initial rollout.
- Rollout policy: default-off feature flag with explicit operator enablement and immediate rollback via env toggle.

**Further Considerations**
1. Model finalization gate: pick one default and one fallback after quick benchmark on your real S510AD and Harusame query sets.
2. Device policy: keep device auto-detection with explicit override to avoid surprises on mixed deployment hosts.
3. Optional phase-2 enhancement: per-collection reranker config if different corpora require different model/latency tradeoffs.
