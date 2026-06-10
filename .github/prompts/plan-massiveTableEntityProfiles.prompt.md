## Plan: Massive Table Entity Profiles

Build massive-table ingestion on top of the existing DocumentLoader and RagIndexer pipeline by adding a normalization stage for wide/transposed tables, then emit two retrieval-friendly artifacts per sheet/file: entity-centric column profiles as the primary chunk type and bounded comparison chunks as secondary support for cross-entity questions. Reuse the current XLSX path, extend CSV into the same structured flow, keep vector indexing as the first-class target, and postpone any graph-schema redesign until retrieval quality is validated.

**Steps**
1. Phase 1 — Table shape discovery. Inspect how wide tables arrive from [document_loader.py](d:/Github/Fujinami/document_loader.py) in both `_parse_excel()` and the non-XLSX passthrough path, then define explicit heuristics for "massive table" detection: sparse first rows, transposed layout, first column as attribute names, and repeated entity columns. This step blocks the rest because later normalization depends on reliably recognizing sheets/files that should bypass generic table narration.
2. Phase 2 — Unified structured ingestion for XLSX and CSV. Route `.csv` files into the same tabular preprocessing path as `.xlsx` instead of raw passthrough text, producing a canonical in-memory table model with source filename, sheet/file name, header rows, attribute column, entity columns, and inferred units. Reuse the current [document_loader.py](d:/Github/Fujinami/document_loader.py) ownership boundary so the indexer continues to receive chunk dicts from one loader abstraction.
3. Phase 3 — Entity-centric serialization. Add a dedicated transformation stage in DocumentLoader after table parsing and before generic `_stage2_tables()` narration that converts each entity column into a self-contained profile chunk. Each chunk should carry: entity name, source sheet/file, optional table family, normalized metric/value pairs, unit hints, and enough surrounding context to stand alone in embeddings. Prefer one chunk per entity per logical table section; if the profile is too large, split by coherent row groups rather than token-only splitting.
4. Phase 4 — Comparison chunk generation. For the same canonical table model, emit secondary comparison chunks covering a limited number of entity columns per chunk with stable overlap. Use the reference approach from massiveTable.md: markdown sub-tables or compact narrative matrices that preserve row labels and unit context. Mark these chunks distinctly in metadata so retrieval tuning can prioritize entity profiles and fall back to comparison chunks for compare/rank/trend queries.
5. Phase 5 — Metadata contract update. Extend the chunk metadata emitted by DocumentLoader and stored through [indexer/store.py](d:/Github/Fujinami/indexer/store.py) so each row in LanceDB can expose retrieval filters and diagnostics such as `chunk_type`, `table_strategy`, `entity_name`, `entity_group`, `sheet_name`, `metric_keys`, and `comparison_scope`. This depends on steps 2-4 because the metadata should reflect the final serialized artifacts, not raw table text.
6. Phase 6 — Indexer compatibility and retrieval review. Keep [indexer/pipeline.py](d:/Github/Fujinami/indexer/pipeline.py) unchanged at the orchestration level, but verify that `index_documents()` and `_embed_and_save()` preserve the new chunk payloads end-to-end and that `upsert_from_embedded_json()` stores the richer metadata without schema breakage. This can run in parallel with step 7 once the metadata contract is defined.
7. Phase 7 — Query-path validation. Review how query retrieval consumes chunk text and metadata so the new chunk mix does not degrade answerability. The main focus is vector retrieval behavior, because current graph extraction in [graph_engine/pipeline.py](d:/Github/Fujinami/graph_engine/pipeline.py) still operates on concatenated free text and is not yet shaped for metric-level tabular relations. Keep graph changes out of phase 1 except for ensuring the new serialized text remains extractable if graph indexing is enabled.
8. Phase 8 — Tests and fixtures. Add focused loader tests for: transposed XLSX massive table detection, CSV normalization into the same canonical model, entity-profile chunk creation, comparison chunk windowing, and metadata persistence. Reuse the style of [tests/test_document_loader_table_chunking.py](d:/Github/Fujinami/tests/test_document_loader_table_chunking.py) for stage-local assertions and add at least one end-to-end indexing assertion that confirms the stored metadata shape through [indexer/store.py](d:/Github/Fujinami/indexer/store.py).
9. Phase 9 — Rollout controls and fallback behavior. Add a narrow feature flag or environment toggle for the new massive-table strategy so ordinary small tables still use the current `_stage2_tables()` narration path. This reduces regression risk and makes A/B evaluation against the current behavior straightforward.

**Relevant files**
- [document_loader.py](d:/Github/Fujinami/document_loader.py) — primary implementation surface; extend `_parse_excel()`, add structured CSV handling, and introduce the massive-table normalization/serialization branch before or within `_stage2_tables()`.
- [indexer/pipeline.py](d:/Github/Fujinami/indexer/pipeline.py) — preserve orchestration while validating that `index_documents()` and `_embed_and_save()` pass the new chunk forms through unchanged.
- [indexer/store.py](d:/Github/Fujinami/indexer/store.py) — widen stored metadata keys so entity profile and comparison chunks remain queryable and debuggable after upsert.
- [graph_engine/pipeline.py](d:/Github/Fujinami/graph_engine/pipeline.py) — reference only; confirm phase-1 serialized text remains compatible with current free-text graph extraction, but do not redesign graph modeling yet.
- [tests/test_document_loader_table_chunking.py](d:/Github/Fujinami/tests/test_document_loader_table_chunking.py) — closest existing template for stage-scoped table behavior tests.
- [docs/rag-architecture.md](d:/Github/Fujinami/docs/rag-architecture.md) — update the indexing-pipeline documentation after implementation so the new massive-table strategy is discoverable.
- [.github/reference/massiveTable.md](d:/Github/Fujinami/.github/reference/massiveTable.md) — source design reference for entity-centric and comparison serialization rules.

**Verification**
1. Unit-test the loader branch directly with synthetic wide/transposed tables for both XLSX and CSV inputs, asserting chunk counts, chunk ids, serialized text structure, and metadata fields.
2. Run the existing table-chunking test suite plus the new massive-table tests to confirm generic table behavior still passes.
3. Index a representative workbook/CSV export into a temp collection and inspect the resulting embedded metadata to verify entity profile chunks and comparison chunks are both stored.
4. Run a small retrieval check with queries like "What CPU does S510AD use?" and "Compare average hybrid boot time across selected systems" to confirm the expected chunk type is retrieved first.
5. If graph indexing is enabled in validation, confirm there is no extraction crash or pathological triple explosion from the new serialized text.

**Decisions**
- Include in scope: `.xlsx` sheets and standalone `.csv` exports.
- Include in phase 1: entity-centric chunks plus comparison chunks.
- Exclude from phase 1: graph-schema redesign, UI changes, and broad retrieval-ranking retuning beyond metadata and sanity checks.
- Recommended primary retrieval object: entity profile chunk. Comparison chunks should be supplemental and lower-priority.

**Further Considerations**
1. Prefer an explicit `table_strategy=massive_entity_profile` marker in metadata so retrieval evaluation can separate new chunks from legacy narrated tables.
2. If one workbook contains several related sheets, define a stable entity identity rule early, for example `entity_name + sheet family`, to avoid accidental cross-sheet merges.
3. For very wide comparison sheets, cap comparison windows to a small fixed width with overlap; otherwise the comparison chunks will become embedding-noisy and duplicate-heavy.
