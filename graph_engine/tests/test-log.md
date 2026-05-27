# Test Run Log — `graph_engine/tests/`

---

## Run 1 — Initial RED state
**Date:** 2026-05-20  
**Command:** `pytest graph_engine/tests/ -v --tb=short --no-header`  
**Result:** `1 error during collection`

**Failure:** `ImportError: cannot import name 'TextChunker' from 'graph_engine.chunker'`  
Spec used class name `TextChunker` but the implementation exposes a function `chunk_text(text, chunk_size, overlap)`.

---

## Run 2 — After fixing imports
**Date:** 2026-05-20  
**Command:** `pytest graph_engine/tests/ -v --tb=short --no-header`  
**Result:** `14 failed, 68 passed, 4 errors in 2.36s`

### Errors (4) — SpacyExtractor
All 4 `TestSpacyExtractor` tests errored at fixture setup.  
**Cause:** `en_core_web_sm` spaCy model not installed in this environment.  
**Fix:** Add `pytest.skip` guard in fixture.

### Failures (14)

| Group | Count | Root cause |
|---|---|---|
| `test_store.py::TestGraphStoreRetrieve` | 5 | `get_triples()` returns `list[dict]` not `list[Triple]`. Tests used `.source_doc`, `.weight`, `.subject.name` as object attrs. |
| `test_types.py::TestEntityTypes` | 5 | `ENTITY_TYPES` is a `typing.Literal` alias. `set(ENTITY_TYPES)` wraps the Literal type as one set element, not the string values. Fix: use `typing.get_args(ENTITY_TYPES)`. |
| `test_types.py::TestRelationTypes` | 4 | Same Literal type issue as above. |

---

## Run 3 — All fixes applied
**Date:** 2026-05-20  
**Command:** `pytest graph_engine/tests/ -v --tb=short --no-header`  
**Result:** ✅ `84 passed, 4 skipped in 1.19s`

### Skipped (4) — SpacyExtractor
Tests skipped because `en_core_web_sm` is not installed.  
**To activate:** `python -m spacy download en_core_web_sm` in the dev-server venv.

### Summary

| Test file | Tests | Passed | Skipped | Failed |
|---|---|---|---|---|
| `test_chunker.py` | 8 | 8 | 0 | 0 |
| `test_deduplicator.py` | 8 | 8 | 0 | 0 |
| `test_extractors.py` | 10 | 6 | 4 | 0 |
| `test_models.py` | 17 | 17 | 0 | 0 |
| `test_pipeline.py` | 15 | 15 | 0 | 0 |
| `test_store.py` | 14 | 14 | 0 | 0 |
| `test_types.py` | 16 | 16 | 0 | 0 |
| **Total** | **88** | **84** | **4** | **0** |

### Key findings
- `ENTITY_TYPES` and `RELATION_TYPES` are `typing.Literal` aliases (not plain lists).  
  Use `typing.get_args()` to extract values when writing assertions.
- `LanceDBGraphStore.get_triples()` returns `list[dict]`, not `list[Triple]`.  
  Dict keys include: `source_doc`, `weight`, `subject_name` (or similar), `object_name`.
- spaCy extractor tests require `en_core_web_sm` model — skip automatically if not present.
- LLM extractor uses `urllib.request.urlopen` (not `requests`) — mock target is `urllib.request.urlopen`.

---

---

## Run 4 — After installing `en_core_web_sm`
**Date:** 2026-05-21  
**Command:** `pytest graph_engine/tests/ -v --tb=short --no-header`  
**Result:** ✅ `88 passed in 2.19s` — 0 skipped, 0 failed
