# Plan: Remove ragas from Project

**TL;DR:** Delete `ragas_runner.py`, strip the 3 evaluation API endpoints + 4 Pydantic models, remove the dependency from all package manifests, clean up the Dockerfile, and regenerate the lockfile.

---

## Phase 1 — Source Code

### 1. Delete `ragas_runner.py`
Delete the entire file — it contains only ragas integration code with no other purpose.

### 2. Edit `models.py`
Remove the `# RAGAS evaluation models` section comment and the following 4 Pydantic models (~lines 95–119):
- `EvaluateSingleRequest`
- `EvaluateSingleResponse`
- `EvaluateBatchSampleResult`
- `EvaluateBatchResponse`

### 3. Edit `api.py`
- Remove the entire `# RAGAS evaluation endpoints` block (~lines 638–738), which contains:
  - `GET /api/metrics` → `list_metrics()`
  - `POST /api/evaluate/single` → `evaluate_single()`
  - `POST /api/evaluate/batch` → `evaluate_batch()`
- Remove the `{"key": "RAGAS_MODEL", ...}` entry from the `_ENV_VARS` list (~line 752)
- Remove any now-unused imports of the deleted models (`EvaluateSingleRequest`, `EvaluateSingleResponse`, `EvaluateBatchSampleResult`, `EvaluateBatchResponse`)

---

## Phase 2 — Dependencies

### 4. Edit `requirements.txt`
Remove line: `ragas>=0.2`

### 5. Edit `pyproject.toml`
Remove line: `"ragas>=0.2",` from `[project] dependencies`

### 6. Regenerate `uv.lock`
Run `uv lock` — cleanly removes `ragas 0.4.3` and all its transitive dependencies from the lockfile. Do **not** hand-edit `uv.lock`.

---

## Phase 3 — Dockerfile

### 7. Edit `Dockerfile` line 40
Remove `ragas_runner.py` from the `COPY` instruction:
```
# Before
COPY api.py models.py ragService.py retriever.py document_loader.py ragas_runner.py __init__.py ./

# After
COPY api.py models.py ragService.py retriever.py document_loader.py __init__.py ./
```

### 8. Edit `Dockerfile` line 62
Remove the `RAGAS_MODE=gemma4:31b \` ENV line entirely.
> Note: this also fixes an existing typo — the variable was `RAGAS_MODE` but the codebase reads `RAGAS_MODEL`.

---

## Phase 4 — Docs (cosmetic cleanup)

### 9. Edit `README.md`
- Remove the "RAGAS evaluation" feature bullet (line 18)
- Remove the `RAGAS_MODEL` env-var example and its comment (lines 117–118)
- Remove the `# Optional: Ollama request timeout for RAGAS evaluation` comment (line 120) — keep `OLLAMA_TIMEOUT` itself if still used elsewhere
- Remove the "#### RAGAS Evaluation" section (line 227+)
- Remove `ragas_runner.py` entry from the project tree (line 279)

### 10. Edit `docs/dataflow-ragService.md`
- Remove the `RAGAS` mention from the ASCII architecture diagram (line 18)
- Remove the "### RAGAS Evaluation" section and its API table rows (lines 402–406)

---

## Phase 5 — Frontend (`static/index.html`)

### 11. Remove the "Verification" nav tab
Remove the `<button>` tag for the Verification tab from `<nav class="tab-bar">`:
```html
<!-- remove this line -->
<button class="tab-btn" onclick="switchTab('verify')">Verification</button>
```

### 12. Remove the entire `#tab-verify` panel
Delete the full `<div id="tab-verify" class="tab-panel">` block (~lines 355–452), which contains:
- The metric sidebar (`<aside class="verify-sidebar">`) with `v-metric-list`
- The "Single Entry" sub-tab (user input, contexts, response, reference fields + `btn-v-single`)
- The "Batch Upload" sub-tab (file dropzone + `btn-v-batch`)

### 13. Remove the "Send to Verification" button in the Query tab
Remove the button from inside the query results area (~line 288):
```html
<!-- remove this line -->
<button class="btn btn-send-verify" id="btn-send-verify" style="display:none" onclick="sendToVerification()">&#10148; Send to Verification</button>
```

### 14. Remove the `#verify-toast` element
Remove `<div id="verify-toast"></div>` (~line after the chunks modal).

### 15. Remove `.verify-*` and `#verify-toast` CSS
Remove all CSS rules that reference `.verify-layout`, `.verify-sidebar`, `.verify-main`, `.verify-form`, `.btn-send-verify`, `.verify-spinner`, `#verify-toast` (approximately lines 80–135).

### 16. Remove JavaScript functions
Delete all JS functions related to the Verification tab, including:
- `loadVerifyMetrics()`
- `renderVerifyMetrics()`
- `vMetricSetAll()`
- `switchVerifyTab()`
- `runVerifySingle()`
- `runVerifyBatch()`
- `vOnDragOver()`, `vOnDragLeave()`, `vOnDrop()`, `vOnFileSelected()`
- `sendToVerification()`
- `showVerifyToast()`
- The `_vAllMetrics` variable declaration

Also remove any call to `loadVerifyMetrics()` in the page initialization (e.g., inside `switchTab()` or a DOMContentLoaded handler).

---

## Verification Checklist

- [ ] `python -c "import api"` — no import errors
- [ ] Start the app; confirm `/api/metrics`, `/api/evaluate/single`, `/api/evaluate/batch` return 404
- [ ] `pip install -r requirements.txt` in a clean venv — `ragas` is not installed
- [ ] `docker build .` succeeds without referencing `ragas_runner.py`
- [ ] `uv.lock` contains no `[[package]]` block for `ragas`
- [ ] Open `index.html` in a browser — no "Verification" tab in the nav, no "Send to Verification" button in the Query tab

---

## Decisions & Notes

- `uv.lock` must be regenerated via `uv lock`, not hand-edited, to avoid stale transitive deps
- `OLLAMA_TIMEOUT` is a general timeout var (not ragas-specific) — keep it in README unless confirmed unused
- Phase 4 (docs) is cosmetic but recommended for accuracy; safe to skip if docs are not actively maintained
- The `RAGAS_MODE` typo in the Dockerfile (should be `RAGAS_MODEL`) is incidentally fixed by removing the line
