# Plan: Migrate DocumentLoader to Docling

Replace all custom format parsers (`pypdf`, `pymupdf`, `python-docx`, COM automation) and the hand-rolled VLM pipeline with docling's `DocumentConverter`. The public interface stays identical — only `document_loader.py` and `requirements.txt` change.

---

## Phase 1 — Dependencies

1. In `requirements.txt`: add `docling[asr]`; remove `pypdf`, `pymupdf`, `python-docx`, `pywin32`

## Phase 2 — Rewrite document_loader.py

2. Expand `SUPPORTED_EXTENSIONS` to the full docling input set:
   - **Documents**: `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.md`, `.markdown`, `.adoc`, `.asciidoc`, `.tex`, `.html`, `.htm`, `.xhtml`, `.csv`, `.txt`, `.text`, `.vtt`
   - **Images**: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp`
   - **Audio** (ASR extra): `.wav`, `.mp3`, `.m4a`, `.aac`, `.ogg`, `.flac`
   - **Video** (ASR extra + ffmpeg): `.mp4`, `.avi`, `.mov`

3. Add `_build_converter()` method — creates a `DocumentConverter` with:
   - `PdfPipelineOptions(do_picture_description=True, enable_remote_services=True)`
   - `PictureDescriptionVlmEngineOptions` (or equivalent) using `ApiVlmEngineOptions(runtime_type=VlmEngineType.API, url=f"{ollama_base_url}/v1/chat/completions", timeout=request_timeout)` with `vlm_model` set — replaces the custom `_detect_diagram_type` / `_describe_image` pipeline
   - Same options applied to `InputFormat.IMAGE` for image files

4. Rewrite `__init__`: keep same parameters (`ollama_base_url`, `vlm_model`, `request_timeout`); call `_build_converter()` and store the converter as `self._converter`

5. Rewrite `load()`: remove the `if/elif` format dispatch; call `self._converter.convert(str(path))` → `result.document.export_to_markdown()` for all formats

6. Keep `load_directory()` unchanged (it calls `load()` and filters by `SUPPORTED_EXTENSIONS`)

7. Delete private methods: `_load_pdf`, `_load_docx`, `_load_doc`, `_detect_diagram_type`, `_build_vlm_prompt`, `_describe_image`

## Phase 3 — No changes needed elsewhere

8. `ragService.py` — `DocumentLoader` and `SUPPORTED_EXTENSIONS` imports remain valid; `_compute_delta` / `_save_manifest` / upload validation all pick up the new extension set automatically
9. `api.py` — no changes; imports `SUPPORTED_EXTENSIONS` from `ragService` which re-exports from `document_loader`

---

## Relevant files

- `document_loader.py` — full rewrite of internals; `SUPPORTED_EXTENSIONS`, `DocumentLoader.__init__`, `load()`, `_build_converter()` are the core changes
- `requirements.txt` — swap 4 deps for `docling[asr]`
- `ragService.py` — read-only reference; instantiation at line 212, `SUPPORTED_EXTENSIONS` usage at lines 25, 378, 404
- `api.py` — read-only reference; validation at line 234

---

## Verification

1. `pip install -r requirements.txt` completes without conflicts
2. `python -c "from document_loader import DocumentLoader, SUPPORTED_EXTENSIONS; print(len(SUPPORTED_EXTENSIONS))"` — should print ~30
3. `DocumentLoader().load('data/harusame2/roles-overview.md')` returns non-empty markdown text
4. Upload a `.xlsx` and `.png` via the API endpoint — both should pass extension validation (line 234 in `api.py`)
5. Full indexing cycle: `RagService.index_documents()` with an XLSX or PPTX in the data folder completes and shows up in search

---

## Decisions

- Audio/video included (`docling[asr]` extra); ffmpeg must be installed separately (OS package or added to Dockerfile)
- `.doc` support dropped — no COM fallback
- Export format: `export_to_markdown()` — preserves table structure and headings for richer GraphRAG extraction vs. flat `export_to_text()`

---

## Further Considerations

1. **VLM preset compatibility**: Docling presets (`granite_vision`, `smoldocling`) embed specific prompt templates and model names. Since this project uses `llava:7b`, the preset may not match — need to either find a compatible preset or configure `PictureDescriptionVlmEngineOptions` without a preset, passing the model name directly. This is the trickiest part of the implementation.
2. **Dockerfile model pre-bake**: Docling downloads layout/OCR models (~1 GB) on first use. The Dockerfile should add a `RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"` step after installing deps to bake models into the image and avoid cold-start delays.
