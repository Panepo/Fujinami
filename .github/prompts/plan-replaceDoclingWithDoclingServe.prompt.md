# Plan: Replace local docling with docling-serve

Replace the in-process `docling` Python library with HTTP calls to a `docling-serve` container. The serve API's `/v1/convert/file` endpoint returns JSON (`DoclingDocument`) or Markdown — the internal document structure is nearly identical to the Python API objects, just as plain dicts instead of Python classes.

---

## Phase 1 — document_loader.py (core refactor)

**1.1 Remove `_build_converter()`**
- Delete the method and its deferred `from docling...` imports (~L1454–1476)
- Remove `self._converter = self._build_converter()` from `__init__` (~L622)

**1.2 Add docling-serve URL to `__init__`**
- New param `docling_url = os.environ.get("DOCLING_URL", "http://localhost:5001")`
- Store as `self._docling_url`

**1.3 New helper: `_convert_file(path, to_format)`**
- POSTs to `{self._docling_url}/v1/convert/file` as `multipart/form-data`
- `"md"` → `{"to_formats": ["md"]}` → returns `response["document"]["md_content"]`
- `"json"` → `{"to_formats": ["json"], "image_export_mode": "embedded", "do_ocr": false, "include_images": true}` → returns `response["document"]["json_content"]`

**1.4 Replace `_load_passthrough(path)`**
- Replace `self._converter.convert()` + `export_to_markdown()` with `self._convert_file(path, "md")`
- Chunking logic below it is unchanged

**1.5 Replace `_parse_document(path, tmp_dir)`**
- Replace `self._converter.convert()` with `self._convert_file(path, "json")`
- XLSX branch (`_parse_excel`) is unchanged — it doesn't use docling

**1.6 Rewrite `_extract_elements_and_pictures` → `_extract_elements_and_pictures_from_json(json_doc, tmp_dir)`**
- Build `$ref → item` lookup from `json_doc["texts"]`, `["tables"]`, `["pictures"]`
- Walk `json_doc["body"]["children"]` recursively, following `$ref` links (order-preserving)
- Item access changes: `item["label"]`, `item["prov"][0]["page_no"]`, `item["text"]`, `item.get("level")`, `item.get("marker")`

**1.7 Replace `_table_to_text()`**
- Replace `table_item.export_to_markdown(doc=doc)` with a local grid-to-pipe-table renderer using `table_item["data"]["grid"]`

**1.8 Replace `_save_picture()`**
- Replace `pic_item.image.pil_image` with base64-decode of `pic_item["image"]["uri"]` (`data:image/png;base64,...`) and write bytes to disk

**1.9 Update `_item_label()`, `_page_no()`, `_heading_level()`, `_is_ordered()` helpers**
- Replace duck-typed attribute access with plain dict `.get()` calls

**1.10 Remove `_fallback_extract()`**
- Used `doc.texts`/`doc.tables` Python attributes — not applicable to JSON dicts; delete it

---

## Phase 2 — docker-compose.yml

**2.1 Add `docling-serve` service** (after `ollama`)
- Image: `quay.io/docling-project/docling-serve-cu130:main`
- Port: `5001:5001`
- GPU reservation (same `nvidia / all / gpu` block as ollama)
- Env: `DOCLING_SERVE_ENABLE_UI=0`

**2.2 Update `fujinami` service**
- Add `DOCLING_URL=http://docling-serve:5001` to `environment`
- Add `docling-serve` to `depends_on`

**2.3 Update scheduler labels**
- Add `docling-serve` to the midnight-restart docker command

---

## Phase 3 — nginx.conf

Add a `/docling/` location block in the `443` server with `proxy_read_timeout 300s` / `proxy_send_timeout 300s` (document conversion can be slow):

```nginx
location /docling/ {
    proxy_pass http://docling-serve:5001/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_http_version 1.1;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
}
```

---

## Phase 4 — .env (new file)

```env
DOCLING_URL=http://localhost:5001
```

For local dev without docker, point directly at a running docling-serve instance on port 5001.

---

## Phase 5 — requirements.txt

- Remove `docling[asr]`
- Add `requests` (for multipart form upload; `urllib` alone can't do this cleanly)

---

## Relevant files
- `document_loader.py` — all docling logic (lines ~600–1500)
- `docker-compose.yml` — service definitions
- `nginx.conf` — reverse proxy
- `requirements.txt` — remove `docling[asr]`, add `requests`
- `.env` — new file

---

## Verification
1. `pip install -r requirements.txt` — confirm `docling` is gone, no import errors
2. Run `docker run -p 5001:5001 quay.io/docling-project/docling-serve-cu130:main` locally + set `.env`
3. Test `DocumentLoader().load("sample.pdf")` — check chunks with headings, tables, images
4. Test passthrough: `DocumentLoader().load("sample.md")`
5. `docker compose up -d` on target server — confirm docling-serve starts, GPU visible in logs

---

## Decisions
- **CUDA image tag**: `main` (rolling head) — CUDA images intentionally don't carry `latest` per upstream policy
- **OCR disabled**: `do_ocr: false` to match current `PdfPipelineOptions(do_ocr=False)`
- **Images**: `image_export_mode: "embedded"` → base64 in JSON, no file serving needed

---

## Further Considerations
1. **Image tag pinning**: `main` is always latest. Pinning to `v1.20.0` gives reproducibility. Which do you prefer?
2. **Async conversion**: Large PDFs could hit HTTP timeouts on the sync endpoint. `/v1/convert/file/async` + polling avoids this but adds complexity. Include in scope?
3. **nginx exposure**: The plan adds a `/docling/` nginx route. If you prefer docling-serve to be internal-only (not accessible from outside), the nginx change can be skipped entirely.
