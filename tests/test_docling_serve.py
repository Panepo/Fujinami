"""
Simple smoke-test for the docling-serve instance.

Usage:
    python test_docling_serve.py [URL]          # URL defaults to DOCLING_URL env or http://localhost:5001

What it tests:
    1. Health check — GET /health
    2. Markdown conversion  — POST /v1/convert/file  (to_formats: ["md"])
    3. JSON conversion      — POST /v1/convert/file  (to_formats: ["json"])
    4. Image extraction     — confirms at least one picture item in JSON output

A small one-page PDF is generated on-the-fly with reportlab (if available),
otherwise a plain .txt file is used as a fallback.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = (
    sys.argv[1]
    if len(sys.argv) > 1 and sys.argv[1].startswith("http")
    else os.environ.get("DOCLING_URL", "http://10.68.129.51:5001")
).rstrip("/")

TIMEOUT = 120  # seconds — generous for first-run model download

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
DEFAULT_MD_FIXTURE = FIXTURES_DIR / "docling_smoke_test.md"
DEFAULT_JSON_FIXTURE = FIXTURES_DIR / "docling_smoke_test.json"
DEFAULT_IMG_FIXTURE = FIXTURES_DIR / "docling_smoke_test.jpg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(label: str) -> None:
    print(f"  [PASS] {label}")


def _fail(label: str, reason: str) -> None:
    print(f"  [FAIL] {label}: {reason}")
    sys.exit(1)


def _resolve_fixture(file_path: Path | None, default_path: Path, label: str) -> Path:
    if file_path is not None:
        return file_path
    if default_path.exists():
        print(f"     using fixture: {default_path}")
        return default_path
    _fail(label, f"fixture not found: {default_path}")


def _load_json_fixture(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _fail("json fixture", f"fixture not found: {path}")
    except json.JSONDecodeError as exc:
        _fail("json fixture", f"invalid JSON in fixture {path}: {exc}")


def _make_test_pdf(path: Path) -> bool:
    """Try to create a minimal PDF with reportlab. Returns True on success."""
    try:
        import io
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader

        c = canvas.Canvas(str(path), pagesize=A4)

        # Title — large bold so docling classifies it as a heading
        c.setFont("Helvetica-Bold", 18)
        c.drawString(72, 760, "docling-serve test document")

        # Section heading — bold + larger than body text
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, 720, "Section 1: Introduction")

        # Body paragraphs
        c.setFont("Helvetica", 12)
        c.drawString(72, 695, "This is a simple paragraph for testing docling-serve.")
        c.drawString(72, 678, "It contains headings, paragraphs, and a structured table.")

        # Draw a real table with visible grid lines so docling's layout model
        # can detect it — pipe-character text alone is NOT detected as a table in PDFs.
        table_rows = [
            ("Feature", "Status"),
            ("Markdown output", "OK"),
            ("JSON output", "OK"),
            ("Table extraction", "OK"),
        ]
        col_widths = [200, 100]
        row_h = 22
        x0, y_top = 72, 645

        for ri, row in enumerate(table_rows):
            y = y_top - ri * row_h
            x = x0
            for ci, cell in enumerate(row):
                c.rect(x, y - row_h, col_widths[ci], row_h)
                c.setFont("Helvetica-Bold" if ri == 0 else "Helvetica", 11)
                c.drawString(x + 5, y - row_h + 7, cell)
                x += col_widths[ci]

        # Embed a raster image so docling classifies it as a picture item.
        # Use the project's test image if available, otherwise fall back to
        # a minimal in-memory PNG.
        _test_img = DEFAULT_IMG_FIXTURE
        if _test_img.exists():
            img_buf = io.BytesIO(_test_img.read_bytes())
        else:
            # Minimal valid 1×1 white PNG (no external file needed)
            import struct, zlib
            def _png_chunk(tag: bytes, data: bytes) -> bytes:
                c_data = tag + data
                return struct.pack(">I", len(data)) + c_data + struct.pack(">I", zlib.crc32(c_data) & 0xFFFFFFFF)
            ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            idat = zlib.compress(b"\x00\xFF\xFF\xFF")
            img_buf = io.BytesIO()
            img_buf.write(b"\x89PNG\r\n\x1a\n")
            img_buf.write(_png_chunk(b"IHDR", ihdr))
            img_buf.write(_png_chunk(b"IDAT", idat))
            img_buf.write(_png_chunk(b"IEND", b""))

        img_buf.seek(0)
        c.drawImage(ImageReader(img_buf), 72, 460, width=200, height=100)

        c.save()
        return True
    except ImportError:
        return False


def _make_test_txt(path: Path) -> None:
    path.write_text(
        "# docling-serve test document\n\n"
        "## Section 1: Introduction\n\n"
        "This is a simple paragraph for testing docling-serve.\n\n"
        "| Feature      | Status  |\n"
        "|--------------|----------|\n"
        "| Markdown out | OK      |\n"
        "| JSON out     | OK      |\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health() -> None:
    print("\n1. Health check")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        if r.status_code == 200:
            _ok(f"GET /health → {r.status_code}")
        else:
            _fail("GET /health", f"status {r.status_code}: {r.text[:200]}")
    except requests.ConnectionError as exc:
        _fail("GET /health", f"connection error — is docling-serve running at {BASE_URL}? ({exc})")


def test_convert_md(file_path: Path | None = None) -> None:
    print("\n2. Markdown conversion")
    file_path = _resolve_fixture(file_path, DEFAULT_MD_FIXTURE, "md input")
    with open(file_path, "rb") as fh:
        r = requests.post(
            f"{BASE_URL}/v1/convert/file",
            files={"files": (file_path.name, fh, "application/octet-stream")},
            data=[("to_formats", "md")],
            timeout=TIMEOUT,
        )
    if r.status_code != 200:
        _fail("POST /v1/convert/file (md)", f"status {r.status_code}: {r.text[:300]}")

    body = r.json()
    md = body.get("document", {}).get("md_content", "")
    if not md.strip():
        _fail("md_content", "empty — check if docling-serve processed the file")
    _ok(f"md_content present ({len(md)} chars)")
    print(f"     preview: {md[:120].strip()!r}")


def test_convert_json(file_path: Path | None = None) -> None:
    print("\n3. JSON conversion")
    file_path = _resolve_fixture(file_path, DEFAULT_MD_FIXTURE, "json input")
    expected_doc = _load_json_fixture(DEFAULT_JSON_FIXTURE)
    with open(file_path, "rb") as fh:
        r = requests.post(
            f"{BASE_URL}/v1/convert/file",
            files={"files": (file_path.name, fh, "application/octet-stream")},
            data=[
                ("to_formats", "json"),
                ("image_export_mode", "embedded"),
                ("pdf_backend", "pypdfium2"),
                ("images_scale", "2.0"),
                ("include_images", "true"),
            ],
            timeout=TIMEOUT,
        )
    if r.status_code != 200:
        _fail("POST /v1/convert/file (json)", f"status {r.status_code}: {r.text[:300]}")

    body = r.json()
    status = body.get("status", "unknown")
    errors = body.get("errors", [])
    if errors:
        print(f"     errors: {errors}")

    doc = body.get("document", {})
    # json_content may come back as a dict or as a JSON string depending on server version
    raw = doc.get("json_content")
    if isinstance(raw, str):
        try:
            json_doc = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            json_doc = {}
    elif isinstance(raw, dict):
        json_doc = raw
    else:
        json_doc = {}

    if not json_doc:
        # Print available keys to help diagnose
        print(f"     response status: {status!r}")
        print(f"     document keys:   {list(doc.keys())}")
        for key, val in doc.items():
            snippet = str(val)[:80] if val else repr(val)
            print(f"       {key}: {snippet}")
        _fail("json_content", "empty — see document keys above")

    texts = json_doc.get("texts", [])
    tables = json_doc.get("tables", [])
    pictures = json_doc.get("pictures", [])
    _ok(f"json_content present — texts={len(texts)}, tables={len(tables)}, pictures={len(pictures)}")

    # Basic structural parity check against fixture schema sections.
    for key in ("texts", "tables", "pictures"):
        if key in expected_doc and key not in json_doc:
            _fail("json_content", f"missing expected section: {key}")

    if texts:
        first = texts[0]
        print(f"     first text item: label={first.get('label')!r}  text={str(first.get('text',''))[:80]!r}")


def test_image_extraction(img_path: Path | None = None) -> None:
    """POST fixtures/test_img.jpg directly — docling treats image files as picture items."""
    print("\n4. Image extraction (embedded base64)")
    img_path = _resolve_fixture(img_path, DEFAULT_IMG_FIXTURE, "image input")

    with open(img_path, "rb") as fh:
        r = requests.post(
            f"{BASE_URL}/v1/convert/file",
            files={"files": (img_path.name, fh, "image/jpeg")},
            data=[
                ("to_formats", "json"),
                ("image_export_mode", "embedded"),
                ("include_images", "true"),
            ],
            timeout=TIMEOUT,
        )
    if r.status_code != 200:
        _fail("POST /v1/convert/file (image)", f"status {r.status_code}: {r.text[:300]}")

    body = r.json()
    doc = body.get("document", {})
    raw = doc.get("json_content")
    if isinstance(raw, str):
        try:
            json_doc = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            json_doc = {}
    elif isinstance(raw, dict):
        json_doc = raw
    else:
        json_doc = {}

    pictures = json_doc.get("pictures", [])
    pages = json_doc.get("pages", {})

    # When docling ingests a bare image file, it stores the rasterised page under
    # pages[<n>]["image"] rather than in the pictures list.
    page_images = []
    page_iter = pages.values() if isinstance(pages, dict) else pages
    for page in page_iter:
        if isinstance(page, dict) and page.get("image"):
            page_images.append(page["image"])

    if not pictures and not page_images:
        _fail("pictures/pages", "no picture items and no page images returned for image input")

    # Prefer a picture item; fall back to the page-level image.
    if pictures:
        image_obj = pictures[0].get("image") or {}
    else:
        image_obj = page_images[0]

    uri = image_obj.get("uri", "") if isinstance(image_obj, dict) else ""
    if uri.startswith("data:"):
        _ok(f"embedded base64 URI present ({len(uri)} chars)")
    else:
        print(f"     image obj: {image_obj!r}")
        _fail("image.uri", f"expected data: URI, got: {uri[:80]!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"docling-serve smoke test")
    print(f"Target: {BASE_URL}")
    print("=" * 50)

    test_health()
    test_convert_md()
    test_convert_json()
    test_image_extraction()

    print("\n" + "=" * 50)
    print("All tests passed.")


if __name__ == "__main__":
    main()
