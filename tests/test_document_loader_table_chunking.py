from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from document_loader import DocumentLoader


def _build_markdown_table(row_count: int) -> str:
    header = "| Item | Description |"
    sep = "| --- | --- |"
    rows = [
        f"| R{i} | This is a long cell value for row {i} to force splitting. |"
        for i in range(row_count)
    ]
    return "\n".join([header, sep] + rows)


def test_stage2_splits_large_table_and_preserves_markdown_structure(monkeypatch) -> None:
    monkeypatch.setenv("TABLE_CHUNK_SIZE", "180")
    loader = DocumentLoader()
    loader._narrate_table = lambda _markdown, _section: None  # type: ignore[method-assign]

    text = _build_markdown_table(row_count=6)
    elements = [
        {
            "id": "e1",
            "type": "table",
            "text": text,
            "page": 1,
            "section_path": ["Sheet1"],
        }
    ]

    chunks = loader._stage2_tables(elements, doc_stem="book")

    assert len(chunks) >= 2
    for idx, chunk in enumerate(chunks):
        lines = chunk["text"].splitlines()
        assert lines[0] == "| Item | Description |"
        assert lines[1] == "| --- | --- |"
        assert len(lines) >= 3
        assert chunk["chunk_id"] == f"book_table_0_part_{idx}"
