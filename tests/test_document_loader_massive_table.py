from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from document_loader import DocumentLoader
from indexer.store import upsert_from_embedded_json


def test_parse_csv_routes_into_structured_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "massive.csv"
    csv_path.write_text(
        "Metric,S510AD,UX10\n"
        "CPU model name,Ryzen,Intel\n"
        "Average System Boot time (Unit second),18.13,20.21\n",
        encoding="utf-8",
    )

    loader = DocumentLoader()
    elements = loader._parse_csv(csv_path)

    assert len(elements) == 2
    assert elements[0]["type"] == "heading"
    assert elements[1]["type"] == "table"
    assert elements[1]["sheet_name"] == "massive"
    assert elements[1]["table_rows"][0][0] == "Metric"
    assert "| Metric | S510AD | UX10 |" in elements[1]["text"]


def test_massive_table_emits_entity_and_comparison_chunks(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_MASSIVE_TABLE_STRATEGY", "1")
    loader = DocumentLoader()

    rows = [
        ["", "", "", ""],
        ["Metric", "S510AD", "UX10", "F110"],
        ["CPU model name", "Ryzen AI 7 350", "Core Ultra", "Core i7"],
        ["Memroy 1 model name", "DDR5 32GB", "DDR5 16GB", "DDR4 16GB"],
        ["Memroy 2 model name", "DDR5 32GB", "", ""],
        ["Average System Boot time (Hybrid Shutdown, Unit second)", "18.13", "20.21", "22.01"],
        ["Average BIOS Post Time (Hybrid Shutdown, Unit second)", "9.46", "11.20", "12.11"],
        ["BIOS Version", "R0.52", "R0.49", "R0.38"],
    ]
    element = {
        "id": "e1",
        "type": "table",
        "text": loader._rows_to_markdown(rows),
        "page": None,
        "section_path": ["TestConfiguration"],
        "sheet_name": "TestConfiguration",
        "table_rows": rows,
    }

    chunks = loader._stage2_tables([element], doc_stem="book")

    assert any(c.get("chunk_type") == "entity_profile" for c in chunks)
    assert any(c.get("chunk_type") == "table_comparison" for c in chunks)
    entity = next(c for c in chunks if c.get("chunk_type") == "entity_profile")
    assert entity.get("table_strategy") == "massive_entity_profile"
    assert entity.get("entity_name")
    assert entity.get("sheet_name") == "TestConfiguration"
    assert entity.get("metric_keys")


def test_upsert_persists_massive_table_metadata(tmp_path: Path) -> None:
    class FakeTable:
        def __init__(self) -> None:
            self.batch = None

        def add(self, typed_batch) -> None:
            self.batch = typed_batch

    class FakeDB:
        def __init__(self) -> None:
            self.created = None

        def create_table(self, _name, data=None, schema=None):
            table = FakeTable()
            table.batch = data
            self.created = table
            return table

    embedded = {
        "doc_stem": "demo",
        "filename": "demo.csv",
        "chunks": [
            {
                "chunk_text_original": "Entity Profile: S510AD",
                "chunk_text_embedded": "[Document: demo]\nEntity Profile: S510AD",
                "chunk_type": "entity_profile",
                "table_strategy": "massive_entity_profile",
                "entity_name": "S510AD",
                "entity_group": "S510AD::part_0",
                "sheet_name": "TestConfiguration",
                "metric_keys": ["CPU model name", "BIOS Version"],
                "comparison_scope": ["S510AD", "UX10"],
                "language": "EN",
                "chunk_hash": "abc",
                "embedding": [0.0] * 768,
            }
        ],
    }
    embedded_path = tmp_path / "demo.embedded.json"
    embedded_path.write_text(json.dumps(embedded), encoding="utf-8")

    db = FakeDB()
    table = upsert_from_embedded_json(db, None, embedded_path)

    row = table.batch.to_pylist()[0]
    meta = json.loads(row["metadata"])
    assert meta["chunk_type"] == "entity_profile"
    assert meta["table_strategy"] == "massive_entity_profile"
    assert meta["entity_name"] == "S510AD"
    assert meta["sheet_name"] == "TestConfiguration"
    assert meta["metric_keys"] == ["CPU model name", "BIOS Version"]
