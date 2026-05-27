"""LanceDB helpers — open/create table, remove rows, upsert from embedded.json."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_LANCEDB_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 768)),
        pa.field("metadata", pa.string()),
    ]
)

_TABLE_NAME = "documents"


# ---------------------------------------------------------------------------
# Table management
# ---------------------------------------------------------------------------


def open_or_create_table(db: Any, schema: pa.Schema) -> Any:
    """Open the existing LanceDB table, dropping it on vector schema mismatch.

    Returns the table object, or ``None`` if the table does not yet exist.
    """
    if _TABLE_NAME not in db.table_names():
        logger.info("LanceDB table '%s' will be created on first index", _TABLE_NAME)
        return None

    existing = db.open_table(_TABLE_NAME)
    existing_vector_type = existing.schema.field("vector").type
    expected_vector_type = schema.field("vector").type
    if existing_vector_type != expected_vector_type:
        logger.warning(
            "LanceDB table '%s' has schema mismatch (vector %s vs expected %s). "
            "Dropping and recreating.",
            _TABLE_NAME,
            existing_vector_type,
            expected_vector_type,
        )
        db.drop_table(_TABLE_NAME)
        logger.info("LanceDB table '%s' will be recreated on first index", _TABLE_NAME)
        return None

    logger.info("Opened existing LanceDB table '%s'", _TABLE_NAME)
    return existing


# ---------------------------------------------------------------------------
# Row removal
# ---------------------------------------------------------------------------


def remove_from_lancedb(table: Any, doc_ids: list[str]) -> None:
    """Delete all LanceDB rows whose ``doc_id`` is in *doc_ids*."""
    if table is None:
        return
    for doc_id in doc_ids:
        try:
            safe_id = doc_id.replace("'", "\\'")
            table.delete(f"doc_id = '{safe_id}'")
            logger.info("Removed LanceDB rows for doc_id '%s'", doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Delete failed for doc_id '%s': %s", doc_id, exc)


# ---------------------------------------------------------------------------
# Upsert from embedded.json
# ---------------------------------------------------------------------------


def upsert_from_embedded_json(db: Any, table: Any, path: Path) -> Any:
    """Read *path* (embedded.json), build a PyArrow table, and add rows to LanceDB.

    The embedded.json must include a top-level ``"filename"`` field (full
    document filename with extension) and a ``"chunks"`` array where each
    element has at minimum ``"chunk_text_original"`` and ``"embedding"``
    fields.

    Returns the (possibly newly created) LanceDB table.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    # Prefer "filename" (full name with extension); fall back to doc_stem
    filename: str = data.get("filename") or data["doc_stem"]
    chunks: list[dict] = data.get("chunks", [])

    if not chunks:
        logger.warning("No chunks in embedded.json '%s', skipping upsert", path)
        return table

    rows: list[dict] = []
    for i, chunk in enumerate(chunks):
        meta = {
            "source": filename,
            "chunk_index": i,
            "chunk_type": chunk.get("chunk_type", "text"),
            "page_number": chunk.get("page_number"),
            "section_title": chunk.get("section_title"),
            "language": chunk.get("language"),
            "chunk_hash": chunk.get("chunk_hash"),
        }
        rows.append(
            {
                "id": f"{filename}#{i}",
                "doc_id": filename,
                "text": chunk.get("chunk_text_original", ""),
                "vector": chunk.get("embedding", []),
                "metadata": json.dumps(meta),
            }
        )

    typed_batch = pa.Table.from_pylist(rows, schema=_LANCEDB_SCHEMA)
    if table is not None:
        table.add(typed_batch)
        logger.info("Added %d rows to LanceDB table '%s'", len(rows), _TABLE_NAME)
    else:
        table = db.create_table(_TABLE_NAME, data=typed_batch, schema=_LANCEDB_SCHEMA)
        logger.info("Created LanceDB table '%s' with %d rows", _TABLE_NAME, len(rows))

    return table
