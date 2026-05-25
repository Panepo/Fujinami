"""
GraphStore — triple persistence layer.

Current backend: LanceDB (already used by ragService for vectors).
Triples are stored in a separate ``graph_triples`` table so they
never interfere with the existing ``documents`` vector table.

Future backend: Apache AGE (PostgreSQL graph DB).
Swap by replacing ``LanceDBGraphStore`` with ``AGEGraphStore``
implementing the same ``GraphStore`` interface.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pyarrow as pa

from graph_engine.models import Triple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LanceDB schema for triples
# Stored as flat rows — one row per triple.
# Subject and object node data are JSON-serialized for simplicity.
# ---------------------------------------------------------------------------

_TRIPLE_SCHEMA = pa.schema(
    [
        pa.field("triple_id", pa.string()),          # sha256 of method+subj_id+pred+obj_id
        pa.field("source_doc", pa.string()),
        pa.field("method", pa.string()),             # extraction method: spacy, llm, hybrid
        pa.field("subject_id", pa.string()),
        pa.field("subject_name", pa.string()),
        pa.field("subject_type", pa.string()),
        pa.field("predicate", pa.string()),
        pa.field("object_id", pa.string()),
        pa.field("object_name", pa.string()),
        pa.field("object_type", pa.string()),
        pa.field("weight", pa.float32()),
        pa.field("subject_specs", pa.string()),       # JSON string
        pa.field("object_specs", pa.string()),        # JSON string
    ]
)

_TABLE_NAME = "graph_triples"


# ---------------------------------------------------------------------------
# Abstract interface — swap backends here
# ---------------------------------------------------------------------------


class GraphStore(ABC):
    """Interface for triple storage backends."""

    @abstractmethod
    def add_triples(self, triples: list[Triple]) -> int:
        """
        Persist triples. Upserts on triple_id (subject+predicate+object).
        Returns the number of rows written.
        """
        ...

    @abstractmethod
    def get_triples(
        self,
        source_doc: str | None = None,
        subject_name: str | None = None,
        subject_type: str | None = None,
        predicate: str | None = None,
        method: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query triples. All filters are AND-combined.
        Returns list of plain dicts (JSON-serializable).
        """
        ...

    @abstractmethod
    def delete_by_source(self, source_doc: str) -> int:
        """Delete all triples produced from *source_doc* (all methods). Returns deleted count."""
        ...

    @abstractmethod
    def delete_by_source_and_method(self, source_doc: str, method: str) -> int:
        """Delete triples for a specific (source_doc, method) pair. Returns deleted count."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored triples."""
        ...


# ---------------------------------------------------------------------------
# LanceDB backend
# ---------------------------------------------------------------------------


class LanceDBGraphStore(GraphStore):
    """
    Store triples in a LanceDB table named ``graph_triples``.

    Uses the same LanceDB database path as ragService but in a
    separate table — the existing ``documents`` vector table is
    never touched.

    Parameters
    ----------
    lance_db_path:
        Path to the LanceDB directory (same as ragService's lance_path).
    """

    def __init__(self, lance_db_path: str | Path) -> None:
        import lancedb  # noqa: PLC0415

        self._path = Path(lance_db_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._path))

        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)
            logger.info("Opened existing LanceDB graph_triples table")
        else:
            self._table = None
            logger.info("graph_triples table will be created on first write")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_triples(self, triples: list[Triple]) -> int:
        if not triples:
            return 0

        rows = [self._triple_to_row(t) for t in triples]

        if self._table is None:
            self._table = self._db.create_table(
                _TABLE_NAME, data=rows, schema=_TRIPLE_SCHEMA
            )
            logger.info("Created graph_triples table with %d rows", len(rows))
        else:
            # Upsert: delete existing rows with same triple_id first
            ids = [r["triple_id"] for r in rows]
            id_list = ", ".join(f"'{i}'" for i in ids)
            try:
                self._table.delete(f"triple_id IN ({id_list})")
            except Exception:
                pass
            self._table.add(rows)
            logger.info("Upserted %d triples into graph_triples", len(rows))

        return len(rows)

    def delete_by_source(self, source_doc: str) -> int:
        if self._table is None:
            return 0
        before = self._table.count_rows()
        self._table.delete(f"source_doc = '{source_doc}'")
        after = self._table.count_rows()
        deleted = before - after
        logger.info("Deleted %d triples for source_doc='%s'", deleted, source_doc)
        return deleted

    def delete_by_source_and_method(self, source_doc: str, method: str) -> int:
        if self._table is None:
            return 0
        before = self._table.count_rows()
        self._table.delete(f"source_doc = '{source_doc}' AND method = '{method}'")
        after = self._table.count_rows()
        deleted = before - after
        logger.info("Deleted %d triples for source_doc='%s', method='%s'", deleted, source_doc, method)
        return deleted

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_triples(
        self,
        source_doc: str | None = None,
        subject_name: str | None = None,
        subject_type: str | None = None,
        predicate: str | None = None,
        method: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._table is None:
            return []

        filters: list[str] = []
        if source_doc:
            filters.append(f"source_doc = '{source_doc}'")
        if subject_name:
            filters.append(f"subject_name = '{subject_name}'")
        if subject_type:
            filters.append(f"subject_type = '{subject_type}'")
        if predicate:
            filters.append(f"predicate = '{predicate}'")
        if method:
            filters.append(f"method = '{method}'")

        query = self._table.search()
        if filters:
            query = query.where(" AND ".join(filters))

        rows = query.to_list()
        return [self._row_to_dict(r) for r in rows]

    def count(self) -> int:
        if self._table is None:
            return 0
        return self._table.count_rows()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _triple_to_row(t: Triple) -> dict[str, Any]:
        import hashlib  # noqa: PLC0415

        triple_id = hashlib.sha256(
            f"{t.method}::{t.subject.id}::{t.predicate}::{t.object.id}".encode()
        ).hexdigest()[:24]

        return {
            "triple_id": triple_id,
            "source_doc": t.source_doc,
            "method": t.method,
            "subject_id": t.subject.id,
            "subject_name": t.subject.name,
            "subject_type": t.subject.type,
            "predicate": t.predicate,
            "object_id": t.object.id,
            "object_name": t.object.name,
            "object_type": t.object.type,
            "weight": float(t.weight),
            "subject_specs": json.dumps(t.subject.specs),
            "object_specs": json.dumps(t.object.specs),
        }

    @staticmethod
    def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "triple_id": row["triple_id"],
            "source_doc": row["source_doc"],
            "method": row.get("method", "unknown"),
            "subject": {
                "id": row["subject_id"],
                "name": row["subject_name"],
                "type": row["subject_type"],
                "specs": json.loads(row.get("subject_specs") or "{}"),
            },
            "predicate": row["predicate"],
            "object": {
                "id": row["object_id"],
                "name": row["object_name"],
                "type": row["object_type"],
                "specs": json.loads(row.get("object_specs") or "{}"),
            },
            "weight": row["weight"],
        }
