"""
Pydantic data models for the graph engine.

These models are the shared contract between all components
(extractors, store, API) and match the schema in graph-design-spec.md.
"""
from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Closed entity type list  (from graph-design-spec.md §3)
# ---------------------------------------------------------------------------

ENTITY_TYPES = Literal[
    # General
    "Person",
    "Organization",
    "Location",
    "Date",
    "Event",
    "Concept",
    # Technical / device-spec domain
    "Device",
    "Component",
    "Specification",
    "Protocol",
    "Interface",
    "Software",
    "Standard",
    "Error",
    "Configuration",
    "Product",
    "Version",
    "Vendor",
    # Fallback
    "Other",
]

# ---------------------------------------------------------------------------
# Closed relation type list  (from graph-design-spec.md §4)
# ---------------------------------------------------------------------------

RELATION_TYPES = Literal[
    "is_a",
    "part_of",
    "causes",
    "governs",
    "opposes",
    "related_to",
    # Domain-specific additions
    "supports_protocol",
    "certified_by",
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Node(BaseModel):
    """A single entity node in the knowledge graph."""

    id: str = Field(default="", description="UUID or hash of name+type — auto-computed if empty")
    name: str = Field(..., description="Human-readable entity label")
    type: str = Field(..., description="Entity category from ENTITY_TYPES")
    source_doc: str = Field(..., description="Which document produced this node")
    specs: dict[str, str] = Field(
        default_factory=dict,
        description="Numeric / technical specs stored as key-value (e.g. voltage='24V')",
    )

    @model_validator(mode="after")
    def _auto_id(self) -> "Node":
        if not self.id:
            key = f"{self.name.lower().strip()}::{self.type.lower().strip()}"
            self.id = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self


class Edge(BaseModel):
    """A directed relationship between two nodes."""

    source: str = Field(..., description="Node id of the subject")
    target: str = Field(..., description="Node id of the object")
    relation: str = Field(..., description="Predicate label from RELATION_TYPES")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score 0–1")
    source_doc: str = Field(..., description="Which document produced this edge")


class Triple(BaseModel):
    """
    The fundamental unit: (Subject) --[Predicate]--> (Object).

    Maps directly to graph-design-spec.md §2 triple structure.
    """

    subject: Node
    predicate: str = Field(..., description="Relation label — same as Edge.relation")
    object: Node
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    source_doc: str
    method: str = Field(default="unknown", description="Extraction method: spacy, llm, hybrid, or unknown")

    def to_edge(self) -> Edge:
        return Edge(
            source=self.subject.id,
            target=self.object.id,
            relation=self.predicate,
            weight=self.weight,
            source_doc=self.source_doc,
        )
