"""
Pydantic request/response models for the FastAPI RAG server.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class CollectionCreateRequest(BaseModel):
    name: str


class CollectionRenameRequest(BaseModel):
    new_name: str


class CollectionInfo(BaseModel):
    name: str
    doc_count: int
    index_status: Literal["not_indexed", "indexed", "new_docs"] = "not_indexed"
    vector_indexed: bool = False
    graph_indexed: bool = False


class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int


ALL_ENTITY_TYPES = [
    "organization",
    "person",
    "geo",
    "event",
    "concept",
    "technology",
    "product",
    "process",
    "system",
]


class IndexRequest(BaseModel):
    entity_types: list[str] | None = None
    mode: Literal["vector", "graph", "all"] = "all"
    force: bool = False


class IndexResponse(BaseModel):
    collection: str
    status: str
    task_id: str


class IndexStatusResponse(BaseModel):
    task_id: str
    collection: str = ""
    status: Literal["pending", "running", "done", "error"]
    detail: str | None = None


class QueryRequest(BaseModel):
    query: str
    method: Literal["vector", "graph", "hybrid"] = "hybrid"
    top_k: int = 5
    stream: bool = False


class SourceChunk(BaseModel):
    doc_id: str
    chunk_index: int
    excerpt: str
    full_text: str = ""


class QueryResponse(BaseModel):
    collection: str
    method: str
    answer: str
    sources: list[SourceChunk] | None = None
    graphrag_context: str | None = None


class DocumentChunk(BaseModel):
    chunk_index: int
    text: str
    chunk_type: str | None = None
    section_title: str | None = None
    page_number: int | None = None


# ---------------------------------------------------------------------------
# RAGAS evaluation models
# ---------------------------------------------------------------------------


class EvaluateSingleRequest(BaseModel):
    user_input: str = ""
    retrieved_contexts: list[str] = []
    response: str = ""
    reference: str = ""
    metrics: list[str]


class EvaluateSingleResponse(BaseModel):
    scores: dict[str, float | None]


class EvaluateBatchSampleResult(BaseModel):
    sample: dict
    scores: dict[str, float | None]


class EvaluateBatchResponse(BaseModel):
    results: list[EvaluateBatchSampleResult]
