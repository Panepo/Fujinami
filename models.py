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


class IndexRequest(BaseModel):
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


class RewriteMeta(BaseModel):
    mode: str
    original_query: str
    rewritten_queries: list[str]
    hypothetical_document: str | None = None


class QueryRequest(BaseModel):
    query: str
    method: Literal["vector", "graph", "hybrid"] = "hybrid"
    top_k: int = 5
    stream: bool = False
    self_rag: bool = False
    rewrite: Literal["hyde", "multi_query", "step_back"] | None = None


class SourceChunk(BaseModel):
    doc_id: str
    chunk_index: int
    excerpt: str
    full_text: str = ""


class SelfRagStep(BaseModel):
    step: str                    # machine key, e.g. "retrieval_check"
    label: str                   # human-readable label
    detail: str | None = None    # extra info / query text / counts
    result: str | None = None    # outcome string
    ok: bool | None = None       # pass/fail indicator (None = neutral)


class SelfRagMeta(BaseModel):
    needed: bool
    relevant_chunks: int
    grounded: bool
    iterations: int
    process_log: list[SelfRagStep] = []


class QueryResponse(BaseModel):
    collection: str
    method: str
    answer: str
    sources: list[SourceChunk] | None = None
    graphrag_context: str | None = None
    self_rag_meta: Optional[SelfRagMeta] = None
    rewrite_meta: Optional[RewriteMeta] = None


class DocumentChunk(BaseModel):
    chunk_index: int
    text: str
    chunk_type: str | None = None
    section_title: str | None = None
    page_number: int | None = None

