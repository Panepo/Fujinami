"""
Pydantic request/response models for the Fujinami FastAPI RAG server.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class CollectionCreateRequest(BaseModel):
    name: str


class CollectionRenameRequest(BaseModel):
    new_name: str


class CollectionInfo(BaseModel):
    name: str
    doc_count: int


class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int


class IndexResponse(BaseModel):
    collection: str
    status: str
    task_id: str


class IndexStatusResponse(BaseModel):
    task_id: str
    status: Literal["pending", "running", "done", "error"]
    detail: str | None = None


class QueryRequest(BaseModel):
    query: str
    method: Literal["vector", "global", "hybrid"] = "hybrid"
    top_k: int = 5
    stream: bool = False


class SourceChunk(BaseModel):
    doc_id: str
    chunk_index: int
    excerpt: str


class QueryResponse(BaseModel):
    collection: str
    method: str
    answer: str
    sources: list[SourceChunk] | None
