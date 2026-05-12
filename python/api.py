"""
FastAPI HTTP server for the Fujinami RAG system.

Run from the workspace root:
    uvicorn python.api:app --reload
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

# Ensure python/ is on sys.path so sibling modules import as plain names.
_pkg_dir = Path(__file__).parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from models import (
    CollectionCreateRequest,
    CollectionInfo,
    CollectionRenameRequest,
    DocumentInfo,
    IndexRequest,
    IndexResponse,
    IndexStatusResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from ragService import SUPPORTED_EXTENSIONS, RagService

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_registry: dict[str, RagService] = {}
_tasks: dict[str, IndexStatusResponse] = {}

_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_ROOT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Scan existing ragdata/ sub-dirs on startup and populate the registry."""
    ragdata_root = _ROOT_DIR / "ragdata"
    if ragdata_root.exists():
        for entry in sorted(ragdata_root.iterdir()):
            if entry.is_dir() and entry.name not in {"__pycache__"}:
                _registry[entry.name] = RagService(collection_name=entry.name)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Fujinami RAG API", lifespan=lifespan)

_static_dir = _ROOT_DIR / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_service(name: str) -> RagService:
    svc = _registry.get(name)
    if svc is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return svc


def _doc_count(name: str) -> int:
    data_dir = _ROOT_DIR / "data" / name
    if not data_dir.exists():
        return 0
    return sum(
        1
        for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _index_status(name: str) -> str:
    """Return 'not_indexed', 'indexed', or 'new_docs' for the collection."""
    manifest_path = _ROOT_DIR / "ragdata" / name / "lancedb" / "file_manifest.json"
    data_dir = _ROOT_DIR / "data" / name

    # Load manifest
    stored: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    if not stored:
        return "not_indexed"

    # Compare current files to manifest
    if data_dir.exists():
        for f in data_dir.iterdir():
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                prev = stored.get(f.name)
                if prev is None:
                    return "new_docs"
                stat = f.stat()
                if stat.st_mtime != prev["mtime"] or stat.st_size != prev["size"]:
                    return "new_docs"

    return "indexed"


def _validate_name(name: str) -> None:
    if not _COLLECTION_NAME_RE.match(name):
        raise HTTPException(
            status_code=422,
            detail="Collection name must contain only letters, digits, hyphens, or underscores.",
        )


# ---------------------------------------------------------------------------
# Collection endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    return FileResponse(str(_static_dir / "index.html"))


@app.get("/collections", response_model=list[CollectionInfo])
async def list_collections() -> list[CollectionInfo]:
    return [
        CollectionInfo(name=name, doc_count=_doc_count(name), index_status=_index_status(name))
        for name in sorted(_registry)
    ]


@app.post("/collections", response_model=CollectionInfo, status_code=201)
async def create_collection(body: CollectionCreateRequest) -> CollectionInfo:
    _validate_name(body.name)
    if body.name in _registry:
        raise HTTPException(status_code=409, detail=f"Collection '{body.name}' already exists")
    svc = RagService(collection_name=body.name)
    (_ROOT_DIR / "data" / body.name).mkdir(parents=True, exist_ok=True)
    _registry[body.name] = svc
    return CollectionInfo(name=body.name, doc_count=0)


@app.patch("/collections/{name}", response_model=CollectionInfo)
async def rename_collection(name: str, body: CollectionRenameRequest) -> CollectionInfo:
    _get_service(name)
    _validate_name(body.new_name)
    if body.new_name in _registry:
        raise HTTPException(status_code=409, detail=f"Collection '{body.new_name}' already exists")

    old_data = _ROOT_DIR / "data" / name
    new_data = _ROOT_DIR / "data" / body.new_name
    old_ragdata = _ROOT_DIR / "ragdata" / name
    new_ragdata = _ROOT_DIR / "ragdata" / body.new_name

    if old_data.exists():
        old_data.rename(new_data)
    if old_ragdata.exists():
        old_ragdata.rename(new_ragdata)

    del _registry[name]
    _registry[body.new_name] = RagService(collection_name=body.new_name)
    return CollectionInfo(name=body.new_name, doc_count=_doc_count(body.new_name))


@app.delete("/collections/{name}", status_code=204)
async def delete_collection(name: str) -> None:
    _get_service(name)
    del _registry[name]

    data_dir = _ROOT_DIR / "data" / name
    ragdata_dir = _ROOT_DIR / "ragdata" / name
    if data_dir.exists():
        shutil.rmtree(data_dir)
    if ragdata_dir.exists():
        shutil.rmtree(ragdata_dir)


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------


@app.get("/collections/{name}/documents", response_model=list[DocumentInfo])
async def list_documents(name: str) -> list[DocumentInfo]:
    _get_service(name)
    data_dir = _ROOT_DIR / "data" / name
    if not data_dir.exists():
        return []
    return [
        DocumentInfo(filename=f.name, size_bytes=f.stat().st_size)
        for f in sorted(data_dir.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


@app.post("/collections/{name}/documents", response_model=DocumentInfo, status_code=201)
async def upload_document(name: str, file: UploadFile) -> DocumentInfo:
    _get_service(name)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    data_dir = _ROOT_DIR / "data" / name
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / (file.filename or "upload")
    content = await file.read()
    dest.write_bytes(content)
    return DocumentInfo(filename=dest.name, size_bytes=dest.stat().st_size)


@app.delete("/collections/{name}/documents/{filename}", status_code=204)
async def delete_document(name: str, filename: str) -> None:
    _get_service(name)
    target = (_ROOT_DIR / "data" / name / filename).resolve()
    # Guard against path traversal
    if not str(target).startswith(str((_ROOT_DIR / "data" / name).resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
    target.unlink()


# ---------------------------------------------------------------------------
# Index endpoints
# ---------------------------------------------------------------------------


async def _run_index(task_id: str, rag: RagService, entity_types: list[str] | None = None) -> None:
    _tasks[task_id].status = "running"
    try:
        await rag.index_documents(entity_types=entity_types)
        _tasks[task_id].status = "done"
    except Exception as exc:  # noqa: BLE001
        _tasks[task_id].status = "error"
        _tasks[task_id].detail = str(exc)


@app.post("/collections/{name}/index", response_model=IndexResponse, status_code=202)
async def trigger_index(
    name: str,
    background_tasks: BackgroundTasks,
    body: IndexRequest | None = None,
) -> IndexResponse:
    rag = _get_service(name)
    entity_types = (body.entity_types if body else None) or None
    task_id = str(uuid.uuid4())
    _tasks[task_id] = IndexStatusResponse(task_id=task_id, collection=name, status="pending")
    background_tasks.add_task(_run_index, task_id, rag, entity_types)
    return IndexResponse(collection=name, status="pending", task_id=task_id)


@app.get("/tasks", response_model=list[IndexStatusResponse])
async def list_tasks() -> list[IndexStatusResponse]:
    """Return all tasks that are still pending or running."""
    return [
        t for t in _tasks.values()
        if t.status in {"pending", "running"}
    ]


@app.get("/collections/{name}/index/{task_id}", response_model=IndexStatusResponse)
async def get_index_status(name: str, task_id: str) -> IndexStatusResponse:
    _get_service(name)
    status = _tasks.get(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return status


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------


async def _stream_answer(
    rag: RagService,
    query_req: QueryRequest,
    sources: list[SourceChunk] | None,
) -> AsyncGenerator[str, None]:
    """Yield SSE events: token*, sources (optional), done."""
    from semantic_kernel.contents import ChatHistory  # noqa: PLC0415

    if query_req.method == "global":
        # GraphRAG global — no SK streaming; run as subprocess then emit as single token
        answer = await rag.global_search(query_req.query)
        yield f"event: token\ndata: {json.dumps(answer)}\n\n"
    else:
        # Build context then stream via SK
        if query_req.method == "vector":
            context = await rag._raw_vector_context(query_req.query, query_req.top_k)
        else:
            vector_ctx = await rag._raw_vector_context(query_req.query, query_req.top_k)
            graphrag_ctx = await rag._graphrag_search(query_req.query, method="local")
            context = f"Vector Search Results:\n{vector_ctx}\n\nGraph Search Results:\n{graphrag_ctx}"

        history = ChatHistory()
        history.add_system_message(
            "You are a helpful assistant. Answer the user's question using only "
            "the provided context. If the context does not contain enough information, say so."
        )
        history.add_user_message(f"Context:\n{context}\n\nQuestion: {query_req.query}")

        from semantic_kernel.connectors.ai.ollama import (  # noqa: PLC0415
            OllamaChatPromptExecutionSettings,
        )

        async for chunk in rag._chat_service.get_streaming_chat_message_contents(
            history, OllamaChatPromptExecutionSettings()
        ):
            text = str(chunk[0]) if chunk else ""
            if text:
                yield f"event: token\ndata: {json.dumps(text)}\n\n"

    if sources is not None:
        sources_data = json.dumps([s.model_dump() for s in sources])
        yield f"event: sources\ndata: {sources_data}\n\n"

    yield "event: done\ndata: \"\"\n\n"


@app.post("/collections/{name}/query")
async def query_collection(name: str, body: QueryRequest):
    rag = _get_service(name)

    # Retrieve sources for non-global methods
    sources: list[SourceChunk] | None = None
    if body.method != "global":
        raw_rows = await rag._raw_vector_results(body.query, body.top_k)
        sources = []
        for row in raw_rows:
            try:
                meta = json.loads(row.get("metadata", "{}"))
            except (json.JSONDecodeError, TypeError):
                meta = {}
            sources.append(
                SourceChunk(
                    doc_id=row.get("doc_id", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    excerpt=row.get("text", "")[:200],
                )
            )

    if body.stream:
        return StreamingResponse(
            _stream_answer(rag, body, sources),
            media_type="text/event-stream",
        )

    # Non-streaming path
    if body.method == "vector":
        answer = await rag.vector_search(body.query, body.top_k)
    elif body.method == "global":
        answer = await rag.global_search(body.query)
    else:
        answer = await rag.hybrid_search(body.query, body.top_k)

    return QueryResponse(
        collection=name,
        method=body.method,
        answer=answer,
        sources=sources,
    )
