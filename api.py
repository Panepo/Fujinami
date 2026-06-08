"""
FastAPI HTTP server for the RAG system.

Run from the workspace root:
    uvicorn python.api:app --reload
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shutil
import sys
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

# Ensure python/ is on sys.path so sibling modules import as plain names.
_pkg_dir = Path(__file__).parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

import csv
import io

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from models import (
    CollectionCreateRequest,
    CollectionInfo,
    CollectionRenameRequest,
    DocumentChunk,
    DocumentInfo,
    IndexRequest,
    IndexResponse,
    IndexStatusResponse,
    QueryRequest,
    QueryResponse,
    RewriteMeta,
    SourceChunk,
)
from rag_service import SUPPORTED_EXTENSIONS, RagService

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

app = FastAPI(title="RAG API", lifespan=lifespan)

_logger = logging.getLogger(__name__)

_static_dir = _ROOT_DIR / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    tb = traceback.extract_tb(exc.__traceback__)
    origin = tb[-1] if tb else None
    location = (
        f"{origin.filename}:{origin.lineno} in {origin.name}"
        if origin
        else "unknown location"
    )
    _logger.exception("Unhandled error at %s – %s: %s", location, type(exc).__name__, exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
            "location": location,
        },
    )


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
                current_hash = hashlib.sha256(f.read_bytes()).hexdigest()
                if current_hash != prev:
                    return "new_docs"

    return "indexed"


def _get_index_flags(name: str) -> dict:
    """Return ``{vector_indexed, graph_indexed}`` for a collection."""
    flags_path = _ROOT_DIR / "ragdata" / name / "index_flags.json"
    if not flags_path.exists():
        return {"vector_indexed": False, "graph_indexed": False}
    try:
        return json.loads(flags_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"vector_indexed": False, "graph_indexed": False}


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
    result = []
    for name in sorted(_registry):
        flags = _get_index_flags(name)
        result.append(CollectionInfo(
            name=name,
            doc_count=_doc_count(name),
            index_status=_index_status(name),
            vector_indexed=flags.get("vector_indexed", False),
            graph_indexed=flags.get("graph_indexed", False),
        ))
    return result


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


@app.get("/collections/{name}/documents/{filename}/download")
async def download_document(name: str, filename: str) -> FileResponse:
    target = (_ROOT_DIR / "data" / name / filename).resolve()
    # Guard against path traversal
    if not str(target).startswith(str((_ROOT_DIR / "data" / name).resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
    return FileResponse(path=target, filename=filename, media_type="application/octet-stream")


@app.get("/collections/{name}/documents/{filename}/embedded")
async def download_embedded(name: str, filename: str) -> FileResponse:
    """Return the per-document ``embedded.json`` as a file download.

    The file is written by the indexer after a document is embedded.
    Returns ``404`` if the document has not yet been indexed.
    """
    _get_service(name)
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    doc_stem = Path(filename).stem
    embedded_dir = (_ROOT_DIR / "ragdata" / name / "embedded").resolve()
    target = (embedded_dir / f"{doc_stem}.embedded.json").resolve()
    # Guard against path traversal
    if not str(target).startswith(str(embedded_dir)):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not target.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Embedded JSON for '{filename}' not found (document not yet indexed)",
        )
    return FileResponse(
        path=target,
        filename=f"{doc_stem}.embedded.json",
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{doc_stem}.embedded.json"'},
    )


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


@app.get("/collections/{name}/documents/{filename}/chunks", response_model=list[DocumentChunk])
async def get_document_chunks(name: str, filename: str) -> list[DocumentChunk]:
    rag = _get_service(name)
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    try:
        chunks = rag.get_document_chunks(filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"chunk lookup failed: {exc}") from exc
    return [
        DocumentChunk(
            chunk_index=c["chunk_index"],
            text=c["text"] if isinstance(c.get("text"), str) else "",
            chunk_type=c.get("chunk_type"),
            section_title=c.get("section_title"),
            page_number=c.get("page_number"),
        )
        for c in chunks
    ]


@app.get("/collections/{name}/debug/table")
async def debug_table(name: str) -> JSONResponse:
    """Diagnostic: return LanceDB table row count and unique doc_ids."""
    rag = _get_service(name)
    retriever = rag.retriever
    try:
        if not retriever._ensure_table():
            return JSONResponse({"error": "table does not exist"})
        tbl = retriever._table
        arrow = tbl.to_arrow()
        total_rows = arrow.num_rows
        doc_ids = sorted(set(arrow.column("doc_id").to_pylist()))
        return JSONResponse({"total_rows": total_rows, "doc_ids": doc_ids})
    except Exception as exc:
        return JSONResponse({"error": str(exc)})


# ---------------------------------------------------------------------------
# Index endpoints
# ---------------------------------------------------------------------------


async def _run_index(task_id: str, rag: RagService, mode: str = "all", force: bool = False) -> None:
    _tasks[task_id].status = "running"
    try:
        await rag.index_documents(mode=mode, force=force)
        _tasks[task_id].status = "done"
    except Exception as exc:  # noqa: BLE001
        tb = traceback.extract_tb(exc.__traceback__)
        origin = tb[-1] if tb else None
        location = (
            f"{origin.filename}:{origin.lineno} in {origin.name}"
            if origin
            else "unknown location"
        )
        _logger.exception(
            "Index task %s failed at %s – %s: %s",
            task_id,
            location,
            type(exc).__name__,
            exc,
        )
        _tasks[task_id].status = "error"
        _tasks[task_id].detail = f"{type(exc).__name__}: {exc} (at {location})"


@app.post("/collections/{name}/index", response_model=IndexResponse, status_code=202)
async def trigger_index(
    name: str,
    background_tasks: BackgroundTasks,
    body: IndexRequest | None = None,
) -> IndexResponse:
    rag = _get_service(name)
    mode = (body.mode if body else None) or "all"
    force = (body.force if body else False)
    task_id = str(uuid.uuid4())
    _tasks[task_id] = IndexStatusResponse(task_id=task_id, collection=name, status="pending")
    background_tasks.add_task(_run_index, task_id, rag, mode, force)
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


async def _run_rebuild(task_id: str, rag: RagService) -> None:
    _tasks[task_id].status = "running"
    try:
        count = await rag._indexer.rebuild_from_embedded()
        rag._retriever.reload_table()
        _tasks[task_id].status = "done"
        _tasks[task_id].detail = f"Rebuilt from {count} embedded.json file(s)"
    except Exception as exc:  # noqa: BLE001
        tb = traceback.extract_tb(exc.__traceback__)
        origin = tb[-1] if tb else None
        location = (
            f"{origin.filename}:{origin.lineno} in {origin.name}"
            if origin
            else "unknown location"
        )
        _logger.exception(
            "Rebuild task %s failed at %s – %s: %s",
            task_id,
            location,
            type(exc).__name__,
            exc,
        )
        _tasks[task_id].status = "error"
        _tasks[task_id].detail = f"{type(exc).__name__}: {exc} (at {location})"


@app.post("/collections/{name}/rebuild", response_model=IndexResponse, status_code=202)
async def trigger_rebuild(name: str, background_tasks: BackgroundTasks) -> IndexResponse:
    """Rebuild the LanceDB table from cached embedded.json files (no re-embedding).

    Use this after an indexer fix to re-populate the vector store without
    re-parsing or re-embedding documents.
    """
    rag = _get_service(name)
    task_id = str(uuid.uuid4())
    _tasks[task_id] = IndexStatusResponse(task_id=task_id, collection=name, status="pending")
    background_tasks.add_task(_run_rebuild, task_id, rag)
    return IndexResponse(collection=name, status="pending", task_id=task_id)




async def _stream_answer(
    rag: RagService,
    query_req: QueryRequest,
    sources: list[SourceChunk] | None,
    vector_context: str | None = None,
    graphrag_context: str | None = None,
    rewrite_meta: RewriteMeta | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE events: chunks (optional), graphrag (optional), token*, sources (optional), done."""
    try:
        async for event in _stream_answer_inner(rag, query_req, sources, vector_context, graphrag_context, rewrite_meta):
            yield event
    except Exception as exc:  # noqa: BLE001
        tb = traceback.extract_tb(exc.__traceback__)
        origin = tb[-1] if tb else None
        location = (
            f"{origin.filename}:{origin.lineno} in {origin.name}"
            if origin
            else "unknown location"
        )
        _logger.exception(
            "Streaming query failed at %s – %s: %s",
            location,
            type(exc).__name__,
            exc,
        )
        yield f"event: error\ndata: {json.dumps({'detail': str(exc), 'type': type(exc).__name__, 'location': location})}\n\n"


async def _stream_answer_inner(
    rag: RagService,
    query_req: QueryRequest,
    sources: list[SourceChunk] | None,
    vector_context: str | None = None,
    graphrag_context: str | None = None,
    rewrite_meta: RewriteMeta | None = None,
) -> AsyncGenerator[str, None]:
    import time  # noqa: PLC0415
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    # Emit rewrite metadata before context so the UI can display it immediately
    if rewrite_meta is not None:
        yield f"event: rewrite_meta\ndata: {json.dumps(rewrite_meta.model_dump())}\n\n"

    # Emit retrieved context before tokens so the UI can display it immediately
    if sources is not None:
        chunks_data = json.dumps([s.model_dump() for s in sources])
        yield f"event: chunks\ndata: {chunks_data}\n\n"
    if graphrag_context is not None:
        yield f"event: graphrag\ndata: {json.dumps(graphrag_context)}\n\n"

    if query_req.method == "graph":
        # Graph-only: emit node flow events + answer
        t0 = time.time()
        yield f"event: node_enter\ndata: {json.dumps({'node': 'graph_retrieve', 'timestamp': t0})}\n\n"
        answer = graphrag_context or await rag.global_search(query_req.query)
        yield f"event: node_complete\ndata: {json.dumps({'node': 'graph_retrieve', 'duration_ms': int((time.time()-t0)*1000)})}\n\n"
        yield f"event: routing_decision\ndata: {json.dumps({'needs_graph': True})}\n\n"

        t1 = time.time()
        yield f"event: node_enter\ndata: {json.dumps({'node': 'generate_answer', 'timestamp': t1})}\n\n"
        yield f"event: token\ndata: {json.dumps(answer)}\n\n"
        yield f"event: node_complete\ndata: {json.dumps({'node': 'generate_answer', 'duration_ms': int((time.time()-t1)*1000)})}\n\n"
    else:
        # Vector-first path
        t_vr = time.time()
        yield f"event: node_enter\ndata: {json.dumps({'node': 'vector_retrieve', 'timestamp': t_vr})}\n\n"
        if vector_context is None:
            vector_context = await rag._raw_vector_context(query_req.query, query_req.top_k)
        yield f"event: node_complete\ndata: {json.dumps({'node': 'vector_retrieve', 'duration_ms': int((time.time()-t_vr)*1000)})}\n\n"

        # Evaluate context
        t_ev = time.time()
        yield f"event: node_enter\ndata: {json.dumps({'node': 'evaluate_context', 'timestamp': t_ev})}\n\n"
        needs_graph = query_req.method == "hybrid"
        yield f"event: node_complete\ndata: {json.dumps({'node': 'evaluate_context', 'duration_ms': int((time.time()-t_ev)*1000)})}\n\n"
        yield f"event: routing_decision\ndata: {json.dumps({'needs_graph': needs_graph})}\n\n"

        context = vector_context or ""
        if needs_graph:
            t_gr = time.time()
            yield f"event: node_enter\ndata: {json.dumps({'node': 'graph_retrieve', 'timestamp': t_gr})}\n\n"
            if graphrag_context is None:
                graphrag_context = await asyncio.to_thread(rag._retriever._graph_context, query_req.query)
            yield f"event: node_complete\ndata: {json.dumps({'node': 'graph_retrieve', 'duration_ms': int((time.time()-t_gr)*1000)})}\n\n"
            parts_ctx: list[str] = []
            if vector_context:
                parts_ctx.append(f"Vector Search Results:\n{vector_context}")
            if graphrag_context:
                parts_ctx.append(f"Graph Search Results:\n{graphrag_context}")
            context = "\n\n".join(parts_ctx)

        # Generate answer with token streaming
        t_ga = time.time()
        yield f"event: node_enter\ndata: {json.dumps({'node': 'generate_answer', 'timestamp': t_ga})}\n\n"

        messages = [
            SystemMessage(content=(
                "You are a helpful assistant. Answer the user's question using only "
                "the provided context. If the context does not contain enough information, say so."
            )),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query_req.query}"),
        ]

        async for chunk in rag._chat_service.astream(messages):
            thinking = (chunk.additional_kwargs or {}).get("reasoning_content") or ""
            if thinking:
                yield f"event: think\ndata: {json.dumps(thinking)}\n\n"
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            if text:
                yield f"event: token\ndata: {json.dumps(text)}\n\n"

        yield f"event: node_complete\ndata: {json.dumps({'node': 'generate_answer', 'duration_ms': int((time.time()-t_ga)*1000)})}\n\n"

    if sources is not None:
        sources_data = json.dumps([s.model_dump() for s in sources])
        yield f"event: sources\ndata: {sources_data}\n\n"

    yield "event: done\ndata: \"\"\n\n"


@app.post("/collections/{name}/query")
async def query_collection(name: str, body: QueryRequest):
    rag = _get_service(name)

    if _index_status(name) == "new_docs":
        raise HTTPException(
            status_code=409,
            detail=(
                "Collection has unindexed documents. "
                "Please re-index the collection before querying."
            ),
        )

    sources: list[SourceChunk] | None = None
    vector_context: str | None = None
    graphrag_context: str | None = None
    rewrite_meta: RewriteMeta | None = None
    _rewrite_queries: list[str] | None = None
    _hyde_embedding: list[float] | None = None

    # Step 1 — Compute rewrite metadata (always, so it can be returned in every path)
    if body.rewrite and body.method != "graph":
        from rewriter import QueryRewriter  # noqa: PLC0415

        rewriter = QueryRewriter(rag._chat_service, rag._retriever._query_embedding_service)
        _rewrite_queries, _hyde_embedding, rewrite_meta = await rewriter.rewrite(body.query, body.rewrite)

    # Step 2 — Self-RAG path (includes rewrite_meta for display)
    if body.self_rag:
        from self_reflector import SelfReflector  # noqa: PLC0415

        reflector = SelfReflector(rag)
        answer, sources, graphrag_context, self_rag_meta = await reflector.query(
            body.query, body.method, body.top_k
        )
        return QueryResponse(
            collection=name,
            method=body.method,
            answer=answer,
            sources=sources,
            graphrag_context=graphrag_context,
            self_rag_meta=self_rag_meta,
            rewrite_meta=rewrite_meta,
        )

    # Step 3 — Regular retrieval (using rewrite results when available)
    if body.rewrite and body.method != "graph":
        if body.rewrite == "hyde" and _hyde_embedding is not None:
            # HyDE: use the hypothetical document embedding for retrieval
            raw_rows = await rag._raw_vector_results_from_embedding(_hyde_embedding, body.top_k)
        else:
            # multi_query / step_back: parallel searches then deduplicate
            assert _rewrite_queries is not None
            all_results = await asyncio.gather(
                *[rag._raw_vector_results(q, body.top_k) for q in _rewrite_queries]
            )
            seen_keys: set[tuple] = set()
            raw_rows = []
            for batch in all_results:
                for row in batch:
                    try:
                        meta_dict = json.loads(row.get("metadata", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        meta_dict = {}
                    key = (row.get("doc_id", ""), meta_dict.get("chunk_index", 0))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        raw_rows.append(row)

        vector_context = "\n\n".join(r.get("text", "") for r in raw_rows)
        sources = []
        for row in raw_rows:
            try:
                meta_dict = json.loads(row.get("metadata", "{}"))
            except (json.JSONDecodeError, TypeError):
                meta_dict = {}
            sources.append(
                SourceChunk(
                    doc_id=row.get("doc_id", ""),
                    chunk_index=meta_dict.get("chunk_index", 0),
                    excerpt=row.get("text", "")[:200],
                    full_text=row.get("text", ""),
                )
            )

    # Fetch vector chunks for non-graph methods (when rewrite is not active)
    elif body.method != "graph":
        raw_rows = await rag._raw_vector_results(body.query, body.top_k)
        vector_context = "\n\n".join(r.get("text", "") for r in raw_rows)
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
                    full_text=row.get("text", ""),
                )
            )

    # Fetch GraphRAG context for hybrid and global methods
    if body.method == "hybrid":
        graphrag_context = await rag._graphrag_search(body.query, method="local")
    elif body.method == "graph":
        graphrag_context = await rag._graphrag_search(body.query, method="global")

    if body.stream:
        return StreamingResponse(
            _stream_answer(rag, body, sources, vector_context, graphrag_context, rewrite_meta),
            media_type="text/event-stream",
        )

    # Non-streaming path — generate answer from pre-fetched context
    if body.method == "vector":
        answer = await rag._generate_response(body.query, vector_context or "")
    elif body.method == "graph":
        answer = graphrag_context or ""
    else:  # hybrid
        parts: list[str] = []
        if vector_context:
            parts.append(f"Vector Search Results:\n{vector_context}")
        if graphrag_context:
            parts.append(f"Graph Search Results:\n{graphrag_context}")
        merged = "\n\n".join(parts)
        answer = await rag._generate_response(body.query, merged)

    return QueryResponse(
        collection=name,
        method=body.method,
        answer=answer,
        sources=sources,
        graphrag_context=graphrag_context,
        rewrite_meta=rewrite_meta,
    )


# ---------------------------------------------------------------------------
# Graph endpoints
# ---------------------------------------------------------------------------


@app.get("/collections/{name}/graph/stats")
async def get_graph_stats(name: str) -> dict:
    """Return triple count for the collection's graph store."""
    from graph_engine.store import LanceDBGraphStore  # noqa: PLC0415

    svc = _get_service(name)
    lance_path = svc._retriever._lance_path
    try:
        store = LanceDBGraphStore(lance_path)
        return {"triple_count": store.count()}
    except Exception:
        return {"triple_count": 0}


@app.get("/collections/{name}/graph")
async def get_graph_triples(
    name: str,
    source_doc: str | None = None,
    subject_type: str | None = None,
    predicate: str | None = None,
) -> list[dict]:
    """Return triples from the knowledge graph, with optional filters."""
    from graph_engine.store import LanceDBGraphStore  # noqa: PLC0415

    svc = _get_service(name)
    lance_path = svc._retriever._lance_path
    try:
        store = LanceDBGraphStore(lance_path)
        return store.get_triples(
            source_doc=source_doc or None,
            subject_type=subject_type or None,
            predicate=predicate or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Environment config endpoint
# ---------------------------------------------------------------------------

_ENV_VARS: list[dict] = [
    # key, default (None = required), description
    {"key": "OLLAMA_INDEX_URL",  "default": None,        "description": "Ollama server used for indexing / embeddings"},
    {"key": "OLLAMA_CHAT_URL",   "default": None,        "description": "Ollama server used for chat / query"},
    {"key": "CHAT_MODEL",        "default": None,        "description": "Model used for answering queries"},
    {"key": "EMBEDDING_MODEL",   "default": None,        "description": "Embedding model for vector search"},
    {"key": "VLM_MODEL",         "default": None,        "description": "Vision-language model for image documents"},
    {"key": "VLM_TIMEOUT",       "default": "180",       "description": "Timeout (s) for VLM requests"},
    {"key": "INDEX_MODEL",       "default": "",          "description": "Override model used during indexing"},
    {"key": "CHUNK_SIZE",        "default": "800",       "description": "Token size per chunk"},
    {"key": "CHUNK_OVERLAP",     "default": "80",        "description": "Overlap between consecutive chunks"},
    {"key": "GRAPH_EXTRACTOR",   "default": "hybrid",    "description": "Entity extractor: hybrid | llm | spacy"},
    {"key": "EXTRACT_MODEL",     "default": "",          "description": "Model used by the LLM graph extractor"},

    {"key": "OLLAMA_TIMEOUT",    "default": "1800",      "description": "Timeout (s) for Ollama API calls"},
]


@app.get("/api/env")
async def get_env_config() -> list[dict]:
    """Return the current values of known application environment variables."""
    import os  # noqa: PLC0415

    result = []
    for entry in _ENV_VARS:
        key = entry["key"]
        raw = os.environ.get(key)
        result.append({
            "key": key,
            "value": raw if raw is not None else entry["default"],
            "set": raw is not None,
            "default": entry["default"],
            "description": entry["description"],
        })
    return result
