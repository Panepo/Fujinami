# Fujinami RAG Service Architecture

## Core Stack
- **FastAPI Server**: api.py (entry point)
- **LLM Client**: langchain-ollama (ChatOllama + OllamaEmbeddings)
- **Graph Query Orchestration**: LangGraph `QueryGraph` (`graph_engine/query_graph.py`)
- **Chat LLM**: Ollama (OLLAMA_CHAT_URL, CHAT_MODEL)
- **Embeddings**: Ollama (EMBEDDING_MODEL)
- **Vector Store**: LanceDB (persistent, file-based)
- **Graph Engine**: graph_engine (local triples, spaCy NER)
- **Document Processing**: docling + Ollama VLM

## Environment Variables
- `OLLAMA_INDEX_URL` - Ollama for indexing/embeddings
- `OLLAMA_CHAT_URL` - Ollama for chat
- `CHAT_MODEL` - Model ID for chat
- `EMBEDDING_MODEL` - Model ID for embeddings
- `VLM_MODEL` - Vision model for images
- `TOP_K` - Default top-k for vector search (default: 5)
- `CHUNK_SIZE` - Tokens per chunk (default: 800)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 80)
- `GRAPH_EXTRACTOR` - "hybrid"|"llm"|"spacy"
- Optional: `EXTRACT_MODEL`, `INDEX_MODEL`, `VLM_TIMEOUT`, `OLLAMA_TIMEOUT`

## Key Data Models (models.py)
- `QueryRequest`: query, method ("vector"|"graph"|"hybrid"), top_k, stream
- `QueryResponse`: collection, method, answer, sources, graphrag_context
- `SourceChunk`: doc_id, chunk_index, excerpt, full_text
- `CollectionInfo`: name, doc_count, index_status, vector_indexed, graph_indexed
- `IndexRequest`: mode ("vector"|"graph"|"all"), force
- `DocumentChunk`: chunk_index, text, chunk_type, section_title, page_number

## Query Endpoint
- **Path**: POST `/collections/{name}/query`
- **Request**: QueryRequest body
- **Response**: QueryResponse (or SSE stream if `stream: true`)
- **Methods**:
  - `vector`: LanceDB semantic search only
  - `graph`: Knowledge graph triples only
  - `hybrid`: Both (merged context)
- **Streaming**: Server-Sent Events with events: chunks, graphrag, token*, sources, done

## Query Flow (Non-Streaming)
1. Validate collection exists and has no unindexed docs
2. For vector/hybrid: `_raw_vector_results()` — ANN search merged with title-keyword match → create SourceChunk list
3. For graph/hybrid: `_graph_context()` with 3 cascading strategies:
   - Strategy 1: spaCy NER + noun-chunks → LIKE-based LanceDBGraphStore lookup
   - Strategy 2: raw query tokens → LIKE-based lookup (fallback)
   - Strategy 3: `OllamaEmbeddings` similarity vs all stored entity names (fallback)
4. Merge contexts as needed
5. `ChatOllama.ainvoke()` (langchain-ollama) → generate response

## Query Flow (Streaming)
1. Same pre-fetching as non-streaming
2. Emit SSE: `chunks` (retrieved SourceChunks), `graphrag` (graph context)
3. Emit `node_enter` / `node_complete` / `routing_decision` events as nodes execute
4. Stream tokens via `ChatOllama.astream()` → emit `token` events
5. Emit `sources`, then `done`; on error emit `error` event

## Indexing Pipeline
- RagIndexer delegates to:
  - DocumentLoader: parse → table classification → VLM vision → text chunking → metadata
  - graph_engine: entity extraction (spaCy/LLM/hybrid) → triple storage
  - LanceDB upsert
- Supports: PDF, DOCX, PPTX, XLSX, images, markdown, HTML, text, audio/video

## UI Form (index.html)
- **Query Tab**:
  - Collection selector
  - Query textarea
  - Method radio (hybrid|vector|graph)
  - Stream checkbox
  - Ask button
  - Results: answer box, context details, sources table, graphrag context
- **Manage Tab**: collections, documents, index controls, entity types checkboxes
- **Graph Tab**: knowledge graph visualization with filters

## Supported Document Extensions
**Pipeline** (full processing): .pdf, .docx, .xlsx, .pptx, images
**Passthrough** (markdown export): .md, .adoc, .tex, .html, .csv, .txt, .vtt, audio/video

## Response Streaming
Uses `text/event-stream` media type:
- `event: chunks\ndata: [SourceChunk JSON array]` — emitted before tokens
- `event: graphrag\ndata: [context string]` — hybrid/graph only
- `event: node_enter\ndata: {node, timestamp}` — node lifecycle
- `event: node_complete\ndata: {node, duration_ms}` — node lifecycle
- `event: routing_decision\ndata: {needs_graph: bool}` — after evaluate_context
- `event: token\ndata: [token string]` (repeats) — via `ChatOllama.astream()`
- `event: sources\ndata: [SourceChunk JSON array]`
- `event: done\ndata: ""`
- `event: error\ndata: {detail, type, location}` — on streaming failure

## Self-RAG (self_rag=true)
- `SelfReflector` delegates to `QueryGraph` (LangGraph `StateGraph`)
- Nodes: `vector_retrieve_node` → `evaluate_context_node` (LLM YES/NO) → conditional → `graph_retrieve_node` (if context insufficient) → `generate_answer_node`
- For method="graph": skips vector+evaluate, goes directly to `graph_retrieve_node`
- `node_trace` entries translated to `SelfRagStep` list (step key = node name, result = duration ms)
- Takes precedence over `stream: true`

## Additional Endpoints
- `GET /tasks` — list all pending/running index tasks
- `POST /collections/{name}/rebuild` — rebuild LanceDB from cached embedded.json (no re-embedding)
- `GET /collections/{name}/debug/table` — diagnostic: row count + doc_ids
