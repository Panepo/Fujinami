# Fujinami RAG Service Architecture

## Core Stack
- **FastAPI Server**: api.py (entry point)
- **LLM Client**: Semantic Kernel with Ollama plugin
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
- `IndexRequest`: entity_types[], mode ("vector"|"graph"|"all"), force
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
2. For vector/hybrid: fetch raw LanceDB results → create SourceChunk list
3. For graph/hybrid: spaCy NER on query → LanceDBGraphStore lookup
4. Merge contexts as needed
5. Call Semantic Kernel chat service → generate response

## Query Flow (Streaming)
1. Same pre-fetching as non-streaming
2. Emit SSE events: chunks, graphrag (context)
3. Get streaming tokens from OllamaChatPromptExecutionSettings
4. Emit "token" events, then "sources" and "done"

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
- `event: chunks\ndata: [SourceChunk JSON array]`
- `event: graphrag\ndata: [context string]`
- `event: token\ndata: [token string]` (repeats)
- `event: sources\ndata: [SourceChunk JSON array]`
- `event: done\ndata: ""`
