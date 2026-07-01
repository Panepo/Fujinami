# Plan: Add Image Upload to Query Page

## TL;DR
Add an image upload widget to the query page so users can upload an image and ask questions about it. The image will be encoded as base64, sent alongside the text query to the LLM, and combined with collection-retrieved context. Temporary images are stored in memory during the session and discarded after the query completes.

## Requirements Snapshot
- **Image scope**: Simple pass-through (encode + send to LLM, no VLM processing)
- **Image context**: Combined (image + collection retrieval context used together)
- **Storage**: Temporary session storage (in-memory, deleted after query)
- **UX location**: Image upload widget on query page (alongside query textarea)

## Phase 1: Frontend UI (Query Page)

### 1.1 Modify `static/index.html` — Add Image Upload Section to Query Tab

**Location**: Query tab (before or alongside the query textarea)

**Components to add**:
- File input with `accept="image/*"` (supports .png, .jpg, .jpeg, .gif, .webp, .bmp, .tiff)
- Image preview thumbnail (displays selected image or placeholder)
- Clear/remove button (removes selected image from preview)
- File info text (filename, optional file size display)
- Styling to match existing card-based design (dark header, light content)

**HTML structure**:
```html
<div class="image-upload-section">
  <label for="image-input">Upload Image (Optional)</label>
  <input type="file" id="image-input" accept="image/*" />
  <div id="image-preview-container" style="display: none;">
    <img id="image-preview" alt="Selected image preview" />
    <button id="clear-image-btn">Clear Image</button>
    <span id="image-info"></span>
  </div>
</div>
```

### 1.2 Update JavaScript Form Submission Logic in `index.html`

**FileReader integration**:
- Use `FileReader.readAsDataURL()` to convert image file → base64-encoded data URI
- Store base64 string in JavaScript variable (e.g., `currentImageBase64`)
- Handle file selection event listener to update preview
- Handle clear button to reset state

**Form state tracking**:
- Store image data in form state alongside query text
- Support three scenarios:
  - Query only (no image, backward compatible)
  - Image only (no query text, search-by-image)
  - Image + text (most common use case)

**API request modification**:
- On form submit (query button), check if image is selected
- If present, include `image_base64` field in JSON request body
- If not present, omit field (null/undefined)

---

## Phase 2: Backend API (Query Endpoint)

### 2.1 Modify `api.py` — Extend QueryRequest Model

**Current QueryRequest** (example structure):
```python
@dataclass
class QueryRequest:
    query: str
    method: str = "hybrid"  # hybrid, vector, graph
    top_k: int = 5
    stream: bool = False
    self_rag: bool = False
    rewrite: Optional[str] = None
```

**Updated QueryRequest**:
```python
@dataclass
class QueryRequest:
    query: str
    method: str = "hybrid"  # hybrid, vector, graph
    top_k: int = 5
    stream: bool = False
    self_rag: bool = False
    rewrite: Optional[str] = None
    image_base64: Optional[str] = None  # NEW: Base64-encoded image data
```

### 2.2 Extend `POST /collections/{name}/query` Endpoint Handler

**No new route needed** — extend existing handler:
- Accept the new `image_base64` field from request body
- Validate image field (optional, can be None)
- Pass `image_base64` to the retrieval/generation pipeline (see Phase 3)
- Return response with metadata indicating if image was processed

**Optional validation**:
- Check if base64 string is valid (try to decode)
- Optional: Check file size limit (e.g., reject > 10MB)
- Return 400 Bad Request if validation fails

---

## Phase 3: LLM Integration

### 3.1 Modify `retriever.py` — Update Prompt Construction

**Current behavior**: Query context passed to LLM as text chunks only

**New behavior**:
- Check if `image_base64` is present in query request
- If present:
  - Convert base64 to data URI format: `data:image/png;base64,{base64_string}`
  - Include image in system/user message to LLM
  - Optionally add instruction like: "An image has been uploaded. Please analyze it along with the context below."
- If not present:
  - Use existing prompt logic (no changes)

**LLM compatibility note**:
- ✅ **Qwen 3.6:35b confirmed to support vision natively**
  - Verified: `ollama show qwen3.6:35b` shows vision capability in Capabilities output
  - Image data URIs are natively supported (standard for multimodal LLMs)
  - No fallback needed; images are ready for production use

**Prompt structure example**:
```
System: You are a helpful assistant that answers questions about documents and images.

User: I've uploaded an image: ![image](data:image/png;base64,{base64_string})

Here is relevant context from the document collection:
[retrieved text chunks]

Question: {query_text}

Please analyze the image and context to provide an answer.
```

### 3.2 Update `graph_engine/query_graph.py` — QueryGraph State Machine

**Current query state** (example):
```python
class QueryState:
    collection_name: str
    query: str
    method: str
    top_k: int
    stream: bool
    self_rag: bool
    rewrite: Optional[str]
    # ... other state fields
```

**Updated query state**:
```python
class QueryState:
    collection_name: str
    query: str
    method: str
    top_k: int
    stream: bool
    self_rag: bool
    rewrite: Optional[str]
    image_base64: Optional[str] = None  # NEW
    # ... other state fields
```

**Pipeline integration**:
- Pass `image_base64` from initial request → state initialization
- Thread through all retrieval nodes (vector_retrieve, graph_retrieve)
- Pass to generation node (where LLM prompt is constructed)
- No changes needed to retrieval logic itself (image doesn't affect vector/graph search)

---

## Phase 4: Response & Cleanup

### 4.1 Extend QueryResponse (Optional Metadata)

**Optional addition** to [api.py](api.py):
```python
@dataclass
class QueryResponse:
    collection: str
    method: str
    answer: str
    sources: list[DocumentChunk]
    graphrag_context: Optional[dict] = None
    self_rag_meta: Optional[dict] = None
    rewrite_meta: Optional[dict] = None
    image_used: bool = False  # NEW: Indicates image was processed
```

**Frontend can use this flag**:
- Display acknowledgment like "Image processed" in response header
- Log that image was used in query (for UI feedback)

### 4.2 Session Cleanup

**No additional work required**:
- Image only lives in memory during request
- After response is sent, base64 string is garbage collected
- No temp files created
- No database updates needed
- Automatic cleanup via Python memory management

---

## Relevant Files to Modify

| File | Change | Lines |
|------|--------|-------|
| [static/index.html](static/index.html) | Add image upload UI, FileReader logic, form submission | HTML section + inline JS |
| [api.py](api.py) | Extend QueryRequest model, pass image_base64 to handler | Query endpoint, dataclass |
| [retriever.py](retriever.py) | Include image in LLM prompt construction | Prompt building logic |
| [graph_engine/query_graph.py](graph_engine/query_graph.py) | Add image_base64 to QueryState, thread through pipeline | State init, node invocations |

---

## Verification Checklist

### UI Verification
- [ ] Image file input appears on Query tab
- [ ] Image preview displays selected image correctly
- [ ] Clear button removes image from preview
- [ ] File type validation (HTML5 accept attribute restricts to images)
- [ ] Styling matches existing card design

### API Integration Test
- [ ] POST `/collections/{collection}/query` with `image_base64` field → 200 OK
- [ ] Request without image (backward compatibility) → 200 OK
- [ ] Large image handling (validation error or size limit)
- [ ] Invalid base64 → 400 Bad Request

### LLM Processing
- [ ] Query: "What is in this image?" with only image → LLM returns image-aware answer
- [ ] Query: "Describe this with context" with image + text → answer references both
- [ ] LLM logs/thinking show that image was received and processed
- [ ] Streaming still works with image (SSE events flow correctly)

### Edge Cases
- [ ] Unsupported image format (e.g., .svg) → graceful error
- [ ] Query with image but no collection selected → behavior defined (proceed or error)
- [ ] Multiple image uploads in same session → last image overwrites previous
- [ ] Network error during query with image → error handling intact
- [ ] LLM doesn't support images → fallback strategy (ignore image, query text only, or error)

---

## Architecture Decisions Rationale

### Why Simple Pass-Through?
- Avoids added latency from VLM processing (3-pass vision pipeline)
- Reduces infrastructure complexity (no separate Ollama VLM call)
- Modern LLMs can analyze images directly without preprocessing
- Simpler debugging: image → LLM directly, no intermediate transformation

### Why Base64 Encoding?
- Stateless: no temp files or file cleanup needed
- Fits naturally in JSON request body
- Compatible with data URI format (standard for multimodal LLMs)
- Easy to parse client-side (FileReader.readAsDataURL())

### Why In-Memory Only?
- Simple implementation (no database schema changes)
- Automatic cleanup (request lifetime = image lifetime)
- Prevents accumulation of orphaned temp files
- Privacy: image not persisted to disk

### Why Extend Existing Endpoint?
- Reuses query infrastructure (no new routes, no routing changes)
- Image is optional, so backward compatible
- Same auth/rate-limit logic applies
- Simpler test coverage

### Why Data URI in Prompt?
- Standard for multimodal LLM APIs (Claude, GPT-4V, Llama Vision)
- Self-contained: no need to pass file path or handle separate image upload
- Works with streaming (entire payload in request body)
- Easy to debug: can inspect base64 in request logs

---

## Further Considerations

### 1. LLM Multimodal Support
✅ **Qwen 3.6:35b is confirmed vision-capable** — no additional capability checks needed

The model includes built-in vision support (verified: `ollama show` output includes `vision` in Capabilities section). Features:
- Native image understanding (no separate VLM or preprocessing needed)
- Built-in extended thinking capability for complex image analysis
- Production-ready for image query workloads immediately

### 2. Image Size Limits
Consider adding constraints:
- Client-side: HTML5 input validation or file size check before upload
- Server-side: Reject base64 strings > 10MB (configurable)
- UI feedback: Show file size and warning if too large

### 3. CORS/Security
If frontend and API are on separate domains:
- Ensure CORS headers allow POST requests with image_base64 payload
- Consider rate limiting for image uploads (prevent abuse)
- Validate base64 format on server side (no arbitrary file uploads)

### 4. Performance & Scalability
- Base64 encoding increases payload size ~33% (not ideal for very large images)
- Consider alternative: multipart/form-data upload instead (more efficient)
- Current approach acceptable for typical images (< 5MB)

### 5. Testing Strategy
- Unit test: FileReader → base64 conversion
- Integration test: POST with image_base64 → LLM response includes image analysis
- E2E test: Upload image via UI → submit query → verify answer references image
- Test with various image types: .png, .jpg, .webp, animated .gif

---

## Rollout Plan

### Stage 1: Development & Testing (Local)
1. Implement Phase 1–4 in local branch
2. Run verification checklist
3. Test with multiple image types and scenarios

### Stage 2: Code Review
1. Review HTML/JS changes (image handling, FileReader)
2. Review API/backend changes (dataclass, prompt construction)
3. Ensure backward compatibility (existing queries without images)

### Stage 3: Staging Deployment
1. Deploy to staging environment
2. Load test with multiple concurrent image uploads
3. Verify LLM response quality with images
4. Monitor error rates and response times

### Stage 4: Production Rollout
1. Deploy to production
2. Monitor image query usage (metrics: count, avg size, success rate)
3. Gather user feedback on UX
4. Iterate based on feedback (e.g., add file size limit, improve preview)

---

## Success Criteria

✅ Users can upload an image on the query page
✅ Image is visible in a preview before submitting query
✅ Query with image + text returns an answer that addresses the image
✅ No errors when image_base64 is omitted (backward compatible)
✅ LLM response latency < 10s for typical images (< 5MB)
✅ Image data is not persisted (only in-memory)
✅ All existing query features (streaming, self-RAG, rewrite) work with images

---
