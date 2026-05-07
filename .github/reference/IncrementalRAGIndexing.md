# Plan: Incremental RAG Indexing (Per-File Delta)

## TL;DR
Replace the single-fingerprint full-rebuild strategy with a per-file manifest that tracks each file independently. On re-index, only new/modified files are chunked, embedded, and entity-extracted. Deleted files have their contributions surgically removed from all stores. Community detection (Louvain) always runs on the full merged graph (fast, no LLM), but community *summaries* are only re-generated for communities whose membership changed.

---

## Phase 1 — Per-File Manifest & Delta Detection

**Step 1.1 — Replace `.fingerprint` with `file_manifest.json`**
- Old: single JSON string of all files' stats → compare whole blob
- New: `Record<"folder/filename", { mtime: number; size: number }>` stored at `data/lancedb/file_manifest.json`
- New constant: `manifestPath = path.resolve(process.cwd(), 'data/lancedb/file_manifest.json')`

**Step 1.2 — Add `loadManifest()` private method**
- Read `file_manifest.json` → parse and return `Record<string, { mtime: number; size: number }>`
- Return `{}` if file doesn't exist

**Step 1.3 — Add `computeFileDelta()` private method**
- Input: `storedManifest`
- Walk each `folder/file` on disk (same logic as `computeFingerprint()`)
- Return: `{ newFiles: string[], modifiedFiles: string[], deletedFiles: string[], unchanged: string[] }`
  - `newFiles`: on disk, not in manifest
  - `modifiedFiles`: in both, but mtime or size differs
  - `deletedFiles`: in manifest, not on disk
  - `unchanged`: in both, stats match
- If `newFiles + modifiedFiles + deletedFiles = 0` → no-op (skip reindex)

---

## Phase 2 — Extend Sidecar Formats to Track Source File

**Step 2.1 — Add `source` field to `entity_chunks.json` entries**
- Old: `Record<entityId, { text: string; accessLevel: string }[]>`
- New: `Record<entityId, { text: string; accessLevel: string; source: string }[]>`
- `source` = `"folder/filename"` (e.g., `"common/guide.md"`)

**Step 2.2 — Add `source` field to `relationships.json` entries**
- Old: `{ sourceId: string; targetId: string; type: string }[]`
- New: `{ sourceId: string; targetId: string; type: string; source: string }[]`
- `source` = same `"folder/filename"` key

**Step 2.3 — Update `extractEntitiesAndRels()` signature**
- Each chunk has `metadata.source` (filename) and `metadata.accessLevel` (folder)
- Construct `source` key as `"${chunk.metadata.accessLevel}/${chunk.metadata.source}"`
- Pass `source` key through to `EntityRecord` (add `source: string` field to type)
- Pass `source` into each `RelRecord`

---

## Phase 3 — Incremental Vector Store Updates

**Step 3.1 — Add `removeFromVectorStore(sources: string[])` private method**
- Uses LanceDB `Table.delete(filter)` API: `await this.rawTable.delete(\`source IN (...)\`)`
- Handles both deleted and modified files (modified = delete old + add new)

**Step 3.2 — Add `addToVectorStore(docs: Document[])` private method**
- Uses `await this.vectorStore.addDocuments(docs)` (LangChain LanceDB wrapper method)
- For first-run (table doesn't exist): fall through to existing `LanceDB.fromDocuments()`

**Step 3.3 — FTS index handling**
- After adding new docs, attempt `rawTable.createIndex('text', ...)` — gracefully ignore "already exists" error

---

## Phase 4 — Incremental Entity & Relationship Store

**Step 4.1 — Add `loadEntitySidecars()` private method**
- Load current `entity_chunks.json` → `Map<entityId, ChunkEntry[]>`
- Load current `relationships.json` → `RelRecord[]`
- Return both (or empty structures if files don't exist)

**Step 4.2 — Add `pruneEntitiesForSources(sources: string[])` private method**
- Remove all chunk entries where `entry.source` is in `sources` from `entityChunkMap`
- Remove entities with no remaining chunks
- Remove all relationships where `rel.source` is in `sources`
- Return pruned `{ entityChunkMap, relationships }`

**Step 4.3 — Modify `buildEntityStore()` to accept and merge existing data**
- New signature: `buildEntityStore(newEntities, newRelationships, existingChunkMap, existingRelationships)`
- Merge new entity chunks into `existingChunkMap` (deduplicate by text+source)
- Only embed entities that are **new or updated** (not already in the entity table with same description)
- Merge relationships arrays (deduplicate by sourceId+targetId+source)
- Full rebuild of `rag_entities` table from merged data (drop + recreate — simpler than incremental LanceDB entity table update)

---

## Phase 5 — Incremental Community Summaries

**Step 5.1 — Add `community_members.json` sidecar**
- Format: `Record<communityId, { memberIds: string[]; summaryHash: string }>`
- Stores which entity IDs belong to each community + a hash of the summary
- Stored at `data/lancedb/community_members.json`

**Step 5.2 — Modify `buildCommunitySummaries()` to skip unchanged communities**
- After Louvain, for each community: compute set of member entity IDs
- Compare against stored `community_members.json`
- If member set is identical (same IDs) → reuse existing summary (no LLM call)
- If changed → run LLM summary prompt, update entry
- For removed communities (in stored but not in new graph) → skip
- Rebuild `rag_communities` table from merged (reused + new) summaries

---

## Phase 6 — Refactor `initVectorStore()` Orchestration

**Step 6.1 — New flow in `initVectorStore()`**
1. Load manifest → `storedManifest`
2. `computeFileDelta(storedManifest)` → `{ newFiles, modifiedFiles, deletedFiles, unchanged }`
3. If all arrays empty → load existing LanceDB + graph state, return (same as current fingerprint match)
4. Log which files changed
5. Determine `changedSources = [...newFiles, ...modifiedFiles]` and `removedSources = [...deletedFiles, ...modifiedFiles]`
6. If `removedSources` not empty → `removeFromVectorStore(removedSources)`, `pruneEntitiesForSources(removedSources)`
7. Load and chunk only `changedSources` files → `newChunks`
8. If `newChunks` not empty:
   - If table exists → `addToVectorStore(newChunks)`
   - If table doesn't exist → `LanceDB.fromDocuments(newChunks, ...)` (first run)
9. Extract entities from `newChunks` only → `{ newEntities, newRelationships }`
10. Load existing sidecars → `{ existingChunkMap, existingRelationships }`
11. `buildEntityStore(newEntities, newRelationships, existingChunkMap, existingRelationships)`
12. Run Louvain on full merged graph → `communityMap`
13. `buildCommunitySummaries(entities, communityMap)` with membership cache
14. `loadGraphState()`
15. Update manifest: merge `storedManifest` + new stats for `changedSources`, remove `removedSources`
16. Save manifest

**Step 6.2 — Update `loadAndChunkDocs()` to accept a file filter**
- New parameter: `filesToProcess?: Set<string>` (set of `"folder/filename"` keys)
- If provided, only process files in that set; otherwise process all (first run fallback)

---

## Relevant Files
- `src/common/services/rag.service.ts` — only file modified
- `data/lancedb/file_manifest.json` — new sidecar (replaces `.fingerprint`)
- `data/lancedb/entity_chunks.json` — extended with `source` field
- `data/lancedb/relationships.json` — extended with `source` field
- `data/lancedb/community_members.json` — new sidecar for community membership cache

---

## Verification
1. Start with 0 files → confirm `file_manifest.json` is created empty/nonexistent
2. Add 5 files → confirm all 5 indexed, manifest written with 5 entries
3. Add 2 more files → confirm only 2 new files are chunked/embedded/entity-extracted (check logs: entity extraction chunk counter should be `1/N` where N = new file chunk count only)
4. Delete 1 file → confirm its chunks removed from `rag_docs` (query by source), its entities pruned from `entity_chunks.json`
5. Modify 1 file → confirm old chunks removed and new chunks added
6. Confirm community summaries are **not** re-generated for communities with unchanged membership (check logs for LLM call count)

---

## Decisions
- **Scope**: Only `rag.service.ts` is modified — no new modules, controllers, or DTOs
- **Entity table rebuild**: Full drop+recreate for `rag_entities` (simpler than incremental LanceDB row-level ops); only re-embedding delta entities
- **Community rebuild**: Louvain always runs (fast, pure graph math); LLM summaries only for changed communities
- **Breaking change**: `entity_chunks.json` and `relationships.json` format changes — first run after upgrade will detect manifest mismatch (`.fingerprint` vs `file_manifest.json`) and do a full reindex automatically
- **LanceDB delete API**: Relies on `Table.delete(filter: string)` from `@lancedb/lancedb` — verify this API exists before implementation
