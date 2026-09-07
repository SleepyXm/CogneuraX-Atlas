# CogneuraX Atlas contract

The user owns architecture and scope. Implement the smallest clear change requested; do not add adjacent features, files, dependencies, abstractions, ranking rules, or services without approval. Function names must describe their complete observable effect. Keep cohesive workflows together and keep shared types in `atlas/types.go`.

## Runtime boundary

- `docker compose up --build` must start the complete system.
- Go owns the public API, namespaces, collections, PostgreSQL state, River jobs, original storage, Qdrant access, and retrieval orchestration.
- Python owns only Docling HybridChunker, LlamaIndex TreeSummarize, and Qdrant FastEmbed BM25.
- Qdrant is the only vector and evidence store. PostgreSQL never stores document bodies, chunks, or vectors.
- Dense inference uses configurable TEI. Routing inference uses configurable llama.cpp. A model change requires a new index version.
- Apple Embedding Atlas has no runtime role until the user explicitly approves that integration.

## Indexing invariant

Persist Docling chunks before indexing. Every document in a frozen generation must finish its multiple typed dense routes before any member begins sparse indexing. Every member must finish sparse indexing before that generation is activated. PostgreSQL readiness remains authoritative.

## Retrieval invariant

Dense routes gate eligible documents by a calibrated threshold. FastEmbed BM25 searches complete chunks within those documents. Whole-collection sparse fallback is allowed only when scoped search has zero accepted chunks. Do not add fusion, custom ranking, a reranker, dense chunk vectors, or a query-time LLM without an explicit architecture decision.

The existing frozen evaluation exposed a real semantic handoff failure and false-positive fallback. Preserve that evidence. Do not tune thresholds or expected answers against observed holdout results; use a separate development set and then a fresh unseen holdout.

## Layout

Keep the Go package limited to `service.go`, `ingestion.go`, `retrieval.go`, `storage.go`, and `types.go`, with tests beside them. Keep the Python processor in `processor/`. Packaging files belong at the repository root. Do not restore the removed visualization, Svelte, Rust, Node, benchmark-corpus, or Apple Atlas code unless explicitly requested.
