# CogneuraX Atlas contract

The user owns architecture and scope. Implement the smallest clear change requested; do not add adjacent features, files, dependencies, abstractions, ranking rules, or services without approval. Function names must describe their complete observable effect. Keep cohesive workflows together and keep shared types in `atlas/types.go`.

## Runtime boundary

- `docker compose up --build` must start the complete system.
- Go owns the public API, namespaces, collections, PostgreSQL state, River jobs, original storage, Qdrant access, and retrieval orchestration.
- Python owns only Docling HybridChunker and Qdrant FastEmbed BM25.
- Qdrant is the only vector and evidence store. PostgreSQL never stores document bodies, chunks, or vectors.
- Dense inference uses configurable TEI. A model change requires a new index version.
- Apple Embedding Atlas has no runtime role until the user explicitly approves that integration.

## Indexing invariant

Persist ordered Docling contextual evidence regions before indexing. Every document in a frozen generation must finish dense indexing of every region before any member begins sparse indexing. Every member must finish sparse indexing before that generation is activated. PostgreSQL readiness remains authoritative.

## Retrieval invariant

Dense search gates every ready evidence region by a calibrated threshold. FastEmbed BM25 searches only those exact admitted region IDs and applies its calibrated threshold. Evidence must pass both stages; an empty dense or scoped sparse result returns no evidence. Do not add a whole-collection fallback, fusion, custom ranking, a reranker, a verifier, or a query-time LLM without an explicit architecture decision.

Preserve the historical document-routing evaluation and the evidence-region evaluation separately. Do not tune thresholds or expected answers against observed holdout results; use a separate development set and then a fresh unseen holdout.

## Layout

Keep the Go package limited to `service.go`, `ingestion.go`, `retrieval.go`, `storage.go`, and `types.go`, with tests beside them. Keep the Python processor in `processor/`. Packaging files belong at the repository root. Do not restore the removed visualization, Svelte, Rust, Node, benchmark-corpus, or Apple Atlas code unless explicitly requested.
