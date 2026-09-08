# CogneuraX Atlas

CogneuraX Atlas is a self-contained document-indexing and retrieval service. Applications call one Go API; the packaged runtime owns PostgreSQL metadata, Qdrant vectors, dense embeddings, document processing, and retrieval orchestration.

Apple Embedding Atlas is not a runtime dependency. Visual inspection may be integrated later without changing indexing or retrieval behaviour.

## Run the complete system

Docker and Docker Compose are the only host requirements.

```sh
docker compose up --build
```

The API listens on `http://localhost:8090`. The first start downloads the configured models and is substantially slower than later starts. The Compose default targets Apple Silicon; on x86_64, copy `.env.example` to `.env` and enable its x86 TEI image setting.

Every API request uses a service token and a caller-owned UUID namespace:

```sh
curl -X POST http://localhost:8090/v1/collections \
  -H 'Authorization: Bearer local-development-token' \
  -H 'X-Atlas-Namespace-ID: 00000000-0000-0000-0000-000000000001' \
  -H 'Content-Type: application/json' \
  -d '{"name":"manuals","description":"Product manuals"}'
```

The returned collection ID is used to upload and inspect documents:

```sh
curl -X POST http://localhost:8090/v1/collections/COLLECTION_ID/documents \
  -H 'Authorization: Bearer local-development-token' \
  -H 'X-Atlas-Namespace-ID: 00000000-0000-0000-0000-000000000001' \
  -F 'file=@manual.pdf'

curl 'http://localhost:8090/v1/collections/COLLECTION_ID/documents?limit=20&offset=0' \
  -H 'Authorization: Bearer local-development-token' \
  -H 'X-Atlas-Namespace-ID: 00000000-0000-0000-0000-000000000001'
```

Retrieve evidence without invoking a user-owned generation model:

```sh
curl -X POST http://localhost:8090/v1/retrieve \
  -H 'Authorization: Bearer local-development-token' \
  -H 'X-Atlas-Namespace-ID: 00000000-0000-0000-0000-000000000001' \
  -H 'Content-Type: application/json' \
  -d '{"collection_id":"COLLECTION_ID","query":"What does the manual say about refunds?"}'
```

## Pipeline

Each uploaded document joins a collection generation. Docling produces ordered contextual evidence regions with peer merging disabled and a 128-token default cap. Atlas persists those regions, and TEI embeds every region's contextual source text in batches. Once every document in the frozen generation has its inactive dense regions, FastEmbed creates one paired Qdrant BM25 sparse vector per complete region. A generation becomes searchable only after every member completes both stages.

Retrieval embeds a question once with TEI. Dense Qdrant search admits every ready region at or above the configured threshold, with its limit sized from PostgreSQL's complete ready-region count. FastEmbed then encodes the question for BM25, which searches only those exact region IDs and applies its own threshold. Evidence must pass both stages; an empty dense or scoped BM25 result returns no evidence. Sparse score determines the returned order. Atlas has no whole-collection fallback, Qwen or spaCy router, reranker, verifier, score fusion, or query-time LLM. Answer generation belongs to the calling application.

## Current accuracy boundary

The evidence-region architecture was tested in Synapse on 35 SQuAD articles and 2,290 Docling regions, but that result is not yet a standalone Atlas reproduction. See `V0.1-Metrics.md` for the exact historical and experimental boundaries. Do not describe Atlas as successfully reproduced until its own complete Compose path and frozen evaluation have run, and do not tune against the observed holdout. Any reranker, verifier, fusion, body-only sparse representation, or query-time model pass remains a separate architecture experiment.
