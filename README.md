# CogneuraX Atlas

CogneuraX Atlas is a self-contained document-indexing and retrieval service. Applications call one Go API; the packaged runtime owns PostgreSQL metadata, Qdrant vectors, dense embeddings, document processing, and ingestion-time routing inference.

Apple Embedding Atlas is not a runtime dependency. Visual inspection may be integrated later without changing indexing or retrieval behaviour.

## Run the complete system

Docker and Docker Compose are the only host requirements.

```sh
docker compose up --build
```

The API listens on `http://localhost:8090`. The first start downloads the configured models and is substantially slower than later starts. On Apple Silicon, copy `.env.example` to `.env` and enable its ARM64 TEI image setting.

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

Each uploaded document joins a collection generation. Docling produces ordered contextual chunks. LlamaIndex and Qwen reduce all of those chunks into separate typed routing facets. TEI embeds the facets. Once every document in the generation has dense routes, FastEmbed creates one Qdrant BM25 sparse vector per complete chunk. A generation becomes searchable only after every member completes both stages.

Retrieval embeds a question once for dense document routing and once for sparse chunk search. It searches sparse evidence inside the eligible documents, then permits a whole-collection sparse fallback only when scoped search returns no accepted chunks. Atlas returns evidence and citations; answer generation belongs to the calling application.

## Current accuracy boundary

The first mixed-document evaluation found 75% answer-bearing retrieval and 50% rejection of unanswerable questions. Hardware affects ingestion latency, not this result. The remaining failure is a measured semantic gap between dense document routing and thresholded lexical chunk search. Do not describe the current retriever as production-accurate or tune against the frozen test prompts. Any reranker, dense chunk vectors, score fusion, or query-time model pass is an explicit architecture experiment, not an invisible patch.
