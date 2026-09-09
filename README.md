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

Retrieval deterministically separates the prompt into three representations without calling a model: the intact grammatical question for TEI, copied content terms for BM25, and exact anchors limited to numbers and multiword proper names. Dense Qdrant search admits every ready region at or above `0.55`, with its limit sized from PostgreSQL's complete ready-region count. Every exact anchor must occur somewhere in that dense semantic set. FastEmbed then searches only those exact region IDs and accepts evidence at or above the selected BM25 threshold of `12.5`. Evidence must pass both stages; an empty dense or scoped BM25 result returns no evidence. Sparse score determines the returned order. Atlas has no whole-collection fallback, Qwen or spaCy router, reranker, verifier, score fusion, or query-time LLM. Answer generation belongs to the calling application.

## Current accuracy boundary

The current retrieval pass was measured in Synapse on 35 SQuAD articles, 2,290 Docling regions, and 420 questions before the same implementation was reconciled into Atlas. At the selected `12.5` threshold it returned answer-bearing evidence for 124/140 answerable questions, ranked the correct source first for 127/140, and rejected 131/280 unsupported questions. This is implementation-equivalent reference evidence, not an independent Atlas full-stack reproduction. See `V0.1-Metrics.md` and `testdata/outputs.json` for the complete threshold sweep, limitations, and saved unsupported-query evidence.
