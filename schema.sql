CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE atlas_collections (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace_id uuid NOT NULL,
    name text NOT NULL,
    description text NOT NULL DEFAULT '',
    index_version text NOT NULL DEFAULT 'v2',
    index_generation bigint NOT NULL DEFAULT 1,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT atlas_collections_name_length CHECK (char_length(name) BETWEEN 1 AND 100),
    CONSTRAINT atlas_collections_description_length CHECK (char_length(description) <= 1000),
    CONSTRAINT atlas_collections_namespace_name_key UNIQUE (namespace_id, name)
);

CREATE TABLE atlas_documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id uuid NOT NULL REFERENCES atlas_collections(id) ON DELETE CASCADE,
    filename text NOT NULL,
    mime_type text NOT NULL,
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    sha256 char(64) NOT NULL,
    storage_key text NOT NULL,
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
    index_generation bigint NOT NULL,
    index_stage text NOT NULL DEFAULT 'pending' CHECK (index_stage IN ('pending', 'dense', 'sparse', 'ready')),
    failure_reason text,
    chunk_count integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT atlas_documents_collection_sha256_key UNIQUE (collection_id, sha256)
);

CREATE INDEX atlas_documents_collection_created_idx ON atlas_documents(collection_id, created_at);
CREATE INDEX atlas_documents_generation_stage_idx ON atlas_documents(collection_id, index_generation, index_stage);
