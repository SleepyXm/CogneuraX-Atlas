import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.document_converter import DocumentConverter
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastembed import SparseTextEmbedding
from pydantic import BaseModel
from transformers import AutoTokenizer


EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
REGION_MAX_TOKENS = int(os.getenv("ATLAS_REGION_MAX_TOKENS", "128"))

app = FastAPI(title="CogneuraX Atlas processor")


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]


class SparseQuery(BaseModel):
    query: str


class SparseDocumentRequest(BaseModel):
    regions: list[dict]


@lru_cache
def load_processing_components():
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model_limit = min(tokenizer.model_max_length, REGION_MAX_TOKENS)
    chunk_tokenizer = HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=model_limit)
    # Each returned region remains one coherent Docling unit. Disabling peer
    # merging prevents unrelated statements under one heading accumulating a
    # misleading lexical score, while Docling still preserves source context.
    chunker = HybridChunker(
        tokenizer=chunk_tokenizer,
        merge_peers=False,
        repeat_table_header=True,
    )
    return DocumentConverter(), chunker


@lru_cache
def load_sparse_model():
    return SparseTextEmbedding(model_name="Qdrant/bm25", language="english")


def page_number(chunk):
    for item in getattr(chunk.meta, "doc_items", []):
        provenance = getattr(item, "prov", [])
        if provenance:
            return getattr(provenance[0], "page_no", None)
    return None


def process_document(file: UploadFile):
    converter, chunker = load_processing_components()
    file.file.seek(0)
    suffix = Path(file.filename or "document.txt").suffix.lower()
    # Docling's plain-text converter treats comparison operators as list syntax;
    # its Markdown converter preserves the same unformatted text verbatim.
    if suffix == ".txt":
        suffix = ".md"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary_path = temporary.name
            shutil.copyfileobj(file.file, temporary)
        converted = converter.convert(temporary_path)
    finally:
        if temporary_path is not None:
            os.remove(temporary_path)
    docling_chunks = list(chunker.chunk(converted.document))
    # Docling contextualization folds headings, captions, and table structure
    # into the evidence text; page and order remain explicit citation fields.
    texts = [chunker.contextualize(chunk=chunk) for chunk in docling_chunks]
    if not texts:
        raise ValueError("Docling produced no chunks")

    oversized = [
        index
        for index, text in enumerate(texts)
        if chunker.tokenizer.count_tokens(text) > chunker.tokenizer.get_max_tokens()
    ]
    if oversized:
        raise ValueError(f"Docling regions exceed the embedding token limit: {oversized}")

    regions = [
        {"text": text, "page": page_number(chunk), "region_index": index}
        for index, (chunk, text) in enumerate(zip(docling_chunks, texts, strict=True))
    ]
    return {"regions": regions}


@app.post("/process")
async def process(file: UploadFile):
    try:
        return await run_in_threadpool(process_document, file)
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/sparse-document")
async def sparse_document(request: SparseDocumentRequest):
    texts = [region.get("text", "") for region in request.regions]
    if not texts or any(not text.strip() for text in texts):
        raise HTTPException(status_code=400, detail="evidence regions are required")
    vectors = await run_in_threadpool(lambda: list(load_sparse_model().embed(texts)))
    regions = [dict(region) for region in request.regions]
    for region, vector in zip(regions, vectors, strict=True):
        region["sparse"] = {"indices": vector.indices.tolist(), "values": vector.values.tolist()}
    return {"regions": regions}


@app.post("/sparse-query", response_model=SparseVector)
async def sparse_query(request: SparseQuery):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    def encode():
        vector = next(load_sparse_model().query_embed(request.query))
        return SparseVector(indices=vector.indices.tolist(), values=vector.values.tolist())

    return await run_in_threadpool(encode)
