from time import perf_counter
from typing import List
from uuid import uuid4

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from app.config import get_settings
from app.core.llm_client import get_embedding
from app.core.opensearch_client import get_opensearch_client

router = APIRouter()


class TextDocumentIn(BaseModel):
    text: str
    title: str | None = None


class IngestionResult(BaseModel):
    doc_id: str
    num_chunks: int
    ingest_latency_ms: float


def _simple_chunk(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """
    Very simple character-based chunker to keep week-1 logic minimal.
    """
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = end - overlap

    return chunks


@router.post("/text", response_model=IngestionResult)
async def ingest_text_document(payload: TextDocumentIn) -> IngestionResult:
    """
    Text ingestion pipeline:
    - chunk text
    - embed each chunk
    - store in OpenSearch with basic metadata
    """
    start = perf_counter()

    settings = get_settings()
    client = get_opensearch_client()

    doc_id = str(uuid4())
    chunks = _simple_chunk(payload.text)

    for idx, chunk in enumerate(chunks):
        embedding = await get_embedding(chunk)
        body = {
            "doc_id": doc_id,
            "chunk_index": idx,
            "text": chunk,
            "embedding": embedding,
        }
        client.index(index=settings.opensearch_index, body=body)

    num_chunks = len(chunks)
    elapsed_ms = (perf_counter() - start) * 1000
    return IngestionResult(
        doc_id=doc_id, num_chunks=num_chunks, ingest_latency_ms=elapsed_ms
    )


@router.post("/file", response_model=IngestionResult)
async def ingest_file_document(file: UploadFile = File(...)) -> IngestionResult:
    """
    File ingestion pipeline (week-1 version):
    - read bytes
    - treat as UTF-8 text (placeholder for real PDF parsing)
    - reuse text ingestion logic
    """
    raw_bytes = await file.read()
    text = raw_bytes.decode("utf-8", errors="ignore")

    payload = TextDocumentIn(text=text, title=file.filename)
    return await ingest_text_document(payload)
