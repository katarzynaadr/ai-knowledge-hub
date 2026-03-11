from uuid import uuid4
from time import perf_counter

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel


router = APIRouter()


class TextDocumentIn(BaseModel):
    text: str
    title: str | None = None


class IngestionResult(BaseModel):
    doc_id: str
    num_chunks: int
    ingest_latency_ms: float


@router.post("/text", response_model=IngestionResult)
async def ingest_text_document(payload: TextDocumentIn) -> IngestionResult:
    """
    Minimal stub for text document ingestion.

    For now it just simulates ingestion and returns a fake result.
    """
    start = perf_counter()

    # TODO: implement real text extraction, chunking, embeddings, and OpenSearch indexing
    doc_id = str(uuid4())
    num_chunks = 0

    elapsed_ms = (perf_counter() - start) * 1000
    return IngestionResult(doc_id=doc_id, num_chunks=num_chunks, ingest_latency_ms=elapsed_ms)


@router.post("/file", response_model=IngestionResult)
async def ingest_file_document(file: UploadFile = File(...)) -> IngestionResult:
    """
    Minimal stub for file (e.g. PDF) ingestion.

    For now it reads the content but does not process it.
    """
    start = perf_counter()

    _ = await file.read()

    # TODO: implement real PDF text extraction, chunking, embeddings, and OpenSearch indexing
    doc_id = str(uuid4())
    num_chunks = 0

    elapsed_ms = (perf_counter() - start) * 1000
    return IngestionResult(doc_id=doc_id, num_chunks=num_chunks, ingest_latency_ms=elapsed_ms)

