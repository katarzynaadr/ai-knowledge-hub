from time import perf_counter

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()


class QARequest(BaseModel):
    query: str
    top_k: int = 5


class RetrievedContext(BaseModel):
    doc_id: str
    chunk_index: int
    score: float
    text: str


class QAMetrics(BaseModel):
    latency_ms_total: float
    latency_ms_retrieval: float
    latency_ms_llm: float


class QAResponse(BaseModel):
    answer: str
    contexts: list[RetrievedContext]
    metrics: QAMetrics


@router.post("/query", response_model=QAResponse)
async def query_qa(payload: QARequest) -> QAResponse:
    """
    Minimal stub for QA endpoint.

    For now it returns a canned answer and empty contexts.
    """
    start_total = perf_counter()

    # TODO: implement real query embedding and OpenSearch retrieval
    start_retrieval = perf_counter()
    contexts: list[RetrievedContext] = []
    latency_retrieval_ms = (perf_counter() - start_retrieval) * 1000

    # TODO: implement real LLM call for answer synthesis
    start_llm = perf_counter()
    answer = f"(stub) No documents yet, but you asked: {payload.query!r}"
    latency_llm_ms = (perf_counter() - start_llm) * 1000

    latency_total_ms = (perf_counter() - start_total) * 1000

    metrics = QAMetrics(
        latency_ms_total=latency_total_ms,
        latency_ms_retrieval=latency_retrieval_ms,
        latency_ms_llm=latency_llm_ms,
    )

    return QAResponse(answer=answer, contexts=contexts, metrics=metrics)

