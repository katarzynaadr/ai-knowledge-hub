from time import perf_counter

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings
from app.core.llm_client import generate_answer, get_embedding
from app.core.opensearch_client import get_opensearch_client

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
    start_total = perf_counter()

    settings = get_settings()
    client = get_opensearch_client()

    # Embed query
    query_embedding = await get_embedding(payload.query)

    # Vector search in OpenSearch (using knn search API)
    start_retrieval = perf_counter()
    search_body = {
        "size": payload.top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": payload.top_k,
                }
            }
        },
    }
    search_resp = client.search(index=settings.opensearch_index, body=search_body)

    contexts: list[RetrievedContext] = []
    context_texts: list[str] = []

    hits = search_resp.get("hits", {}).get("hits", []) or []
    for hit in hits:
        source = hit.get("_source", {})
        score = float(hit.get("_score") or 0.0)
        ctx = RetrievedContext(
            doc_id=str(source.get("doc_id", "")),
            chunk_index=int(source.get("chunk_index", 0)),
            score=score,
            text=str(source.get("text", "")),
        )
        contexts.append(ctx)
        context_texts.append(ctx.text)

    latency_retrieval_ms = (perf_counter() - start_retrieval) * 1000

    # LLM answer synthesis
    start_llm = perf_counter()
    answer = await generate_answer(payload.query, context_texts)
    latency_llm_ms = (perf_counter() - start_llm) * 1000

    latency_total_ms = (perf_counter() - start_total) * 1000

    metrics = QAMetrics(
        latency_ms_total=latency_total_ms,
        latency_ms_retrieval=latency_retrieval_ms,
        latency_ms_llm=latency_llm_ms,
    )

    return QAResponse(answer=answer, contexts=contexts, metrics=metrics)
