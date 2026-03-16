from __future__ import annotations

import time
from typing import Any

from opensearchpy import ConnectionError as OpenSearchConnectionError
from opensearchpy import OpenSearch, RequestsHttpConnection

from app.core.config import get_settings

_client: OpenSearch | None = None


def get_opensearch_client() -> OpenSearch:
    global _client
    if _client is not None:
        return _client

    settings = get_settings()

    auth = None
    if settings.opensearch_username and settings.opensearch_password:
        auth = (settings.opensearch_username, settings.opensearch_password)

    _client = OpenSearch(
        hosts=[settings.opensearch_host],
        http_auth=auth,
        use_ssl=settings.opensearch_host.startswith("https"),
        verify_certs=False,
        connection_class=RequestsHttpConnection,
    )
    return _client


def ensure_index(retries: int = 10, delay_seconds: float = 2.0) -> None:
    """
    Ensure the RAG index exists with a dense_vector field.

    Retries a few times to allow OpenSearch to start up,
    which is important in Docker Compose where the API
    container may come up before OpenSearch is ready.
    """
    settings = get_settings()
    index_name = settings.opensearch_index

    last_exc: Exception | None = None

    for _ in range(retries):
        client = get_opensearch_client()
        try:
            if client.indices.exists(index=index_name):
                return

            body: dict[str, Any] = {
                "settings": {
                    "index": {
                        "knn": True,
                    }
                },
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "text": {"type": "text"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": 1536,  # match your embedding model
                        },
                    }
                },
            }
            client.indices.create(index=index_name, body=body)
            return
        except OpenSearchConnectionError as exc:
            last_exc = exc
            time.sleep(delay_seconds)

    if last_exc is not None:
        raise last_exc
