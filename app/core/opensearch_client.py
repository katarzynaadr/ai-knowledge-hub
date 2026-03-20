from __future__ import annotations

import time
from typing import Any

from opensearchpy import AuthorizationException, OpenSearch, RequestsHttpConnection
from opensearchpy import ConnectionError as OpenSearchConnectionError

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


def _relax_disk_watermarks(client: OpenSearch) -> None:
    """Raise disk watermarks so Docker dev environments don't block index creation."""
    try:
        client.cluster.put_settings(
            body={
                "persistent": {
                    "cluster.routing.allocation.disk.watermark.low": "99%",
                    "cluster.routing.allocation.disk.watermark.high": "99%",
                    "cluster.routing.allocation.disk.watermark.flood_stage": "99%",
                }
            }
        )
    except Exception:
        pass


def ensure_index(retries: int = 10, delay_seconds: float = 2.0) -> None:
    settings = get_settings()
    index_name = settings.opensearch_index

    last_exc: Exception | None = None

    for _ in range(retries):
        client = get_opensearch_client()
        try:
            _relax_disk_watermarks(client)

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
                            "dimension": settings.embedding_dimension,
                        },
                    }
                },
            }
            client.indices.create(index=index_name, body=body)
            return
        except (OpenSearchConnectionError, AuthorizationException) as exc:
            last_exc = exc
            time.sleep(delay_seconds)

    if last_exc is not None:
        raise last_exc
