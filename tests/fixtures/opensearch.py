"""OpenSearch mock fixtures for tests."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_opensearch_client():
    """Mock OpenSearch client with minimal index/search behavior."""
    mock_client = MagicMock()

    # Mock index.exists to return True (index already exists)
    mock_client.indices.exists.return_value = True

    # Mock search to return empty hits by default
    mock_client.search.return_value = {
        "hits": {
            "hits": [],
            "total": {"value": 0},
        }
    }

    return mock_client


@pytest.fixture
def patch_opensearch_client(mock_opensearch_client):
    """Patch get_opensearch_client in API modules (ingestion, qa)."""
    with (
        patch("app.api.ingestion.get_opensearch_client", return_value=mock_opensearch_client),
        patch("app.api.qa.get_opensearch_client", return_value=mock_opensearch_client),
    ):
        yield mock_opensearch_client