"""LLM client mock fixtures for tests."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def mock_get_embedding():
    """Mock get_embedding for API layer (ingestion, qa)."""
    mock = AsyncMock(return_value=[0.1] * 768)
    with (
        patch("app.api.ingestion.get_embedding", mock),
        patch("app.api.qa.get_embedding", mock),
    ):
        yield mock


@pytest.fixture
def mock_generate_answer():
    """Mock generate_answer for QA API."""
    mock = AsyncMock(return_value="This is a test answer based on the context.")
    with patch("app.api.qa.generate_answer", mock):
        yield mock


@pytest.fixture
def mock_llm_client(mock_get_embedding, mock_generate_answer):
    """Mock both get_embedding and generate_answer for API tests."""
    yield