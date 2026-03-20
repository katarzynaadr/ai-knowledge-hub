from unittest.mock import patch

import pytest

from app.core.config import Settings, get_settings


@pytest.fixture
def mock_settings() -> Settings:
    """Provide test settings with safe defaults (no external services)."""
    return Settings(
        app_name="AI Knowledge Hub - Test",
        env="test",
        opensearch_host="http://localhost:9200",
        opensearch_username=None,
        opensearch_password=None,
        opensearch_index="test_rag_chunks",
        embedding_model="nomic-embed-text",
        llm_model="llama3.2:3b",
        llm_provider="ollama",
        embedding_dimension=768,
        ollama_base_url="http://localhost:11434",
        openai_api_key=None,
        gemini_api_key=None,
    )


@pytest.fixture
def override_settings(mock_settings: Settings) -> None:
    """Override get_settings to return test settings for the duration of the test."""
    with (
        patch("app.core.config.get_settings", return_value=mock_settings),
        patch("app.config.get_settings", return_value=mock_settings),
        patch("app.api.ingestion.get_settings", return_value=mock_settings),
        patch("app.api.qa.get_settings", return_value=mock_settings),
        patch("app.main.get_settings", return_value=mock_settings),
        patch("app.core.llm_client.get_settings", return_value=mock_settings),
        patch("app.core.opensearch_client.get_settings", return_value=mock_settings),
    ):
        yield