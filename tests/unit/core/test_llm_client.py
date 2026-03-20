"""Unit tests for app.core.llm_client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.llm_client import generate_answer, get_embedding


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_embedding_returns_vector(mock_settings, override_settings):
    """get_embedding returns a list of floats when Ollama responds correctly."""
    mock_response = {"embedding": [0.1] * 768}

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_response_obj = MagicMock()
        mock_response_obj.raise_for_status = MagicMock()
        mock_response_obj.json = lambda: mock_response
        mock_client.post = AsyncMock(return_value=mock_response_obj)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = await get_embedding("test text")

    assert len(result) == 768
    assert all(isinstance(x, float) for x in result)
    assert result[0] == 0.1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_embedding_raises_for_unsupported_provider(
    mock_settings, override_settings
):
    """get_embedding raises ValueError for unsupported provider."""
    mock_settings.llm_provider = "unsupported_provider"

    with pytest.raises(ValueError, match="Unsupported provider"):
        await get_embedding("test text")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_answer_returns_string(mock_settings, override_settings):
    """generate_answer returns a string when Ollama responds correctly."""
    mock_response = {"response": "The answer is 42."}

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_response_obj = MagicMock()
        mock_response_obj.raise_for_status = MagicMock()
        mock_response_obj.json = lambda: mock_response
        mock_client.post = AsyncMock(return_value=mock_response_obj)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = await generate_answer("What is the answer?", ["Context: 42"])

    assert result == "The answer is 42."


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_answer_includes_context_in_prompt(
    mock_settings, override_settings
):
    """generate_answer sends context in the prompt to Ollama."""
    mock_response = {"response": "Answer"}

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_response_obj = MagicMock()
        mock_response_obj.raise_for_status = MagicMock()
        mock_response_obj.json = lambda: mock_response
        mock_client.post = AsyncMock(return_value=mock_response_obj)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        await generate_answer("Question?", ["Context A", "Context B"])

        call_args = mock_client.post.call_args
        sent_json = call_args.kwargs["json"]
        assert "Context A" in sent_json["prompt"]
        assert "Context B" in sent_json["prompt"]
        assert "Question?" in sent_json["prompt"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_answer_raises_for_unsupported_provider(
    mock_settings, override_settings
):
    """generate_answer raises ValueError for unsupported provider."""
    mock_settings.llm_provider = "unknown"

    with pytest.raises(ValueError, match="Unsupported provider"):
        await generate_answer("query", ["context"])
