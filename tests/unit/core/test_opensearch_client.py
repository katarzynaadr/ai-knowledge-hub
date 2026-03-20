"""Unit tests for app.core.opensearch_client."""

from unittest.mock import MagicMock, patch

import pytest

from app.core.opensearch_client import get_opensearch_client


@pytest.mark.unit
def test_get_opensearch_client_returns_client(mock_settings, override_settings):
    """get_opensearch_client returns an OpenSearch client when patched."""
    with patch("app.core.opensearch_client.OpenSearch") as mock_opensearch_cls:
        mock_client = MagicMock()
        mock_opensearch_cls.return_value = mock_client

        # Reset global to ensure we get fresh client
        import app.core.opensearch_client as opensearch_module
        opensearch_module._client = None

        client = get_opensearch_client()

        assert client is mock_client
        mock_opensearch_cls.assert_called_once()


@pytest.mark.unit
def test_get_opensearch_client_is_singleton(mock_settings, override_settings):
    """get_opensearch_client returns same instance on multiple calls."""
    with patch("app.core.opensearch_client.OpenSearch") as mock_opensearch_cls:
        mock_client = MagicMock()
        mock_opensearch_cls.return_value = mock_client

        import app.core.opensearch_client as opensearch_module
        opensearch_module._client = None

        client1 = get_opensearch_client()
        client2 = get_opensearch_client()

        assert client1 is client2
        mock_opensearch_cls.assert_called_once()


@pytest.mark.unit
def test_get_opensearch_client_uses_auth_when_credentials_provided(mock_settings, override_settings):
    """get_opensearch_client passes auth tuple when username/password are set."""
    mock_settings.opensearch_username = "user"
    mock_settings.opensearch_password = "pass"

    import app.core.opensearch_client as opensearch_module

    opensearch_module._client = None  # Reset singleton so new client is created

    with patch("app.core.opensearch_client.OpenSearch") as mock_opensearch_cls:
        mock_client = MagicMock()
        mock_opensearch_cls.return_value = mock_client

        get_opensearch_client()

        call_kwargs = mock_opensearch_cls.call_args.kwargs
        assert call_kwargs["http_auth"] == ("user", "pass")
