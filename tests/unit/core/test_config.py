"""Unit tests for app.core.config."""

import pytest

from app.core.config import Settings, get_settings


@pytest.mark.unit
def test_get_settings_returns_settings_instance():
    """get_settings returns a Settings instance."""
    settings = get_settings()

    assert settings is not None
    assert isinstance(settings, Settings)


@pytest.mark.unit
def test_get_settings_is_singleton():
    """get_settings returns the same instance on multiple calls."""
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2


@pytest.mark.unit
def test_settings_default_values():
    """Settings have expected structure and default-like values."""
    settings = get_settings()

    assert isinstance(settings.app_name, str)
    assert isinstance(settings.env, str)
    assert isinstance(settings.opensearch_index, str)
    assert isinstance(settings.embedding_dimension, int)
    assert settings.embedding_dimension > 0
    assert settings.llm_provider in ("ollama", "openai", "gemini")


@pytest.mark.unit
def test_settings_can_be_constructed_with_overrides():
    """Settings can be constructed with custom values (model_construct bypasses env)."""
    # Use model_construct to bypass env loading and test field assignment
    settings = Settings.model_construct(
        app_name="Custom App",
        opensearch_index="custom_index",
        embedding_dimension=384,
    )

    assert settings.app_name == "Custom App"
    assert settings.opensearch_index == "custom_index"
    assert settings.embedding_dimension == 384
