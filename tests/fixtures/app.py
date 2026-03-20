from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def app():
    """Create FastAPI app with startup event mocked (no OpenSearch connection)."""
    with patch("app.main.ensure_index"):
        yield create_app()


@pytest.fixture
def client(app) -> TestClient:
    """Provide test client for HTTP requests."""
    return TestClient(app)