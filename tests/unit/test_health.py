"""Unit tests for health endpoint."""

import pytest


@pytest.mark.unit
def test_health_returns_ok(client):
    """Health endpoint returns status ok."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.unit
def test_health_returns_json(client):
    """Health endpoint returns JSON content type."""
    response = client.get("/health")

    assert response.headers["content-type"] == "application/json"
