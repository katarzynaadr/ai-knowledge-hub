import pytest

from app.api.ingestion import (
    IngestionResult,
    TextDocumentIn,
    _simple_chunk,
    ingest_text_document,
)


@pytest.mark.unit
def test_simple_chunk_splits_long_text():
    """_simple_chunk splits text into chunks of max_chars with overlap."""
    text = "a" * 2500
    chunks = _simple_chunk(text, max_chars=1000, overlap=200)

    assert len(chunks) >= 2
    assert all(len(c) <= 1000 for c in chunks)
    # Overlap: second chunk should start 200 chars before end of first
    assert chunks[0][-200:] == chunks[1][:200]


@pytest.mark.unit
def test_simple_chunk_returns_single_chunk_for_short_text():
    """_simple_chunk returns single chunk when text is shorter than max_chars."""
    text = "short"
    chunks = _simple_chunk(text, max_chars=1000)

    assert len(chunks) == 1
    assert chunks[0] == "short"


@pytest.mark.unit
def test_simple_chunk_empty_text_returns_empty_list():
    """_simple_chunk returns empty list for empty text."""
    chunks = _simple_chunk("", max_chars=1000)

    assert chunks == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_text_document_success(
    mock_settings,
    override_settings,
    mock_get_embedding,
    patch_opensearch_client,
):
    """POST /documents/text returns IngestionResult with doc_id and num_chunks."""
    payload = TextDocumentIn(text="This is a test document for ingestion.")

    result = await ingest_text_document(payload)

    assert isinstance(result, IngestionResult)
    assert result.doc_id
    assert result.num_chunks >= 1
    assert result.ingest_latency_ms >= 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_text_document_calls_embedding_per_chunk(
    mock_settings,
    override_settings,
    mock_get_embedding,
    patch_opensearch_client,
):
    """Ingestion calls get_embedding once per chunk."""
    # Text long enough to create multiple chunks
    payload = TextDocumentIn(text="x" * 1500)

    await ingest_text_document(payload)

    assert mock_get_embedding.call_count >= 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_text_document_indexes_to_opensearch(
    mock_settings,
    override_settings,
    mock_get_embedding,
    patch_opensearch_client,
):
    """Ingestion indexes documents to OpenSearch."""
    payload = TextDocumentIn(text="Index me.")

    await ingest_text_document(payload)

    assert patch_opensearch_client.index.called


@pytest.mark.unit
def test_ingest_text_via_api(
    client,
    override_settings,
    mock_get_embedding,
    patch_opensearch_client,
    mock_settings,
):
    """POST /api/documents/text returns 200 and IngestionResult."""
    response = client.post(
        "/api/documents/text",
        json={"text": "API test document", "title": "Test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert "num_chunks" in data
    assert "ingest_latency_ms" in data


@pytest.mark.unit
def test_ingest_file_via_api(
    client,
    override_settings,
    mock_get_embedding,
    patch_opensearch_client,
    mock_settings,
):
    """POST /api/documents/file accepts file upload and returns IngestionResult."""
    response = client.post(
        "/api/documents/file",
        files={"file": ("test.txt", b"File content for ingestion", "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert "num_chunks" in data
