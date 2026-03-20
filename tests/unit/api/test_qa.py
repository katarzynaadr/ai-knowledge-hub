import pytest


@pytest.mark.unit
def test_query_qa_returns_response(
    client,
    override_settings,
    mock_get_embedding,
    mock_generate_answer,
    patch_opensearch_client,
    mock_settings,
):
    """POST /api/qa/query returns QAResponse with answer and metrics."""
    # Configure mock to return some hits
    patch_opensearch_client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_score": 0.95,
                    "_source": {
                        "doc_id": "doc-1",
                        "chunk_index": 0,
                        "text": "Relevant context about the topic.",
                    },
                },
            ],
            "total": {"value": 1},
        }
    }

    response = client.post(
        "/api/qa/query",
        json={"query": "What is the topic?", "top_k": 5},
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "contexts" in data
    assert "metrics" in data
    assert "latency_ms_total" in data["metrics"]
    assert "latency_ms_retrieval" in data["metrics"]
    assert "latency_ms_llm" in data["metrics"]


@pytest.mark.unit
def test_query_qa_with_empty_results(
    client,
    override_settings,
    mock_get_embedding,
    mock_generate_answer,
    patch_opensearch_client,
    mock_settings,
):
    """POST /api/qa/query handles empty search results."""
    patch_opensearch_client.search.return_value = {
        "hits": {"hits": [], "total": {"value": 0}},
    }

    response = client.post(
        "/api/qa/query",
        json={"query": "Unknown topic?", "top_k": 5},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"]  # LLM still generates answer (e.g. "context does not contain")
    assert data["contexts"] == []


@pytest.mark.unit
def test_query_qa_passes_top_k_to_search(
    client,
    override_settings,
    mock_get_embedding,
    mock_generate_answer,
    patch_opensearch_client,
    mock_settings,
):
    """POST /api/qa/query passes top_k to OpenSearch search."""
    patch_opensearch_client.search.return_value = {
        "hits": {"hits": [], "total": {"value": 0}},
    }

    client.post(
        "/api/qa/query",
        json={"query": "test", "top_k": 10},
    )

    call_args = patch_opensearch_client.search.call_args
    search_body = call_args.kwargs.get("body", call_args.args[1] if len(call_args.args) > 1 else {})
    assert search_body.get("size") == 10
    assert search_body["query"]["knn"]["embedding"]["k"] == 10


@pytest.mark.unit
def test_query_qa_calls_embedding_for_query(
    client,
    override_settings,
    mock_get_embedding,
    mock_generate_answer,
    patch_opensearch_client,
    mock_settings,
):
    """POST /api/qa/query calls get_embedding with the query."""
    patch_opensearch_client.search.return_value = {
        "hits": {"hits": [], "total": {"value": 0}},
    }

    client.post(
        "/api/qa/query",
        json={"query": "my search query", "top_k": 5},
    )

    mock_get_embedding.assert_called_once_with("my search query")


@pytest.mark.unit
def test_query_qa_calls_generate_answer_with_contexts(
    client,
    override_settings,
    mock_get_embedding,
    mock_generate_answer,
    patch_opensearch_client,
    mock_settings,
):
    """POST /api/qa/query passes retrieved contexts to generate_answer."""
    patch_opensearch_client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_score": 0.9,
                    "_source": {
                        "doc_id": "d1",
                        "chunk_index": 0,
                        "text": "Context text A",
                    },
                },
            ],
            "total": {"value": 1},
        }
    }

    client.post(
        "/api/qa/query",
        json={"query": "Question?", "top_k": 5},
    )

    mock_generate_answer.assert_called_once()
    call_args = mock_generate_answer.call_args
    assert call_args.args[0] == "Question?"
    assert "Context text A" in call_args.args[1]
