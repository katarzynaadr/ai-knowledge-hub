from __future__ import annotations

from typing import Sequence

import httpx

from app.core.config import get_settings


async def get_embedding(text: str) -> list[float]:
    """
    Return an embedding vector for the given text.

    For now this is a stub you can implement against OpenAI or Gemini later.
    """
    settings = get_settings()

    # Example placeholder: raise if not configured
    if settings.llm_provider not in {"openai", "gemini"}:
        raise ValueError(f"Unsupported provider: {settings.llm_provider}")

    # TODO: implement real HTTP call to provider's embedding endpoint
    # For now, return a fake fixed-dimension vector for wiring.
    dim = 1536
    return [0.0] * dim


async def generate_answer(query: str, contexts: Sequence[str]) -> str:
    """
    Call the LLM to generate an answer based on the query and retrieved contexts.
    """
    settings = get_settings()

    # Simple prompt template
    prompt = (
        "You are a helpful assistant answering questions based on the provided context.\n\n"
        "Context:\n" + "\n\n".join(contexts) + "\n\nQuestion:\n" + query + "\n\nAnswer:"
    )

    # TODO: implement real call to OpenAI or Gemini; example shape:
    async with httpx.AsyncClient(timeout=30.0) as client:
        _ = client  # placeholder to avoid unused variable

    # Stubbed answer for now
    return f"(stubbed answer) You asked: {query!r} with {len(contexts)} context chunks."
