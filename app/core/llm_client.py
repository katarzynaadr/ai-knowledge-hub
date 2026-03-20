from __future__ import annotations

from typing import Sequence

import httpx

from app.core.config import get_settings


async def get_embedding(text: str) -> list[float]:
    settings = get_settings()

    if settings.llm_provider == "ollama":
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/embeddings",
                json={"model": settings.embedding_model, "prompt": text},
            )
            r.raise_for_status()
            data = r.json()
            return data["embedding"]

    if settings.llm_provider in {"openai", "gemini"}:
        raise NotImplementedError(
            f"Provider {settings.llm_provider} not implemented; use ollama for local."
        )

    raise ValueError(f"Unsupported provider: {settings.llm_provider}")


async def generate_answer(query: str, contexts: Sequence[str]) -> str:
    settings = get_settings()

    prompt = (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say so.\n\n"
        "Context:\n"
        + "\n\n".join(contexts)
        + "\n\nQuestion: "
        + query
        + "\n\nAnswer:"
    )

    if settings.llm_provider == "ollama":
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/generate",
                json={
                    "model": settings.llm_model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()

    if settings.llm_provider in {"openai", "gemini"}:
        raise NotImplementedError(
            f"Provider {settings.llm_provider} not implemented; use ollama for local."
        )

    raise ValueError(f"Unsupported provider: {settings.llm_provider}")
