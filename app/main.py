from fastapi import FastAPI

from app.api import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Knowledge Hub - Minimal RAG",
        version="0.1.0",
    )

    app.include_router(api_router, prefix="/api")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
