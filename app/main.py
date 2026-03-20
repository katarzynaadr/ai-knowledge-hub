from fastapi import FastAPI

from app.api import router as api_router
from app.core.config import get_settings
from app.core.opensearch_client import ensure_index


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
    )

    @app.on_event("startup")
    async def on_startup() -> None:
        ensure_index()

    app.include_router(api_router, prefix="/api")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
