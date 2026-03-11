from fastapi import APIRouter

from . import ingestion, qa

router = APIRouter()

router.include_router(ingestion.router, prefix="/documents", tags=["documents"])
router.include_router(qa.router, prefix="/qa", tags=["qa"])
