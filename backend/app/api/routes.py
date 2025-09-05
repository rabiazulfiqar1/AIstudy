from fastapi import APIRouter
from app.api.notes import router as notes

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}

router.include_router(notes, prefix="/api")