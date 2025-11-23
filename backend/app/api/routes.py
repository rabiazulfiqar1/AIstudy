from fastapi import APIRouter
from app.api.notes import router as notes
from app.api.rs import router as recommendations
from app.api.profile import router as profile

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}

router.include_router(notes, prefix="/api")
router.include_router(recommendations, prefix="/api") 
router.include_router(profile, prefix="/api")