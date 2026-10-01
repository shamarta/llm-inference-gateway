"""HTTP routes of the gateway."""

from fastapi import APIRouter

from gateway import __version__
from gateway.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok", version=__version__)