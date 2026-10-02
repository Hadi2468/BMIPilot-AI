"""FastAPI service exposing the deterministic assessment and the full agent.

Run locally:  uvicorn bmipilot.api:app --reload
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from bmipilot import __version__
from bmipilot.config import Settings, get_settings
from bmipilot.growth import Sex, Source, reference_curves
from bmipilot.observability import log_tracing_status
from bmipilot.schemas import Assessment, CoachResult, Profile
from bmipilot.service import BMIPilotService

logger = logging.getLogger(__name__)


# ============================================================
# Dependencies
# ============================================================


def get_service(request: Request) -> BMIPilotService:
    return request.app.state.service


def require_token(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(HTTPBearer(auto_error=False))],
) -> None:
    expected = request.app.state.settings.api_token
    if not expected:
        return
    if creds is None or not secrets.compare_digest(creds.credentials, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing bearer token")


ServiceDep = Annotated[BMIPilotService, Depends(get_service)]


# ============================================================
# Routes
# ============================================================

# Sync handlers: FastAPI runs them in a threadpool, so a long LLM run
# doesn't block the event loop.
router = APIRouter(dependencies=[Depends(require_token)])


@router.post("/assess", response_model=Assessment, tags=["bmi"])
def assess(profile: Profile, svc: ServiceDep) -> Assessment:
    """Deterministic BMI assessment for any age (no LLM, instant)."""
    return svc.assess(profile)


@router.post("/coach", response_model=CoachResult, tags=["bmi"])
def coach(profile: Profile, svc: ServiceDep) -> CoachResult:
    """Full agent run: assessment plus exercise, diet and lifestyle plan."""
    return svc.coach(profile)


@router.get("/reference/{source}/{sex}", tags=["reference"])
def reference(source: Source, sex: Sex) -> list[dict[str, float]]:
    """Growth curves for charts: WHO (0-2 years, z-score lines) or CDC (2-20, percentiles)."""
    return reference_curves(source, sex)


# ============================================================
# App factory
# ============================================================


def create_app(service: BMIPilotService | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO)
        log_tracing_status()
        # Build lazily so importing this module never needs API keys.
        app.state.service = service or BMIPilotService.from_settings(settings)
        yield

    app = FastAPI(title="BMIPilot AI", version=__version__, lifespan=lifespan)
    app.state.settings = settings

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(router)
    return app


app = create_app()
