"""Application composition root.

Layering, outermost first::

    app.api        HTTP: routing, schemas, status codes        (depends inward)
    app.services   use cases: spooling, concurrency, orchestration
    app.detection  detector seam + the OpenCV scan pipeline
    app.priority   the signal-priority state machine (pure, no I/O)
    app.config     settings, parsed once from the environment

Nothing below ``app.api`` imports FastAPI, which is what makes the detection
pipeline and the priority controller testable on their own.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import router
from app.config import Settings
from app.detection import VideoScanner, build_detector
from app.priority import PriorityPolicy, SignalPriorityController
from app.services.prediction import PredictionService

__all__ = ["app", "create_app"]

logger = logging.getLogger(__name__)


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def _policy_from(settings: Settings) -> PriorityPolicy:
    return PriorityPolicy(
        min_confidence=settings.priority_min_confidence,
        corroborating_sightings=settings.priority_corroborating_sightings,
        corroboration_window_s=settings.priority_corroboration_window_s,
        min_hold_s=settings.priority_min_hold_s,
        max_hold_s=settings.priority_max_hold_s,
        clear_after_s=settings.priority_clear_after_s,
        cooldown_s=settings.priority_cooldown_s,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Everything heavy happens in the lifespan hook."""
    settings = settings or Settings.from_env()
    _configure_logging(settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        # Weights are resolved and loaded exactly once, at startup, and the
        # loaded model is then shared by every request.
        detector = build_detector(settings)
        detector.warmup()
        scanner = VideoScanner(detector, settings)
        application.state.settings = settings
        application.state.scanner = scanner
        application.state.prediction_service = PredictionService(scanner, settings)
        application.state.priority_controller = SignalPriorityController(
            _policy_from(settings)
        )
        logger.info(
            "ITCS %s ready: detector=%s real=%s ambulance_class=%s env=%s",
            __version__,
            detector.name,
            detector.performs_real_detection,
            detector.identifies_ambulances,
            settings.environment,
        )
        yield

    application = FastAPI(
        title="Intelligent Traffic Control System",
        version=__version__,
        summary=(
            "Emergency-vehicle detection over uploaded traffic clips, plus a "
            "signal-priority decision layer. Controls no hardware."
        ),
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()
