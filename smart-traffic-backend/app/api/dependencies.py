"""Wiring between the FastAPI app state and the request handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

from app.config import Settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.detection import VideoScanner
    from app.priority import SignalPriorityController
    from app.services.prediction import PredictionService

__all__ = ["get_controller", "get_prediction_service", "get_scanner", "get_settings"]


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_scanner(request: Request) -> VideoScanner:
    return request.app.state.scanner


def get_controller(request: Request) -> SignalPriorityController:
    return request.app.state.priority_controller


def get_prediction_service(request: Request) -> PredictionService:
    return request.app.state.prediction_service
