"""Request/response schemas (pydantic v2)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LocationUpdate(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    source: Literal["browser", "manual", "exif", "route"] = "manual"
    accuracy_m: float | None = Field(default=None, ge=0)


class ModelSelect(BaseModel):
    weights: str = Field(min_length=1, max_length=255)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    model_error: str | None
    database: str
    cuda_available: bool
    version: str
