"""ORM models. Uses only portable column types so PostgreSQL can replace SQLite."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Inference(Base):
    """One inference event: an uploaded image, a webcam snapshot or a processed video."""

    __tablename__ = "inferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))  # image | video | webcam
    status: Mapped[str] = mapped_column(String(16), default="completed")  # queued|processing|completed|failed
    progress: Mapped[float] = mapped_column(Float, default=1.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    annotated_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    video_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    output_video_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    location_source: Mapped[str | None] = mapped_column(String(16), nullable=True)  # browser|exif|manual|route
    location_accuracy_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(128))
    conf_threshold: Mapped[float] = mapped_column(Float)
    inference_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_detections: Mapped[int] = mapped_column(Integer, default=0)
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    detections: Mapped[list["Detection"]] = relationship(back_populates="inference", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inference_id: Mapped[int] = mapped_column(ForeignKey("inferences.id", ondelete="CASCADE"), index=True)
    class_code: Mapped[str] = mapped_column(String(8), index=True)
    class_name: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(8), index=True)
    severity_score: Mapped[float] = mapped_column(Float)
    severity_detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    location_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    video_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    crop_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    model_version: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    inference: Mapped[Inference] = relationship(back_populates="detections")

    __table_args__ = (Index("ix_detections_latlon", "latitude", "longitude"),)
