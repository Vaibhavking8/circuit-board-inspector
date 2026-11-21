"""
api/schemas.py — Pydantic schemas at the API boundary.

These are the only place Pydantic is used. Core CV code uses plain dataclasses.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class PartSchema(BaseModel):
    id: str
    name: str
    box: tuple[float, float, float, float]  # normalised (x, y, w, h)


class ReferenceUploadResponse(BaseModel):
    reference_id: str
    width: int
    height: int


class PartsUpdate(BaseModel):
    parts: list[PartSchema]


class PartsResponse(BaseModel):
    parts: list[PartSchema]


class JobResponse(BaseModel):
    job_id: str
    status: str           # "queued" | "running" | "complete" | "failed"
    progress: float       # 0.0 – 1.0
    result: dict | None = None
    error: str | None = None


class PartResultSchema(BaseModel):
    part_id: str
    status: str           # "present" | "missing" | "uncertain"
    signals: dict[str, float]
    threshold: float
    margin: float
    inspection_box: tuple[int, int, int, int]
    reason: str | None = None


class AlignmentInfoSchema(BaseModel):
    strategy: str
    transform_type: str
    quality: float


class ImageResultResponse(BaseModel):
    status: str           # "ok" | "missing_parts" | "could_not_verify"
    alignment: AlignmentInfoSchema
    part_results: list[PartResultSchema]
    runtime_ms: float
    messages: list[str] = []


class InspectResponse(BaseModel):
    job_id: str


class ReEvaluateRequest(BaseModel):
    sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    # stored inspection result to re-evaluate
    part_results: list[PartResultSchema]
    alignment_status: str    # "aligned" | "failed" | "identity"
    inspection_width: int
    inspection_height: int


class ReEvaluateResponse(BaseModel):
    part_results: list[PartResultSchema]
    status: str


class SetupSummarySchema(BaseModel):
    setup_id: str
    part_count: int
    created_at: float


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
