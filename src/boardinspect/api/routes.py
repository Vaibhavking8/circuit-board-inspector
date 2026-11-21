"""
api/routes.py — all BoardInspect API routes.

In-memory reference store: dict[reference_id -> ReferenceState].
Jobs run in background threads via the JOBS singleton; GET /api/jobs/{id} polls them.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from boardinspect.api.jobs import JOBS
from boardinspect.api.schemas import (
    AlignmentInfoSchema,
    ImageResultResponse,
    JobResponse,
    PartResultSchema,
    PartsResponse,
    PartsUpdate,
    PartSchema,
    ReEvaluateRequest,
    ReEvaluateResponse,
    ReferenceUploadResponse,
    SetupSummarySchema,
)
from boardinspect.api.services.inspection_service import run_inspection
from boardinspect.api.services.reference_service import process_upload
from boardinspect.api.services.setup_service import list_all as list_all_setups
from boardinspect.api.services.setup_service import load as load_setup_svc
from boardinspect.api.services.setup_service import save as save_setup_svc
from boardinspect.config import DEFAULT_CONFIG
from boardinspect.decision.sensitivity import re_evaluate
from boardinspect.errors import DuplicateNameError, InvalidBoxError
from boardinspect.models import (
    ImageResult,
    Part,
    PartResult,
    PartStatus,
    ReferenceSetup,
)
from boardinspect.pipeline import calibrate_reference
from boardinspect.registration.interface import RegistrationStatus
from boardinspect.viz import render_result_overlay

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# ── In-memory stores ─────────────────────────────────────────────────────────

@dataclass
class ReferenceState:
    reference_id: str
    image_bytes: bytes
    image_array: np.ndarray
    setup: ReferenceSetup
    last_inspection_result: Any | None = field(default=None)
    last_inspection_array: np.ndarray | None = field(default=None)


_references: dict[str, ReferenceState] = {}
_lock = threading.Lock()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_ref(reference_id: str) -> ReferenceState:
    state = _references.get(reference_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Reference '{reference_id}' not found.")
    return state


def _part_result_to_schema(pr: PartResult) -> PartResultSchema:
    return PartResultSchema(
        part_id=pr.part_id,
        status=pr.status.value,
        signals=pr.signals,
        threshold=pr.threshold,
        margin=pr.margin,
        inspection_box=pr.inspection_box,
        reason=pr.reason,
    )


def _image_result_to_response(ir: ImageResult) -> ImageResultResponse:
    return ImageResultResponse(
        status=ir.status.value,
        alignment=AlignmentInfoSchema(
            strategy=ir.alignment.strategy,
            transform_type=ir.alignment.transform_type.value,
            quality=ir.alignment.quality,
        ),
        part_results=[_part_result_to_schema(pr) for pr in ir.part_results],
        runtime_ms=ir.runtime_ms,
        messages=ir.messages,
    )


# ── Reference endpoints ──────────────────────────────────────────────────────

@router.post("/reference", response_model=ReferenceUploadResponse)
async def upload_reference(request: Request, file: UploadFile = File(...)):
    """Upload and validate a reference board image."""
    config = getattr(request.app.state, "config", DEFAULT_CONFIG)
    raw = await file.read()
    reference_id, validated, setup = process_upload(raw, file.filename or "<upload>", config)

    with _lock:
        _references[reference_id] = ReferenceState(
            reference_id=reference_id,
            image_bytes=raw,
            image_array=validated.array,
            setup=setup,
        )

    logger.debug(
        "Reference uploaded: id=%s, size=%dx%d",
        reference_id,
        validated.original_width,
        validated.original_height,
    )
    return ReferenceUploadResponse(
        reference_id=reference_id,
        width=validated.original_width,
        height=validated.original_height,
    )


@router.get("/reference/{reference_id}/parts", response_model=PartsResponse)
async def get_parts(reference_id: str):
    """Return the annotated parts for a reference."""
    state = _get_ref(reference_id)
    return PartsResponse(parts=[
        PartSchema(id=p.id, name=p.name, box=p.box)
        for p in state.setup.parts
    ])


@router.put("/reference/{reference_id}/parts", response_model=PartsResponse)
async def set_parts(reference_id: str, body: PartsUpdate):
    """Replace the full parts list for a reference.

    Raises DuplicateNameError (→ 400) if any two parts share a name.
    Raises InvalidBoxError (→ 400) if any box is outside [0, 1] or has
    non-positive dimensions.
    """
    state = _get_ref(reference_id)

    # Validate: no duplicate names
    names = [p.name for p in body.parts]
    if len(names) != len(set(names)):
        seen: set[str] = set()
        dupes = [n for n in names if n in seen or seen.add(n)]  # type: ignore[func-returns-value]
        raise DuplicateNameError(
            f"Duplicate part name: '{dupes[0]}'. Each part must have a unique name."
        )

    # Validate boxes and build Part objects
    parts: list[Part] = []
    for ps in body.parts:
        x, y, w, h = ps.box
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            raise InvalidBoxError(
                f"Part '{ps.name}' has an invalid box {ps.box}. "
                "Coordinates must be normalised in [0, 1] with positive width and height."
            )
        try:
            parts.append(Part(id=ps.id, name=ps.name, box=tuple(ps.box)))  # type: ignore[arg-type]
        except ValueError as exc:
            raise InvalidBoxError(str(exc)) from exc

    with _lock:
        _references[reference_id] = dataclasses.replace(
            state,
            setup=dataclasses.replace(state.setup, parts=parts, calibration=None),
        )

    return PartsResponse(parts=[PartSchema(id=p.id, name=p.name, box=p.box) for p in parts])


# ── Calibration job ──────────────────────────────────────────────────────────

@router.post("/reference/{reference_id}/calibrate", response_model=JobResponse)
async def start_calibrate(reference_id: str, request: Request):
    """Start a background calibration job for the reference."""
    state = _get_ref(reference_id)
    config = getattr(request.app.state, "config", DEFAULT_CONFIG)
    job_id = JOBS.create()

    def _run() -> None:
        def _progress(idx: int, total_parts: int) -> None:
            JOBS.set_progress(job_id, idx / max(total_parts, 1))

        updated_setup = calibrate_reference(
            state.image_array, state.setup, config, _progress
        )
        with _lock:
            _references[reference_id] = dataclasses.replace(state, setup=updated_setup)
        JOBS.set_complete(job_id, {
            "part_count": len(updated_setup.parts),
            "calibrated": {
                pid: {"threshold": cal.threshold, "reliable": cal.reliable}
                for pid, cal in (updated_setup.calibration or {}).items()
            },
        })

    JOBS.run_in_background(job_id, _run)
    return JobResponse(job_id=job_id, status="queued", progress=0.0)


# ── Inspection job ───────────────────────────────────────────────────────────

@router.post("/reference/{reference_id}/inspect", response_model=JobResponse)
async def start_inspect(
    reference_id: str,
    request: Request,
    file: UploadFile = File(...),
):
    """Upload an inspection image and start a background inspection job."""
    state = _get_ref(reference_id)
    config = getattr(request.app.state, "config", DEFAULT_CONFIG)
    raw = await file.read()
    sensitivity = float(request.query_params.get("sensitivity", "0.5"))
    job_id = JOBS.create()

    def _run() -> None:
        result = run_inspection(state.setup, state.image_array, raw, sensitivity, config)
        # Decode the inspection image for the export endpoint
        arr = np.frombuffer(raw, dtype=np.uint8)
        insp_arr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        with _lock:
            _references[reference_id] = dataclasses.replace(
                state,
                last_inspection_result=result,
                last_inspection_array=insp_arr,
            )
        resp = _image_result_to_response(result)
        JOBS.set_complete(job_id, resp.model_dump())

    JOBS.run_in_background(job_id, _run)
    return JobResponse(job_id=job_id, status="queued", progress=0.0)


# ── Job polling ──────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Poll the status, progress and result of a background job."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        result=job.result,
        error=job.error,
    )


# ── Re-evaluate ──────────────────────────────────────────────────────────────

@router.post("/reference/{reference_id}/re-evaluate", response_model=ReEvaluateResponse)
async def re_evaluate_endpoint(
    reference_id: str,
    body: ReEvaluateRequest,
    request: Request,
):
    """Re-decide all parts from stored signals at a new sensitivity level.

    This is synchronous — it only re-applies the decision engine with shifted
    thresholds and returns immediately without re-running alignment or comparison.
    """
    state = _get_ref(reference_id)
    config = getattr(request.app.state, "config", DEFAULT_CONFIG)

    calibration = state.setup.calibration or {}

    stored_prs = [
        PartResult(
            part_id=pr.part_id,
            status=PartStatus(pr.status),
            signals=pr.signals,
            threshold=pr.threshold,
            margin=pr.margin,
            inspection_box=tuple(pr.inspection_box),  # type: ignore[arg-type]
            reason=pr.reason,
        )
        for pr in body.part_results
    ]

    alignment_status = RegistrationStatus(body.alignment_status)

    new_prs, new_status = re_evaluate(
        stored_prs,
        calibration,
        state.setup,
        body.inspection_width,
        body.inspection_height,
        alignment_status,
        body.sensitivity,
        config,
    )

    return ReEvaluateResponse(
        part_results=[_part_result_to_schema(pr) for pr in new_prs],
        status=new_status.value,
    )


# ── Setups (save / load / list) ───────────────────────────────────────────────

@router.get("/setups", response_model=list[SetupSummarySchema])
async def get_setups(request: Request):
    """List all setups saved to disk."""
    config = getattr(request.app.state, "config", DEFAULT_CONFIG)
    summaries = list_all_setups(config)
    return [
        SetupSummarySchema(
            setup_id=s.setup_id,
            part_count=s.part_count,
            created_at=s.created_at,
        )
        for s in summaries
    ]


@router.post("/setups/{reference_id}", response_model=SetupSummarySchema)
async def save_setup_endpoint(reference_id: str, request: Request):
    """Persist the current reference + calibration to disk."""
    state = _get_ref(reference_id)
    config = getattr(request.app.state, "config", DEFAULT_CONFIG)
    setup_id = save_setup_svc(state.setup, state.image_bytes, config)
    summaries = list_all_setups(config)
    match = next((s for s in summaries if s.setup_id == setup_id), None)
    if match is None:
        return SetupSummarySchema(
            setup_id=setup_id,
            part_count=len(state.setup.parts),
            created_at=0.0,
        )
    return SetupSummarySchema(
        setup_id=match.setup_id,
        part_count=match.part_count,
        created_at=match.created_at,
    )


@router.get("/setups/{setup_id}/load", response_model=ReferenceUploadResponse)
async def load_setup_endpoint(setup_id: str, request: Request):
    """Load a saved setup from disk into the in-memory reference store."""
    config = getattr(request.app.state, "config", DEFAULT_CONFIG)
    setup, image_array = load_setup_svc(setup_id, config)
    _, enc = cv2.imencode(".png", image_array)
    image_bytes = enc.tobytes()
    reference_id = setup.content_hash[:16]
    with _lock:
        _references[reference_id] = ReferenceState(
            reference_id=reference_id,
            image_bytes=image_bytes,
            image_array=image_array,
            setup=setup,
        )
    return ReferenceUploadResponse(
        reference_id=reference_id,
        width=setup.original_width,
        height=setup.original_height,
    )


# ── Export ───────────────────────────────────────────────────────────────────

@router.get("/reference/{reference_id}/export")
async def export_result(reference_id: str, request: Request):
    """Return the last inspection result as an annotated PNG overlay.

    Uses viz.render_result_overlay to draw status banner, bounding boxes and
    labels onto the inspection image. Returns 404 if no inspection has been
    run yet for this reference.
    """
    state = _get_ref(reference_id)
    config = getattr(request.app.state, "config", DEFAULT_CONFIG)
    if state.last_inspection_result is None or state.last_inspection_array is None:
        raise HTTPException(
            status_code=404,
            detail="No inspection result available yet for this reference.",
        )
    overlay = render_result_overlay(state.last_inspection_array, state.last_inspection_result, config)
    _, enc = cv2.imencode(".png", overlay)
    return Response(content=enc.tobytes(), media_type="image/png")


# ── Reset ────────────────────────────────────────────────────────────────────

@router.post("/reference/{reference_id}/reset")
async def reset_reference(reference_id: str):
    """Remove a reference from the in-memory store."""
    with _lock:
        removed = _references.pop(reference_id, None)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"Reference '{reference_id}' not found.")
    return {"ok": True}
