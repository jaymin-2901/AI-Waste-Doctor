"""Render web API for AI Waste Doctor image classification."""

import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai.classifier import WasteClassifier


app = FastAPI(title="AI Waste Doctor API", version="1.2.0")

# Keep synchronized with streamlit/app.py -> EXPECTED_API_BUILD.
API_BUILD = "2026-10-03-science-fair-bottle-fix-v6"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

# IMPORTANT: the Streamlit live preview draws a 60% scan square.  The API now
# uses exactly the same 60% crop, so the model sees the object users were told
# to place inside the guide instead of extra bowl/background pixels.
SCAN_ZONE_RATIO = 0.60

allowed_origins = [
    origin.strip()
    for origin in os.getenv("AI_WASTE_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

classifier = WasteClassifier(smoothing_frames=1)


def get_scan_zone(width: int, height: int, ratio: float = SCAN_ZONE_RATIO):
    box_size = int(min(width, height) * ratio)
    return (
        (width - box_size) // 2,
        (height - box_size) // 2,
        box_size,
        box_size,
    )


def service_payload() -> dict:
    return {
        "build": API_BUILD,
        "model_loaded": not classifier.demo_mode,
        "labels": classifier.labels,
        "model_status": classifier.status_message,
        "model_error": classifier.model_error,
        "scan_zone_ratio": SCAN_ZONE_RATIO,
        "semantic_guard_enabled": classifier.semantic_guard_enabled,
        "semantic_guard_status": classifier.semantic_status,
        "semantic_guard_error": classifier.semantic_error,
    }


@app.get("/")
def health_check() -> dict:
    return {
        "service": "AI Waste Doctor API",
        "status": "ok",
        **service_payload(),
    }


@app.get("/health")
def health() -> dict:
    payload = {
        "status": "ok" if not classifier.demo_mode else "degraded",
        "ready": not classifier.demo_mode,
        **service_payload(),
    }
    if classifier.demo_mode:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if classifier.demo_mode:
        raise HTTPException(status_code=503, detail="The classification model is not ready")

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")

    content_length = file.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header") from error
        if declared_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit")

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The image could not be decoded")

    crop_box = get_scan_zone(image.shape[1], image.shape[0])

    # Each HTTP request is independent.  Streamlit performs live multi-frame
    # stability itself, so do not carry probability history between requests.
    classifier.history_buffer.clear()

    result = classifier.predict(
        image,
        crop_box=crop_box,
        normalization_mode="-1_to_1",
    )

    return {
        "filename": file.filename,
        "build": API_BUILD,
        "crop_box": crop_box,
        "scan_zone_ratio": SCAN_ZONE_RATIO,
        "top_class": result["top_class"] if result["is_confident"] else "Uncertain",
        "top_confidence": result["top_confidence"],
        "is_confident": result["is_confident"],
        "status": result["status"],
        "predictions": result["smoothed_predictions"],
        "raw_predictions": result["raw_predictions"],
        "model_loaded": not result["demo_mode"],
        "semantic_guard_applied": result.get("semantic_guard_applied", False),
        "semantic_guard_status": result.get("semantic_guard_status", classifier.semantic_status),
        "semantic_hint": result.get("semantic_hint"),
    }
