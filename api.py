"""Render web API for AI Waste Doctor image classification."""

from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ai.classifier import WasteClassifier

app = FastAPI(title="AI Waste Doctor API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

classifier = WasteClassifier(smoothing_frames=1)


@app.get("/")
def health_check() -> dict:
    return {
        "service": "AI Waste Doctor API",
        "status": "ok",
        "model_loaded": not classifier.demo_mode,
        "labels": classifier.labels,
        "model_status": classifier.status_message,
        "model_error": classifier.model_error,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": not classifier.demo_mode,
        "model_status": classifier.status_message,
        "model_error": classifier.model_error,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a valid image file")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")

    image = cv2.imdecode(np.frombuffer(BytesIO(image_bytes).getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The image could not be decoded")

    classifier.history_buffer.clear()
    result = classifier.predict(image, normalization_mode="-1_to_1")
    return {
        "filename": file.filename,
        "top_class": result["top_class"] if result["is_confident"] else "Uncertain",
        "top_confidence": result["top_confidence"],
        "is_confident": result["is_confident"],
        "status": result["status"],
        "predictions": result["smoothed_predictions"],
        "model_loaded": not result["demo_mode"],
    }
