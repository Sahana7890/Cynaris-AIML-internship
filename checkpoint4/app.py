"""
MedScan Diagnostics - Checkpoint 4
FastAPI model serving layer.

Run from the project root:

    uvicorn checkpoint4.api:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health
    POST /predict

The API expects the selected Checkpoint 3 model at:

    models/best_model.pth
"""

from pathlib import Path
import io
import sys

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

# Allow imports from src/
sys.path.insert(0, str(ROOT / "src"))

from config import CLASS_NAMES, MODEL_DIR
from model import build_model
from preprocessing import eval_transform


# -------------------------------------------------------------------
# FastAPI application
# -------------------------------------------------------------------

app = FastAPI(
    title="MedScan Diagnostics API",
    description=(
        "Research prototype for chest X-ray classification "
        "using a ResNet-18 model."
    ),
    version="1.0.0",
)


# -------------------------------------------------------------------
# Device
# -------------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -------------------------------------------------------------------
# Model configuration
# -------------------------------------------------------------------

MODEL_PATH = MODEL_DIR / "best_model.pth"

model = None


# -------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------

def load_model():
    """
    Load the trained ResNet-18 model from disk.
    """

    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Place the selected Checkpoint 3 model at "
            "models/best_model.pth."
        )

    model = build_model(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    # Support both:
    # 1. Raw state_dict
    # 2. Dictionary containing model_state_dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()


# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    """
    Attempt to load the trained model when the API starts.
    """

    try:
        load_model()

        print(
            f"Model loaded successfully from: {MODEL_PATH}"
        )
        print(f"Using device: {DEVICE}")

    except FileNotFoundError as exc:
        print(f"WARNING: {exc}")


# -------------------------------------------------------------------
# Health endpoint
# -------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Check whether the API and model are available.
    """

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
    }


# -------------------------------------------------------------------
# Prediction endpoint
# -------------------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classify an uploaded chest X-ray image.

    Returns:
        predicted_class
        class_probabilities
        model information
    """

    # Check model availability
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. "
                "Check models/best_model.pth."
            ),
        )

    # Validate content type
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    # Read image
    try:
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read image: {exc}",
        )

    # Preprocess image
    transform = eval_transform()

    tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    # Model inference
    with torch.inference_mode():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    # Find predicted class
    predicted_index = int(
        probabilities.argmax().item()
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    # Create probability dictionary
    probability_dict = {
        CLASS_NAMES[index]: round(
            float(
                probabilities[index].cpu()
            ),
            6,
        )
        for index in range(
            len(CLASS_NAMES)
        )
    }

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "class_probabilities": probability_dict,
        "model": "ResNet-18",
        "device": str(DEVICE),
        "note": (
            "Research prototype; "
            "not a clinical diagnosis."
        ),
    }