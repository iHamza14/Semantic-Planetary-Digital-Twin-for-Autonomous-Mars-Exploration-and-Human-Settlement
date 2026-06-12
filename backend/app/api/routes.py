from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..schemas.prediction import PredictionResponse
from ..services.inference_service import InferenceService

router = APIRouter()
inference_service = InferenceService()


class PathPredictionRequest(BaseModel):
    image_path: str
    auto_ground_truth: bool = True


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "device": inference_service.device}


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    ground_truth_mask: UploadFile | None = File(default=None),
) -> dict:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await image.read()
    mask_bytes = await ground_truth_mask.read() if ground_truth_mask else None
    return inference_service.predict_from_bytes(image_bytes=image_bytes, mask_bytes=mask_bytes)


@router.post("/predict-from-path", response_model=PredictionResponse)
def predict_from_path(request: PathPredictionRequest) -> dict:
    img_path = Path(request.image_path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {img_path}")

    try:
        return inference_service.predict_from_path(
            image_path=img_path,
            auto_ground_truth=request.auto_ground_truth,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
