# Offroad Segmentation FastAPI Backend

This backend extracts the image inference pipeline from `demo.py` into a modular FastAPI service.

## Structure

- `app/main.py` - FastAPI app bootstrap.
- `app/api/routes.py` - API routes.
- `app/services/inference_service.py` - Model loading, preprocessing, inference, confidence, and optional accuracy logic.
- `app/core/config.py` - Runtime configuration constants.
- `app/schemas/prediction.py` - Response schemas.

## Run

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /health` - Service/device status.
- `POST /predict` - Upload image and optional ground-truth mask.
- `POST /predict-from-path` - Predict from local filesystem path with optional auto GT detection from `Color_Images -> Segmentation` mapping.
