from pydantic import BaseModel


class PredictionResponse(BaseModel):
    latency_ms: float
    mean_confidence: float
    accuracy: float | None = None
    predicted_shape: tuple[int, int]
    prediction_map: list[list[int]]


class HealthResponse(BaseModel):
    status: str
    device: str
