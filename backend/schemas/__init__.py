"""
Pydantic Schemas and DTOs for Request / Response Validation.
"""
from backend.schemas.prediction import (
    ReconstructionRequest,
    ReconstructionResponse,
    ProfilePredictionRequest,
    ProfilePredictionResponse,
    TimeSeriesForecastRequest,
    TimeSeriesForecastResponse,
    ModelEvaluationResponse,
)
from backend.schemas.ocean import (
    GeoCoordinate,
    BoundingBox,
    GridDataResponse,
    TransectRequest,
    TransectResponse,
    ArgoFloatDTO,
    ArgoComparisonResponse,
    MarineHeatwaveEventDTO,
    MarineHeatwaveResponse,
)

__all__ = [
    "ReconstructionRequest",
    "ReconstructionResponse",
    "ProfilePredictionRequest",
    "ProfilePredictionResponse",
    "TimeSeriesForecastRequest",
    "TimeSeriesForecastResponse",
    "ModelEvaluationResponse",
    "GeoCoordinate",
    "BoundingBox",
    "GridDataResponse",
    "TransectRequest",
    "TransectResponse",
    "ArgoFloatDTO",
    "ArgoComparisonResponse",
    "MarineHeatwaveEventDTO",
    "MarineHeatwaveResponse",
]
