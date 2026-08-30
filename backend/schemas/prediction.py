"""
backend/schemas/prediction.py
=============================
Pydantic Schemas for AI prediction, reconstruction, and forecasting endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ReconstructionRequest(BaseModel):
    depth: float = Field(default=75.0, description="Subsurface target depth in meters")
    date: Optional[str] = Field(default="2024-05-20", description="Target date (YYYY-MM-DD)")
    model_type: Optional[str] = Field(default="conv_lstm", description="conv_lstm or cnn_lstm")
    region: Optional[str] = Field(default="Entire Indian Ocean", description="Target geographic region")


class ReconstructionResponse(BaseModel):
    status: str = "success"
    depth: float
    model_type: str
    grid_shape: List[int]
    min_temp: float
    max_temp: float
    mean_temp: float
    spearman_corr: float
    rmse: float
    mae: float
    inference_time_ms: float
    data: Optional[List[List[float]]] = None


class ProfilePredictionRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude")
    date: Optional[str] = Field(default="2024-05-20", description="Date string")
    depths: Optional[List[float]] = Field(default=None, description="Custom depth array")


class ProfilePredictionResponse(BaseModel):
    lat: float
    lon: float
    depths: List[float]
    conv_lstm_temp: List[float]
    glorys_temp: List[float]
    argo_obs_temp: Optional[List[float]] = None
    thermocline_depth: float
    mixed_layer_depth: float
    surface_temp: float


class TimeSeriesForecastRequest(BaseModel):
    lat: float = Field(default=15.0, ge=-90.0, le=90.0)
    lon: float = Field(default=65.0, ge=-180.0, le=180.0)
    depth: float = Field(default=75.0, ge=0.0)
    horizon_days: int = Field(default=7, ge=1, le=30)
    variable: str = Field(default="Temperature")


class ForecastPoint(BaseModel):
    date: str
    historical_obs: Optional[float] = None
    glorys_baseline: Optional[float] = None
    ai_forecast: Optional[float] = None
    upper_bound: Optional[float] = None
    lower_bound: Optional[float] = None
    point_type: str


class TimeSeriesForecastResponse(BaseModel):
    lat: float
    lon: float
    depth: float
    horizon_days: int
    current_temp: float
    predicted_temp: float
    change_temp: float
    confidence_pct: int
    anomaly_val: float
    series: List[ForecastPoint]


class ModelEvaluationResponse(BaseModel):
    model_name: str
    spearman_correlation: float
    pearson_correlation: float
    rmse: float
    mae: float
    ssim: float
    total_eval_samples: int
    device_used: str
