"""
backend/api/routes/predict.py
=============================
REST Endpoints for AI Subsurface Predictions and ConvLSTM Inferences.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List

from backend.schemas.prediction import (
    ReconstructionRequest,
    ReconstructionResponse,
    ProfilePredictionRequest,
    ProfilePredictionResponse,
    TimeSeriesForecastRequest,
    TimeSeriesForecastResponse,
    ModelEvaluationResponse
)
from backend.services.inference_service import InferenceService

router = APIRouter(prefix="/predict", tags=["AI Prediction & Forecasting"])


@router.post("/reconstruct", response_model=ReconstructionResponse)
def reconstruct_subsurface(request: ReconstructionRequest):
    """
    Run ConvLSTM Deep Learning Inference to reconstruct 2D Subsurface Temperature (ST) field.
    """
    try:
        result = InferenceService.reconstruct_subsurface_grid(
            depth=request.depth,
            model_type=request.model_type or "conv_lstm",
            target_date=request.date or "2024-05-20"
        )
        return ReconstructionResponse(
            status=result["status"],
            depth=result["depth"],
            model_type=result["model_type"],
            grid_shape=result["grid_shape"],
            min_temp=result["min_temp"],
            max_temp=result["max_temp"],
            mean_temp=result["mean_temp"],
            spearman_corr=result["spearman_corr"],
            rmse=result["rmse"],
            mae=result["mae"],
            inference_time_ms=result["inference_time_ms"],
            data=result["reconstructed_grid"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profile", response_model=ProfilePredictionResponse)
def get_predicted_profile(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude")
):
    """
    Get deep learning vertical temperature profile comparison at a specific ocean coordinate.
    """
    try:
        res = InferenceService.predict_vertical_profile(lat=lat, lon=lon)
        return ProfilePredictionResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/timeseries", response_model=TimeSeriesForecastResponse)
def get_timeseries_forecast(request: TimeSeriesForecastRequest):
    """
    Generate AI subsurface time-series forecast with confidence intervals.
    """
    try:
        res = InferenceService.forecast_timeseries(
            lat=request.lat,
            lon=request.lon,
            depth=request.depth,
            horizon_days=request.horizon_days,
            variable=request.variable
        )
        return TimeSeriesForecastResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation", response_model=ModelEvaluationResponse)
def get_evaluation_metrics():
    """
    Retrieve validation metrics (Spearman correlation, RMSE, MAE, SSIM) of current active model.
    """
    try:
        metrics = InferenceService.evaluate_model_performance()
        return ModelEvaluationResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
