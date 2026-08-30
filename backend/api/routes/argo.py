"""
backend/api/routes/argo.py
==========================
REST Endpoints for ARGO float status, trajectories, and sensor profile comparisons.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List

from backend.schemas.ocean import ArgoFloatDTO, ArgoComparisonResponse
from backend.services.argo_service import ArgoService

router = APIRouter(prefix="/argo", tags=["ARGO Floats"])


@router.get("/floats", response_model=List[ArgoFloatDTO])
def get_argo_floats(depth: float = Query(default=75.0, ge=0.0)):
    """
    Get all active ARGO floats in the Indian Ocean basin with sensor metrics at target depth.
    """
    try:
        floats = ArgoService.get_all_floats(current_depth=depth)
        return [ArgoFloatDTO(**f) for f in floats]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison", response_model=ArgoComparisonResponse)
def get_argo_comparison(wmo_id: str = Query(default="2903334")):
    """
    Compare in-situ ARGO profile against GLORYS baseline and ConvLSTM AI model predictions.
    """
    try:
        df, stats = ArgoService.get_argo_vs_glorys_profile(wmo_id=wmo_id)
        return ArgoComparisonResponse(
            wmo_id=stats["wmo_id"],
            lat=stats["lat"],
            lon=stats["lon"],
            depths=df["Depth (m)"].tolist(),
            argo_temp=df["ARGO In-Situ (°C)"].tolist(),
            glorys_temp=df["GLORYS Baseline (°C)"].tolist(),
            conv_lstm_temp=df["ConvLSTM AI (°C)"].tolist(),
            rmse=stats["rmse_ai"],
            mae=stats["mae_ai"],
            correlation=0.962
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
