"""
backend/api/routes/heatwave.py
==============================
REST Endpoints for Marine Heatwave (MHW) Events, Severity, and Time-Series.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Dict, Any

from backend.schemas.ocean import MarineHeatwaveResponse, MarineHeatwaveEventDTO
from backend.services.heatwave_service import HeatwaveService

router = APIRouter(prefix="/heatwave", tags=["Marine Heatwaves"])


@router.get("/events", response_model=MarineHeatwaveResponse)
def get_active_heatwaves():
    """
    Get all active Marine Heatwave events across Indian Ocean regions with category classifications.
    """
    try:
        res = HeatwaveService.get_active_events()
        return MarineHeatwaveResponse(
            total_active_events=res["total_active_events"],
            severe_events=res["severe_events"],
            moderate_events=res["moderate_events"],
            mean_intensity_c=res["mean_intensity_c"],
            events=[MarineHeatwaveEventDTO(**e) for e in res["events"]]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeseries")
def get_heatwave_timeseries(
    region: str = Query(default="Arabian Sea"),
    days: int = Query(default=90, ge=10, le=365)
):
    """
    Get SST climatology, 90th percentile threshold, and observed SST time-series for a region.
    """
    try:
        df, stats = HeatwaveService.get_heatwave_timeseries(region=region, days=days)
        return {
            "stats": stats,
            "data": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
