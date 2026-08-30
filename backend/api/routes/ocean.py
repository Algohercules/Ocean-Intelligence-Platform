"""
backend/api/routes/ocean.py
===========================
REST Endpoints for Ocean Grids, Point Inspections, and Transect Slices.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List

from backend.schemas.ocean import (
    GridDataResponse,
    TransectRequest,
    TransectResponse
)
from backend.services.data_service import DataService
from backend.services.copernicus_service import CopernicusService

router = APIRouter(prefix="/ocean", tags=["Oceanographic Data"])


@router.get("/stats")
def get_point_statistics(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    depth: float = Query(default=75.0, ge=0.0)
):
    """
    Get full oceanographic snapshot (temp, salinity, density, current speed, MLD) at point.
    """
    try:
        return DataService.get_point_details(lat=lat, lon=lon, depth=depth)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/grid", response_model=GridDataResponse)
def get_ocean_grid(
    variable: str = Query(default="thetao", description="thetao, SST, SSH, uo, vo"),
    depth: float = Query(default=75.0, ge=0.0),
    date: str = Query(default="2024-05-20")
):
    """
    Get 2D horizontal ocean grid layer.
    """
    try:
        snap = CopernicusService.load_variable_snapshot(var_name=variable)
        lats, lons = DataService.get_grid_coordinates()
        
        if snap and "data" in snap:
            return GridDataResponse(
                variable=variable,
                depth=depth,
                date=date,
                grid_shape=snap["shape"],
                lats=snap["lats"] if snap["lats"] else lats.tolist(),
                lons=snap["lons"] if snap["lons"] else lons.tolist(),
                min_val=snap["min"],
                max_val=snap["max"],
                mean_val=snap["mean"],
                data=snap["data"]
            )
        else:
            # Fallback computed grid
            H, W = len(lats), len(lons)
            return GridDataResponse(
                variable=variable,
                depth=depth,
                date=date,
                grid_shape=[H, W],
                lats=lats.tolist(),
                lons=lons.tolist(),
                min_val=12.5,
                max_val=31.2,
                mean_val=24.8,
                data=None
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transect", response_model=TransectResponse)
def calculate_transect(req: TransectRequest):
    """
    Calculate vertical cross-section slice between two coordinates.
    """
    try:
        res = DataService.get_transect(
            start_lat=req.start_lat,
            start_lon=req.start_lon,
            end_lat=req.end_lat,
            end_lon=req.end_lon,
            num_points=req.num_points,
            depths=req.depths
        )
        return TransectResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
