"""
backend/schemas/ocean.py
========================
Pydantic Schemas for spatial coordinates, transects, ARGO floats, and marine heatwaves.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class GeoCoordinate(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    depth: Optional[float] = Field(default=0.0)


class BoundingBox(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class GridDataResponse(BaseModel):
    variable: str
    depth: float
    date: str
    grid_shape: List[int]
    lats: List[float]
    lons: List[float]
    min_val: float
    max_val: float
    mean_val: float
    data: Optional[List[List[float]]] = None


class TransectRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    num_points: int = Field(default=50, ge=10, le=200)
    depths: Optional[List[float]] = None


class TransectResponse(BaseModel):
    num_points: int
    distances_km: List[float]
    lats: List[float]
    lons: List[float]
    depths: List[float]
    temperature_matrix: List[List[float]]


class ArgoFloatDTO(BaseModel):
    wmo_id: str
    lat: float
    lon: float
    last_reported: str
    status: str
    cycle_number: int
    data_points: int
    surface_temp: float
    temp_at_depth: float
    trajectory: Optional[List[Dict[str, Any]]] = None


class ArgoComparisonResponse(BaseModel):
    wmo_id: str
    lat: float
    lon: float
    depths: List[float]
    argo_temp: List[float]
    glorys_temp: List[float]
    conv_lstm_temp: List[float]
    rmse: float
    mae: float
    correlation: float


class MarineHeatwaveEventDTO(BaseModel):
    event_id: str
    region: str
    lat: float
    lon: float
    start_date: str
    duration_days: int
    peak_intensity_c: float
    category: str
    cumulative_intensity: float
    affected_area_km2: float


class MarineHeatwaveResponse(BaseModel):
    total_active_events: int
    severe_events: int
    moderate_events: int
    mean_intensity_c: float
    events: List[MarineHeatwaveEventDTO]
