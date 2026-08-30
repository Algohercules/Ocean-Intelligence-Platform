"""
frontend/client.py
==================
Backend API / Service Client Adapter for Streamlit UI.
Supports dual execution modes:
1. REST Mode: Communicates with FastAPI backend server via HTTP endpoints.
2. Direct Mode: Direct Python invocations of backend services when API server is not running.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import requests

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.config import API_URL, DEPTH_LEVELS
from backend.services.inference_service import InferenceService
from backend.services.data_service import DataService
from backend.services.argo_service import ArgoService
from backend.services.heatwave_service import HeatwaveService


class OceanClient:
    """
    Unified Client Adapter for Streamlit pages and components.
    """
    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url.rstrip("/")
        self._check_api_availability()

    def _check_api_availability(self) -> bool:
        try:
            r = requests.get(f"{self.api_url}/", timeout=0.8)
            self.use_rest = (r.status_code == 200)
        except Exception:
            self.use_rest = False
        return self.use_rest

    # ==================== AI PREDICTION & RECONSTRUCTION ====================

    def reconstruct_grid(self, depth: float = 75.0, model_type: str = "conv_lstm", target_date: str = "2024-05-20") -> Dict[str, Any]:
        """Runs ConvLSTM AI inference for 2D subsurface temperature grid."""
        if self.use_rest:
            try:
                r = requests.post(f"{self.api_url}/api/predict/reconstruct", json={
                    "depth": depth,
                    "model_type": model_type,
                    "date": target_date
                }, timeout=10)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
        return InferenceService.reconstruct_subsurface_grid(depth=depth, model_type=model_type, target_date=target_date)

    def get_predicted_profile(self, lat: float, lon: float) -> Dict[str, Any]:
        """Gets AI vertical temperature profile prediction vs baseline."""
        if self.use_rest:
            try:
                r = requests.get(f"{self.api_url}/api/predict/profile", params={"lat": lat, "lon": lon}, timeout=5)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
        return InferenceService.predict_vertical_profile(lat=lat, lon=lon)

    def get_timeseries_forecast(self, lat: float = 15.0, lon: float = 65.0, depth: float = 75.0, horizon_days: int = 7) -> Dict[str, Any]:
        """AI multi-step time series forecast with confidence bounds."""
        if self.use_rest:
            try:
                r = requests.post(f"{self.api_url}/api/predict/timeseries", json={
                    "lat": lat,
                    "lon": lon,
                    "depth": depth,
                    "horizon_days": horizon_days
                }, timeout=5)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
        return InferenceService.forecast_timeseries(lat=lat, lon=lon, depth=depth, horizon_days=horizon_days)

    def get_model_evaluation(self) -> Dict[str, Any]:
        """Retrieves model evaluation benchmark scores."""
        if self.use_rest:
            try:
                r = requests.get(f"{self.api_url}/api/predict/evaluation", timeout=5)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
        return InferenceService.evaluate_model_performance()

    # ==================== OCEANOGRAPHIC DATA & PROFILES ====================

    def get_point_details(self, lat: float, lon: float, depth: float = 75.0) -> Dict[str, Any]:
        """Fetches temperature, salinity, density, current speed for coordinate."""
        if self.use_rest:
            try:
                r = requests.get(f"{self.api_url}/api/ocean/stats", params={"lat": lat, "lon": lon, "depth": depth}, timeout=5)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
        return DataService.get_point_details(lat=lat, lon=lon, depth=depth)

    def get_vertical_profile(self, lat: float, lon: float) -> pd.DataFrame:
        """Returns DataFrame of depth vs physical variables."""
        return DataService.get_vertical_profile(lat=lat, lon=lon)

    def get_transect(self, start_lat: float, start_lon: float, end_lat: float, end_lon: float, num_points: int = 50) -> Dict[str, Any]:
        """Returns vertical 2D transect data across ocean slice."""
        if self.use_rest:
            try:
                r = requests.post(f"{self.api_url}/api/ocean/transect", json={
                    "start_lat": start_lat,
                    "start_lon": start_lon,
                    "end_lat": end_lat,
                    "end_lon": end_lon,
                    "num_points": num_points
                }, timeout=5)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
        return DataService.get_transect(start_lat, start_lon, end_lat, end_lon, num_points)

    # ==================== ARGO FLOATS ====================

    def get_argo_floats(self, current_depth: float = 75.0) -> List[Dict[str, Any]]:
        """Returns active ARGO floats in the Indian Ocean."""
        if self.use_rest:
            try:
                r = requests.get(f"{self.api_url}/api/argo/floats", params={"depth": current_depth}, timeout=5)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
        return ArgoService.get_all_floats(current_depth=current_depth)

    def get_argo_comparison(self, wmo_id: str = "2903334") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Returns ARGO vs GLORYS vs ConvLSTM AI comparison profile."""
        return ArgoService.get_argo_vs_glorys_profile(wmo_id=wmo_id)

    # ==================== MARINE HEATWAVES ====================

    def get_heatwave_events(self) -> Dict[str, Any]:
        """Returns list and stats of active Marine Heatwaves."""
        if self.use_rest:
            try:
                r = requests.get(f"{self.api_url}/api/heatwave/events", timeout=5)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
        return HeatwaveService.get_active_events()

    def get_heatwave_timeseries(self, region: str = "Arabian Sea", days: int = 90) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Returns MHW climatology and threshold time series."""
        return HeatwaveService.get_heatwave_timeseries(region=region, days=days)


# Singleton client instance
client = OceanClient()
