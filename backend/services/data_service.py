"""
backend/services/data_service.py
================================
Data ingestion, land-mask caching, grid interpolation, and depth profile generation.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

from backend.config import (
    OCEAN_BOUNDS,
    DEPTH_LEVELS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    DATA_DIR
)


class DataService:
    """
    Core data service for ocean spatial grids, vertical profiles,
    transects, and landmask operations.
    """
    _land_mask: Optional[np.ndarray] = None
    _grid_lats: Optional[np.ndarray] = None
    _grid_lons: Optional[np.ndarray] = None

    @classmethod
    def get_grid_coordinates(cls) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the 1D latitude and longitude arrays for the Indian Ocean grid."""
        if cls._grid_lats is None or cls._grid_lons is None:
            cls._grid_lats = np.linspace(
                OCEAN_BOUNDS["lat_min"],
                OCEAN_BOUNDS["lat_max"],
                OCEAN_BOUNDS["grid_height"]
            )
            cls._grid_lons = np.linspace(
                OCEAN_BOUNDS["lon_min"],
                OCEAN_BOUNDS["lon_max"],
                OCEAN_BOUNDS["grid_width"]
            )
        return cls._grid_lats, cls._grid_lons

    @classmethod
    def get_land_mask(cls) -> np.ndarray:
        """
        Loads or creates a binary land mask for the Indian Ocean (180 x 360).
        True = Land, False = Ocean.
        """
        if cls._land_mask is not None:
            return cls._land_mask

        mask_file = PROCESSED_DATA_DIR / "land_mask.npy"
        if mask_file.exists():
            try:
                cls._land_mask = np.load(mask_file)
                return cls._land_mask
            except Exception as e:
                print(f"[DataService] Could not load cached landmask: {e}")

        # Synthesize realistic Indian landmass mask over 30E-120E, -40S to 30N
        H, W = OCEAN_BOUNDS["grid_height"], OCEAN_BOUNDS["grid_width"]
        lats, lons = cls.get_grid_coordinates()
        LON, LAT = np.meshgrid(lons, lats)

        # Base ocean = False
        mask = np.zeros((H, W), dtype=bool)

        # 1. Indian subcontinent polygon approximation (lat 8 to 30, lon 68 to 89)
        india_subcontinent = (LAT >= 8) & (LAT <= 28) & (LON >= 68) & (LON <= 88)
        # Triangular tapering towards south
        india_triangle = india_subcontinent & (LON >= (77.0 - (LAT - 8.0) * 0.6)) & (LON <= (77.0 + (LAT - 8.0) * 0.6))
        mask |= india_triangle

        # 2. Arabian Peninsula / Africa Horn (lat -5 to 30, lon 30 to 55)
        africa_arabia = (LON <= 50) & (LAT >= -15) & (LAT <= 30)
        mask |= africa_arabia

        # 3. Southeast Asia / Indonesia / Australia
        se_asia = (LAT >= 0) & (LON >= 98) & (LON <= 120)
        australia = (LAT <= -12) & (LON >= 113) & (LON <= 120)
        mask |= se_asia | australia

        # 4. Madagascar (lat -25 to -12, lon 43 to 50)
        madagascar = (LAT >= -25) & (LAT <= -12) & (LON >= 43) & (LON <= 51)
        mask |= madagascar

        try:
            np.save(mask_file, mask)
        except Exception:
            pass

        cls._land_mask = mask
        return cls._land_mask

    @classmethod
    def get_point_details(cls, lat: float, lon: float, depth: float = 75.0) -> Dict[str, Any]:
        """Calculates oceanographic parameters at specific coordinate and depth."""
        # Realistic Indian Ocean thermal stratification
        surface_temp = 29.2 - 0.12 * abs(lat) + 0.04 * (lon - 65.0)
        deep_temp = 3.8 + 0.5 * np.cos(np.radians(lat))
        
        # Exponential thermocline drop
        k = 0.018 + 0.004 * np.sin(np.radians(lat * 2))
        temp_val = deep_temp + (surface_temp - deep_temp) * np.exp(-k * depth)
        
        # Salinity (PSU)
        salinity = 35.2 + 0.8 * np.sin(np.radians(lat)) - (depth / 2000.0) * 0.4
        # Current speed (m/s)
        current_speed = max(0.05, 0.45 * np.exp(-depth / 150.0) + 0.1 * np.cos(np.radians(lon * 3)))
        
        # Density kg/m3 (UNESCO EOS-80 approximation)
        density = 1024.0 + 0.7 * (salinity - 35.0) - 0.2 * (temp_val - 20.0) + 0.0045 * depth

        # Mixed layer depth (MLD)
        mld = max(15.0, 45.0 - 0.8 * lat + 5.0 * np.sin(np.radians(lon)))
        # Thermocline depth (depth of max d(temp)/dz)
        thermocline_depth = 110.0 + 15.0 * np.sin(np.radians(lat))

        return {
            "lat": lat,
            "lon": lon,
            "depth": depth,
            "temperature_c": round(float(temp_val), 2),
            "surface_temp_c": round(float(surface_temp), 2),
            "salinity_psu": round(float(salinity), 2),
            "current_speed_ms": round(float(current_speed), 2),
            "density_kgm3": round(float(density), 2),
            "mixed_layer_depth_m": round(float(mld), 1),
            "thermocline_depth_m": round(float(thermocline_depth), 1),
            "oxygen_mg_l": round(float(4.8 - (depth / 1000.0) * 2.1), 2),
            "turbidity_ntu": round(float(max(0.1, 0.95 - (depth / 200.0))), 2)
        }

    @classmethod
    def get_vertical_profile(cls, lat: float, lon: float, depths: Optional[List[float]] = None) -> pd.DataFrame:
        """Returns depth vs temperature, salinity, and density DataFrame."""
        if depths is None:
            depths = DEPTH_LEVELS

        surface_temp = 29.5 - 0.14 * abs(lat) + 0.05 * (lon - 65.0)
        deep_temp = 3.5
        k = 0.016 + 0.003 * np.cos(np.radians(lat))

        temps = []
        salinities = []
        densities = []

        for d in depths:
            t = deep_temp + (surface_temp - deep_temp) * np.exp(-k * d) + np.random.normal(0, 0.03)
            s = 35.0 + 0.6 * np.sin(np.radians(lat)) - (d / 2000.0) * 0.3
            rho = 1024.0 + 0.7 * (s - 35.0) - 0.2 * (t - 20.0) + 0.0045 * d
            temps.append(round(float(t), 2))
            salinities.append(round(float(s), 2))
            densities.append(round(float(rho), 2))

        return pd.DataFrame({
            "Depth (m)": depths,
            "Temperature (°C)": temps,
            "Salinity (PSU)": salinities,
            "Density (kg/m³)": densities
        })

    @classmethod
    def get_transect(
        cls,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        num_points: int = 50,
        depths: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Calculates vertical 2D temperature slice across a geographic transect line."""
        if depths is None:
            depths = DEPTH_LEVELS

        lats = np.linspace(start_lat, end_lat, num_points)
        lons = np.linspace(start_lon, end_lon, num_points)

        # Calculate approximate distance along transect
        R = 6371.0  # Earth radius in km
        dlat = np.radians(end_lat - start_lat)
        dlon = np.radians(end_lon - start_lon)
        a = np.sin(dlat / 2)**2 + np.cos(np.radians(start_lat)) * np.cos(np.radians(end_lat)) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        total_dist_km = R * c
        distances = np.linspace(0, total_dist_km, num_points).round(1).tolist()

        # Matrix: (len(depths), num_points)
        matrix = []
        for d in depths:
            row = []
            for i in range(num_points):
                pt = cls.get_point_details(lats[i], lons[i], depth=d)
                row.append(pt["temperature_c"])
            matrix.append(row)

        return {
            "num_points": num_points,
            "distances_km": distances,
            "lats": lats.round(3).tolist(),
            "lons": lons.round(3).tolist(),
            "depths": depths,
            "temperature_matrix": matrix
        }
