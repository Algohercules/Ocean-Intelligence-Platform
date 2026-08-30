"""
backend/services/copernicus_service.py
======================================
Copernicus Marine Service NetCDF loader, parser, and spatial interpolator.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from backend.config import RAW_DATA_DIR, DATA_DIR


class CopernicusService:
    """
    Service to locate, parse, and extract real Copernicus Marine NetCDF datasets (thetao, uo, vo).
    """

    @staticmethod
    def get_nc_files() -> List[Path]:
        """Find all NetCDF files in data/raw and data/copernicus."""
        search_dirs = [RAW_DATA_DIR, DATA_DIR / "copernicus", DATA_DIR]
        found_files = []
        for d in search_dirs:
            if d.exists():
                found_files.extend(list(d.glob("*.nc")))
        return sorted(list(set(found_files)))

    @classmethod
    def load_variable_snapshot(
        cls,
        var_name: str = "thetao",
        depth_idx: int = 0,
        time_idx: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Loads a snapshot of a 3D/4D ocean variable from available NetCDF files.
        """
        nc_files = cls.get_nc_files()
        if not nc_files:
            return None

        # Look for file matching var_name
        target_file = None
        for f in nc_files:
            if var_name.lower() in f.name.lower():
                target_file = f
                break
        if target_file is None and nc_files:
            target_file = nc_files[0]

        try:
            import netCDF4
            with netCDF4.Dataset(target_file, "r") as ds:
                # Find variable name in dataset
                found_var = None
                for v in [var_name, "thetao", "uo", "vo", "SST", "SSH", "uSSW", "vSSW", "ST"]:
                    if v in ds.variables:
                        found_var = v
                        break
                if not found_var:
                    # Pick first non-dimension variable
                    non_dim_vars = [k for k in ds.variables.keys() if k not in ["lat", "latitude", "lon", "longitude", "depth", "time"]]
                    if non_dim_vars:
                        found_var = non_dim_vars[0]

                if not found_var:
                    return None

                var_data = ds.variables[found_var]
                dims = var_data.dimensions
                shape = var_data.shape

                # Extract slice
                if len(shape) == 4:  # (time, depth, lat, lon)
                    t = min(time_idx, shape[0] - 1)
                    d = min(depth_idx, shape[1] - 1)
                    arr = var_data[t, d, :, :]
                elif len(shape) == 3:  # (time, lat, lon) or (depth, lat, lon)
                    t = min(time_idx, shape[0] - 1)
                    arr = var_data[t, :, :]
                elif len(shape) == 2:  # (lat, lon)
                    arr = var_data[:, :]
                else:
                    arr = np.array(var_data)

                # Extract coords if available
                lats, lons = None, None
                for lat_key in ["latitude", "lat"]:
                    if lat_key in ds.variables:
                        lats = np.array(ds.variables[lat_key][:])
                        break
                for lon_key in ["longitude", "lon"]:
                    if lon_key in ds.variables:
                        lons = np.array(ds.variables[lon_key][:])
                        break

                arr = np.ma.filled(arr, fill_value=np.nan)
                valid_vals = arr[~np.isnan(arr)]
                min_v = float(np.min(valid_vals)) if len(valid_vals) > 0 else 0.0
                max_v = float(np.max(valid_vals)) if len(valid_vals) > 0 else 30.0
                mean_v = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 20.0

                return {
                    "filename": target_file.name,
                    "variable": found_var,
                    "shape": list(arr.shape),
                    "min": min_v,
                    "max": max_v,
                    "mean": mean_v,
                    "lats": lats.tolist() if lats is not None else [],
                    "lons": lons.tolist() if lons is not None else [],
                    "data": np.nan_to_num(arr, nan=0.0).tolist()
                }
        except Exception as e:
            print(f"[CopernicusService] Error parsing NetCDF {target_file}: {e}")
            return None
