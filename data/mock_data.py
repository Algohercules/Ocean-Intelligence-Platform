"""
data/mock_data.py
=================
Indian Ocean Intelligence Platform - Mock Data Service & Backend Integration Layer.
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# Default spatial boundaries for Indian Ocean
BOUNDS = {
    'lat_min': -30.0,
    'lat_max': 25.0,
    'lon_min': 35.0,
    'lon_max': 105.0
}

DEPTH_LEVELS = [0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 700, 850, 1000]

REGIONS = {
    'All Indian Ocean': {'lat': (-30.0, 25.0), 'lon': (35.0, 105.0), 'center': [10.0, 78.0], 'zoom': 4.3},
    'Indian Ocean': {'lat': (-30.0, 25.0), 'lon': (35.0, 105.0), 'center': [10.0, 78.0], 'zoom': 4.3},
    'Arabian Sea': {'lat': (5.0, 25.0), 'lon': (45.0, 77.0), 'center': [15.0, 62.0], 'zoom': 5.0},
    'Bay of Bengal': {'lat': (5.0, 22.0), 'lon': (78.0, 98.0), 'center': [14.0, 88.0], 'zoom': 5.0},
    'Equatorial Indian Ocean': {'lat': (-10.0, 5.0), 'lon': (50.0, 95.0), 'center': [-2.0, 75.0], 'zoom': 4.6},
    'Selected Area': {'lat': (10.0, 20.0), 'lon': (60.0, 70.0), 'center': [15.0, 65.0], 'zoom': 5.2}
}


@st.cache_data(show_spinner=False)
def get_temperature_map(dataset="ARGO vs GLORYS", variable="Temperature (°C)", depth=75, date_str="2024-05-20", region="All Indian Ocean"):
    """
    BACKEND PLACEHOLDER: Fetch 2D temperature/spatial grid dataframe for ocean maps.
    Strictly filters out mainland India land points so ocean temperatures fit ocean waters.
    """
    np.random.seed(42 + int(depth))
    
    reg_info = REGIONS.get(region, REGIONS['All Indian Ocean'])
    lat_min, lat_max = reg_info['lat']
    lon_min, lon_max = reg_info['lon']
    
    # Generate spatial grid points
    lats = np.linspace(lat_min, lat_max, 45)
    lons = np.linspace(lon_min, lon_max, 55)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    flat_lat = lat_grid.flatten()
    flat_lon = lon_grid.flatten()
    
    # Land Mask Filter: Exclude mainland India inland coordinates (approx 14°N - 28°N, 77°E - 85°E)
    is_india_land = (flat_lat >= 14.0) & (flat_lat <= 28.0) & (flat_lon >= 77.0) & (flat_lon <= 85.0)
    # Exclude Africa inland (approx lat -30 to 12, lon 30 to 38)
    is_africa_land = (flat_lat >= -30.0) & (flat_lat <= 12.0) & (flat_lon >= 30.0) & (flat_lon <= 38.0)
    
    ocean_mask = ~(is_india_land | is_africa_land)
    
    flat_lat = flat_lat[ocean_mask]
    flat_lon = flat_lon[ocean_mask]
    
    # Realistic ocean temperature decay with depth
    base_surface = 28.5 - 0.2 * np.abs(flat_lat) + 1.2 * np.exp(-((flat_lat - 5)**2 + (flat_lon - 80)**2) / 400.0)
    decay_factor = np.exp(-depth / 220.0)
    deep_water = 4.2
    
    temp = deep_water + (base_surface - deep_water) * decay_factor
    noise = np.random.normal(0, 0.35, len(flat_lat))
    temp = np.clip(temp + noise, 4.0, 32.0)
    anomaly = np.random.normal(0.8, 0.5, len(flat_lat))
    
    # RGB Color mapping for PyDeck or scatter points
    norm_temp = np.clip((temp - 4.0) / (32.0 - 4.0), 0.0, 1.0)
    r = np.clip((norm_temp * 255 * 1.4).astype(int), 0, 255)
    g = np.clip(((1.0 - np.abs(norm_temp - 0.5) * 2) * 255).astype(int), 0, 255)
    b = np.clip(((1.0 - norm_temp) * 255 * 1.2).astype(int), 0, 255)
    
    df = pd.DataFrame({
        'latitude': flat_lat,
        'longitude': flat_lon,
        'temperature': np.round(temp, 2),
        'anomaly': np.round(anomaly, 2),
        'r': r, 'g': g, 'b': b,
        'color': [[int(r[i]), int(g[i]), int(b[i]), 200] for i in range(len(temp))]
    })
    
    return df


@st.cache_data(show_spinner=False)
def get_argo_floats(region="All Indian Ocean"):
    """
    BACKEND PLACEHOLDER: Fetch active ARGO float metadata and coordinates.
    """
    np.random.seed(101)
    n_floats = 124
    
    reg_info = REGIONS.get(region, REGIONS['All Indian Ocean'])
    lat_min, lat_max = reg_info['lat']
    lon_min, lon_max = reg_info['lon']
    
    lats = np.random.uniform(lat_min + 2, lat_max - 2, n_floats)
    lons = np.random.uniform(lon_min + 2, lon_max - 2, n_floats)
    
    # Exclude land points for floats
    is_india_land = (lats >= 8.5) & (lats <= 28.0) & (lons >= 73.0) & (lons <= 87.0)
    lats = np.where(is_india_land, lats - 12.0, lats)
    
    float_ids = [f"WMO_{6903000 + i}" for i in range(n_floats)]
    last_updates = [(datetime.now() - timedelta(days=int(np.random.randint(0, 10)))).strftime("%Y-%m-%d") for _ in range(n_floats)]
    surface_temps = np.round(np.random.uniform(24.5, 30.5, n_floats), 2)
    depth_1000_temps = np.round(np.random.uniform(6.0, 8.5, n_floats), 2)
    cycle_numbers = np.random.randint(15, 240, n_floats)
    
    df = pd.DataFrame({
        'float_id': float_ids,
        'latitude': np.round(lats, 3),
        'longitude': np.round(lons, 3),
        'last_update': last_updates,
        'surface_temp': surface_temps,
        'temp_1000m': depth_1000_temps,
        'cycle_number': cycle_numbers,
        'status': np.random.choice(['Active', 'Reporting', 'Calibrating'], n_floats, p=[0.8, 0.15, 0.05])
    })
    return df


@st.cache_data(show_spinner=False)
def get_selected_area_stats(region="Arabian Sea", depth=75, bounds=None):
    return {
        "area_name": "Selected Area Box A-1",
        "bounds": "10.0°N – 20.0°N, 60.0°E – 70.0°E",
        "region": region if region != "Selected Area" else "Arabian Sea",
        "area_km2": "1.23 Million km²",
        "avg_temp": 25.2,
        "min_temp": 23.8,
        "max_temp": 27.9,
        "anomaly": +1.4,
        "status": "Moderate Warm Anomaly",
        "argo_profiles": 124,
        "data_coverage": "68%",
        "ai_coverage": "32%",
        "selected_location": {
            "latitude": "15.2° N",
            "longitude": "70.4° E",
            "depth": f"{depth} m",
            "date": "20 May 2024",
            "temperature": "25.2 °C"
        }
    }


@st.cache_data(show_spinner=False)
def get_point_details(lat=15.0, lon=65.0, depth=75):
    """
    BACKEND PLACEHOLDER: Continuous spatial lookup for ANY arbitrary (lat, lon) coordinate.
    """
    lat_str = f"{abs(lat):.2f}° {'N' if lat >= 0 else 'S'}"
    lon_str = f"{abs(lon):.2f}° {'E' if lon >= 0 else 'W'}"
    
    # Sub-basin region detection
    if lat >= 5.0 and lon <= 77.0:
        reg_name = "Arabian Sea"
    elif lat >= 5.0 and lon > 77.0:
        reg_name = "Bay of Bengal"
    elif lat >= -10.0 and lat < 5.0:
        reg_name = "Equatorial Indian Ocean"
    else:
        reg_name = "Southern Indian Ocean"
        
    # Temperature calculation for coordinate
    base_surf = 28.5 - 0.18 * abs(lat) + 1.1 * np.cos(lon / 8.0)
    decay_factor = np.exp(-depth / 220.0)
    deep_water = 4.2
    depth_temp = np.round(deep_water + (base_surf - deep_water) * decay_factor, 1)
    surf_temp = np.round(base_surf, 1)
    min_temp = np.round(deep_water + 1.8, 1)
    max_temp = np.round(base_surf + 1.4, 1)
    anomaly = np.round(0.8 + 0.6 * np.sin((lat + lon) / 5.0), 1)
    
    # Nearest ARGO float calculation
    seed_val = int(abs(lat * 100) + abs(lon * 100)) % 1000
    nearest_dist = (seed_val % 45) + 12
    float_wmo = f"WMO_{6903000 + (seed_val % 124)}"
    
    return {
        "area_name": f"Target Point ({lat_str}, {lon_str})",
        "bounds": f"Point: {lat_str}, {lon_str}",
        "region": reg_name,
        "area_km2": "Single Grid Coordinate",
        "avg_temp": depth_temp,
        "surface_temp": surf_temp,
        "min_temp": min_temp,
        "max_temp": max_temp,
        "anomaly": anomaly,
        "status": "Target Coordinate Active",
        "argo_profiles": 1,
        "nearest_argo_id": float_wmo,
        "nearest_argo_dist": f"{nearest_dist} km",
        "data_coverage": "94%",
        "ai_coverage": "100%",
        "searched_lat": lat,
        "searched_lon": lon,
        "selected_location": {
            "latitude": lat_str,
            "longitude": lon_str,
            "depth": f"{depth} m",
            "date": "20 May 2024",
            "temperature": f"{depth_temp} °C"
        }
    }


def get_avg_temp_by_depth(region="All Indian Ocean", date_str="2024-05-20"):
    base_temps = [28.6, 28.2, 27.8, 27.4, 26.4, 25.2, 23.7, 21.3, 19.5, 16.8, 14.8, 13.1, 9.8, 8.4, 7.5]
    mins = [27.1, 26.9, 26.5, 26.1, 25.0, 23.8, 22.0, 19.4, 17.6, 15.0, 13.0, 11.4, 8.6, 7.4, 6.7]
    maxs = [30.2, 30.0, 29.8, 29.5, 28.7, 27.9, 26.5, 24.1, 22.3, 19.2, 17.0, 15.2, 11.4, 9.6, 8.5]
    anomalies = [+0.9, +0.8, +0.8, +0.7, +0.7, +1.4, +1.3, +1.2, +1.0, +0.8, +0.7, +0.6, +0.4, +0.3, +0.2]
    
    df = pd.DataFrame({
        'Depth (m)': DEPTH_LEVELS,
        'Avg Temp (°C)': base_temps,
        'Min Temp (°C)': mins,
        'Max Temp (°C)': maxs,
        'Anomaly (°C)': anomalies
    })
    return df


def get_temperature_profile(lat=15.2, lon=70.4):
    depths = np.array(DEPTH_LEVELS)
    temp = 4.5 + (28.8 - 4.5) / (1.0 + (depths / 160.0)**1.4)
    temp = np.round(temp + np.random.normal(0, 0.15, len(depths)), 2)
    return pd.DataFrame({'Depth (m)': depths, 'Temperature (°C)': temp})


def get_time_series_data(region="All Indian Ocean", start_year=2000, end_year=2025):
    years = np.arange(start_year, end_year + 1)
    dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='ME')
    t_val = np.linspace(0, len(dates)/12, len(dates))
    trend = 0.068 * t_val
    seasonal = 1.1 * np.sin(2 * np.pi * t_val + 0.5)
    cycle = 0.6 * np.cos(2 * np.pi * t_val / 4.2)
    noise = np.random.normal(0, 0.22, len(dates))
    temp_series = 27.2 + trend + seasonal + cycle + noise
    
    df = pd.DataFrame({
        'Date': dates,
        'Year': dates.year,
        'Temperature (°C)': np.round(temp_series, 2),
        'Trend (°C)': np.round(27.2 + trend, 2)
    })
    
    stats = {
        'trend_str': "+0.68 °C / decade",
        'max_temp': f"{np.max(temp_series):.1f} °C",
        'min_temp': f"{np.min(temp_series):.1f} °C",
        'mean_temp': f"{np.mean(temp_series):.1f} °C"
    }
    return df, stats


def get_argo_vs_glorys_profile():
    depths = np.array(DEPTH_LEVELS)
    argo_temp = 4.5 + (28.8 - 4.5) / (1.0 + (depths / 160.0)**1.4)
    glorys_temp = argo_temp + np.array([0.05, 0.12, 0.18, 0.25, 0.31, 0.28, 0.20, 0.15, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01, 0.0])
    
    df = pd.DataFrame({
        'Depth (m)': depths,
        'ARGO (°C)': np.round(argo_temp, 2),
        'GLORYS (°C)': np.round(glorys_temp, 2),
        'Difference (°C)': np.round(glorys_temp - argo_temp, 2)
    })
    
    stats = {
        'rmse': "0.42 °C",
        'mae': "0.31 °C",
        'r2': "0.94",
        'bias': "+0.08 °C"
    }
    return df, stats


def get_ai_reconstruction_data(depth=200):
    np.random.seed(88)
    lats = np.linspace(5, 22, 35)
    lons = np.linspace(60, 95, 45)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    flat_lat = lat_grid.flatten()
    flat_lon = lon_grid.flatten()
    
    base_state = 24.5 - 0.18 * np.abs(flat_lat) + 0.8 * np.cos(flat_lon / 10.0)
    mask = np.random.choice([1, 0], len(flat_lat), p=[0.65, 0.35])
    observed_temp = np.where(mask == 1, base_state + np.random.normal(0, 0.2, len(flat_lat)), np.nan)
    ai_reconstructed_temp = base_state + np.random.normal(0, 0.15, len(flat_lat))
    diff = np.where(~np.isnan(observed_temp), ai_reconstructed_temp - observed_temp, 0.0)
    
    df_obs = pd.DataFrame({'latitude': flat_lat, 'longitude': flat_lon, 'temperature': np.round(observed_temp, 2)})
    df_ai = pd.DataFrame({'latitude': flat_lat, 'longitude': flat_lon, 'temperature': np.round(ai_reconstructed_temp, 2)})
    df_diff = pd.DataFrame({'latitude': flat_lat, 'longitude': flat_lon, 'temperature': np.round(diff, 2)})
    
    depths = np.array(DEPTH_LEVELS)
    full_profile = 4.5 + (28.8 - 4.5) / (1.0 + (depths / 160.0)**1.4)
    obs_profile = np.where(depths <= 100, full_profile + np.random.normal(0, 0.1, len(depths)), np.nan)
    ai_profile = full_profile + np.random.normal(0, 0.08, len(depths))
    missing_profile = np.where(depths > 100, full_profile, np.nan)
    
    df_profile = pd.DataFrame({
        'Depth (m)': depths,
        'Observed (0-100m)': np.round(obs_profile, 2),
        'AI Predicted (150-1000m)': np.round(ai_profile, 2),
        'Missing Reference': np.round(missing_profile, 2)
    })
    
    return df_obs, df_ai, df_diff, df_profile


def get_validation_metrics():
    return {'rmse': "0.42 °C", 'mae': "0.31 °C", 'r2': "0.94", 'bias': "+0.08 °C"}


def get_heatwave_data(region="All Indian Ocean"):
    np.random.seed(999)
    lats = np.linspace(-25, 20, 40)
    lons = np.linspace(40, 100, 50)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    flat_lat = lat_grid.flatten()
    flat_lon = lon_grid.flatten()
    
    anom = 1.2 + 2.5 * np.exp(-((flat_lat - 15)**2 + (flat_lon - 65)**2) / 80.0)
    anom += np.random.normal(0, 0.3, len(flat_lat))
    
    df_heatwave = pd.DataFrame({'latitude': flat_lat, 'longitude': flat_lon, 'anomaly': np.round(anom, 2)})
    events = pd.DataFrame({
        'Event ID': ['MHW-2024-01', 'MHW-2024-02', 'MHW-2024-03', 'MHW-2024-04'],
        'Region': ['Arabian Sea', 'Bay of Bengal', 'Equatorial IO', 'Mozambique Channel'],
        'Severity': ['Extreme (Category IV)', 'Strong (Category III)', 'Moderate (Category II)', 'Strong (Category III)'],
        'Max Anomaly (°C)': ['+3.8 °C', '+3.1 °C', '+2.2 °C', '+2.9 °C'],
        'Duration (Days)': [28, 19, 14, 22],
        'Area Impacted (km²)': ['450,000', '320,000', '180,000', '290,000']
    })
    return df_heatwave, events


def get_regional_analysis_data():
    regions = ['Arabian Sea', 'Bay of Bengal', 'Equatorial Indian Ocean', 'Southern Indian Ocean']
    return pd.DataFrame({
        'Region': regions,
        'Avg Temp (°C)': [26.8, 28.2, 28.5, 21.4],
        'Min Temp (°C)': [23.1, 25.4, 26.2, 14.8],
        'Max Temp (°C)': [30.4, 31.2, 30.8, 26.2],
        'Anomaly (°C)': [+1.4, +1.8, +0.9, +0.4],
        'Heat Content (10⁹ J/m²)': [1.45, 1.62, 1.58, 1.12],
        'ARGO Floats': [42, 38, 26, 18]
    })


def get_transect_data(slice_type="Zonal (Latitude Transect)", target_val=15.0):
    depths = np.array(DEPTH_LEVELS)
    coords = np.linspace(35.0, 105.0, 40) if "Latitude" in slice_type else np.linspace(-30.0, 25.0, 40)
    coord_name = "Longitude (°E)" if "Latitude" in slice_type else "Latitude (°N)"
    grid_coords, grid_depths = np.meshgrid(coords, depths)
    eddy = 4.0 * np.sin(grid_coords / 6.0)
    temp_grid = 4.5 + (28.5 - 4.5) / (1.0 + ((grid_depths + eddy) / 160.0)**1.4)
    return coords, depths, np.round(temp_grid, 2), coord_name


def get_mhw_timeseries_data(region="Arabian Sea"):
    dates = pd.date_range(start="2024-01-01", end="2024-05-20", freq="D")
    t = np.linspace(0, len(dates), len(dates))
    climatology = 26.5 + 1.2 * np.sin(2 * np.pi * t / 365.0)
    thresh_90th = climatology + 1.5
    spike = 3.5 * np.exp(-((t - 110)**2) / 120.0)
    observed = climatology + spike + np.random.normal(0, 0.25, len(dates))
    return pd.DataFrame({
        'Date': dates,
        'Observed Temp (°C)': np.round(observed, 2),
        'Climatology Mean (°C)': np.round(climatology, 2),
        '90th Percentile Threshold (°C)': np.round(thresh_90th, 2),
        'MHW Active': observed > thresh_90th
    })


def get_ai_forecast_timeseries(lat=15.0, lon=65.0, depth=75, horizon_days=7, variable="Temperature"):
    from datetime import datetime, timedelta
    base_date = datetime(2024, 5, 20)
    
    hist_dates = [base_date - timedelta(days=i) for i in range(14, 0, -1)]
    fc_dates = [base_date + timedelta(days=i) for i in range(0, horizon_days + 1)]
    
    depth_factor = 4.5 + (28.8 - 4.5) / (1.0 + (depth / 160.0)**1.4)
    lat_adj = -0.15 * (lat - 15.0) + 0.05 * (lon - 65.0)
    curr_temp = depth_factor + lat_adj
    
    t_hist = np.linspace(-14, 0, len(hist_dates))
    hist_obs = curr_temp + 0.3 * np.sin(t_hist / 3.0) + np.random.normal(0, 0.08, len(hist_dates))
    glorys_hist = hist_obs - 0.12 + np.random.normal(0, 0.05, len(hist_dates))
    
    t_fc = np.linspace(0, horizon_days, len(fc_dates))
    trend = 0.12 * (t_fc / 7.0)
    ai_fc = curr_temp + trend + 0.15 * np.sin(t_fc / 4.0)
    glorys_fc = ai_fc - 0.15 + np.random.normal(0, 0.05, len(fc_dates))
    
    uncertainty = 0.1 + 0.05 * t_fc
    upper_bound = ai_fc + uncertainty
    lower_bound = ai_fc - uncertainty
    
    df_hist = pd.DataFrame({
        'Date': hist_dates,
        'Historical ARGO (°C)': np.round(hist_obs, 2),
        'GLORYS Baseline (°C)': np.round(glorys_hist, 2),
        'AI Forecast (°C)': [np.nan] * len(hist_dates),
        'Upper Bound (°C)': [np.nan] * len(hist_dates),
        'Lower Bound (°C)': [np.nan] * len(hist_dates),
        'Type': ['Historical'] * len(hist_dates)
    })
    
    df_fc = pd.DataFrame({
        'Date': fc_dates,
        'Historical ARGO (°C)': [np.nan] * len(fc_dates),
        'GLORYS Baseline (°C)': np.round(glorys_fc, 2),
        'AI Forecast (°C)': np.round(ai_fc, 2),
        'Upper Bound (°C)': np.round(upper_bound, 2),
        'Lower Bound (°C)': np.round(lower_bound, 2),
        'Type': ['Forecast'] * len(fc_dates)
    })
    
    df_combined = pd.concat([df_hist, df_fc], ignore_index=True)
    
    current_val = float(hist_obs[-1])
    predicted_val = float(ai_fc[-1])
    change_val = predicted_val - current_val
    confidence_pct = int(max(60, 96 - 1.2 * horizon_days))
    anomaly_val = float(change_val + 0.6)
    
    stats = {
        'current_temp': f"{current_val:.1f} °C",
        'predicted_temp': f"{predicted_val:.1f} °C",
        'change_temp': f"{'+' if change_val >= 0 else ''}{change_val:.1f} °C",
        'confidence': f"{confidence_pct} %",
        'anomaly': f"{'+' if anomaly_val >= 0 else ''}{anomaly_val:.1f} °C",
        'forecast_horizon': f"{horizon_days} Days",
        'selected_depth': f"{depth} m",
        'data_source': "ARGO + GLORYS + AI",
        'forecast_start_date': base_date.strftime("%Y-%m-%d")
    }
    
    return df_combined, stats


def get_ai_depth_heatmap_matrix(lat=15.0, lon=65.0, horizon_days=7):
    from datetime import datetime, timedelta
    base_date = datetime(2024, 5, 20)
    fc_dates = [(base_date + timedelta(days=i)).strftime("%b %d") for i in range(horizon_days + 1)]
    depths = np.array(DEPTH_LEVELS)
    
    date_grid, depth_grid = np.meshgrid(np.arange(len(fc_dates)), depths)
    temp_grid = 4.5 + (28.8 - 4.5) / (1.0 + (depth_grid / 160.0)**1.4) + 0.15 * date_grid / max(1, horizon_days)
    
    return fc_dates, depths, np.round(temp_grid, 2)

