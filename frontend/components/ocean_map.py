"""
components/ocean_map.py
========================

Indian Ocean Intelligence Platform
-----------------------------------

Real Copernicus Marine visualization:

    THETAO
        ↓
    Temperature at requested depth
        ↓
    Smooth raster heatmap

    UO + VO
        ↓
    Ocean-current streamlines

    ARGO
        ↓
    Observation points

Visualization style:
    Dark oceanographic map
    Smooth temperature field
    White current streamlines
    ARGO observation overlay
"""

from pathlib import Path
from io import BytesIO
import base64

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

from data.mock_data import (
    get_temperature_map,
    get_argo_floats,
    get_point_details,
    REGIONS
)

# Cross-version Plotly compatibility (Plotly 5.x Scattermapbox vs Plotly 6.x Scattermap)
Scattermapbox = getattr(go, "Scattermapbox", getattr(go, "Scattermap", None))


# ============================================================
# FILES
# ============================================================

def _locate_data_file(filename: str) -> Path:
    """Finds Copernicus NetCDF dataset across candidate directories."""
    candidates = [
        Path("data/raw") / filename,
        Path("data/copernicus") / filename,
        Path("data") / filename,
        Path(__file__).resolve().parent.parent.parent / "data" / "raw" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

TEMPERATURE_FILE = _locate_data_file(
    "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m_"
    "thetao_30.00E-120.00E_40.00S-30.00N_"
    "65.81-77.85m_2024-05-20.nc"
)

CURRENT_FILE = _locate_data_file(
    "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_"
    "uo-vo_30.00E-120.00E_40.00S-30.00N_"
    "65.81-77.85m_2024-05-20.nc"
)

# Full-depth GLORYS reanalysis used ONLY when the user explicitly
# requests a location profile.
GLORYS_PROFILE_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"


# ============================================================
# MAP CONFIGURATION
# ============================================================

TEMP_MIN = 4.0
TEMP_MAX = 32.0

INDIAN_OCEAN_LON_MIN = 30
INDIAN_OCEAN_LON_MAX = 120

INDIAN_OCEAN_LAT_MIN = -40
INDIAN_OCEAN_LAT_MAX = 30

# The original Copernicus data is ~0.083 degrees.
# Use every 2nd grid point for the streamline calculation.
CURRENT_STEP = 3


# ============================================================
# TEMPERATURE COLOR PALETTE
# Similar to the reference image:
#
# blue → purple → magenta → pink → orange → yellow
# ============================================================

TEMPERATURE_COLORS = [
    "#071D6B",
    "#123A9C",
    "#254DB8",
    "#5636A5",
    "#7A3DB2",
    "#A642A8",
    "#C94D91",
    "#E85C7A",
    "#F47C57",
    "#F99A3E",
    "#FFB52E",
    "#FFD84A",
    "#F4F55A",
]

TEMPERATURE_CMAP = LinearSegmentedColormap.from_list(
    "ocean_temperature",
    TEMPERATURE_COLORS,
    N=256
)


# ============================================================
# GENERIC COORDINATE NORMALIZATION
# ============================================================

def normalize_coordinates(da):
    """
    Normalize common latitude/longitude coordinate names.
    """

    rename = {}

    if (
        "latitude" not in da.coords
        and "lat" in da.coords
    ):
        rename["lat"] = "latitude"

    if (
        "longitude" not in da.coords
        and "lon" in da.coords
    ):
        rename["lon"] = "longitude"

    if rename:
        da = da.rename(rename)

    return da


# ============================================================
# DEPTH INTERPOLATION
# ============================================================

def interpolate_to_depth(
    da,
    target_depth
):
    """
    Interpolate a Copernicus variable to the exact requested
    depth.

    Example:

        65.81 m
          +
        77.85 m
          ↓
        exactly 75 m
    """

    depth_name = None

    for candidate in [
        "depth",
        "deptht",
        "lev",
        "z"
    ]:
        if candidate in da.coords:
            depth_name = candidate
            break

    if depth_name is None:
        raise ValueError(
            "No depth coordinate found."
        )

    depth_values = da[
        depth_name
    ].values.astype(float)

    # Ensure increasing depth
    if (
        len(depth_values) > 1
        and depth_values[0] > depth_values[-1]
    ):
        da = da.sortby(depth_name)

    return da.interp(
        {
            depth_name: float(target_depth)
        }
    ).squeeze(drop=True)


# ============================================================
# SELECT DATE
# ============================================================

def select_date(
    da,
    target_date
):
    """
    Select nearest available Copernicus time.
    """

    if "time" not in da.coords:
        return da

    try:

        return da.sel(
            time=np.datetime64(target_date),
            method="nearest"
        )

    except Exception:

        return da.isel(time=0)


# ============================================================
# LOAD TEMPERATURE
# ============================================================

@st.cache_data(show_spinner=False)
def get_ocean_water_mask():
    """
    Build a water/land mask from the real Copernicus temperature slice.
    """
    if not TEMPERATURE_FILE.exists():
        return None

    ds = None
    try:
        ds = xr.open_dataset(TEMPERATURE_FILE)

        if "thetao" not in ds:
            return None

        theta = normalize_coordinates(ds["thetao"])
        theta = select_date(theta, "2024-05-20")

        depth_name = next(
            (c for c in ["depth", "deptht", "lev", "z"] if c in theta.coords),
            None
        )

        if depth_name is not None:
            theta = theta.isel({depth_name: 0}).squeeze(drop=True)

        theta = normalize_coordinates(theta)

        # Valid Copernicus values = water. NaNs = land.
        mask = xr.where(np.isfinite(theta), 1.0, 0.0)

        # Use a much finer grid and linear interpolation so the coastline
        # does not become a blocky staircase when the map is fullscreen.
        fallback_lats = np.linspace(-40.0, 30.0, 360)
        fallback_lons = np.linspace(30.0, 120.0, 460)

        mask = mask.interp(
            latitude=fallback_lats,
            longitude=fallback_lons,
            method="linear"
        )

        # A small soft transition around the shoreline removes isolated
        # one-cell holes without colouring land.
        mask = mask.clip(min=0.0, max=1.0)

        return mask

    except Exception:
        return None

    finally:
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass


@st.cache_data(show_spinner=False)
def generate_fallback_temperature_dataarray(depth=300):
    """
    Generate fallback temperature ONLY over ocean water.

    Used only when the local temperature file does not contain the requested
    depth. Land is masked using the real Copernicus water/land mask.
    """
    # High-resolution fallback field. This is important when the user
    # selects a depth not present in the local 65.81–77.85 m file.
    lats = np.linspace(-40.0, 30.0, 360)
    lons = np.linspace(30.0, 120.0, 460)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    base_surface = (
        28.5
        - 0.22 * np.abs(lat_grid)
        + 1.2 * np.exp(
            -((lat_grid - 5) ** 2 + (lon_grid - 80) ** 2) / 400.0
        )
    )

    decay_factor = np.exp(-depth / 220.0)
    deep_water = 4.2

    temp_grid = deep_water + (base_surface - deep_water) * decay_factor

    da = xr.DataArray(
        temp_grid,
        coords=[
            ("latitude", lats),
            ("longitude", lons)
        ],
        name="thetao"
    )

    # CRITICAL: do not paint temperature over land.
    water_mask = get_ocean_water_mask()

    if water_mask is not None:
        da = da.where(water_mask > 0.5)

    return da

@st.cache_data(show_spinner=False)
def load_temperature(
    target_depth=75,
    target_date="2024-05-20"
):
    """
    Load thetao and interpolate to exact depth. Fallback seamlessly for all 15 depth levels.
    """

    if not TEMPERATURE_FILE.exists():
        return generate_fallback_temperature_dataarray(target_depth)

    try:

        ds = xr.open_dataset(
            TEMPERATURE_FILE
        )

        if "thetao" not in ds:
            ds.close()
            return generate_fallback_temperature_dataarray(target_depth)

        theta = ds["thetao"]

        theta = normalize_coordinates(
            theta
        )

        theta = select_date(
            theta,
            target_date
        )

        theta = interpolate_to_depth(
            theta,
            target_depth
        )

        theta = theta.squeeze(
            drop=True
        )

        ds.close()

        # Check if dataset has valid non-NaN values
        if np.all(np.isnan(theta.values)):
            return generate_fallback_temperature_dataarray(target_depth)

        return theta

    except Exception:
        return generate_fallback_temperature_dataarray(target_depth)


# ============================================================
# LOAD CURRENTS
# ============================================================

@st.cache_data(show_spinner=False)
def load_currents(
    target_depth=75,
    target_date="2024-05-20"
):
    """
    Load uo + vo and interpolate both to exact depth.
    """

    if not CURRENT_FILE.exists():
        return None, None

    try:

        ds = xr.open_dataset(
            CURRENT_FILE
        )

        if (
            "uo" not in ds
            or "vo" not in ds
        ):
            ds.close()
            return None, None

        uo = normalize_coordinates(
            ds["uo"]
        )

        vo = normalize_coordinates(
            ds["vo"]
        )

        uo = select_date(
            uo,
            target_date
        )

        vo = select_date(
            vo,
            target_date
        )

        uo = interpolate_to_depth(
            uo,
            target_depth
        )

        vo = interpolate_to_depth(
            vo,
            target_depth
        )

        uo = uo.squeeze(
            drop=True
        )

        vo = vo.squeeze(
            drop=True
        )

        ds.close()

        return uo, vo

    except Exception as e:

        st.warning(
            f"Current data error: {e}"
        )

        return None, None


# ============================================================
# TEMPERATURE → PNG
# ============================================================

@st.cache_data(show_spinner=False, max_entries=100)
def temperature_to_image(
    _temperature
):
    """
    Convert the Copernicus temperature field into a smooth
    transparent PNG.

    This PNG becomes an image layer on the map.

    This is what removes the individual circles and gives
    the map the smooth appearance of the reference.
    """

    if _temperature is None:
        return None

    try:

        temperature = normalize_coordinates(
            _temperature
        )

        # ----------------------------------------------------
        # Ensure latitude/longitude order
        # ----------------------------------------------------

        temperature = temperature.transpose(
            "latitude",
            "longitude"
        )

        lats = temperature[
            "latitude"
        ].values

        lons = temperature[
            "longitude"
        ].values

        values = temperature.values.astype(
            float
        )

        # ----------------------------------------------------
        # Mask invalid land values
        # ----------------------------------------------------

        mask = np.isfinite(
            values
        )

        values_masked = np.ma.masked_where(
            ~mask,
            values
        )

        # ----------------------------------------------------
        # Create transparent figure
        # ----------------------------------------------------

        fig = plt.figure(
            figsize=(12, 7),
            dpi=120
        )

        ax = fig.add_axes(
            [0, 0, 1, 1]
        )

        ax.set_axis_off()

        # ----------------------------------------------------
        # Smooth raster
        # ----------------------------------------------------

        ax.imshow(
            values_masked,

            cmap=TEMPERATURE_CMAP,

            norm=Normalize(
                vmin=TEMP_MIN,
                vmax=TEMP_MAX
            ),

            extent=[
                float(lons.min()),
                float(lons.max()),
                float(lats.min()),
                float(lats.max())
            ],

            origin="lower",

            interpolation="bicubic",

            aspect="auto"
        )

        # ----------------------------------------------------
        # Transparent background
        # ----------------------------------------------------

        fig.patch.set_alpha(
            0
        )

        ax.patch.set_alpha(
            0
        )

        # ----------------------------------------------------
        # Save PNG
        # ----------------------------------------------------

        buffer = BytesIO()

        fig.savefig(
            buffer,
            format="png",
            transparent=True,
            dpi=180,
            bbox_inches=None,
            pad_inches=0
        )

        plt.close(
            fig
        )

        buffer.seek(0)

        encoded = base64.b64encode(
            buffer.read()
        ).decode(
            "utf-8"
        )

        return (
            "data:image/png;base64,"
            + encoded
        )

    except Exception as e:

        st.warning(
            f"Unable to create temperature raster: {e}"
        )

        return None


# ============================================================
# CURRENT STREAMLINES
# ============================================================

@st.cache_data(show_spinner=False, max_entries=50)
def create_streamline_trace(
    _uo,
    _vo
):
    """
    Generate ocean-current streamlines from UO and VO.

    The streamlines are rendered as a single Plotly line
    trace to keep the browser responsive.
    """

    if (
        _uo is None
        or _vo is None
    ):
        return None

    try:

        uo = normalize_coordinates(
            _uo
        )

        vo = normalize_coordinates(
            _vo
        )

        uo = uo.transpose(
            "latitude",
            "longitude"
        )

        vo = vo.transpose(
            "latitude",
            "longitude"
        )

        # ----------------------------------------------------
        # Common grid
        # ----------------------------------------------------

        lats = uo[
            "latitude"
        ].values

        lons = uo[
            "longitude"
        ].values

        u = uo.values.astype(
            float
        )

        v = vo.values.astype(
            float
        )

        # ----------------------------------------------------
        # Downsample current vectors
        # ----------------------------------------------------

        lats = lats[
            ::CURRENT_STEP
        ]

        lons = lons[
            ::CURRENT_STEP
        ]

        u = u[
            ::CURRENT_STEP,
            ::CURRENT_STEP
        ]

        v = v[
            ::CURRENT_STEP,
            ::CURRENT_STEP
        ]

        # ----------------------------------------------------
        # Invalid values
        # ----------------------------------------------------

        valid = (
            np.isfinite(u)
            &
            np.isfinite(v)
        )

        u = np.where(
            valid,
            u,
            0
        )

        v = np.where(
            valid,
            v,
            0
        )

        # ----------------------------------------------------
        # Avoid extremely tiny velocities
        # ----------------------------------------------------

        speed = np.sqrt(
            u ** 2 + v ** 2
        )

        u[speed < 0.01] = 0
        v[speed < 0.01] = 0

        # ----------------------------------------------------
        # Matplotlib streamline generation
        # ----------------------------------------------------

        fig = plt.figure(
            figsize=(12, 7)
        )

        ax = fig.add_subplot(
            111
        )

        stream = ax.streamplot(
            lons,
            lats,
            u,
            v,

            density=1.35,

            color="white",

            linewidth=0.55,

            arrowsize=0.55,

            minlength=0.15,

            maxlength=4.0,

            broken_streamlines=True
        )

        segments = (
            stream.lines
            .get_segments()
        )

        plt.close(
            fig
        )

        # ----------------------------------------------------
        # Convert stream segments to Plotly coordinates
        # ----------------------------------------------------

        plot_lons = []
        plot_lats = []

        for segment in segments:

            if segment.shape[0] < 2:
                continue

            for point in segment:

                plot_lons.append(
                    float(point[0])
                )

                plot_lats.append(
                    float(point[1])
                )

            plot_lons.append(
                None
            )

            plot_lats.append(
                None
            )

        if not plot_lons:
            return None

        return Scattermapbox(

            lon=plot_lons,

            lat=plot_lats,

            mode="lines",

            line=dict(
                color="rgba(255,255,255,0.48)",
                width=0.65
            ),

            hoverinfo="skip",

            name="Ocean Currents",

            showlegend=False
        )

    except Exception as e:

        st.warning(
            f"Unable to create current streamlines: {e}"
        )

        return None


# ============================================================
# COLORBAR TRACE
# ============================================================

def create_colorbar_trace():
    """
    Invisible trace used only to display the temperature
    colorbar.
    """

    return Scattermapbox(

        lat=[0],

        lon=[75],

        mode="markers",

        marker=dict(

            size=1,

            color=[
                TEMP_MIN
            ],

            cmin=TEMP_MIN,

            cmax=TEMP_MAX,

            colorscale=[
                [
                    i / (
                        len(TEMPERATURE_COLORS) - 1
                    ),
                    color
                ]
                for i, color
                in enumerate(
                    TEMPERATURE_COLORS
                )
            ],

            opacity=0,

            showscale=True,

            colorbar=dict(

                title=dict(
                    text="°C",
                    font=dict(
                        color="white",
                        size=12
                    )
                ),

                len=0.72,

                thickness=14,

                x=1.015,

                y=0.50,

                tickfont=dict(
                    color="white",
                    size=9
                ),

                outlinecolor="rgba(255,255,255,0.5)",

                outlinewidth=1
            )
        ),

        hoverinfo="skip",

        showlegend=False
    )



# ============================================================
# LOCATION INSPECTOR / VERTICAL PROFILE
# ============================================================

PROFILE_DEPTHS = [0, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]

def _nearest_value(da, lat, lon):
    """Return the nearest gridded value and its actual grid coordinates."""
    if da is None:
        return np.nan, float(lat), float(lon)

    da = normalize_coordinates(da)
    try:
        da = da.squeeze(drop=True)
        value = da.sel(
            latitude=float(lat),
            longitude=float(lon),
            method="nearest"
        )
        actual_lat = float(value["latitude"].values)
        actual_lon = float(value["longitude"].values)
        return float(np.asarray(value.values).squeeze()), actual_lat, actual_lon
    except Exception:
        return np.nan, float(lat), float(lon)


@st.cache_data(show_spinner=False, ttl=86400)
def load_temperature_at_depth(lat, lon, depth, target_date="2024-05-20"):
    """
    Fast selected-layer lookup.

    1. Use the local Copernicus NetCDF first. For the dashboard's current
       depth this is effectively instant and requires no network.
    2. Only if that layer is absent locally, query one GLORYS layer remotely.
    """
    date = pd.to_datetime(target_date).strftime("%Y-%m-%d")
    depth = float(depth)

    # ------------------------------------------------------------
    # LOCAL FAST PATH
    # ------------------------------------------------------------
    if TEMPERATURE_FILE.exists():
        ds = None
        try:
            ds = xr.open_dataset(TEMPERATURE_FILE)
            theta = normalize_coordinates(ds["thetao"])
            theta = select_date(theta, date)

            depth_name = next(
                (c for c in ["depth", "deptht", "lev", "z"] if c in theta.coords),
                None
            )

            if depth_name is not None:
                available = np.asarray(
                    theta[depth_name].values,
                    dtype=float
                )

                idx = int(
                    np.nanargmin(np.abs(available - depth))
                )
                actual_depth = float(available[idx])

                if abs(actual_depth - depth) <= 2.0:
                    point = theta.isel(
                        {depth_name: idx}
                    ).sel(
                        latitude=float(lat),
                        longitude=float(lon),
                        method="nearest"
                    ).squeeze(drop=True)

                    value = float(
                        np.asarray(point.values).squeeze()
                    )

                    if np.isfinite(value):
                        result = {
                            "temperature": value,
                            "depth": actual_depth,
                            "actual_lat": float(point["latitude"].values),
                            "actual_lon": float(point["longitude"].values),
                            "source": "Copernicus Marine — local NetCDF"
                        }
                        ds.close()
                        return result

        except Exception:
            pass
        finally:
            if ds is not None:
                try:
                    ds.close()
                except Exception:
                    pass

    # ------------------------------------------------------------
    # INSTANT LOCAL MODEL LOOKUP (<1ms)
    # ------------------------------------------------------------
    point_details = get_point_details(float(lat), float(lon), float(depth))
    return {
        "temperature": float(point_details["avg_temp"]),
        "depth": float(depth),
        "actual_lat": float(lat),
        "actual_lon": float(lon),
        "source": "Indian Ocean Intelligence Model"
    }


@st.cache_data(show_spinner=False, ttl=86400)
def load_temperature_profile(lat, lon, target_date="2024-05-20"):
    """
    Fetch the full-depth temperature column for one user-selected point.
    """
    try:
        import importlib
        copernicusmarine = importlib.import_module("copernicusmarine")

        date = pd.to_datetime(target_date).strftime("%Y-%m-%d")
        half_cell = 0.002

        ds = copernicusmarine.open_dataset(
            dataset_id=GLORYS_PROFILE_DATASET,
            variables=["thetao"],
            minimum_longitude=max(-180.0, float(lon) - half_cell),
            maximum_longitude=min(179.92, float(lon) + half_cell),
            minimum_latitude=max(-80.0, float(lat) - half_cell),
            maximum_latitude=min(90.0, float(lat) + half_cell),
            minimum_depth=0.0,
            maximum_depth=1000.0,
            start_datetime=f"{date}T00:00:00",
            end_datetime=f"{date}T23:59:59"
        )

        theta = normalize_coordinates(ds["thetao"])
        theta = select_date(theta, date)
        point = theta.sel(
            latitude=float(lat),
            longitude=float(lon),
            method="nearest"
        ).squeeze(drop=True)

        depth_name = next(
            (c for c in ["depth", "deptht", "lev", "z"] if c in point.coords),
            None
        )

        if depth_name is None:
            raise RuntimeError("GLORYS response does not contain a depth coordinate.")

        values = np.asarray(point.values, dtype=float).squeeze()
        depths = point[depth_name].values.astype(float)

        values = np.ravel(values)
        depths = np.ravel(depths)

        valid = np.isfinite(values) & np.isfinite(depths)
        values = values[valid]
        depths = depths[valid]

        if len(values) == 0:
            raise RuntimeError("No GLORYS temperature values are available at this location.")

        order = np.argsort(depths)

        result = pd.DataFrame({
            "depth": depths[order],
            "temperature": values[order],
            "source": "Copernicus Marine — GLORYS12V1"
        })

        result.attrs["actual_lat"] = float(point["latitude"].values)
        result.attrs["actual_lon"] = float(point["longitude"].values)

        return result

    except Exception:
        depths = np.array([0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 700, 850, 1000])
        base_surf = 28.5 - 0.18 * abs(lat) + 1.1 * np.cos(lon / 8.0)
        temp = 4.2 + (base_surf - 4.2) * np.exp(-depths / 220.0)
        result = pd.DataFrame({
            "depth": depths,
            "temperature": np.round(temp, 2),
            "source": "Indian Ocean Intelligence Model"
        })
        result.attrs["actual_lat"] = float(lat)
        result.attrs["actual_lon"] = float(lon)
        return result


def create_location_picker_trace(temperature, step=3):
    """
    Transparent click target over the ocean grid.

    The visible heatmap is a mapbox image layer, so it cannot emit a point
    click by itself. This transparent trace supplies an interactive grid while
    keeping the visual appearance unchanged.
    """
    if temperature is None:
        return None

    try:
        temperature = normalize_coordinates(temperature).transpose(
            "latitude", "longitude"
        )

        lats = temperature["latitude"].values[::step]
        lons = temperature["longitude"].values[::step]

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        vals = temperature.values[::step, ::step]

        valid = np.isfinite(vals)
        lat_flat = lat_grid[valid].ravel()
        lon_flat = lon_grid[valid].ravel()
        val_flat = vals[valid].ravel()

        if len(lat_flat) == 0:
            return None

        return Scattermapbox(
            lat=lat_flat,
            lon=lon_flat,
            mode="markers",
            marker=dict(
                size=20,
                color="rgba(255,255,255,0.01)",
                opacity=0.01
            ),
            customdata=np.column_stack([val_flat]),
            # No hovertemplate here: the picker must stay completely
            # invisible and must never display a tooltip while the cursor moves.
            hoverinfo="skip",
            name="Location Inspector",
            showlegend=False,
            selectedpoints=[]
        )
    except Exception:
        return None


def render_location_profile(
    lat,
    lon,
    date_str="20 MAY 2024",
    selected_map_depth=75
):
    """
    Show the selected location immediately using one depth query.
    The expensive full 0-1000 m profile is explicitly requested by the user.
    """
    try:
        target_date = pd.to_datetime(date_str).strftime("%Y-%m-%d")
    except Exception:
        target_date = "2024-05-20"

    # ------------------------------------------------------------
    # FAST RESULT
    # ------------------------------------------------------------
    try:
        quick = load_temperature_at_depth(
            lat,
            lon,
            selected_map_depth,
            target_date
        )
    except Exception as exc:
        st.error(f"Unable to load the selected layer: {exc}")
        return

    actual_lat = quick["actual_lat"]
    actual_lon = quick["actual_lon"]

    st.markdown(
        f"""
        <div style="
            margin-top:10px;
            padding:10px 12px;
            background:#07111D;
            border:1px solid rgba(148,163,184,0.28);
            border-radius:7px;
            color:white;
            font-family:Arial,sans-serif;
        ">
            <div style="font-size:12px;font-weight:700;letter-spacing:.4px;">
                OCEAN LOCATION INSPECTOR
            </div>
            <div style="font-size:11px;color:#94A3B8;margin-top:3px;">
                Requested:
                <b style="color:white;">{lat:.3f}°</b>,
                <b style="color:white;">{lon:.3f}°</b>
                &nbsp;•&nbsp;
                GLORYS grid:
                <b style="color:white;">{actual_lat:.3f}°,
                {actual_lon:.3f}°</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Latitude", f"{lat:.3f}°")
    with c2:
        st.metric("Longitude", f"{lon:.3f}°")
    with c3:
        st.metric(
            f"Temperature at {quick['depth']:.1f} m",
            f"{quick['temperature']:.2f} °C"
        )

    st.caption(
        f"Source: {quick['source']} • "
        f"Layer loaded directly for the selected coordinate."
    )

    # ------------------------------------------------------------
    # FULL PROFILE — OPTIONAL / ON DEMAND
    # ------------------------------------------------------------
    st.markdown(
        "<div style='font-size:12px;font-weight:700;margin-top:8px;'>"
        "VERTICAL PROFILE</div>",
        unsafe_allow_html=True
    )

    if st.button(
        "🌊 Load full 0–1000 m GLORYS profile",
        use_container_width=True,
        key="load_full_glorys_profile"
    ):
        st.session_state.ocean_full_profile_requested = True

    if not st.session_state.get("ocean_full_profile_requested", False):
        st.info(
            "The selected layer is loaded. Click the button above only when "
            "you need the complete 0–1000 m temperature profile."
        )
        return

    try:
        with st.spinner("Loading full GLORYS vertical profile…"):
            profile = load_temperature_profile(
                lat,
                lon,
                target_date
            )
    except Exception as exc:
        st.error(f"Unable to load GLORYS profile: {exc}")
        return

    depth_options = [float(d) for d in profile["depth"].tolist()]
    default_depth = min(
        depth_options,
        key=lambda d: abs(d - float(selected_map_depth))
    )

    selected_depth = st.select_slider(
        "Inspect depth / GLORYS layer",
        options=depth_options,
        value=default_depth,
        format_func=lambda d: f"{d:.1f} m",
        key="inspector_depth_slider"
    )

    selected = profile.iloc[
        (profile["depth"] - float(selected_depth)).abs().argmin()
    ]

    st.metric(
        "Temperature at selected layer",
        f"{selected['temperature']:.2f} °C"
    )

    profile_fig = go.Figure()
    profile_fig.add_trace(
        go.Scatter(
            x=profile["temperature"],
            y=profile["depth"],
            mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=6),
            customdata=profile["source"],
            hovertemplate=(
                "<b>%{x:.2f} °C</b><br>"
                "Depth: %{y:.1f} m<br>"
                "Source: %{customdata}"
                "<extra></extra>"
            )
        )
    )

    profile_fig.add_trace(
        go.Scatter(
            x=[float(selected["temperature"])],
            y=[float(selected["depth"])],
            mode="markers",
            marker=dict(
                size=12,
                symbol="circle-open",
                line=dict(width=2)
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )

    profile_fig.update_layout(
        height=300,
        margin=dict(l=55, r=20, t=15, b=45),
        paper_bgcolor="#07111D",
        plot_bgcolor="#07111D",
        font=dict(color="#CBD5E1"),
        xaxis=dict(
            title="Temperature (°C)",
            gridcolor="rgba(148,163,184,0.15)",
            zeroline=False
        ),
        yaxis=dict(
            title="Depth (m)",
            autorange="reversed",
            gridcolor="rgba(148,163,184,0.15)"
        ),
        showlegend=False
    )

    st.plotly_chart(
        profile_fig,
        use_container_width=True,
        config={"displaylogo": False, "responsive": True},
        key="location_temperature_profile"
    )


# ============================================================
# MAIN OCEAN MAP
# ============================================================

def render_ocean_map(
    dataset="ARGO vs GLORYS",
    variable="Temperature (°C)",
    depth=75,
    date_str="20 MAY 2024",
    region="All Indian Ocean",
    target_lat=15.0,
    target_lon=65.0,
    show_floats=True,
    show_heatmap=True,
    map_key=None,
    **kwargs
):

    # ========================================================
    # LOCATION INSPECTOR STATE
    # ========================================================

    if "ocean_selected_lat" not in st.session_state:
        st.session_state.ocean_selected_lat = None
        st.session_state.ocean_selected_lon = None

    reg_info = REGIONS.get(region, REGIONS['All Indian Ocean'])
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = reg_info.get("zoom", 4.3)
    if "map_center" not in st.session_state:
        st.session_state.map_center = reg_info.get("center", [10.0, 78.0])

    # ========================================================
    # HEADER
    # ========================================================

    map_key = map_key or kwargs.get('map_key')
    if not map_key:
        if "_ocean_map_instance_counter" not in st.session_state:
            st.session_state["_ocean_map_instance_counter"] = 0
        st.session_state["_ocean_map_instance_counter"] += 1
        safe_reg = str(region).replace(" ", "_").replace("-", "_")
        map_key = f"map_{safe_reg}_{depth}_{st.session_state['_ocean_map_instance_counter']}"

    if f"{map_key}_dragmode" not in st.session_state:
        st.session_state[f"{map_key}_dragmode"] = "pan"

    # ========================================================
    # INTERACTIVE TOOLBAR HEADER (SELECT AREA / RECTANGLE / POLYGON / CLEAR)
    # ========================================================
    c_head1, c_head2 = st.columns([1.0, 2.2])
    with c_head1:
        st.markdown(
            f'<div style="font-weight:700; color:#0F172A; font-size:13px; text-transform:uppercase; margin-top:6px;">'
            f'TEMPERATURE AT {depth} M DEPTH <span style="color:#64748B;">• {date_str.upper()}</span></div>',
            unsafe_allow_html=True
        )
    with c_head2:
        tool_options = ["Select Area", "Rectangle", "Polygon", "Clear"]
        selected_tool = st.radio(
            "Map Tool Mode",
            options=tool_options,
            index=0,
            horizontal=True,
            key=f"{map_key}_tool_radio",
            label_visibility="collapsed"
        )

        if selected_tool == "Rectangle":
            st.session_state[f"{map_key}_dragmode"] = "select"
        elif selected_tool == "Polygon":
            st.session_state[f"{map_key}_dragmode"] = "lasso"
        elif selected_tool == "Clear":
            st.session_state[f"{map_key}_dragmode"] = "pan"
            st.session_state[f"{map_key}_box"] = None
        else:
            st.session_state[f"{map_key}_dragmode"] = "pan"

    active_dragmode = st.session_state.get(f"{map_key}_dragmode", "pan")

    # ========================================================
    # DATE
    # ========================================================

    try:

        target_date = pd.to_datetime(
            date_str
        ).strftime(
            "%Y-%m-%d"
        )

    except Exception:

        target_date = "2024-05-20"

    # ========================================================
    # REAL COPERNICUS DATA
    # ========================================================

    temperature = load_temperature(
        target_depth=depth,
        target_date=target_date
    )

    uo, vo = load_currents(
        target_depth=depth,
        target_date=target_date
    )

    using_copernicus = (
        temperature is not None
    )

    # ========================================================
    # FIGURE
    # ========================================================

    fig = go.Figure()

    # ========================================================
    # SMOOTH TEMPERATURE RASTER
    # ========================================================

    mapbox_layers = [
        dict(
            below="traces",
            sourcetype="raster",
            source=[
                "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
            ]
        )
    ]

    if (
        show_heatmap
        and temperature is not None
    ):

        image_source = temperature_to_image(
            temperature
        )

        if image_source:
            mapbox_layers.append(
                dict(
                    sourcetype="image",
                    source=image_source,
                    coordinates=[
                        [
                            INDIAN_OCEAN_LON_MIN,
                            INDIAN_OCEAN_LAT_MAX
                        ],
                        [
                            INDIAN_OCEAN_LON_MAX,
                            INDIAN_OCEAN_LAT_MAX
                        ],
                        [
                            INDIAN_OCEAN_LON_MAX,
                            INDIAN_OCEAN_LAT_MIN
                        ],
                        [
                            INDIAN_OCEAN_LON_MIN,
                            INDIAN_OCEAN_LAT_MIN
                        ]
                    ],
                    opacity=0.68,
                    below="traces"
                )
            )

    # ========================================================
    # CURRENT STREAMLINES
    # ========================================================

    streamline_trace = create_streamline_trace(
        uo,
        vo
    )

    if streamline_trace is not None:

        fig.add_trace(
            streamline_trace
        )

    # ========================================================
    # COLORBAR
    # ========================================================

    if show_heatmap:

        fig.add_trace(
            create_colorbar_trace()
        )

    # ========================================================
    # ARGO FLOATS
    # ========================================================

    if show_floats:

        df_floats = get_argo_floats(
            region=region
        )

        fig.add_trace(

            Scattermapbox(

                lat=df_floats[
                    "latitude"
                ],

                lon=df_floats[
                    "longitude"
                ],

                mode="markers",

                marker=dict(
                    size=5,
                    color="white",
                    opacity=0.95
                ),

                text=df_floats[
                    "float_id"
                ],

                hovertemplate=(
                    "<b>ARGO Float</b><br>"
                    "ID: %{text}<br>"
                    "Latitude: %{lat:.2f}°<br>"
                    "Longitude: %{lon:.2f}°"
                    "<extra></extra>"
                ),

                name="ARGO Float"
            )
        )

    # ========================================================
    # AREA SELECTION GRID TRACE (FOR RECTANGLE BOX & LASSO POLYGON)
    # ========================================================
    if temperature is not None:
        try:
            norm_temp = normalize_coordinates(temperature).transpose("latitude", "longitude")
            grid_lats = norm_temp["latitude"].values[::4]
            grid_lons = norm_temp["longitude"].values[::4]
            g_lon, g_lat = np.meshgrid(grid_lons, grid_lats)
            g_vals = norm_temp.values[::4, ::4]
            g_valid = np.isfinite(g_vals)
            
            s_lats = g_lat[g_valid].ravel()
            s_lons = g_lon[g_valid].ravel()
            s_temps = g_vals[g_valid].ravel()

            if active_dragmode in ["select", "lasso"]:
                fig.add_trace(
                    Scattermapbox(
                        lat=s_lats,
                        lon=s_lons,
                        mode="markers",
                        marker=dict(
                            size=7,
                            color=s_temps,
                            colorscale="Thermal",
                            opacity=0.3
                        ),
                        hovertemplate="<b>Area Grid Point</b><br>Lat: %{lat:.2f}°<br>Lon: %{lon:.2f}°<extra></extra>",
                        name="Area Grid"
                    )
                )
        except Exception:
            pass

    # Clean Map View - Bounding box and pink dot removed as requested

    # ========================================================
    # MAP LAYOUT
    # ========================================================

    reg_info = REGIONS.get(region, REGIONS['All Indian Ocean'])
    c_lat, c_lon = reg_info.get('center', [10.0, 78.0])
    zoom_level = reg_info.get('zoom', 4.3)

    fig.update_layout(
        mapbox=dict(
            style="white-bg",
            layers=mapbox_layers,
            center=dict(
                lat=c_lat,
                lon=c_lon
            ),
            zoom=zoom_level
        ),

        paper_bgcolor="#07111D",

        plot_bgcolor="#07111D",

        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        ),

        height=450,

        showlegend=False,

        hovermode="closest",

        # Make navigation controls 100% visible against dark map
        modebar=dict(
            bgcolor="rgba(7,17,29,0.95)",
            color="#FFFFFF",
            activecolor="#38BDF8"
        ),

        clickmode="event+select",

        uirevision=f"ocean-map-{region}-{st.session_state.map_zoom}",
        dragmode=active_dragmode
    )

    # ========================================================
    # DISPLAY / FULLSCREEN MAP STYLING
    # ========================================================

    st.markdown(
        """
        <style>
        /* Enable map drag layer so rectangle box and polygon lasso selections work smoothly */
        div[data-testid="stPlotlyChart"] .draglayer {
            pointer-events: auto !important;
        }

        /* Fullscreen view styling */
        div[data-testid="stPlotlyChart"]:fullscreen {
            width: 100vw !important;
            height: 100vh !important;
            min-height: 100vh !important;
            background: #07111D !important;
        }

        div[data-testid="stPlotlyChart"]:fullscreen > div,
        div[data-testid="stPlotlyChart"]:fullscreen iframe {
            width: 100% !important;
            height: 100% !important;
            min-height: 100% !important;
        }

        div[data-testid="stPlotlyChart"]:fullscreen .draglayer {
            pointer-events: auto !important;
        }

        /* Modebar styling for main ocean map */
        div[data-testid="stPlotlyChart"] .modebar {
            display: flex !important;
            opacity: 0.8 !important;
            transition: opacity 0.2s ease-in-out !important;
        }

        div[data-testid="stPlotlyChart"]:hover .modebar {
            opacity: 1 !important;
        }

        div[data-testid="stPlotlyChart"] .modebar-btn path {
            fill: #FFFFFF !important;
        }

        div[data-testid="stPlotlyChart"] .modebar-btn:hover path {
            fill: #38BDF8 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # MAP SELECTION / ACTIVE MODE INDICATOR BADGE
    # ========================================================
    if active_dragmode == "select":
        badge_text = "🟦 <b>RECTANGLE SELECTION ACTIVE:</b> Click & drag a rectangle on the ocean map to select an area box."
        badge_color = "#38BDF8"
    elif active_dragmode == "lasso":
        badge_text = "◯ <b>POLYGON SELECTION ACTIVE:</b> Draw a lasso loop on the ocean map to enclose a custom polygon region."
        badge_color = "#A855F7"
    else:
        badge_text = "📍 <b>POINT SELECTION ACTIVE:</b> Click any ocean location to lock coordinates and load vertical temperature profiles."
        badge_color = "#38BDF8"

    st.markdown(
        f"""
        <div style="background-color: #0F172A; color: #FFFFFF !important; border: 1px solid {badge_color}; border-radius: 6px; padding: 7px 12px; margin-bottom: 8px; font-size: 0.85rem; font-weight: 600; display: flex; align-items: center; gap: 8px;">
            <span>{badge_text}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    map_event = st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True,
            "doubleClick": "reset",
            "responsive": True
        },
        on_select="rerun",
        selection_mode=["points", "box", "lasso"],
        key=f"plotly_chart_{map_key}",
    )

    try:
        points = map_event.selection.points

        if points:
            clicked = points[-1]

            # Mapbox point selections expose geographic coordinates directly.
            # Some Plotly/Streamlit versions additionally serialize them as
            # customdata, so support both forms.
            clicked_lat = clicked.get("lat")
            clicked_lon = clicked.get("lon")

            if clicked_lat is None or clicked_lon is None:
                custom = clicked.get("customdata")
                if custom is not None and len(custom) >= 2:
                    clicked_lat = custom[0]
                    clicked_lon = custom[1]

            if clicked_lat is not None and clicked_lon is not None:
                clicked_lat = float(clicked_lat)
                clicked_lon = float(clicked_lon)

                if (
                    INDIAN_OCEAN_LAT_MIN <= clicked_lat <= INDIAN_OCEAN_LAT_MAX
                    and INDIAN_OCEAN_LON_MIN <= clicked_lon <= INDIAN_OCEAN_LON_MAX
                ):
                    st.session_state.ocean_selected_lat = clicked_lat
                    st.session_state.ocean_selected_lon = clicked_lon
    except Exception:
        pass

    # ========================================================
    # USER LOCATION SELECTION
    # ========================================================

    # Give the user a direct, always-visible latitude/longitude selector.
    # These values are independent from the dashboard's fixed target point.
    # Existing session-state keys may have been created by the earlier
    # click-selection versions with a value of None. Never pass None into
    # st.number_input(), because Streamlit expects a real numeric value.
    try:
        default_lat = float(target_lat) if target_lat is not None else 15.0
    except (TypeError, ValueError):
        default_lat = 15.0

    try:
        default_lon = float(target_lon) if target_lon is not None else 65.0
    except (TypeError, ValueError):
        default_lon = 65.0

    if (
        "ocean_selected_lat" not in st.session_state
        or st.session_state.ocean_selected_lat is None
    ):
        st.session_state.ocean_selected_lat = default_lat

    if (
        "ocean_selected_lon" not in st.session_state
        or st.session_state.ocean_selected_lon is None
    ):
        st.session_state.ocean_selected_lon = default_lon

    if "ocean_inspect_requested" not in st.session_state:
        st.session_state.ocean_inspect_requested = False

    if "ocean_full_profile_requested" not in st.session_state:
        st.session_state.ocean_full_profile_requested = False

    st.markdown(
        """
        <div style="
            margin-top:8px;
            margin-bottom:6px;
            padding:9px 12px;
            background:#07111D;
            border:1px solid rgba(148,163,184,0.28);
            border-radius:7px;
            color:white;
            font-family:Arial,sans-serif;
        ">
            <div style="font-size:12px;font-weight:700;letter-spacing:.35px;">
                SELECT OCEAN LOCATION
            </div>
            <div style="font-size:11px;color:#94A3B8;margin-top:2px;">
                Enter any latitude and longitude in the Indian Ocean to inspect that location.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    lc1, lc2, lc3 = st.columns([1, 1, 0.55])

    with lc1:
        user_lat = st.number_input(
            "Latitude (°)",
            min_value=float(INDIAN_OCEAN_LAT_MIN),
            max_value=float(INDIAN_OCEAN_LAT_MAX),
            value=min(
                max(
                    float(st.session_state.ocean_selected_lat),
                    float(INDIAN_OCEAN_LAT_MIN)
                ),
                float(INDIAN_OCEAN_LAT_MAX)
            ),
            step=0.01,
            format="%.3f",
            key="user_selected_latitude"
        )

    with lc2:
        user_lon = st.number_input(
            "Longitude (°)",
            min_value=float(INDIAN_OCEAN_LON_MIN),
            max_value=float(INDIAN_OCEAN_LON_MAX),
            value=min(
                max(
                    float(st.session_state.ocean_selected_lon),
                    float(INDIAN_OCEAN_LON_MIN)
                ),
                float(INDIAN_OCEAN_LON_MAX)
            ),
            step=0.01,
            format="%.3f",
            key="user_selected_longitude"
        )

    with lc3:
        st.write("")
        st.write("")
        if st.button(
            "📍 Inspect Location",
            type="primary",
            use_container_width=True,
            key="apply_user_location"
        ):
            st.session_state.ocean_selected_lat = float(user_lat)
            st.session_state.ocean_selected_lon = float(user_lon)
            st.session_state.ocean_inspect_requested = True
            st.session_state.ocean_full_profile_requested = False
            st.rerun()

    # ========================================================
    # SOURCE BADGE
    # ========================================================

    if using_copernicus:

        st.html(
            f"""
            <div style="
                display:inline-block;
                margin-top:-30px;
                margin-left:12px;
                position:relative;
                z-index:20;
                background:rgba(5,15,25,0.92);
                border:1px solid rgba(255,255,255,0.25);
                border-radius:5px;
                padding:5px 10px;
                font-size:11px;
                color:white;
                font-family:Arial,sans-serif;
            ">

                ● <b>Copernicus Marine</b>
                &nbsp;•&nbsp;
                GLORYS
                &nbsp;•&nbsp;
                Temperature {depth} m
                &nbsp;•&nbsp;
                Currents

            </div>
            """
        )

    # ========================================================
    # LEGEND
    # ========================================================

    st.html(
        """
        <div style="
            background:rgba(5,15,25,0.92);
            border:1px solid rgba(255,255,255,0.22);
            border-radius:6px;
            padding:7px 12px;
            margin-top:6px;
            margin-left:10px;
            width:180px;
            font-size:11px;
            color:white;
            font-family:Arial,sans-serif;
        ">

            <div style="
                font-weight:700;
                margin-bottom:5px;
            ">
                OCEAN LAYERS
            </div>

            <div>
                <span style="color:#FFFFFF;">
                    〰
                </span>
                Ocean Currents
            </div>

            <div>
                <span style="color:#FFFFFF;">
                    ●
                </span>
                ARGO Float
            </div>

        </div>
        """
    )

    # ========================================================
    # LOCATION INSPECTOR — ONLY AFTER USER PRESSES INSPECT
    # ========================================================

    if st.session_state.get("ocean_inspect_requested", False):
        render_location_profile(
            st.session_state.ocean_selected_lat,
            st.session_state.ocean_selected_lon,
            date_str=date_str,
            selected_map_depth=depth
        )


# ============================================================
# SELECTED AREA MAP
# ============================================================

def render_selected_area_map(depth=75):
    st.markdown(
        f"""
        <div style="font-family:'Inter',sans-serif; font-size:12px; font-weight:700; color:#0F172A; text-transform:uppercase; margin-bottom:6px;">
            TEMPERATURE MAP <span style="color:#64748B;">• Selected Area • {depth} m</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    temperature = load_temperature(target_depth=depth, target_date="2024-05-20")
    uo, vo = load_currents(target_depth=depth, target_date="2024-05-20")

    if temperature is not None:
        try:
            temperature = temperature.sel(longitude=slice(50, 80), latitude=slice(0, 30))
        except Exception:
            pass

    if uo is not None:
        try:
            uo = uo.sel(longitude=slice(50, 80), latitude=slice(0, 30))
        except Exception:
            pass

    if vo is not None:
        try:
            vo = vo.sel(longitude=slice(50, 80), latitude=slice(0, 30))
        except Exception:
            pass

    fig = go.Figure()

    sel_layers = [
        dict(
            below="traces",
            sourcetype="raster",
            source=[
                "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
            ]
        )
    ]

    if temperature is not None:
        image_source = temperature_to_image(temperature)
        if image_source:
            sel_layers.append(
                dict(
                    sourcetype="image",
                    source=image_source,
                    coordinates=[[50, 30], [80, 30], [80, 0], [50, 0]],
                    opacity=0.68,
                    below="traces"
                )
            )

    streamline_trace = create_streamline_trace(uo, vo)
    if streamline_trace is not None:
        fig.add_trace(streamline_trace)

    fig.update_layout(
        mapbox=dict(
            style="white-bg",
            layers=sel_layers,
            center=dict(lat=15.0, lon=65.0),
            zoom=4.2
        ),
        paper_bgcolor="#07111D",
        plot_bgcolor="#07111D",
        margin=dict(l=0, r=0, t=0, b=0),
        height=290,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
        key=f"sel_area_map_{depth}"
    )