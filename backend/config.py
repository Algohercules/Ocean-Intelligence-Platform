"""
backend/config.py
=================
Central configuration settings for the Ocean Intelligence Platform.
Handles paths, device detection, deep learning hyperparameters,
spatial boundaries, and API server settings.
"""

import os
from pathlib import Path
from typing import List

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

# Ensure runtime directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Device Configuration
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() in ("true", "1", "yes")
try:
    import torch
    DEVICE = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    DEVICE = "cpu"

# Deep Learning Model Hyperparameters
MODEL_CONFIG = {
    "model_type": os.getenv("MODEL_TYPE", "conv_lstm"),  # 'conv_lstm' or 'cnn_lstm'
    "input_dim": int(os.getenv("MODEL_INPUT_DIM", "4")),   # SSH, SST, uSSW, vSSW
    "hidden_dim": int(os.getenv("MODEL_HIDDEN_DIM", "32")),
    "output_dim": int(os.getenv("MODEL_OUTPUT_DIM", "1")),  # ST (Subsurface Temperature)
    "kernel_size": int(os.getenv("MODEL_KERNEL_SIZE", "7")),
    "num_layers": int(os.getenv("MODEL_NUM_LAYERS", "2")),
    "dropout_prob": float(os.getenv("MODEL_DROPOUT", "0.2")),
    "attn_activation": os.getenv("MODEL_ATTN_ACTIVATION", "sigmoid"),
    "sequence_length": int(os.getenv("MODEL_SEQ_LEN", "3")),
}

# Default Model Weights Checkpoint
DEFAULT_CHECKPOINT_NAME = "convlstm_best.pt"
DEFAULT_CHECKPOINT_PATH = CHECKPOINTS_DIR / DEFAULT_CHECKPOINT_NAME

# Indian Ocean Spatial & Grid Configuration
OCEAN_BOUNDS = {
    "lat_min": -40.0,
    "lat_max": 30.0,
    "lon_min": 30.0,
    "lon_max": 120.0,
    "grid_height": 180,
    "grid_width": 360,
}

# Standard Depth Levels for Vertical Subsurface Profiling (in meters)
DEPTH_LEVELS: List[float] = [
    0.5, 5.0, 10.0, 15.0, 25.0, 35.0, 50.0, 65.0, 75.0, 100.0,
    125.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0
]

# API Server Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = os.getenv("API_URL", f"http://127.0.0.1:{API_PORT}")
CORS_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "*"
]
