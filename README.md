# 🌊 Indian Ocean Intelligence Platform

An end-to-end Deep Learning and Oceanographic Intelligence platform for 2D/3D subsurface ocean temperature reconstruction, marine heatwave (MHW) detection, ARGO float tracking, and vertical thermal profiling across the Indian Ocean basin.

---

## 🏗️ Architecture & Project Structure

```
ocean-intelligence-platform/
│
├── backend/                        # Backend & Deep Learning Services
│   ├── __init__.py
│   ├── config.py                   # Central configuration (paths, device, hyperparams, dataset bounds)
│   ├── models/                     # PyTorch Neural Network Architectures
│   │   ├── __init__.py
│   │   ├── conv_lstm.py            # ConvLSTMCell, ConvLSTM with Spatial Attention & Residuals
│   │   ├── cnn_lstm.py             # Hybrid CNN-LSTM Architecture
│   │   └── model_registry.py       # Model loader, weight management, and device placement
│   ├── services/                   # Core Backend Business Logic & Inference Engine
│   │   ├── __init__.py
│   │   ├── inference_service.py    # Deep learning inference pipeline (tensors -> grid ST reconstruction)
│   │   ├── data_service.py         # NetCDF ingestion, spatial interpolations, land-masking & caching
│   │   ├── argo_service.py         # Real-time / simulated ARGO float profiles & tracking
│   │   ├── heatwave_service.py     # Marine Heatwave (MHW) detection & severity analysis
│   │   └── copernicus_service.py   # Copernicus Marine Service NetCDF loader & parser
│   ├── schemas/                    # Pydantic Schemas & DTOs
│   │   ├── __init__.py
│   │   ├── prediction.py           # Request / Response DTOs for AI reconstructions
│   │   └── ocean.py                # Coordinate, spatial bbox, transect & metric models
│   └── api/                        # FastAPI REST API (Modular microservice / backend server)
│       ├── __init__.py
│       ├── main.py                 # FastAPI application entrypoint with CORS & docs
│       └── routes/
│           ├── predict.py          # /api/predict/reconstruct, /api/predict/profile
│           ├── ocean.py            # /api/ocean/grid, /api/ocean/transect, /api/ocean/stats
│           ├── argo.py             # /api/argo/floats, /api/argo/comparison
│           └── heatwave.py         # /api/heatwave/events, /api/heatwave/timeseries
│
├── frontend/                       # Streamlit UI & Data Visualization Layer
│   ├── app.py                      # Main Platform Entrypoint
│   ├── client.py                   # Backend API / Service Client Adapter (Unified direct/REST mode)
│   ├── components/                 # Reusable UI Components
│   │   ├── header.py               # Navbar & title
│   │   ├── sidebar.py              # Filter controls & coordinate picker
│   │   ├── ocean_map.py            # PyDeck 3D & 2D Oceanographic Visualizations
│   │   ├── ai_panel.py             # AI Subsurface Reconstruction & comparison maps
│   │   ├── area_selection.py       # Region & coordinate inspector cards
│   │   ├── comparison_chart.py     # ARGO vs GLORYS vs AI charts
│   │   ├── metric_cards.py         # Dynamic KPIs & model validation scores
│   │   ├── temperature_profile.py  # Depth vs Temperature Profile Visualizer
│   │   ├── time_series.py          # Temporal trend analysis
│   │   ├── data_table.py           # Depth layer data tables
│   │   └── footer.py               # System status and credits
│   ├── pages/                      # Multi-page Dashboard
│   │   ├── 1_Dashboard.py
│   │   ├── 2_Explorer.py
│   │   ├── 3_ARGO.py
│   │   ├── 4_Analysis.py
│   │   ├── 5_AI_Prediction.py      # Powered by ConvLSTM Deep Learning Inference
│   │   ├── 6_Heatwave.py
│   │   └── 7_Reports.py
│   ├── styles/
│   │   └── style.css               # Modern Glassmorphic Dark UI Theme
│   └── assets/                     # Platform logos and graphics
│
├── data/                           # Data storage & artifacts
│   ├── raw/                        # NetCDF datasets (Copernicus, SST, SSH, uSSW, vSSW, ST)
│   ├── processed/                  # Cached landmasks, grids, and normalization scalers
│   └── checkpoints/                # PyTorch saved model weights (.pt files)
│
├── scripts/                        # Automation, training, and evaluation scripts
│   ├── train_model.py              # Standalone ConvLSTM model training CLI
│   ├── evaluate_model.py           # Accuracy, Spearman correlation, and RMSE evaluation
│   └── generate_sample_weights.py  # Generates pretrained model weights for immediate out-of-the-box inference
│
├── tests/                          # Backend unit & integration test suite
│   ├── test_models.py              # PyTorch model forward pass tests
│   ├── test_services.py            # Inference and data service tests
│   └── test_api.py                 # FastAPI endpoint validation tests
│
├── requirements.txt                # Unified dependency management
├── .env.example                    # Environment variable configurations
├── run.py                          # Unified launcher (starts backend API and/or Streamlit UI)
└── README.md                       # Comprehensive deployment, architecture & developer guide
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Model Checkpoint (Out-of-the-Box)
```bash
python scripts/generate_sample_weights.py
```

### 3. Launch Full Stack (FastAPI + Streamlit)
```bash
python run.py
```
- **Streamlit Interactive UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Interactive Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧠 Deep Learning Architecture (ConvLSTM)

The subsurface temperature reconstruction engine uses a multi-layer **Convolutional Long Short-Term Memory (ConvLSTM)** network with **Spatial Attention** and **Residual Connections**:

- **Input Features (4 Channels)**:
  1. `SSH`: Sea Surface Height anomalies
  2. `SST`: Sea Surface Temperature
  3. `uSSW`: Zonal Surface Wind / Current
  4. `vSSW`: Meridional Surface Wind / Current
- **Spatial Attention Module**: Focuses attention maps on oceanographic mesoscale eddies and thermal fronts.
- **Residual Connections**: Skip connections preserve high-frequency spatial gradients across ConvLSTM cells.
- **Ocean Loss Masking**: Training loss calculation strictly masks out continental landmasses using cached 180×360 landmasks.

---

## 🛠️ CLI Utilities & Testing

- **Train ConvLSTM Model**:
  ```bash
  python scripts/train_model.py --epochs 20 --lr 0.001 --hidden_dim 32
  ```
- **Evaluate Model Benchmarks**:
  ```bash
  python scripts/evaluate_model.py
  ```
- **Run Full Test Suite**:
  ```bash
  python -m unittest discover -s tests
  ```

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check and device status |
| `POST` | `/api/predict/reconstruct` | ConvLSTM 2D Subsurface Temperature reconstruction |
| `GET` | `/api/predict/profile` | Vertical AI temperature profile vs baseline |
| `POST` | `/api/predict/timeseries` | Multi-step temporal forecast with uncertainty bounds |
| `GET` | `/api/ocean/stats` | Oceanographic parameters at lat/lon/depth |
| `GET` | `/api/argo/floats` | Active ARGO float metadata and sensor metrics |
| `GET` | `/api/heatwave/events` | Active Marine Heatwaves (MHW) with severity ratings |