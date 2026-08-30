"""
backend/api/main.py
===================
FastAPI Application Entrypoint for Ocean Intelligence Platform.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from backend.config import CORS_ORIGINS, API_HOST, API_PORT, DEVICE, MODEL_CONFIG
from backend.api.routes.predict import router as predict_router
from backend.api.routes.ocean import router as ocean_router
from backend.api.routes.argo import router as argo_router
from backend.api.routes.heatwave import router as heatwave_router

app = FastAPI(
    title="Ocean Intelligence Platform API",
    description="Deep Learning Subsurface Temperature Reconstruction & Ocean Analytics API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(predict_router, prefix="/api")
app.include_router(ocean_router, prefix="/api")
app.include_router(argo_router, prefix="/api")
app.include_router(heatwave_router, prefix="/api")


@app.get("/", tags=["Health"])
def health_check():
    """System health check and runtime information."""
    return {
        "status": "healthy",
        "service": "Ocean Intelligence Platform API",
        "version": "1.0.0",
        "device": DEVICE,
        "model_architecture": MODEL_CONFIG["model_type"],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


if __name__ == "__main__":
    import uvicorn
    print(f"Starting Ocean Intelligence API on {API_HOST}:{API_PORT} [Device: {DEVICE}]...")
    uvicorn.run("backend.api.main:app", host=API_HOST, port=API_PORT, reload=True)
