"""
API Route Handlers.
"""
from backend.api.routes.predict import router as predict_router
from backend.api.routes.ocean import router as ocean_router
from backend.api.routes.argo import router as argo_router
from backend.api.routes.heatwave import router as heatwave_router

__all__ = [
    "predict_router",
    "ocean_router",
    "argo_router",
    "heatwave_router",
]
