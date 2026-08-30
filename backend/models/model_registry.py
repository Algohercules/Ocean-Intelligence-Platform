"""
backend/models/model_registry.py
================================
Model registry, weight loader, checkpoint management, and device placement.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union

from backend.config import (
    DEVICE,
    MODEL_CONFIG,
    DEFAULT_CHECKPOINT_PATH,
    CHECKPOINTS_DIR
)
from backend.models.conv_lstm import ConvLSTM
from backend.models.cnn_lstm import HybridCNNLSTM


class ModelRegistry:
    """
    Singleton registry to instantiate, load, and cache PyTorch ocean models.
    """
    _cached_models: Dict[str, nn.Module] = {}

    @classmethod
    def create_model(
        cls,
        model_type: str = "conv_lstm",
        input_dim: int = 4,
        hidden_dim: int = 32,
        output_dim: int = 1,
        kernel_size: int = 7,
        num_layers: int = 2,
        dropout_prob: float = 0.2,
        attn_activation: str = "sigmoid",
        device: str = DEVICE
    ) -> nn.Module:
        """
        Instantiate model architecture on specified device.
        """
        if model_type.lower() == "conv_lstm":
            model = ConvLSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                num_layers=num_layers,
                dropout_prob=dropout_prob,
                attn_activation=attn_activation,
                output_dim=output_dim
            )
        elif model_type.lower() == "cnn_lstm":
            model = HybridCNNLSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                dropout_prob=dropout_prob
            )
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'conv_lstm' or 'cnn_lstm'.")

        model.to(device)
        return model

    @classmethod
    def load_model(
        cls,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_type: Optional[str] = None,
        device: str = DEVICE,
        force_reload: bool = False
    ) -> nn.Module:
        """
        Loads model with weights from checkpoint. If checkpoint is missing, initializes default model.
        """
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_CHECKPOINT_PATH
        else:
            checkpoint_path = Path(checkpoint_path)

        cache_key = f"{model_type or MODEL_CONFIG['model_type']}_{str(checkpoint_path)}_{device}"
        if not force_reload and cache_key in cls._cached_models:
            return cls._cached_models[cache_key]

        selected_model_type = model_type or MODEL_CONFIG["model_type"]
        model = cls.create_model(
            model_type=selected_model_type,
            input_dim=MODEL_CONFIG["input_dim"],
            hidden_dim=MODEL_CONFIG["hidden_dim"],
            output_dim=MODEL_CONFIG["output_dim"],
            kernel_size=MODEL_CONFIG["kernel_size"],
            num_layers=MODEL_CONFIG["num_layers"],
            dropout_prob=MODEL_CONFIG["dropout_prob"],
            attn_activation=MODEL_CONFIG["attn_activation"],
            device=device
        )

        if checkpoint_path.exists():
            try:
                state_dict = torch.load(checkpoint_path, map_location=device)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                model.load_state_dict(state_dict, strict=False)
                print(f"[ModelRegistry] Successfully loaded weights from {checkpoint_path}")
            except Exception as e:
                print(f"[ModelRegistry] Warning: Could not load weights from {checkpoint_path}: {e}. Running with initialized weights.")
        else:
            print(f"[ModelRegistry] Checkpoint {checkpoint_path} not found. Running with initialized weights.")

        model.eval()
        cls._cached_models[cache_key] = model
        return model


def get_model(device: str = DEVICE) -> nn.Module:
    """Helper function to get cached model instance."""
    return ModelRegistry.load_model(device=device)


def load_model_checkpoint(path: Union[str, Path], device: str = DEVICE) -> nn.Module:
    """Helper function to load specific model checkpoint."""
    return ModelRegistry.load_model(checkpoint_path=path, device=device, force_reload=True)
