"""
scripts/evaluate_model.py
=========================
Accuracy, Spearman rank correlation, RMSE, and MAE evaluation CLI.
"""

import sys
import argparse
from pathlib import Path

# Add repository root to python path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr

from backend.config import DEVICE, DEFAULT_CHECKPOINT_PATH
from backend.models.model_registry import ModelRegistry
from backend.services.inference_service import InferenceService
from backend.services.data_service import DataService


def evaluate(checkpoint_path: str = str(DEFAULT_CHECKPOINT_PATH)):
    print(f"[Evaluate] Loading model from {checkpoint_path}...")
    model = ModelRegistry.load_model(checkpoint_path=checkpoint_path, device=DEVICE)
    model.eval()

    print("[Evaluate] Running benchmark evaluation on Indian Ocean dataset...")
    metrics = InferenceService.evaluate_model_performance()

    print("\n" + "=" * 50)
    print("      CONVLSTM MODEL EVALUATION RESULTS        ")
    print("=" * 50)
    print(f" Model Architecture   : {metrics['model_name']}")
    print(f" Spearman Correlation : {metrics['spearman_correlation']:.4f}")
    print(f" Pearson Correlation  : {metrics['pearson_correlation']:.4f}")
    print(f" RMSE (°C)            : {metrics['rmse']:.4f}")
    print(f" MAE (°C)             : {metrics['mae']:.4f}")
    print(f" SSIM Score           : {metrics['ssim']:.4f}")
    print(f" Evaluation Samples   : {metrics['total_eval_samples']}")
    print(f" Inference Device     : {metrics['device_used']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ConvLSTM Ocean Model")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Path to .pt weights file")
    args = parser.parse_args()
    evaluate(args.checkpoint)
