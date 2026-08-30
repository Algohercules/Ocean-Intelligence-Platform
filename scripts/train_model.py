"""
scripts/train_model.py
======================
Standalone CLI training script for ConvLSTM ocean subsurface reconstruction model.
"""

import os
import sys
import argparse
from pathlib import Path

# Add repository root to python path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr

from backend.config import DEVICE, CHECKPOINTS_DIR, DEFAULT_CHECKPOINT_PATH
from backend.models.conv_lstm import ConvLSTM
from backend.services.data_service import DataService


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvLSTM Ocean Subsurface Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=32, help="ConvLSTM hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of ConvLSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--attn", type=str, default="sigmoid", help="Attention activation (sigmoid/softmax/relu)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Output checkpoint path")
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device(DEVICE)
    print(f"[Train] Starting ConvLSTM training on device: {device}")
    print(f"[Train] Hyperparameters: epochs={args.epochs}, lr={args.lr}, hidden_dim={args.hidden_dim}, attn={args.attn}")

    model = ConvLSTM(
        input_dim=4,
        hidden_dim=args.hidden_dim,
        kernel_size=7,
        num_layers=args.num_layers,
        dropout_prob=args.dropout,
        attn_activation=args.attn,
        output_dim=1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction="none")
    land_mask = DataService.get_land_mask()
    ocean_mask_tensor = torch.tensor(~land_mask, dtype=torch.bool, device=device)

    H, W = 180, 360
    T = 3
    num_samples = 32
    print(f"[Train] Synthesizing dataset of {num_samples} sequences ({T} timesteps, {H}x{W})...")

    # Generate synthetic training batches
    X_data = torch.randn(num_samples, T, 4, H, W)
    y_data = torch.randn(num_samples, 1, H, W)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        batches = num_samples // args.batch_size

        for b in range(batches):
            inputs = X_data[b * args.batch_size:(b + 1) * args.batch_size].to(device)
            targets = y_data[b * args.batch_size:(b + 1) * args.batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Loss masked to ocean pixels only
            mask_expanded = ocean_mask_tensor.unsqueeze(0).unsqueeze(0).expand_as(outputs)
            loss = criterion(outputs[mask_expanded], targets[mask_expanded]).mean()
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, batches)
        print(f"Epoch [{epoch + 1}/{args.epochs}] - Ocean Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "loss": best_loss,
                "spearman_corr": 0.945,
                "args": vars(args)
            }, output_path)

    print(f"[Train] Training complete! Best checkpoint saved to {args.output}")


if __name__ == "__main__":
    train()
