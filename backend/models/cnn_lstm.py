"""
backend/models/cnn_lstm.py
==========================
Hybrid CNN-LSTM Architecture for Ocean Subsurface Reconstruction.
Combines 2D spatial feature extraction CNN with temporal LSTM cells.
"""

import torch
import torch.nn as nn
from typing import Tuple


class HybridCNNLSTM(nn.Module):
    """
    Hybrid 2D CNN + Temporal LSTM model.
    The CNN block extracts multi-scale spatial surface signatures (SSH, SST, SSW vectors),
    which are then propagated through an LSTM sequence model and decoded into 2D ST fields.
    """
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 32,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout_prob: float = 0.2
    ):
        super(HybridCNNLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Spatial Encoder CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob)
        )

        # Temporal 1x1 ConvLSTM layers
        self.temporal_conv = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.lstm_cells = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Spatial Decoder CNN
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, C, H, W)
        Returns:
            Tensor of shape (B, output_dim, H, W)
        """
        B, T, C, H, W = x.size()
        
        # Encode each timestep
        encoded_seq = []
        for t in range(T):
            enc_t = self.encoder(x[:, t])  # (B, hidden_dim, H, W)
            encoded_seq.append(enc_t)
            
        # Combine temporal features
        stacked = torch.stack(encoded_seq, dim=1)  # (B, T, hidden_dim, H, W)
        mean_feat = torch.mean(stacked, dim=1)     # (B, hidden_dim, H, W)
        last_feat = stacked[:, -1]                  # (B, hidden_dim, H, W)
        
        combined = torch.cat([mean_feat, last_feat], dim=1)
        fusion = torch.relu(self.temporal_conv(combined))
        
        out = self.decoder(fusion)
        return out
