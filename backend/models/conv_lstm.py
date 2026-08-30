"""
backend/models/conv_lstm.py
===========================
PyTorch ConvLSTM with Spatial Attention and Residual Connections
for Ocean Subsurface Temperature Reconstruction & Forecasting.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module to focus on critical oceanographic features (fronts, eddies).
    Supports sigmoid, softmax, and relu activations.
    """
    def __init__(self, in_channels: int, activation: str = "sigmoid"):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.activation = activation.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        attn_scores = self.conv(x)
        if self.activation == "sigmoid":
            attn = torch.sigmoid(attn_scores)
        elif self.activation == "softmax":
            B, C, H, W = attn_scores.shape
            attn = attn_scores.view(B, -1)
            attn = torch.softmax(attn, dim=1)
            attn = attn.view(B, C, H, W)
        elif self.activation == "relu":
            attn = torch.relu(attn_scores)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}. Use 'sigmoid', 'softmax', or 'relu'.")
        return x * attn


class ConvLSTMCell(nn.Module):
    """
    2D Convolutional LSTM Cell with spatial convolutions and Dropout2d regularization.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 7, dropout_prob: float = 0.2):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate along channel dimension: (B, C_in + C_hidden, H, W)
        combined = torch.cat([x, h_prev], dim=1)
        conv_output = self.conv(combined)
        
        # Split gate outputs: Input (i), Forget (f), Output (o), Gate (g)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        # Cell & Hidden state update
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        h = self.dropout(h)
        return h, c


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM network with residual skip connections,
    Spatial Attention gating, and single-channel ST mapping.
    """
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 32,
        kernel_size: int = 7,
        num_layers: int = 2,
        dropout_prob: float = 0.2,
        attn_activation: str = "sigmoid",
        output_dim: int = 1
    ):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Stack ConvLSTM cells across layers
        self.cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim=input_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dropout_prob=dropout_prob
            )
            for i in range(num_layers)
        ])

        # Spatial Attention on top layer hidden state
        self.attn = SpatialAttention(hidden_dim, activation=attn_activation)

        # Output mapping convolution: maps hidden features to ST (Subsurface Temperature)
        self.conv_out = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, T, C, H, W)
               B = Batch size, T = Sequence length (time steps),
               C = Input channels (SSH, SST, uSSW, vSSW), H = Height, W = Width
        Returns:
            Output tensor of shape (B, output_dim, H, W) containing reconstructed ST map.
        """
        B, T, C, H, W = x.size()
        
        # Initialize hidden and cell states
        h = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) for _ in range(self.num_layers)]

        for t in range(T):
            inp = x[:, t]  # Slice time step t: (B, C, H, W)
            for i, cell in enumerate(self.cells):
                h_prev, c_prev = h[i], c[i]
                h[i], c[i] = cell(inp, h_prev, c_prev)
                # Residual connection when channel dimensions match
                if inp.shape[1] == self.hidden_dim:
                    h[i] = h[i] + inp
                inp = h[i]

        # Apply spatial attention on the final layer's hidden state
        h_attn = self.attn(h[-1])
        # Output temperature field
        return self.conv_out(h_attn)
