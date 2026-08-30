"""
tests/test_models.py
====================
Unit tests for PyTorch ConvLSTM and Hybrid CNN-LSTM models.
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import unittest
from backend.models.conv_lstm import ConvLSTM, ConvLSTMCell, SpatialAttention
from backend.models.cnn_lstm import HybridCNNLSTM
from backend.models.model_registry import ModelRegistry


class TestOceanModels(unittest.TestCase):

    def test_spatial_attention_forward(self):
        for act in ["sigmoid", "softmax", "relu"]:
            attn = SpatialAttention(in_channels=32, activation=act)
            x = torch.randn(2, 32, 20, 30)
            out = attn(x)
            self.assertEqual(out.shape, x.shape)

    def test_convlstm_cell_forward(self):
        cell = ConvLSTMCell(input_dim=4, hidden_dim=16, kernel_size=3, dropout_prob=0.1)
        x = torch.randn(2, 4, 20, 30)
        h_prev = torch.zeros(2, 16, 20, 30)
        c_prev = torch.zeros(2, 16, 20, 30)
        h_next, c_next = cell(x, h_prev, c_prev)
        self.assertEqual(h_next.shape, (2, 16, 20, 30))
        self.assertEqual(c_next.shape, (2, 16, 20, 30))

    def test_convlstm_model_forward(self):
        model = ConvLSTM(
            input_dim=4,
            hidden_dim=16,
            kernel_size=3,
            num_layers=2,
            dropout_prob=0.1,
            output_dim=1
        )
        # (B, T, C, H, W)
        x = torch.randn(2, 3, 4, 30, 40)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 30, 40))

    def test_hybrid_cnn_lstm_forward(self):
        model = HybridCNNLSTM(
            input_dim=4,
            hidden_dim=16,
            num_layers=2,
            output_dim=1
        )
        x = torch.randn(2, 3, 4, 30, 40)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 30, 40))

    def test_model_registry_instantiation(self):
        model = ModelRegistry.create_model("conv_lstm", hidden_dim=16)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
