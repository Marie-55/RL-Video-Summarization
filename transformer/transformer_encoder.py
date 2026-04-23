"""
Temporal Transformer Encoder for Video Representation Learning.

Extracts contextual features from raw CNN embeddings using multi-head attention.
Optimized for variable-length video sequences.

Architecture:
    CNN features (B, T, 768) [CLIP ViT-L/14]
        → InputProjection: Linear(768→512) + LayerNorm + ReLU
        → PositionalEncoding: x = x + sinusoidal_encoding
        → 3x TransformerEncoderLayer (multi-head self-attention + FFN)
        → contextual_features (B, T, 512)
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Union


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    Adds temporal position information to frame features.
    x = x + positional_encoding
    """
    def __init__(self, d_model: int = 512, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x: (B, T, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class InputProjection(nn.Module):
    """
    Projects CNN features (768-dim for CLIP ViT-L/14) down to d_model (512-dim).
    Required because input_dim (768) != d_model (512).

    Architecture:
        Linear(768 → 512)
        LayerNorm(512)
        ReLU
    """
    def __init__(self, input_dim: int = 768, d_model: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 768) — raw CNN features
        Returns:
            (B, T, 512)   — projected features
        """
        return self.proj(x)


class TemporalTransformerEncoder(nn.Module):
    """
    Temporal Transformer Encoder for video representation learning.
    Designed for CLIP ViT-L/14 embeddings (768-dim).

    Flow:
        CNN features (B, T, 768) [CLIP ViT-L/14]
            → InputProjection: Linear(768→512) + LayerNorm + ReLU
            → PositionalEncoding: x = x + positional_encoding
            → 3x TransformerEncoderLayer (multi-head self-attention + FFN)
            → contextual_features (B, T, 512)

    Config:
        input_dim   = 768   (CLIP ViT-L/14 embeddings)
        d_model     = 512   (transformer hidden dim)
        nhead       = 8     (512/8 = 64 dim per head)
        num_layers  = 3     (same depth as TR-SUM)
        dim_feedfwd = 2048  (4 × d_model)
        dropout     = 0.1
    """
    def __init__(
        self,
        input_dim: int   = 768,
        d_model: int     = 512,
        nhead: int       = 8,
        num_layers: int  = 3,
        dim_feedfwd: int = 2048,
        dropout: float   = 0.1,
        max_len: int     = 5000,
        freeze: bool     = False,
    ):
        super().__init__()
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.d_model    = d_model
        self.input_dim  = input_dim
        self.num_layers = num_layers

        # 1. Project 768 → 512
        self.input_proj = InputProjection(
            input_dim = input_dim,
            d_model   = d_model,
        )

        # 2. Add temporal positional encoding
        self.pos_enc = PositionalEncoding(
            d_model = d_model,
            max_len = max_len,
            dropout = dropout,
        )

        # 3. Stack of Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedfwd,
            dropout         = dropout,
            batch_first     = True,   # (B, T, d_model)
            norm_first      = True,   # pre-norm: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
            norm       = nn.LayerNorm(d_model),
        )

        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad = False
        print("[TemporalEncoder] All parameters frozen.")

    def unfreeze(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True
        print("[TemporalEncoder] All parameters unfrozen.")

    def encode(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Main encoding interface.

        Args:
            features     : (B, T, 768) — precomputed CLIP embeddings
            padding_mask : (B, T) bool, True = padded position to ignore

        Returns:
            contextual_features: (B, T, 512)
        """
        x = self.input_proj(features)           # (B, T, 512)
        x = self.pos_enc(x)                     # (B, T, 512)
        contextual_features = self.transformer(
            x,
            src_key_padding_mask=padding_mask,  # (B, T) or None
        )                                       # (B, T, 512)
        return contextual_features

    def forward(self, features, padding_mask=None):
        return self.encode(features, padding_mask)


def pad_and_mask(
    feature_list: list,
    d_input: int = 768,
):
    """
    Pads a batch of variable-length videos to the same T
    and builds a padding mask.

    Args:
        feature_list : list of tensors OR numpy arrays, each (T_i, d_input)
        d_input      : feature dimension (768 for CLIP ViT-L/14)

    Returns:
        padded : (B, T_max, d_input) torch.FloatTensor
        mask   : (B, T_max) torch.BoolTensor — True = padded (ignore position)
    """
    B     = len(feature_list)
    T_max = max(f.shape[0] for f in feature_list)

    padded = torch.zeros(B, T_max, d_input)
    mask   = torch.ones(B, T_max, dtype=torch.bool)  # all padded initially

    for i, f in enumerate(feature_list):
        T_i = f.shape[0]
        # FIX: convert numpy arrays to torch tensors before assignment.
        # Previously, assigning a numpy array directly into a torch.zeros
        # tensor raised: TypeError: can't assign a numpy.ndarray to a torch.FloatTensor
        if isinstance(f, np.ndarray):
            f = torch.from_numpy(f).float()
        padded[i, :T_i, :] = f
        mask[i, :T_i] = False  # real frames → attend

    return padded, mask