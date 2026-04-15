"""
Feature Pipeline for Temporal Transformer Encoder.

Handles:
    - Loading raw CNN features
    - Encoding to contextual features
    - Device management (CPU/GPU)
    - Batch processing
"""

import torch
import numpy as np
from typing import Union, Optional, Tuple
from pathlib import Path
from transformer.transformer_encoder import TemporalTransformerEncoder


class FeaturePipeline:
    """
    Manages feature encoding through the temporal transformer.
    
    Usage:
        pipeline = FeaturePipeline(config, device='cuda')
        contextual_features = pipeline.encode_video(raw_cnn_features)
    """
    
    def __init__(self, config, device: Optional[str] = None):
        """
        Args:
            config: Config object with transformer parameters
            device: 'cpu', 'cuda', or None (auto-detect)
        """
        self.config = config
        self.device = self._get_device(device)
        
        # Initialize transformer encoder
        self.encoder = TemporalTransformerEncoder(
            input_dim=config.transformer_input_dim,
            d_model=config.transformer_d_model,
            nhead=config.transformer_nhead,
            num_layers=config.transformer_num_layers,
            dim_feedfwd=config.transformer_dim_feedfwd,
            dropout=config.transformer_dropout,
            max_len=config.transformer_max_len,
        ).to(self.device)
        
        print(f"[FeaturePipeline] Transformer initialized on {self.device}")
        self._print_model_summary()
    
    @staticmethod
    def _get_device(device: Optional[str] = None) -> str:
        """Auto-detect device if not specified."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _print_model_summary(self):
        """Print transformer model summary."""
        total_params = sum(p.numel() for p in self.encoder.parameters())
        print("=" * 55)
        print("  Temporal Transformer Encoder")
        print("=" * 55)
        print(f"  Input dimension    : {self.config.transformer_input_dim}")
        print(f"  Output dimension   : {self.config.transformer_d_model}")
        print(f"  Attention heads    : {self.config.transformer_nhead}")
        print(f"  Encoder layers     : {self.config.transformer_num_layers}")
        print(f"  FFN hidden size    : {self.config.transformer_dim_feedfwd}")
        print(f"  Total parameters   : {total_params:,}")
        print(f"  Device             : {self.device}")
        print("=" * 55)
    
    def encode_video(
        self,
        features: Union[torch.Tensor, np.ndarray],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Encode raw CNN features to contextual features for a single video.
        
        Args:
            features: (T, 1024) raw CNN features
                      - Can be torch.Tensor or np.ndarray
            padding_mask: (T,) optional mask for padded frames
            
        Returns:
            contextual_features: (T, 512) numpy array
        """
        # Convert to tensor if needed
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Add batch dimension
        features = features.unsqueeze(0)  # (1, T, 1024)
        features = features.to(self.device)
        
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(0).to(self.device)  # (1, T)
        
        # Encode
        self.encoder.eval()
        with torch.no_grad():
            contextual_features = self.encoder.encode(features, padding_mask)  # (1, T, 512)
        
        # Remove batch dimension and convert to numpy
        contextual_features = contextual_features.squeeze(0)  # (T, 512)
        contextual_features = contextual_features.cpu().numpy()
        
        return contextual_features
    
    def encode_batch(
        self,
        feature_list: list,
        padding_masks: Optional[list] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a batch of videos (variable-length).
        
        Args:
            feature_list: list of (T_i, 1024) tensors/arrays
            padding_masks: optional list of (T_i,) masks
            
        Returns:
            padded_contextual: (B, T_max, 512) numpy array
            batch_mask: (B, T_max) boolean mask
        """
        from transformer.transformer_encoder import pad_and_mask
        
        # Pad to same length
        padded_features, batch_mask = pad_and_mask(feature_list, self.config.transformer_input_dim)
        padded_features = padded_features.to(self.device)
        batch_mask = batch_mask.to(self.device)
        
        # Encode
        self.encoder.eval()
        with torch.no_grad():
            contextual_features = self.encoder.encode(padded_features, batch_mask)  # (B, T_max, 512)
        
        # Convert to numpy
        contextual_features = contextual_features.cpu().numpy()
        batch_mask = batch_mask.cpu().numpy()
        
        return contextual_features, batch_mask
    
    def save_checkpoint(self, path: Union[str, Path]):
        """Save transformer checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), path)
        print(f"[FeaturePipeline] Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Union[str, Path]):
        """Load transformer checkpoint."""
        path = Path(path)
        self.encoder.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[FeaturePipeline] Checkpoint loaded from {path}")
    
    def freeze(self):
        """Freeze transformer parameters (for RL-only training)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("[FeaturePipeline] Transformer frozen.")
    
    def unfreeze(self):
        """Unfreeze transformer parameters (for end-to-end training)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("[FeaturePipeline] Transformer unfrozen.")
    
    def get_optimizer(self, lr: float = 1e-4):
        """Get optimizer for trainable transformer parameters."""
        trainable_params = [p for p in self.encoder.parameters() if p.requires_grad]
        if not trainable_params:
            print("[FeaturePipeline] Warning: No trainable parameters in transformer")
        return torch.optim.Adam(trainable_params, lr=lr)
