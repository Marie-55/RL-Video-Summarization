"""
Feature Pipeline for Temporal Transformer Encoder.

Handles:
    - Loading raw CLIP ViT-L/14 embeddings
    - Encoding to contextual features via transformer
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
        total_params = sum(p.numel() for p in self.encoder.parameters())
        print("  Temporal Transformer Encoder")
        print(f"  Input dimension    : {self.config.transformer_input_dim}")
        print(f"  Output dimension   : {self.config.transformer_d_model}")
        print(f"  Attention heads    : {self.config.transformer_nhead}")
        print(f"  Encoder layers     : {self.config.transformer_num_layers}")
        print(f"  FFN hidden size    : {self.config.transformer_dim_feedfwd}")
        print(f"  Total parameters   : {total_params:,}")
        print(f"  Device             : {self.device}")
    
    def encode_video(
        self,
        features: Union[torch.Tensor, np.ndarray],
        padding_mask: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode CLIP features to contextual features for a single video.
        
        Args:
            features: (T, 768) CLIP ViT-L/14 embeddings
                      - Can be torch.Tensor or np.ndarray
            padding_mask: (T,) optional mask for padded frames
            training: If True, run in train mode and return a Tensor with
                      gradients (for end-to-end training). If False (default),
                      run in eval/no_grad mode and return a numpy array.
            
        Returns:
            contextual_features:
                - (T, 512) numpy array  when training=False
                - (T, 512) torch.Tensor when training=True  (gradients attached)
        """
        # Convert to tensor if needed
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Add batch dimension
        features = features.unsqueeze(0)  # (1, T, 768)
        features = features.to(self.device)
        
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(0).to(self.device)  # (1, T)
        
        if training:
            # Train mode: keep gradients so the transformer can be updated
            self.encoder.train()
            contextual_features = self.encoder.encode(features, padding_mask)  # (1, T, 512)
            contextual_features = contextual_features.squeeze(0)  # (T, 512)
            return contextual_features  # Tensor with grad_fn
        else:
            # Inference / RL-only mode: no gradients needed
            self.encoder.eval()
            with torch.no_grad():
                contextual_features = self.encoder.encode(features, padding_mask)  # (1, T, 512)
            contextual_features = contextual_features.squeeze(0).cpu().numpy()  # (T, 512)
            return contextual_features
    
    def encode_batch(
        self,
        feature_list: list,
        padding_masks: Optional[list] = None,
        training: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a batch of videos (variable-length).
        
        Args:
            feature_list: list of (T_i, 768) tensors/arrays
            padding_masks: optional list of (T_i,) masks
            training: If True, run in train mode with gradients attached.
            
        Returns:
            padded_contextual: (B, T_max, 512) numpy array
            batch_mask: (B, T_max) boolean mask
        """
        from transformer.transformer_encoder import pad_and_mask
        
        # Pad to same length
        padded_features, batch_mask = pad_and_mask(feature_list, self.config.transformer_input_dim)
        padded_features = padded_features.to(self.device)
        batch_mask = batch_mask.to(self.device)
        
        if training:
            self.encoder.train()
            contextual_features = self.encoder.encode(padded_features, batch_mask)  # (B, T_max, 512)
        else:
            self.encoder.eval()
            with torch.no_grad():
                contextual_features = self.encoder.encode(padded_features, batch_mask)  # (B, T_max, 512)
        
        # Convert to numpy
        contextual_features = contextual_features.cpu().numpy() if not training else contextual_features
        batch_mask = batch_mask.cpu().numpy()
        
        return contextual_features, batch_mask
    
    def save_checkpoint(self, path: Union[str, Path]):
        """Save transformer checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), path)
        print(f"[FeaturePipeline] Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Union[str, Path]):
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
        trainable_params = [p for p in self.encoder.parameters() if p.requires_grad]
        if not trainable_params:
            print("[FeaturePipeline] Warning: No trainable parameters in transformer")
        return torch.optim.Adam(trainable_params, lr=lr)