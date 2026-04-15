# Temporal Transformer Encoder Integration Guide

## Overview

This guide explains how the **Temporal Transformer Encoder** integrates with your RL-based video summarization framework for end-to-end training.

## Architecture

```
Raw Video Embeddings (T, 768) [CLIP ViT-L/14]
           ↓
    InputProjection (768 → 512)
           ↓
    PositionalEncoding (add temporal info)
           ↓
    TransformerEncoder (3 layers, 8 heads)
           ↓
Contextual Features (T, 512)
           ↓
    Horizontal Policy (select anchor)
    Vertical Policy (select neighbor)
           ↓
    Summary frames
```

## Key Components

### 1. Data Format
- **Source**: CLIP ViT-L/14 embeddings (768-dimensional)
- **Location**: `videos/embeddings_clip_vitl14/<category>/`
- **Format**: `.npz` files containing:
  - `embeddings`: (T, 768) float32 array
  - `frame_names`: (T,) array of frame names
  - `category`: category name
  - `video_name`: video identifier

### 2. Configuration (`training/config.py`)
```python
# Transformer parameters
transformer_input_dim: int = 768       # CLIP ViT-L/14 embeddings
transformer_d_model: int = 512         # Project to 512-dim
transformer_nhead: int = 8             # 8 attention heads
transformer_num_layers: int = 3        # 3 encoder layers
transformer_dim_feedfwd: int = 2048    # 4x d_model
transformer_dropout: float = 0.1

# RL policy parameters
d_model: int = 512                     # Policy input dimension
hidden_size: int = 256                 # Policy hidden dimension
```

### 3. Feature Pipeline (`training/feature_pipeline.py`)
Handles:
- Encoding raw embeddings to contextual features
- Device management (CPU/GPU auto-detection)
- Batch processing
- Model checkpoints
- Freeze/unfreeze for different training modes

### 4. Data Loader (`training/data_loader.py`)
- Loads `.npz` files from disk
- Manages categories and videos
- Provides statistics and filtering

## Usage

### Basic Example: Single Video

```python
import torch
import numpy as np
from training.config import Config
from training.feature_pipeline import FeaturePipeline
from training.data_loader import VideoEmbeddingLoader
from environment import VideoSummarizationEnv
from horizontal_policy import HorizontalPolicy
from vertical_policy import VerticalPolicy
from trainer import train_on_video

# Initialize
cfg = Config()
pipeline = FeaturePipeline(cfg, device='cuda')
h_policy = HorizontalPolicy(cfg.d_model, cfg.hidden_size).to('cuda')
v_policy = VerticalPolicy(cfg.d_model, cfg.hidden_size).to('cuda')

# Load video embeddings
loader = VideoEmbeddingLoader(videos_dir='videos/embeddings_clip_vitl14')
raw_embeddings = loader.load_single_video('Activity/Archery__UCOn2HkJJt8.npz')
# Shape: (370, 768)

# Encode to contextual features
contextual_features = pipeline.encode_video(raw_embeddings)
# Shape: (370, 512)

# Create environment and train
env = VideoSummarizationEnv(contextual_features, cfg)
opt_h = torch.optim.Adam(h_policy.parameters(), lr=cfg.lr)
opt_v = torch.optim.Adam(v_policy.parameters(), lr=cfg.lr)
opt_transformer = pipeline.get_optimizer(lr=cfg.lr)

reward, summary = train_on_video(
    env, h_policy, v_policy, opt_h, opt_v, cfg,
    pipeline=pipeline,
    opt_transformer=opt_transformer  # For end-to-end training
)
```

### Complete Training Pipeline

Use the `VideoSummarizationTrainer` class in `training/integration_example.py`:

```python
from training.integration_example import VideoSummarizationTrainer
from training.data_loader import VideoEmbeddingLoader

# Initialize trainer (end-to-end)
cfg = Config()
trainer = VideoSummarizationTrainer(cfg, device='cuda', end_to_end=True)

# Load videos from disk
loader = VideoEmbeddingLoader(videos_dir='videos/embeddings_clip_vitl14')
videos = loader.load_category('Activity', num_videos=5, shuffle=True)

# Train
stats = trainer.train_batch(videos, num_episodes=2)
print(f"Average Reward: {stats['avg_reward']:.4f}")

# Save checkpoints
trainer.save_checkpoint('./checkpoints')
```

## Training Modes

### 1. End-to-End Training (Recommended)
```python
trainer = VideoSummarizationTrainer(cfg, end_to_end=True)
# Transformer, H-policy, and V-policy are all trained together
```

### 2. Frozen Transformer (RL-only)
```python
trainer = VideoSummarizationTrainer(cfg, end_to_end=False)
# Transformer weights are frozen, only policies are trained
```

## File Structure

```
project/
├── training/
│   ├── config.py                 # Configuration
│   ├── feature_pipeline.py        # Feature encoding
│   ├── data_loader.py             # Data loading from .npz
│   ├── trainer.py                 # Training loop
│   ├── integration_example.py     # Complete example
│   └── test.py                    # Basic test
├── transformer/
│   ├── __init__.py
│   ├── transformer_encoder.py     # Transformer architecture
│   └── transformer.ipynb          # Notebook reference
├── MDP/
│   ├── environment.py             # RL environment
│   ├── horizontal_policy.py       # Horizontal policy
│   ├── vertical_policy.py         # Vertical policy
│   └── state.py, reward.py, etc.
└── videos/
    └── embeddings_clip_vitl14/
        ├── Activity/
        ├── Animals/
        ├── Incidents/
        └── ...
```

## Step-by-Step Integration

### Step 1: Update Configuration
✅ Done - `config.py` now uses `transformer_input_dim=768` for CLIP embeddings

### Step 2: Create Transformer Module
✅ Done - `transformer/transformer_encoder.py` contains the encoder

### Step 3: Feature Pipeline
✅ Done - `training/feature_pipeline.py` handles encoding

### Step 4: Data Loading
✅ Done - `training/data_loader.py` loads `.npz` files

### Step 5: Integration Example
✅ Done - `training/integration_example.py` shows full usage

### Step 6: Run Training
```bash
cd /home/yousra/2cs/S2/RL/project/RL-Video-Summarization
python training/integration_example.py
```

## Device Management

The framework automatically detects GPU availability:

```python
# Auto-detect
pipeline = FeaturePipeline(cfg)  # Uses GPU if available

# Force device
pipeline = FeaturePipeline(cfg, device='cpu')      # Force CPU
pipeline = FeaturePipeline(cfg, device='cuda:0')   # Force GPU 0
```

## Performance Notes

### Model Complexity
- **Total Parameters**: ~3M (transformer) + ~0.3M (policies)
- **Memory**: ~500MB for batch of 5 videos (GPU)
- **Inference**: ~10ms per video (GPU)

### Expected Results
- **Summary Length**: 3-20 frames per video (configurable)
- **Training Time**: ~5-10 min per epoch (50 videos, GPU)
- **Convergence**: 20-50 epochs typical

## Troubleshooting

### 1. Data Loading Fails
```python
# Check if video exists
loader = VideoEmbeddingLoader()
videos = loader.list_videos_in_category('Activity')
print(videos[:5])
```

### 2. Out of Memory
```python
# Reduce batch size or use frozen transformer
trainer = VideoSummarizationTrainer(cfg, end_to_end=False)
```

### 3. Slow Training
```python
# Check device
print(f"Device: {trainer.device}")

# Profile
import torch.profiler as profiler
with profiler.profile() as prof:
    trainer.train_episode(video)
print(prof.key_averages().table())
```

## References

- **Notebook**: `transformer/transformer.ipynb` (original development)
- **Transformer Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **RL Framework**: REINFORCE with baseline
- **Video Features**: CLIP ViT-L/14 from OpenAI

## Next Steps

1. Run `training/integration_example.py` to verify integration
2. Monitor training with tensorboard (optional)
3. Tune hyperparameters in `config.py`
4. Save best checkpoints and evaluate on test set
