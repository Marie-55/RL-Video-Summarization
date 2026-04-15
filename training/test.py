import numpy as np
import torch
from training.config import Config
from environment import VideoSummarizationEnv
from horizontal_policy import HorizontalPolicy
from vertical_policy import VerticalPolicy
from trainer import train_on_video
from training.feature_pipeline import FeaturePipeline

# 1. Load config & initialize
cfg = Config()
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

# 2. Initialize feature pipeline (transformer encoder)
pipeline = FeaturePipeline(cfg, device='cuda' if torch.cuda.is_available() else 'cpu')

# 3. Instantiate shared policies & optimizers
h_policy = HorizontalPolicy(cfg.d_model, cfg.hidden_size)
v_policy = VerticalPolicy(cfg.d_model, cfg.hidden_size)
opt_h = torch.optim.Adam(h_policy.parameters(), lr=cfg.lr)
opt_v = torch.optim.Adam(v_policy.parameters(), lr=cfg.lr)

# 4. Transformer optimizer (for end-to-end training)
opt_transformer = pipeline.get_optimizer(lr=cfg.lr)

# 5. Simulate video dataset
# TODO: Replace with actual CNN features when available
# For now, simulate raw CNN features (T, 1024)
raw_cnn_features = np.random.randn(150, cfg.transformer_input_dim).astype(np.float32)

# 6. Encode raw features to contextual features using transformer
print("\n[test.py] Encoding raw CNN features with transformer...")
contextual_features = pipeline.encode_video(raw_cnn_features)
print(f"Raw CNN features shape    : {raw_cnn_features.shape}  # (T, 1024)")
print(f"Contextual features shape: {contextual_features.shape}  # (T, 512)")

# 7. Create environment for THIS video
env = VideoSummarizationEnv(contextual_features, cfg)

# 8. Train
print("\n[test.py] Starting training...")
episode_reward, final_summary = train_on_video(
    env, h_policy, v_policy, opt_h, opt_v, cfg,
    pipeline=pipeline,
    opt_transformer=opt_transformer
)

print(f"\n✅ Episode Reward: {episode_reward:.4f}")
print(f"📋 Final Summary Indices: {final_summary}")
print(f"📏 Summary Length: {len(final_summary)} / {len(contextual_features)} frames")
