import numpy as np
import torch
from training.config import Config
from environment import VideoSummarizationEnv
from horizontal_policy import HorizontalPolicy
from vertical_policy import VerticalPolicy
from trainer import train_on_video

# 1. Load config & initialize
cfg = Config()
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

# 2. Instantiate shared policies & optimizers
h_policy = HorizontalPolicy(cfg.d_model, cfg.hidden_size)
v_policy = VerticalPolicy(cfg.d_model, cfg.hidden_size)
opt_h = torch.optim.Adam(h_policy.parameters(), lr=cfg.lr)
opt_v = torch.optim.Adam(v_policy.parameters(), lr=cfg.lr)

# 3. Simulate video dataset (replace with your actual precomputed features)
video_features = np.random.randn(150, cfg.d_model).astype(np.float32)  # T=150 frames

# 4. Create environment for THIS video
env = VideoSummarizationEnv(video_features, cfg)

# 5. Train
episode_reward, final_summary = train_on_video(env, h_policy, v_policy, opt_h, opt_v, cfg)

print(f"✅ Episode Reward: {episode_reward:.4f}")
print(f"📋 Final Summary Indices: {final_summary}")
print(f"📏 Summary Length: {len(final_summary)} / {len(video_features)} frames")