"""
Quick Integration Test: Transformer + RL Framework

This script verifies that all components work together correctly.
"""

import sys
import torch
import numpy as np
from pathlib import Path

print("=" * 70)
print("TRANSFORMER + RL INTEGRATION TEST")
print("=" * 70)

# Test 1: Import all modules
print("\n[Test 1] Importing modules...")
try:
    from training.config import Config
    from training.feature_pipeline import FeaturePipeline
    from training.data_loader import VideoEmbeddingLoader
    from transformer.transformer_encoder import TemporalTransformerEncoder
    from environment import VideoSummarizationEnv
    from horizontal_policy import HorizontalPolicy
    from vertical_policy import VerticalPolicy
    from trainer import train_on_video
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load configuration
print("\n[Test 2] Loading configuration...")
try:
    cfg = Config()
    print(f"  ✓ Config loaded")
    print(f"    - Transformer input dim: {cfg.transformer_input_dim}")
    print(f"    - Transformer output dim: {cfg.transformer_d_model}")
    print(f"    - Policy input dim: {cfg.d_model}")
except Exception as e:
    print(f"  ✗ Config loading failed: {e}")
    sys.exit(1)

# Test 3: Initialize transformer
print("\n[Test 3] Initializing transformer encoder...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = FeaturePipeline(cfg, device=device)
    print(f"  ✓ Transformer initialized on {device}")
except Exception as e:
    print(f"  ✗ Transformer initialization failed: {e}")
    sys.exit(1)

# Test 4: Load video data
print("\n[Test 4] Loading video embeddings...")
try:
    loader = VideoEmbeddingLoader(videos_dir='videos/embeddings_clip_vitl14')
    videos = loader.load_category('Activity', num_videos=3, shuffle=True)
    print(f"  ✓ Loaded {len(videos)} videos from Activity category")
    for i, v in enumerate(videos):
        print(f"    - Video {i+1}: {v.shape}")
except Exception as e:
    print(f"  ✗ Video loading failed: {e}")
    sys.exit(1)

# Test 5: Encode video features
print("\n[Test 5] Encoding video features...")
try:
    raw_features = videos[0]  # (T, 768)
    contextual_features = pipeline.encode_video(raw_features)  # (T, 512)
    print(f"  ✓ Encoding successful")
    print(f"    - Input shape: {raw_features.shape}")
    print(f"    - Output shape: {contextual_features.shape}")
    assert contextual_features.shape[0] == raw_features.shape[0], "Length mismatch!"
    assert contextual_features.shape[1] == cfg.transformer_d_model, "Dimension mismatch!"
except Exception as e:
    print(f"  ✗ Encoding failed: {e}")
    sys.exit(1)

# Test 6: Initialize policies
print("\n[Test 6] Initializing RL policies...")
try:
    h_policy = HorizontalPolicy(cfg.d_model, cfg.hidden_size).to(device)
    v_policy = VerticalPolicy(cfg.d_model, cfg.hidden_size).to(device)
    print(f"  ✓ Policies initialized")
except Exception as e:
    print(f"  ✗ Policy initialization failed: {e}")
    sys.exit(1)

# Test 7: Create environment
print("\n[Test 7] Creating RL environment...")
try:
    env = VideoSummarizationEnv(contextual_features, cfg)
    obs = env.reset()
    print(f"  ✓ Environment created and reset")
    print(f"    - Summary indices: {obs['summary_indices']}")
    print(f"    - Anchor index: {obs['anchor_idx']}")
    print(f"    - Current turn: {obs['turn']}")
except Exception as e:
    print(f"  ✗ Environment creation failed: {e}")
    sys.exit(1)

# Test 8: Run one training episode
print("\n[Test 8] Running one training episode...")
try:
    opt_h = torch.optim.Adam(h_policy.parameters(), lr=cfg.lr)
    opt_v = torch.optim.Adam(v_policy.parameters(), lr=cfg.lr)
    opt_transformer = pipeline.get_optimizer(lr=cfg.lr)
    
    reward, summary = train_on_video(
        env, h_policy, v_policy,
        opt_h, opt_v, cfg,
        pipeline=pipeline,
        opt_transformer=opt_transformer
    )
    print(f"  ✓ Training episode completed")
    print(f"    - Episode reward: {reward:.4f}")
    print(f"    - Summary size: {len(summary)} frames")
except Exception as e:
    print(f"  ✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Save and load checkpoint
print("\n[Test 9] Saving and loading checkpoint...")
try:
    checkpoint_dir = Path('./test_checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save
    torch.save(h_policy.state_dict(), checkpoint_dir / 'h_policy.pt')
    torch.save(v_policy.state_dict(), checkpoint_dir / 'v_policy.pt')
    pipeline.save_checkpoint(checkpoint_dir / 'transformer.pt')
    print(f"  ✓ Checkpoints saved")
    
    # Load
    h_policy_new = HorizontalPolicy(cfg.d_model, cfg.hidden_size).to(device)
    h_policy_new.load_state_dict(torch.load(checkpoint_dir / 'h_policy.pt'))
    pipeline.load_checkpoint(checkpoint_dir / 'transformer.pt')
    print(f"  ✓ Checkpoints loaded")
    
    # Cleanup
    import shutil
    shutil.rmtree(checkpoint_dir)
except Exception as e:
    print(f"  ✗ Checkpoint save/load failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print("\nThe transformer encoder is successfully integrated with your RL framework!")
print("\nNext steps:")
print("  1. Run: python training/integration_example.py")
print("  2. Monitor training metrics")
print("  3. Adjust hyperparameters in training/config.py")
print("=" * 70)
