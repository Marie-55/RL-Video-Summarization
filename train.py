#!/usr/bin/env python3
"""
End-to-End Training Script for RL Video Summarization

Trains the full pipeline (Transformer + Policies) on all available video data using REINFORCE.

Usage:
    python train.py [--num_epochs NUM] [--batch_size SIZE] [--videos_dir PATH] [--checkpoint CKPT]
    
Examples:
    python train.py                                    # Train with defaults (10 epochs)
    python train.py --num_epochs 50                    # Train for 50 epochs
    python train.py --videos_dir /path/to/videos       # Use custom video directory
    python train.py --checkpoint checkpoints/model.pt  # Resume from checkpoint
"""

import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
from typing import Optional


# Add project root and MDP to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'MDP'))

from training.config import Config
from training.data_loader import VideoEmbeddingLoader
from training.feature_pipeline import FeaturePipeline
from training.trainer import train_on_video
from horizontal_policy import HorizontalPolicy
from vertical_policy import VerticalPolicy
from environment import VideoSummarizationEnv


class Trainer:
    """End-to-end trainer for RL video summarization."""
    
    def __init__(
        self,
        config: Config,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'checkpoints',
    ):
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Set seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        print(f"[Trainer] Device: {device}")
        print(f"[Trainer] Checkpoint directory: {self.checkpoint_dir}")
        
    def build_models(self) -> tuple:
        """Build transformer encoder and policies."""
        print("\n" + "=" * 70)
        print("BUILDING MODELS")
        print("=" * 70)
        
        # Transformer encoder
        pipeline = FeaturePipeline(self.config, device=self.device)
        
        # Policies (on CPU by default, will be moved to device during training)
        h_policy = HorizontalPolicy(self.config.d_model, self.config.hidden_size)
        v_policy = VerticalPolicy(self.config.d_model, self.config.hidden_size)
        
        return pipeline, h_policy, v_policy
    
    def load_data(self, videos_dir: str = 'videos/embeddings_clip_vitl14') -> dict:
        """Load all video embeddings organized by category."""
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)
        
        loader = VideoEmbeddingLoader(videos_dir=videos_dir)
        categories = loader.get_categories()
        
        all_data = {}
        total_videos = 0
        
        for category in categories:
            stats = loader.get_statistics(category)
            print(f"\n  Category: {category}")
            print(f"    Videos: {stats['num_videos']}")
            print(f"    Frame range: {stats['min_length']}-{stats['max_length']}")
            print(f"    Avg frames: {stats['avg_length']:.1f}")
            
            embeddings = loader.load_category(category, num_videos=None, shuffle=True)
            all_data[category] = embeddings
            total_videos += len(embeddings)
        
        print(f"\n  Total videos loaded: {total_videos}")
        return all_data
    
    def train_epoch(
        self,
        epoch: int,
        all_data: dict,
        pipeline: FeaturePipeline,
        h_policy: HorizontalPolicy,
        v_policy: VerticalPolicy,
        opt_h: torch.optim.Optimizer,
        opt_v: torch.optim.Optimizer,
        opt_transformer: torch.optim.Optimizer,
    ) -> dict:
        """Train for one epoch over all videos."""
        h_policy.train()
        v_policy.train()
        
        epoch_stats = {
            'epoch': epoch,
            'total_reward': 0.0,
            'num_videos': 0,
            'num_categories': 0,
            'category_stats': {},
        }
        
        pbar = tqdm(total=sum(len(v) for v in all_data.values()), desc=f"Epoch {epoch+1}")
        
        for category, videos in all_data.items():
            category_rewards = []
            
            for video_idx, raw_features in enumerate(videos):
                # Encode video
                ctx = pipeline.encode_video(raw_features)
                
                # Create environment and train
                env = VideoSummarizationEnv(ctx, self.config)
                reward, summary = train_on_video(
                    env, h_policy, v_policy,
                    opt_h, opt_v,
                    self.config,
                    pipeline=pipeline,
                    opt_transformer=opt_transformer,
                )
                
                category_rewards.append(reward)
                epoch_stats['total_reward'] += reward
                epoch_stats['num_videos'] += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'category': category,
                    'reward': f"{reward:.4f}",
                    'summary_size': len(summary),
                })
            
            # Store category statistics
            if category_rewards:
                epoch_stats['category_stats'][category] = {
                    'num_videos': len(category_rewards),
                    'mean_reward': np.mean(category_rewards),
                    'std_reward': np.std(category_rewards),
                    'min_reward': float(np.min(category_rewards)),
                    'max_reward': float(np.max(category_rewards)),
                }
                epoch_stats['num_categories'] += 1
        
        pbar.close()
        
        # Compute epoch averages
        if epoch_stats['num_videos'] > 0:
            epoch_stats['avg_reward'] = epoch_stats['total_reward'] / epoch_stats['num_videos']
        
        return epoch_stats
    
    def save_checkpoint(
        self,
        epoch: int,
        pipeline: FeaturePipeline,
        h_policy: HorizontalPolicy,
        v_policy: VerticalPolicy,
        stats: dict,
        is_best: bool = False,
    ):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'config': self.config.__dict__,
            'h_policy': h_policy.state_dict(),
            'v_policy': v_policy.state_dict(),
            'transformer': pipeline.encoder.state_dict(),
            'stats': stats,
        }
        
        # Save regular checkpoint
        ckpt_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
        torch.save(checkpoint, ckpt_path)
        print(f"\n  ✓ Checkpoint saved: {ckpt_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model updated: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> tuple:
        """Load training checkpoint."""
        print(f"\n[Trainer] Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Rebuild models
        pipeline, h_policy, v_policy = self.build_models()
        h_policy.load_state_dict(checkpoint['h_policy'])
        v_policy.load_state_dict(checkpoint['v_policy'])
        pipeline.encoder.load_state_dict(checkpoint['transformer'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"  ✓ Resuming from epoch {start_epoch}")
        
        return pipeline, h_policy, v_policy, start_epoch
    
    def train(
        self,
        num_epochs: int = 10,
        videos_dir: str = 'videos/embeddings_clip_vitl14',
        checkpoint_path: Optional[str] = None,
    ):
        """Run full training loop."""
        print("\n" + "=" * 70)
        print("RL VIDEO SUMMARIZATION — END-TO-END TRAINING")
        print("=" * 70)
        print(f"Config: {self.config}")
        
        # Load data
        all_data = self.load_data(videos_dir)
        
        # Build or load models
        if checkpoint_path and Path(checkpoint_path).exists():
            pipeline, h_policy, v_policy, start_epoch = self.load_checkpoint(checkpoint_path)
        else:
            pipeline, h_policy, v_policy = self.build_models()
            start_epoch = 0
        
        # Optimizers
        opt_h = torch.optim.Adam(h_policy.parameters(), lr=self.config.lr)
        opt_v = torch.optim.Adam(v_policy.parameters(), lr=self.config.lr)
        opt_transformer = pipeline.get_optimizer(lr=self.config.lr)
        
        # Training loop
        print("\n" + "=" * 70)
        print("TRAINING")
        print("=" * 70)
        
        all_stats = []
        best_reward = -np.inf
        
        for epoch in range(start_epoch, num_epochs):
            epoch_stats = self.train_epoch(
                epoch,
                all_data,
                pipeline, h_policy, v_policy,
                opt_h, opt_v, opt_transformer,
            )
            all_stats.append(epoch_stats)
            
            # Print epoch summary
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{num_epochs} Summary")
            print(f"{'='*70}")
            print(f"  Total videos: {epoch_stats['num_videos']}")
            print(f"  Categories: {epoch_stats['num_categories']}")
            print(f"  Avg reward: {epoch_stats['avg_reward']:.4f}")
            print(f"  Total reward: {epoch_stats['total_reward']:.4f}")
            
            print(f"\n  Per-category breakdown:")
            for category, cat_stats in epoch_stats['category_stats'].items():
                print(f"    {category}:")
                print(f"      Videos: {cat_stats['num_videos']}")
                print(f"      Mean reward: {cat_stats['mean_reward']:.4f} ± {cat_stats['std_reward']:.4f}")
                print(f"      Range: [{cat_stats['min_reward']:.4f}, {cat_stats['max_reward']:.4f}]")
            
            # Save checkpoint
            is_best = epoch_stats['avg_reward'] > best_reward
            if is_best:
                best_reward = epoch_stats['avg_reward']
            
            self.save_checkpoint(epoch, pipeline, h_policy, v_policy, all_stats, is_best=is_best)
        
        # Final summary
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"  Best average reward: {best_reward:.4f}")
        print(f"  Checkpoints saved to: {self.checkpoint_dir}")
        
        # Save final stats
        stats_path = self.checkpoint_dir / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"  Training statistics saved to: {stats_path}")
        
        return all_stats


def main():
    parser = argparse.ArgumentParser(description='Train RL Video Summarization Model')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--videos_dir', type=str, default='videos/embeddings_clip_vitl14',
                        help='Path to video embeddings directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Auto-detected if not specified')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create config
    config = Config(seed=args.seed)
    
    # Initialize trainer
    trainer = Trainer(config, device=device, checkpoint_dir=args.checkpoint_dir)
    
    # Run training
    try:
        trainer.train(
            num_epochs=args.num_epochs,
            videos_dir=args.videos_dir,
            checkpoint_path=args.checkpoint,
        )
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
