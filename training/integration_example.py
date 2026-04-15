"""
Integration Guide: Temporal Transformer Encoder with RL Framework

This module demonstrates how to integrate the temporal transformer encoder
with your RL video summarization system for end-to-end training.

Data Format:
    - Embeddings: CLIP ViT-L/14 (768-dimensional)
    - Format: .npz files in videos/embeddings_clip_vitl14/<category>/
    - Structure: {'embeddings': (T, 768), 'frame_names': (T,), 'category': str, 'video_name': str}
"""

import torch
import numpy as np
from pathlib import Path
from training.config import Config
from training.feature_pipeline import FeaturePipeline
from training.data_loader import VideoEmbeddingLoader
from environment import VideoSummarizationEnv
from horizontal_policy import HorizontalPolicy
from vertical_policy import VerticalPolicy
from trainer import train_on_video


class VideoSummarizationTrainer:
    """
    Complete trainer integrating transformer encoder + RL policies.
    
    Usage:
        trainer = VideoSummarizationTrainer(config)
        trainer.train_episode(raw_cnn_features)
    """
    
    def __init__(self, config: Config, device: str = None, end_to_end: bool = True):
        """
        Args:
            config: Config object with all parameters
            device: 'cpu' or 'cuda', auto-detected if None
            end_to_end: If True, train transformer end-to-end with policies
        """
        self.config = config
        self.device = FeaturePipeline._get_device(device)
        self.end_to_end = end_to_end
        
        # Initialize components
        self.pipeline = FeaturePipeline(config, device=self.device)
        self.h_policy = HorizontalPolicy(config.d_model, config.hidden_size).to(self.device)
        self.v_policy = VerticalPolicy(config.d_model, config.hidden_size).to(self.device)
        
        # Optimizers
        self.opt_h = torch.optim.Adam(self.h_policy.parameters(), lr=config.lr)
        self.opt_v = torch.optim.Adam(self.v_policy.parameters(), lr=config.lr)
        
        if end_to_end:
            self.opt_transformer = self.pipeline.get_optimizer(lr=config.lr)
        else:
            self.pipeline.freeze()
            self.opt_transformer = None
    
    def train_episode(self, raw_cnn_features: np.ndarray) -> tuple:
        """
        Train one episode on a single video.
        
        Args:
            raw_cnn_features: (T, 1024) raw CNN features
            
        Returns:
            (episode_reward, final_summary_indices, contextual_features)
        """
        # Step 1: Encode raw features with transformer
        contextual_features = self.pipeline.encode_video(raw_cnn_features)
        
        # Step 2: Create environment
        env = VideoSummarizationEnv(contextual_features, self.config)
        
        # Step 3: Train
        episode_reward, final_summary = train_on_video(
            env, self.h_policy, self.v_policy,
            self.opt_h, self.opt_v, self.config,
            pipeline=self.pipeline,
            opt_transformer=self.opt_transformer
        )
        
        return episode_reward, final_summary, contextual_features
    
    def train_batch(self, video_features_list: list, num_episodes: int = 1) -> dict:
        """
        Train on a batch of videos for multiple episodes.
        
        Args:
            video_features_list: List of (T_i, 1024) arrays
            num_episodes: Number of training episodes per video
            
        Returns:
            Dictionary with training statistics
        """
        total_reward = 0.0
        all_summaries = []
        
        for episode in range(num_episodes):
            for video_idx, raw_features in enumerate(video_features_list):
                reward, summary, _ = self.train_episode(raw_features)
                total_reward += reward
                all_summaries.append(summary)
                
                print(f"Episode {episode+1}/{num_episodes}, Video {video_idx+1}/{len(video_features_list)}: "
                      f"Reward={reward:.4f}, Summary size={len(summary)}")
        
        avg_reward = total_reward / (num_episodes * len(video_features_list))
        return {
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'summaries': all_summaries,
            'num_episodes': num_episodes,
            'num_videos': len(video_features_list),
        }
    
    def save_checkpoint(self, checkpoint_dir: str):
        """Save all model checkpoints."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        torch.save(self.h_policy.state_dict(), checkpoint_dir / 'h_policy.pt')
        torch.save(self.v_policy.state_dict(), checkpoint_dir / 'v_policy.pt')
        self.pipeline.save_checkpoint(checkpoint_dir / 'transformer.pt')
        
        print(f"[Trainer] Checkpoints saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load all model checkpoints."""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load models
        self.h_policy.load_state_dict(torch.load(checkpoint_dir / 'h_policy.pt'))
        self.v_policy.load_state_dict(torch.load(checkpoint_dir / 'v_policy.pt'))
        self.pipeline.load_checkpoint(checkpoint_dir / 'transformer.pt')
        
        print(f"[Trainer] Checkpoints loaded from {checkpoint_dir}")


# Example usage
if __name__ == "__main__":
    # Initialize config and trainer
    cfg = Config()
    trainer = VideoSummarizationTrainer(cfg, device='cuda', end_to_end=True)
    
    # Load real video embeddings from disk
    print("\n" + "=" * 60)
    print("LOADING VIDEO EMBEDDINGS FROM DISK")
    print("=" * 60)
    
    loader = VideoEmbeddingLoader(videos_dir='videos/embeddings_clip_vitl14')
    
    # Load videos from Activity category (change to any available category)
    category = 'Activity'
    video_features_list = loader.load_category(category, num_videos=5, shuffle=True)
    
    print(f"\nLoaded {len(video_features_list)} videos from '{category}' category")
    for i, features in enumerate(video_features_list):
        print(f"  Video {i+1}: {features.shape[0]} frames, {features.shape[1]} dimensions")
    
    # Train
    print("\n" + "=" * 60)
    print("VIDEO SUMMARIZATION WITH TEMPORAL TRANSFORMER (End-to-End)")
    print("=" * 60)
    
    stats = trainer.train_batch(video_features_list, num_episodes=2)
    
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total Reward    : {stats['total_reward']:.4f}")
    print(f"Average Reward  : {stats['avg_reward']:.4f}")
    print(f"Episodes        : {stats['num_episodes']}")
    print(f"Videos          : {stats['num_videos']}")
    print("=" * 60)
    
    # Save checkpoints
    trainer.save_checkpoint('./checkpoints')
