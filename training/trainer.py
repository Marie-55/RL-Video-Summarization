# trainer.py
import torch
import csv
import os
import numpy as np
from typing import List, Tuple, Optional, Dict
from training.config import Config
from environment import VideoSummarizationEnv
from horizontal_policy import HorizontalPolicy
from vertical_policy import VerticalPolicy
from training.feature_pipeline import FeaturePipeline

def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

def train_on_video(
    env: VideoSummarizationEnv,
    h_policy: HorizontalPolicy,
    v_policy: VerticalPolicy,
    opt_h: torch.optim.Optimizer,
    opt_v: torch.optim.Optimizer,
    config: Config,
    pipeline: Optional[FeaturePipeline] = None,
    opt_transformer: Optional[torch.optim.Optimizer] = None,
    log_path: Optional[str] = None,
    episode_idx: int = 0,
) -> Tuple[float, List[int], Dict]:
    obs = env.reset()
    
    log_probs_h, log_probs_v, rewards = [], [], []
    h_reward_indices, v_reward_indices = [], []
    step_count = 0
    done = False
    
    while not done:
        step_count += 1
        if obs['turn'] == 'H':
            indices = obs['summary_indices']
            feats = torch.tensor(env.contextual_features[indices], dtype=torch.float32)
            anchor_idx, log_p = h_policy.select_anchor(feats, indices)
            log_probs_h.append(log_p)
            obs, r, done, _ = env.step_H(anchor_idx)
            h_reward_indices.append(len(rewards))
            rewards.append(r)
        else:  # 'V'
            anchor = obs['anchor_idx']
            chosen_idx, log_p = v_policy.select_neighbor(anchor, env.contextual_features, config.window_size)
            log_probs_v.append(log_p)
            obs, r, done, _ = env.step_V(chosen_idx)
            v_reward_indices.append(len(rewards))
            rewards.append(r)

    # --- Policy Update (Episodic REINFORCE) ---
    returns = compute_returns(rewards, config.gamma)
    baseline = returns.mean()

    h_returns = returns[h_reward_indices] if h_reward_indices else torch.tensor([])
    v_returns = returns[v_reward_indices] if v_reward_indices else torch.tensor([])

    end_to_end = pipeline is not None and opt_transformer is not None

    opt_h.zero_grad()
    opt_v.zero_grad()
    if end_to_end:
        opt_transformer.zero_grad()

    total_loss = torch.tensor(0.0, requires_grad=True) if not (log_probs_h or log_probs_v) else None

    if log_probs_h and log_probs_v:
        loss_h = -(torch.stack(log_probs_h) * (h_returns - baseline)).sum()
        loss_v = -(torch.stack(log_probs_v) * (v_returns - baseline)).sum()
        total_loss = loss_h + loss_v
    elif log_probs_h:
        loss_h = -(torch.stack(log_probs_h) * (h_returns - baseline)).sum()
        total_loss = loss_h
    elif log_probs_v:
        loss_v = -(torch.stack(log_probs_v) * (v_returns - baseline)).sum()
        total_loss = loss_v

    loss_val = 0.0
    if total_loss is not None and total_loss.requires_grad:
        total_loss.backward()
        opt_h.step()
        opt_v.step()
        if end_to_end:
            opt_transformer.step()
        loss_val = total_loss.item()

    final_summary = env.get_final_summary()
    summary_length = len(final_summary)
    total_reward = float(returns.sum())

    # --- CSV Logging ---
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_exists = os.path.isfile(log_path)
        
        # NOTE: div_reward & rep_reward require environment.py to return them explicitly.
        # For now, we log total_reward. If you modify env.step_V to return (r_total, r_div, r_rep),
        # replace the 0.0 placeholders below.
        div_reward = 0.0
        rep_reward = 0.0
        
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'episode', 'total_reward', 'div_reward', 'rep_reward',
                'summary_length', 'steps_taken', 'loss'
            ])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'episode': episode_idx,
                'total_reward': f"{total_reward:.4f}",
                'div_reward': f"{div_reward:.4f}",
                'rep_reward': f"{rep_reward:.4f}",
                'summary_length': summary_length,
                'steps_taken': step_count,
                'loss': f"{loss_val:.4f}"
            })

    metrics = {
        'total_reward': total_reward,
        'summary_length': summary_length,
        'steps_taken': step_count,
        'loss': loss_val,
        'converged_by_patience': step_count < config.max_steps
    }
    
    return total_reward, final_summary, metrics