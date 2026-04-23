# trainer.py
import torch
import numpy as np
from typing import List, Tuple, Optional
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
) -> Tuple[float, List[int]]:
    obs = env.reset()
    
    log_probs_h, log_probs_v, rewards = [], [], []
    # Track which reward index corresponds to which policy step
    h_reward_indices, v_reward_indices = [], []
    done = False
    
    while not done:
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

    # Zero all gradients upfront
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

    if total_loss is not None and total_loss.requires_grad:
        total_loss.backward()
        opt_h.step()
        opt_v.step()
        if end_to_end:
            opt_transformer.step()

    return float(returns.sum()), env.get_final_summary()