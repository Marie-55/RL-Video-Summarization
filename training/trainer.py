# trainer.py
import torch
import numpy as np
from typing import List, Tuple
from training.config import Config
from environment import VideoSummarizationEnv
from horizontal_policy import HorizontalPolicy
from vertical_policy import VerticalPolicy

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
    config: Config
) -> Tuple[float, List[int]]:
    obs = env.reset()
    
    log_probs_h, log_probs_v, rewards = [], [], []
    done = False
    
    while not done:
        if obs['turn'] == 'H':
            indices = obs['summary_indices']
            feats = torch.tensor(env.contextual_features[indices], dtype=torch.float32)
            anchor_idx, log_p = h_policy.select_anchor(feats, indices)
            log_probs_h.append(log_p)
            obs, r, done, _ = env.step_H(anchor_idx)
        else:  # 'V'
            anchor = obs['anchor_idx']
            chosen_idx, log_p = v_policy.select_neighbor(anchor, env.contextual_features, config.window_size)
            log_probs_v.append(log_p)
            obs, r, done, _ = env.step_V(chosen_idx)
            
        rewards.append(r)
        
    # --- Policy Update (Episodic REINFORCE) ---
    returns = compute_returns(rewards, config.gamma)
    baseline = returns.mean()
    
    # CORRECT ALIGNMENT: H steps are at even indices (0, 2, 4...), V at odd (1, 3, 5...)
    if log_probs_h:
        h_returns = returns[0::2]  # Only H-step returns
        loss_h = -torch.stack(log_probs_h) @ (h_returns - baseline)
        opt_h.zero_grad()
        loss_h.backward()
        opt_h.step()
        
    if log_probs_v:
        v_returns = returns[1::2]  # Only V-step returns
        loss_v = -torch.stack(log_probs_v) @ (v_returns - baseline)
        opt_v.zero_grad()
        loss_v.backward()
        opt_v.step()
        
    return float(returns.sum()), env.get_final_summary()