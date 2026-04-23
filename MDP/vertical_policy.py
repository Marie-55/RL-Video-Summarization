### in this file, we will implement the vertical policy that will find the neighbors of the anchor frame 
### and will select one of them to replace the anchor with or keep it 
### it is also a feedforward neural network that takes the state as an input and outputs the distribution of probabilities over the neighbors to be selected as the new anchor frame
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class VerticalPolicy(nn.Module):
    def __init__(self, d_model: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
    
    def select_neighbor(self, anchor_idx: int, contextual_features: np.ndarray, window_size: int) -> Tuple[int, torch.Tensor]:
        """
        Returns: (chosen_neighbor_or_anchor_index, log_prob)
        """
        T = contextual_features.shape[0]
        start = max(0, anchor_idx - window_size)
        end = min(T, anchor_idx + window_size + 1)
        
        neighbor_indices = [i for i in range(start, end) if i != anchor_idx]
        candidates = [anchor_idx] + neighbor_indices  # Keep anchor is always an option
        
        cand_feats = torch.tensor(contextual_features[candidates], dtype=torch.float32)
        # Move to the same device as the model
        cand_feats = cand_feats.to(next(self.parameters()).device)
        logits = self.forward(cand_feats)
        dist = torch.distributions.Categorical(logits=logits)
        
        idx = dist.sample()
        chosen_idx = candidates[idx.item()]
        return chosen_idx, dist.log_prob(idx)