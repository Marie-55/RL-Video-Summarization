### In this section, we will define the horizontal policy class that will be a feedforward neural network that takes 
### the state as an input and outputs the distribution of probabilities over the frames within the summary(state) to be 
### selected as the anchor frame 

import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np
import state as State

class HorizontalPolicy(nn.Module):
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
        # x: (num_candidates, d_model) -> logits: (num_candidates,)
        return self.net(x).squeeze(-1)
  
    def select_anchor(self, summary_features: torch.Tensor, summary_indices: List[int]) -> Tuple[int, torch.Tensor]:
        """
        Args:
            summary_features: (K, d_model) contextual features of currently selected frames
            summary_indices: corresponding frame indices
        Returns:
            (chosen_anchor_index, log_prob)
        """
        logits = self.forward(summary_features)
        dist = torch.distributions.Categorical(logits=logits)
        idx_in_summary = dist.sample()
        chosen_idx = summary_indices[idx_in_summary.item()]
        return chosen_idx, dist.log_prob(idx_in_summary)