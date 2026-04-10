## in this file , we will define the reward class that computes the reward for a given state , the reward will be computed using the frames in the summary 
## it will include the diversity reward and the representative reward along with the temporal reward 
## in this file we will define the state class , that is a set of frames that represents the state of the summary at a given time 
# reward.py
import numpy as np
from typing import Set
from state import State

class Reward:
    def __init__(self, w_div: float = 0.5, w_rep: float = 0.5):
        self.w_div = w_div
        self.w_rep = w_rep

    @staticmethod
    def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity between rows of A and B."""
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return A_norm @ B_norm.T

    def compute_diversity_reward(self, feat_sel: np.ndarray) -> float:
        """Compute diversity reward as mean(1 - cos_sim) over all summary pairs."""
        K = len(feat_sel)
        
        if K > 1:
            sim_sel = self._cosine_sim_matrix(feat_sel, feat_sel)
            # Zero out diagonal, sum upper triangle
            mask = ~np.eye(K, dtype=bool)
            div_reward = 1.0 - (sim_sel[mask].sum() / mask.sum())
        else:
            div_reward = 0.0
            
        return div_reward

    def compute_representative_reward(self, feat_all: np.ndarray, feat_sel: np.ndarray) -> float:
        """Compute representativeness reward as mean(max_cos_sim) of each video frame to summary."""
        sim_rep = self._cosine_sim_matrix(feat_all, feat_sel)  # (T, K)
        rep_reward = float(np.mean(np.max(sim_rep, axis=1)))
        return rep_reward

    def compute_total_reward(self, contextual_features: np.ndarray, selected_indices: Set[int]) -> float:
        """Compute total reward combining diversity and representativeness."""
        if not selected_indices:
            return 0.0
            
        feat_all = contextual_features
        feat_sel = feat_all[sorted(selected_indices)]  # (K, D)
        
        # Compute individual rewards
        div_reward = self.compute_diversity_reward(feat_sel)
        rep_reward = self.compute_representative_reward(feat_all, feat_sel)
        
        return self.w_div * div_reward + self.w_rep * rep_reward