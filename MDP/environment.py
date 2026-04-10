# environment.py
import numpy as np
from typing import Dict, Any, List, Optional
from state import State
from reward import Reward

class VideoSummarizationEnv:
    def __init__(self, contextual_features: np.ndarray, config):
        self.contextual_features = contextual_features  # (T, D)
        self.T = contextual_features.shape[0]
        self.config = config
        
        self.state: Optional[State] = None
        self.step_count = 0
        self.patience_counter = 0
        self.policy_turn = 'H'
        self.reward_fn = Reward(config.w_div, config.w_rep)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            np.random.seed(seed)
            
        k = max(self.config.min_k, min(int(self.T * self.config.alpha), self.config.max_k))
        init_indices = np.random.choice(self.T, size=k, replace=False).tolist()
        anchor_idx = np.random.choice(init_indices).item()
        
        self.state = State(init_indices, anchor_idx)
        self.step_count = 0
        self.patience_counter = 0
        self.policy_turn = 'H'
        # self._prev_summary = set(self.state.selected_indices)
        
        return self._get_observation()

    def step_H(self, chosen_anchor: int) -> tuple:
        assert self.policy_turn == 'H'
        if chosen_anchor not in self.state.selected_indices:
            raise ValueError("H-policy must select from current summary indices.")
            
        self.state.anchor_idx = chosen_anchor
        self.policy_turn = 'V'
        # H-step doesn't change summary -> reward=0, done=False
        return self._get_observation(), 0.0, False, {"action_type": "H"}

    def step_V(self, chosen_idx: int) -> tuple:
        assert self.policy_turn == 'V'
        
        changed = False
        if chosen_idx not in self.state.selected_indices:
            changed = self.state.replace_anchor(chosen_idx)
            
        if not changed:
            self.patience_counter += 1
        else:
            self.patience_counter = 0
            # self._prev_summary = set(self.state.selected_indices)
            
        self.policy_turn = 'H'
        
        # Compute reward ONLY when summary changes or episode ends
        reward = self.reward_fn.compute_total(self.contextual_features, self.state.selected_indices)
        done = self.step_count >= self.config.max_steps or \
               self.patience_counter >= self.config.stability_patience
               
        self.step_count += 1
        return self._get_observation(), reward, done, {"action_type": "V", "changed": changed}

    def _get_observation(self) -> Dict[str, Any]:
        return {
            'summary_indices': sorted(list(self.state.selected_indices)),
            'anchor_idx': self.state.anchor_idx,
            'contextual_features': self.contextual_features,
            'turn': self.policy_turn
        }

    def get_final_summary(self) -> List[int]:
        return self.state.get_sorted_indices()