from dataclasses import dataclass

@dataclass
class Config:
    # Model Architecture
    d_model: int = 256
    hidden_size: int = 256
    
    # Episode & Summary Control
    alpha: float = 0.02          # Initial summary size ratio (k = alpha * T)
    min_k: int = 3
    max_k: int = 20
    window_size: int = 5         # Vertical policy temporal window
    max_steps: int = 100         # Hard step budget per episode
    stability_patience: int = 5  # Stop if no summary changes for this many V-steps
    
    # Reward
    w_div: float = 0.5
    w_rep: float = 0.5
    
    # Training
    lr: float = 1e-3
    gamma: float = 1.0           # Discount factor for REINFORCE returns
    seed: int = 42