from dataclasses import dataclass

@dataclass
class Config:
    # Transformer Encoder Architecture
    transformer_input_dim: int = 768     # CLIP ViT-L/14 embeddings
    transformer_d_model: int = 512       # Transformer hidden dimension
    transformer_nhead: int = 8           # Number of attention heads (512/8 = 64 per head)
    transformer_num_layers: int = 3      # Number of encoder layers
    transformer_dim_feedfwd: int = 2048  # FFN hidden size (4 × d_model)
    transformer_dropout: float = 0.1     # Dropout rate
    transformer_max_len: int = 5000      # Maximum sequence length
    
    # Policy Architecture
    d_model: int = 512                   # Policy input dimension (from transformer output)
    hidden_size: int = 256               # Policy hidden dimension
    
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