### in this file, we will implement the vertical policy that will find the neighbors of the anchor frame 
### and will select one of them to replace the anchor with or keep it 
### it is also a feedforward neural network that takes the state as an input and outputs the distribution of probabilities over the neighbors to be selected as the new anchor frame
from state import State
from reward import Reward
from frame import Frame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
class VerticalPolicy(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VerticalPolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1) 

    def forward(self, frame_data):
        out = self.fc1(frame_data)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out).squeeze(-1)      
        return torch.softmax(out, dim=-1)
    def select_neighbor(self, anchor: Frame, video: dict, window_size: int = 5):
        video_length = len(video)
        anchor_idx = anchor.index
        neighbor_indices = [
            i for i in range(
                max(0, anchor_idx - window_size),
                min(video_length, anchor_idx + window_size + 1)
            )
            if i != anchor_idx        
            and i in video           
        ]

        if not neighbor_indices:
            return anchor, None       

        neighbor_frames = [video[i] for i in neighbor_indices]
        neighbor_data = torch.stack([
            torch.tensor(f.data, dtype=torch.float32) for f in neighbor_frames
        ])

        probs = self.forward(neighbor_data)
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        selected_frame = neighbor_frames[action.item()]
        log_prob = m.log_prob(action)

        return selected_frame, log_prob
    def update_policy(self, log_probs, returns, optimizer):
        loss = 0.0
        baseline = np.mean(returns)              
        for log_prob, G_t in zip(log_probs, returns):
            loss += -log_prob * (G_t - baseline) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()