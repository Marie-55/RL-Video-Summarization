### In this section, we will define the horizontal policy class that will be a feedforward neural network that takes 
### the state as an input and outputs the distribution of probabilities over the frames within the summary(state) to be 
### selected as the anchor frame 

from state import State
from reward import Reward
from frame import Frame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class HorizontalPolicy(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(HorizontalPolicy, self).__init__()
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

    def select_anchor(self, state: State):
        frames = list(state.frames.values())
        frame_data = torch.stack([
            torch.tensor(f.data, dtype=torch.float32) for f in frames
        ])                                          

        probs = self.forward(frame_data)             
        m = torch.distributions.Categorical(probs)
        action = m.sample()                         
        selected_frame = frames[action.item()]
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