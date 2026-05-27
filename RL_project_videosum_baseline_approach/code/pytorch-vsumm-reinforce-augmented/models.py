import torch
import torch.nn as nn

class DSN(nn.Module):
    """Deep Summarization Network with LSTM"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.cell_name = cell
        
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers, batch_first=True)
        elif cell == 'gru':
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers, batch_first=True)
        else:
            raise ValueError(f"Unsupported cell type: {cell}")
        
        self.fc = nn.Linear(hid_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, in_dim)
        Returns:
            probs: (batch_size, seq_len, 1) - frame importance scores [0, 1]
        """
        self.rnn.flatten_parameters()
        out, _ = self.rnn(x)  # (batch_size, seq_len, hid_dim)
        probs = self.fc(out)  # (batch_size, seq_len, 1)
        probs = self.sigmoid(probs)
        return probs
