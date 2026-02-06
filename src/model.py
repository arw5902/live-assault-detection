import torch
import torch.nn as nn

class HazardGRU(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.25):
        super().__init__()

        # 1-layer GRU (dropout inside GRU has no effect when num_layers=1)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True
        )

        # Apply dropout AFTER GRU (this works with 1 layer)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        """
        x: [batch, window_len, input_dim]
        returns: [batch] hazard score in [0,1]
        """

        out, _ = self.gru(x)          # [B, T, H]
        h_last = out[:, -1, :]        # last timestep [B, H]
        h_last = self.dropout(h_last)

        y = self.fc(h_last)           # [B,1]
        y = torch.sigmoid(y)

        return y.squeeze(1)           # [B]

# Dataset building functions moved to dataset.py
# This file now only contains the GRU model definition
