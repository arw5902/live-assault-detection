import math
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


class HazardLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.25):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        """
        x: [batch, window_len, input_dim]
        returns: [batch] hazard score in [0,1]
        """
        out, _ = self.lstm(x)             # [B, T, H]
        h_last = out[:, -1, :]            # last timestep [B, H]
        h_last = self.dropout(h_last)

        y = self.fc(h_last)              # [B,1]
        y = torch.sigmoid(y)

        return y.squeeze(1)              # [B]


class HazardTransformer(nn.Module):
    """
    Lightweight Transformer encoder for hazard scoring.

    Architecture:
      1. Linear input projection  : input_dim → d_model
      2. Learnable positional emb : window_len positions × d_model
      3. TransformerEncoder       : num_layers × (self-attn + FFN)
      4. Mean pooling over time
      5. Dropout + Linear(d_model, 1) + Sigmoid

    With default d_model=64, nhead=4, dim_feedforward=128, num_layers=1
    the parameter count is ~41 K — between GRU (35 K) and LSTM (47 K).
    """

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 1, dim_feedforward: int = 128,
                 dropout: float = 0.25, max_seq_len: int = 32):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: [batch, window_len, input_dim]
        returns: [batch] hazard score in [0,1]
        """
        B, T, _ = x.shape

        x = self.input_proj(x)                    # [B, T, d_model]
        x = x + self.pos_embed[:, :T, :]          # add positional embedding

        x = self.transformer(x)                   # [B, T, d_model]

        x = x.mean(dim=1)                         # mean pool over time [B, d_model]
        x = self.dropout(x)

        y = self.fc(x)                            # [B, 1]
        y = torch.sigmoid(y)

        return y.squeeze(1)                       # [B]


class HazardCNN(nn.Module):
    """
    Lightweight 1-D temporal CNN for hazard scoring.

    Architecture:
      1. Conv1d(input_dim → hidden, kernel=3, pad=1) + ReLU
      2. Conv1d(hidden → hidden, kernel=3, pad=1) + ReLU
      3. Global average pooling over time
      4. Dropout + Linear(hidden, 1) + Sigmoid

    With default hidden=64 and typical input_dim~20 the parameter count is
    ~16 K — the lightest of the four architectures, well suited to the small
    training corpus and short (≤16-step) windows.  The two stacked conv layers
    give a receptive field of 5 timesteps, enough to capture the rapid
    kinematic bursts that precede a push, without the risk of overfitting that
    learnable positional embeddings bring to the Transformer.
    """

    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.25):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        """
        x: [batch, window_len, input_dim]
        returns: [batch] hazard score in [0,1]
        """
        x = x.permute(0, 2, 1)           # [B, input_dim, T] — Conv1d expects (B, C, L)
        x = self.conv(x)                  # [B, hidden, T]
        x = x.mean(dim=2)                 # global average pool → [B, hidden]
        x = self.dropout(x)
        y = self.fc(x)                    # [B, 1]
        y = torch.sigmoid(y)
        return y.squeeze(1)               # [B]
