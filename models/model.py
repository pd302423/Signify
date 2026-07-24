"""
PyTorch Neural Network Architectures for Sign Language Recognition.

Contains:
1. StaticLandmarkMLP: Lightweight MLP for single-frame ASL alphabet classification.
2. DynamicLandmarkLSTM: Bidirectional LSTM model for dynamic word sequence classification.
3. STGCN_Sign: Spatio-Temporal Graph Convolutional Network for advanced graph-structured landmark sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class StaticLandmarkMLP(nn.Module):
    """
    Shallow Multilayer Perceptron for static sign/letter recognition (A-Z).
    Input: [batch_size, 126] (left hand 63 + right hand 63 landmarks)
    Output: [batch_size, num_classes] (e.g., 26 letters)
    """

    def __init__(self, input_dim: int = 126, hidden_dim: int = 256, num_classes: int = 26, dropout: float = 0.3):
        super(StaticLandmarkMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, input_dim]
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        out = self.fc3(x)
        return out


class DynamicLandmarkLSTM(nn.Module):
    """
    Bidirectional LSTM with attention pooling for dynamic isolated word recognition (WLASL subset).
    Input: [batch_size, sequence_length, feature_dim] e.g. [B, 30, 225]
    Output: [batch_size, num_classes] e.g. 50-100 vocabulary words.
    """

    def __init__(
        self,
        input_dim: int = 225,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 50,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        super(DynamicLandmarkLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention layer for aggregating temporal features
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # [B, seq_len, hidden_dim * directions]

        # Calculate attention weights across frame sequence
        attn_weights = self.attention(lstm_out)  # [B, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted context sum pooling
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [B, hidden_dim * directions]

        # Final classification logits
        logits = self.fc(context)
        return logits


class SimpleGraphConv(nn.Module):
    """Basic Graph Convolutional Layer for skeletal graph modeling."""

    def __init__(self, in_features: int, out_features: int):
        super(SimpleGraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, V, in_features], adj: [V, V]
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return F.relu(output)


if __name__ == "__main__":
    # Sanity checks
    print("Testing PyTorch models initialization...")
    mlp = StaticLandmarkMLP(input_dim=126, num_classes=26)
    dummy_static = torch.randn(8, 126)
    out_mlp = mlp(dummy_static)
    print(f"Static MLP output shape: {out_mlp.shape} (Expected: [8, 26])")

    lstm = DynamicLandmarkLSTM(input_dim=225, num_classes=50)
    dummy_dynamic = torch.randn(8, 30, 225)
    out_lstm = lstm(dummy_dynamic)
    print(f"Dynamic LSTM output shape: {out_lstm.shape} (Expected: [8, 50])")
