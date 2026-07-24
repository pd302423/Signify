"""
Continuous Sign Language Recognition (CSLR) PyTorch Architecture.

Implements a spatio-temporal BiLSTM + CTC (Connectionist Temporal Classification) network.
Translates unsegmented MediaPipe landmark frame sequences into ASL Gloss sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class CSLR_BiLSTM_CTC(nn.Module):
    """
    Spatio-Temporal Sequence Model for Continuous Sign Recognition using CTC Loss.
    
    Pipeline:
    Landmark Sequence [Batch, Time, 225] 
      -> 1D Spatial Convolutions 
      -> Bidirectional LSTM (Temporal Context) 
      -> CTC Linear Projection [Batch, Time, Num_Glosses + 1 (Blank)]
    """

    def __init__(
        self,
        input_dim: int = 225,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_glosses: int = 60,
        dropout: float = 0.3
    ):
        super(CSLR_BiLSTM_CTC, self).__init__()
        
        # Num gloss classes + 1 for CTC blank token (index 0)
        self.num_classes = num_glosses + 1
        self.blank_idx = 0

        # Spatial Feature Extractor (1D Conv across time to capture local motion)
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Temporal Sequence Network (BiLSTM)
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Classifier Projection to Gloss Vocabulary + Blank
        self.fc = nn.Linear(hidden_dim * 2, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch, Time, Features] e.g. [B, T, 225]
        # Reshape for Conv1D: [Batch, Features, Time]
        x_conv = x.transpose(1, 2)
        feats = self.conv1d(x_conv)
        
        # Reshape back for LSTM: [Batch, Time, Features]
        feats = feats.transpose(1, 2)

        lstm_out, _ = self.bilstm(feats)
        
        # Logits shape: [Batch, Time, Num_Classes]
        logits = self.fc(lstm_out)
        return logits

    def decode_greedy(self, logits: torch.Tensor) -> List[List[int]]:
        """
        Greedy CTC Decoding: Takes max index at each frame, collapses consecutive duplicates, and removes blanks.
        """
        probs = F.softmax(logits, dim=-1)
        max_indices = torch.argmax(probs, dim=-1)  # [Batch, Time]

        decoded_sequences = []
        for batch in max_indices:
            seq = []
            prev = self.blank_idx
            for idx in batch.tolist():
                if idx != self.blank_idx and idx != prev:
                    seq.append(idx)
                prev = idx
            decoded_sequences.append(seq)

        return decoded_sequences


def compute_ctc_loss(logits: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
    """
    Computes Connectionist Temporal Classification (CTC) Loss.
    logits: [Batch, Time, Num_Classes] -> Transposed to [Time, Batch, Num_Classes] for PyTorch CTCLoss
    """
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [Time, Batch, Num_Classes]
    loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
    return loss


if __name__ == "__main__":
    print("Testing CSLR BiLSTM CTC Model...")
    model = CSLR_BiLSTM_CTC(input_dim=225, num_glosses=50)
    dummy_sequence = torch.randn(4, 100, 225)  # Batch of 4 sequences with 100 frames
    logits = model(dummy_sequence)
    print(f"Logits Output Shape: {logits.shape} (Expected: [4, 100, 51])")
    
    decoded = model.decode_greedy(logits)
    print(f"Greedy Decoded Sequences Sample: {decoded[0]}")
