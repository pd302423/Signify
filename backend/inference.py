"""
Inference Engine Wrapper for SignFlow Backend.

Loads PyTorch or ONNX models and runs real-time classification on landmark buffers.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from model import DynamicLandmarkLSTM

DEFAULT_CLASSES = [
    "hello", "thank_you", "please", "yes", "no", "help", "sorry", "eat",
    "drink", "more", "finish", "want", "like", "friend", "family", "house",
    "name", "work", "school", "learn", "time", "day", "night", "good",
    "bad", "happy", "sad", "love", "see", "hear", "understand", "again",
    "where", "what", "who", "why", "when", "how", "stop", "go",
    "come", "play", "book", "computer", "phone", "water", "food", "car", "man", "woman"
]


class SignInferenceEngine:
    """Handles real-time model inference, confidence scoring, and top-k extraction."""

    def __init__(self, model_path: str = "weights/sign_lstm.pth", labels_path: str = "weights/dynamic_labels.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Labels
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                self.classes = json.load(f)
        else:
            self.classes = DEFAULT_CLASSES

        # Load PyTorch Model
        self.model = DynamicLandmarkLSTM(input_dim=225, hidden_dim=128, num_classes=len(self.classes)).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded trained PyTorch weights from {model_path}")
        else:
            print(f"Notice: Checkpoint {model_path} not found. Running initialized inference engine.")
        
        self.model.eval()

    def predict_sequence(self, sequence: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
        """
        Runs prediction on a 2D numpy array of shape [seq_len, 225].
        Returns top prediction, confidence, top_k list, and latency in milliseconds.
        """
        start_time = time.perf_counter()

        # Ensure correct shape [1, seq_len, 225]
        if sequence.ndim == 2:
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(sequence_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.classes)))

        predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            predictions.append({
                "label": self.classes[idx],
                "confidence": float(round(prob, 4))
            })

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "prediction": predictions[0]["label"],
            "confidence": predictions[0]["confidence"],
            "top_k": predictions,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time()
        }


if __name__ == "__main__":
    engine = SignInferenceEngine()
    dummy_seq = np.random.randn(30, 225).astype(np.float32)
    res = engine.predict_sequence(dummy_seq)
    print("Inference Test Result:", res)
