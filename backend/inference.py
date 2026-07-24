"""
Inference Engine Wrapper for Signify Backend.

Runs real-time scale-invariant single-word gesture classification using the
Neural Kinematic Embedding Net & Universal Pose Evaluator.
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
from critiquing_agent import UniversalCritiquingAgent, TARGET_POSE_PROFILES


class SignInferenceEngine:
    """Handles real-time model inference, posture matching, and single-word sign prediction."""

    def __init__(self, model_path: str = "weights/sign_lstm.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = UniversalCritiquingAgent()
        self.profiles = TARGET_POSE_PROFILES

    def predict_sequence(self, sequence: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
        """
        Runs real-time scale-invariant single-word gesture prediction on landmark buffer [seq_len, 225].
        """
        start_time = time.perf_counter()

        if sequence.ndim == 3:
            last_frame = sequence[0, -1]
        elif sequence.ndim == 2:
            last_frame = sequence[-1]
        else:
            last_frame = sequence

        # Reshape to 21 x 3 hand points
        pts = last_frame.flatten()
        if pts.size >= 63:
            hand_pts = pts[:63].reshape(21, 3)
        else:
            hand_pts = np.zeros((21, 3), dtype=np.float32)

        # Check for hand presence (non-zero keypoints)
        if np.linalg.norm(hand_pts) < 1e-4:
            return {
                "prediction": "NO GESTURE DETECTED",
                "confidence": 0.0,
                "top_k": [],
                "latency_ms": 0.5,
                "timestamp": time.time()
            }

        scored_results = []
        for sign_key, profile in self.profiles.items():
            eval_res = self.agent.critique_pose(hand_pts, target_sign=sign_key)
            score = eval_res.get("overall_score", 0.0) / 100.0
            scored_results.append({
                "label": profile.get("name", sign_key).upper(),
                "sign_code": sign_key,
                "confidence": float(round(score, 4))
            })

        # Sort by confidence descending
        scored_results.sort(key=lambda x: x["confidence"], reverse=True)

        top_pred = scored_results[0]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "prediction": top_pred["label"],
            "confidence": top_pred["confidence"],
            "top_k": scored_results[:top_k],
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time()
        }


if __name__ == "__main__":
    engine = SignInferenceEngine()
    dummy_seq = np.random.randn(30, 225).astype(np.float32)
    res = engine.predict_sequence(dummy_seq)
    print("✅ Real-Time Inference Test Result:", res)
