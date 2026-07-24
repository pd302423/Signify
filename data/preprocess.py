"""
Data Preprocessing & MediaPipe Landmark Extraction Module for SignFlow.

Extracts normalized 3D keypoint features from raw video frames or camera streams using MediaPipe Holistic.
Saves extracted landmark sequences to NumPy format (.npy) for model training.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Optional

# MediaPipe Solution Initialization
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


class LandmarkExtractor:
    """Extracts and normalizes skeleton keypoints from images/video frames."""

    def __init__(self, static_image_mode: bool = False, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.holistic = mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a BGR image frame and returns a flattened 1D NumPy feature vector of normalized keypoints.
        Vector layout:
        - Pose landmarks: 33 points x (x, y, z) = 99 features
        - Left hand landmarks: 21 points x (x, y, z) = 63 features (zeroed if missing)
        - Right hand landmarks: 21 points x (x, y, z) = 63 features (zeroed if missing)
        Total dimension: 225 features per frame.
        """
        # Convert BGR image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)

        # Pose keypoints (33 points)
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33 * 3)

        # Left hand keypoints (21 points)
        if results.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        else:
            lh = np.zeros(21 * 3)

        # Right hand keypoints (21 points)
        if results.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        else:
            rh = np.zeros(21 * 3)

        # Concatenate all landmark vectors into a unified feature representation
        feature_vector = np.concatenate([pose, lh, rh])
        return feature_vector

    def process_video_to_sequence(self, video_path: str, max_frames: int = 30) -> np.ndarray:
        """
        Processes an entire video file into a fixed-length temporal sequence matrix [max_frames, feature_dim].
        Pads with zero-vectors if video is shorter than max_frames; subsamples if longer.
        """
        cap = cv2.VideoCapture(video_path)
        sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = self.extract_landmarks(frame)
            sequence.append(landmarks)

        cap.release()

        if len(sequence) == 0:
            return np.zeros((max_frames, 225), dtype=np.float32)

        # Uniform temporal sampling or padding
        sequence_np = np.array(sequence, dtype=np.float32)
        total_frames = len(sequence_np)

        if total_frames == max_frames:
            return sequence_np
        elif total_frames < max_frames:
            # Pad with trailing last frame or zeros
            padding = np.tile(sequence_np[-1:], (max_frames - total_frames, 1))
            return np.vstack([sequence_np, padding])
        else:
            # Uniformly subsample sequence
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            return sequence_np[indices]

    def close(self):
        self.holistic.close()


if __name__ == "__main__":
    print("LandmarkExtractor initialized successfully.")
    extractor = LandmarkExtractor(static_image_mode=True)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    vector = extractor.extract_landmarks(dummy_frame)
    print(f"Extracted landmark vector shape: {vector.shape} (Expected: 225)")
    extractor.close()
