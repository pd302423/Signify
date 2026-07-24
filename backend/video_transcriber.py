"""
Video Transcriber Module for SignFlow Studio.

Ingests recorded or uploaded video files (.mp4, .webm, .mov, .avi),
extracts frame-by-frame MediaPipe 3D landmark sequences using HandLandmarker tasks,
and passes them to the Multi-Lingual Sign Transformer CSLR Engine for full sentence transcription.
"""

import os
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Dict, Any, List

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from cslr_inference import CSLRInferenceEngine


class VideoTranscriber:
    """Processes video file streams and extracts landmark sequences for AI transcription."""

    def __init__(self, cslr_engine: CSLRInferenceEngine, model_path: str = "models/hand_landmarker.task"):
        self.engine = cslr_engine
        self.detector = None
        
        abs_model_path = os.path.abspath(model_path)
        if not os.path.exists(abs_model_path):
            abs_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task"))

        if os.path.exists(abs_model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=abs_model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                print(f"✅ Loaded VideoTranscriber MediaPipe HandLandmarker from {abs_model_path}")
            except Exception as e:
                print(f"Notice: HandLandmarker initialization warning ({e})")

    def transcribe_video_bytes(
        self,
        video_bytes: bytes,
        file_extension: str = ".mp4",
        target_lang: str = "en"
    ) -> Dict[str, Any]:
        """
        Saves video bytes to a temporary file, processes video frames with MediaPipe,
        and translates the resulting gesture landmark sequence into text.
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        try:
            temp_file.write(video_bytes)
            temp_file.flush()
            temp_file.close()

            cap = cv2.VideoCapture(temp_file.name)
            if not cap.isOpened():
                return {"status": "error", "message": "Failed to open video file stream."}

            sequence_frames = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                flat_frame = []
                if self.detector is not None:
                    try:
                        results = self.detector.detect(mp_image)
                        if results.hand_landmarks and len(results.hand_landmarks) > 0:
                            first_hand = results.hand_landmarks[0]
                            for lm in first_hand:
                                flat_frame.extend([lm.x, lm.y, lm.z])
                    except Exception:
                        pass

                while len(flat_frame) < 225:
                    flat_frame.append(0.0)

                sequence_frames.append(flat_frame)

            cap.release()

            if len(sequence_frames) < 5:
                return {
                    "status": "error",
                    "message": "Video too short for AI CTC recognition."
                }

            seq_np = np.array(sequence_frames, dtype=np.float32)
            
            res = self.engine.translate_multilingual_sign_sequence(seq_np, target_lang=target_lang)
            final_sentence = cslr_res.get("sentence", "")
            if not final_sentence or final_sentence == ".":
                final_sentence = res.get("translated_text", "")

            return {
                "status": "success",
                "frame_count": len(sequence_frames),
                "translation_result": res,
                "recognized_glosses": cslr_res.get("glosses", []),
                "sentence": final_sentence
            }


        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)


if __name__ == "__main__":
    engine = CSLRInferenceEngine()
    transcriber = VideoTranscriber(engine)
    print("✅ VideoTranscriber Module Ready!")
