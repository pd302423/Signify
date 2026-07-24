"""
Unit tests for ElevenLabs Neural TTS Service and Real-Time Continuous Sentence Streamer.
"""

import sys
import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

from main import app
from tts_elevenlabs import ElevenLabsTTSService
from cslr_inference import CSLRInferenceEngine, ContinuousSentenceStreamer

client = TestClient(app)


def test_elevenlabs_fallback_when_no_api_key():
    service = ElevenLabsTTSService(api_key="")
    res = service.synthesize_speech("Hello world, this is a test sentence.")
    assert res["status"] == "fallback"
    assert "Web Speech TTS fallback" in res["message"]


def test_continuous_sentence_streamer_processing():
    engine = CSLRInferenceEngine()
    streamer = ContinuousSentenceStreamer(engine, window_size=30)
    
    # Process stream of landmark frames
    dummy_frame = np.random.uniform(0.1, 0.9, 225).astype(np.float32)
    for _ in range(20):
        res = streamer.process_frame(dummy_frame)
        assert "type" in res
        assert "sentence" in res
        assert "glosses" in res


def test_elevenlabs_tts_api_endpoint():
    payload = {
        "text": "Hello friend, nice to meet you!",
        "voice_id": "21m00Tcm4TlvDq8ikWAM"
    }
    response = client.post("/tts/elevenlabs", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["success", "fallback", "error"]


def test_elevenlabs_voices_api_endpoint():
    response = client.get("/tts/voices")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Rachel" in data["voices"]
