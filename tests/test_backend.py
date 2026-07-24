"""
Automated pytest test suite for Signify Studio API & Inference Engine.
"""

import sys
import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

from main import app
from cslr_inference import CSLRInferenceEngine

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "PyTorch BiLSTM CTC" in data["features"]


def test_translate_text_to_gloss_endpoint():
    payload = {"text": "Hello friend, what is your name?"}
    response = client.post("/translate_text_to_gloss", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "HELLO" in data["asl_glosses"]
    assert "NAME" in data["asl_glosses"]


def test_cslr_text_to_asl_gloss_logic():
    engine = CSLRInferenceEngine()
    glosses = engine.translate_text_to_asl_gloss("where is your house?")
    assert glosses[-1] == "WHERE"  # WH-word should be shifted to end


def test_multicam_triangulate_endpoint():
    cam1 = [[0.5, 0.5] for _ in range(21)]
    cam2 = [[0.55, 0.48] for _ in range(21)]
    payload = {"cam1_pts": cam1, "cam2_pts": cam2}
    response = client.post("/multicam_triangulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["metric_3d_keypoints"]) == 21


def test_multilingual_languages_endpoint():
    response = client.get("/multilingual_languages")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "ASL" in data["supported_sign_languages"]
    assert "BSL" in data["supported_sign_languages"]
    assert "ISL" in data["supported_sign_languages"]
    assert "CSL" in data["supported_sign_languages"]
    assert "DGS" in data["supported_sign_languages"]


def test_translate_multilingual_sign_endpoint():
    seq = [[0.1 for _ in range(225)] for _ in range(15)]
    payload = {"sequence": seq, "target_lang": "es"}
    response = client.post("/translate_multilingual_sign", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "detected_sign_language" in data["result"]
    assert "translated_text" in data["result"]
