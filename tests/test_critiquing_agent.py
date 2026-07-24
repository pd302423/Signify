"""
Unit and integration tests for the Universal Critiquing Agent.
"""

import sys
import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

from main import app
from critiquing_agent import UniversalCritiquingAgent, TARGET_POSE_PROFILES

client = TestClient(app)


def test_agent_pose_critique_peace():
    agent = UniversalCritiquingAgent()
    # Mock landmarks for 21 points
    dummy_landmarks = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)
    res = agent.critique_pose(dummy_landmarks, target_sign="PEACE")
    
    assert res["status"] == "success"
    assert "target_sign" in res
    assert 0 <= res["overall_score"] <= 100
    assert len(res["finger_breakdown"]) == 5
    assert "subscores" in res


def test_agent_sequence_critique():
    agent = UniversalCritiquingAgent()
    dummy_seq = np.random.uniform(0.1, 0.9, (10, 21, 3)).astype(np.float32)
    res = agent.critique_sequence(dummy_seq, target_sign="OPEN HAND")
    
    assert res["status"] == "success"
    assert res["frame_count"] == 10
    assert "motion_fluidity_score" in res
    assert 0 <= res["peak_pose_score"] <= 100


def test_agent_gloss_grammar_critique():
    agent = UniversalCritiquingAgent()
    # Correct order (WH at end)
    res_good = agent.critique_gloss_grammar(["NAME", "YOU", "WHAT"])
    assert res_good["grammar_score"] == 100.0

    # Incorrect order (WH at start)
    res_bad = agent.critique_gloss_grammar(["WHAT", "NAME", "YOU"])
    assert res_bad["grammar_score"] < 100.0
    assert len(res_bad["issues_found"]) > 0


def test_api_critique_pose_endpoint():
    dummy_pts = [[0.5, 0.5, 0.0] for _ in range(21)]
    payload = {"landmarks": dummy_pts, "target_sign": "I LOVE YOU"}
    response = client.post("/critique_pose", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["target_key"] == "I LOVE YOU"


def test_api_critique_profiles_endpoint():
    response = client.get("/critique_profiles")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "PEACE" in data["profiles"]
    assert "I LOVE YOU" in data["profiles"]
