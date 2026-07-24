"""
Comprehensive System Evaluation & Critiquing Runner for SignFlow Studio.
Executes live critique passes on hand pose profiles, sequence streams, ASL gloss grammar,
and validates backend FastAPI routes & inference engines.
"""

import sys
import os
import json
import numpy as np
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

from main import app
from critiquing_agent import UniversalCritiquingAgent, TARGET_POSE_PROFILES
from inference import SignInferenceEngine
from cslr_inference import CSLRInferenceEngine, ContinuousSentenceStreamer
from multicam_smplx_hand import MultiCameraStereoTriangulator
from easymocap_integration import EasyMocapAdapter

client = TestClient(app)
agent = UniversalCritiquingAgent()

def run_pose_profile_critiques():
    print("==================================================")
    print("1. EVALUATING POSE PROFILES WITH CRITIQUING AGENT")
    print("==================================================")
    results = {}
    
    # Create canonical landmark points for evaluation
    # Standard hand landmarks layout (21 x 3)
    np.random.seed(42)
    
    # Ideal extended hand landmarks layout approximation
    open_hand = np.zeros((21, 3), dtype=np.float32)
    open_hand[0] = [0.0, 0.0, 0.0] # Wrist
    # Fingers pointing straight up
    for i in range(1, 5): open_hand[i] = [0.02 * i, 0.05 * i, 0.0] # Thumb
    for i in range(5, 9): open_hand[i] = [-0.05, 0.1 * (i-4), 0.0] # Index
    for i in range(9, 13): open_hand[i] = [0.0, 0.11 * (i-8), 0.0] # Middle
    for i in range(13, 17): open_hand[i] = [0.05, 0.1 * (i-12), 0.0] # Ring
    for i in range(17, 21): open_hand[i] = [0.1, 0.08 * (i-16), 0.0] # Pinky
    
    for sign_key in TARGET_POSE_PROFILES.keys():
        res = agent.critique_pose(open_hand, target_sign=sign_key)
        results[sign_key] = res
        print(f"Target Sign: {res['target_sign']:<25} | Score: {res['overall_score']:>5.1f}% | Grade: {res['grade']}")
        for f in res['finger_breakdown']:
            print(f"   └─ {f['finger'].capitalize():<7}: Measured {f['measured_deg']:>5.1f}° (Target: {f['target_deg']}°) -> Score: {f['score']:>5.1f}% [{f['status']}]")
        print()
    return results

def run_sequence_critiques():
    print("==================================================")
    print("2. EVALUATING SEQUENCE STREAM CRITIQUES")
    print("==================================================")
    np.random.seed(123)
    
    # Static sequence
    base_frame = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)
    static_seq = np.array([base_frame + np.random.normal(0, 0.001, (21, 3)) for _ in range(20)], dtype=np.float32)
    res_static = agent.critique_sequence(static_seq, target_sign="PEACE")
    print(f"Static Sequence Fluidity Score : {res_static['motion_fluidity_score']}% | Mean Vel: {res_static['mean_velocity']:.4f} | Note: {res_static['fluidity_note']}")
    
    # Smooth moving sequence
    smooth_seq = np.array([base_frame + i * 0.01 for i in range(20)], dtype=np.float32)
    res_smooth = agent.critique_sequence(smooth_seq, target_sign="OPEN HAND")
    print(f"Smooth Sequence Fluidity Score : {res_smooth['motion_fluidity_score']}% | Mean Vel: {res_smooth['mean_velocity']:.4f} | Note: {res_smooth['fluidity_note']}")

    # Fast / Jerky moving sequence
    fast_seq = np.array([base_frame + (i % 2) * 0.2 for i in range(20)], dtype=np.float32)
    res_fast = agent.critique_sequence(fast_seq, target_sign="FIST")
    print(f"Fast Sequence Fluidity Score   : {res_fast['motion_fluidity_score']}% | Mean Vel: {res_fast['mean_velocity']:.4f} | Note: {res_fast['fluidity_note']}")
    print()

def run_gloss_grammar_critiques():
    print("==================================================")
    print("3. EVALUATING ASL GLOSS GRAMMAR CRITIQUES")
    print("==================================================")
    test_cases = [
        ["NAME", "YOU", "WHAT"],
        ["WHAT", "NAME", "YOU"],
        ["ME", "WANT", "WATER"],
        ["IS", "THE", "NAME", "YOU", "WHAT"],
        ["TIME", "GO", "HOME", "NOW"]
    ]
    for glosses in test_cases:
        res = agent.critique_gloss_grammar(glosses)
        print(f"Glosses: {glosses}")
        print(f"   Score: {res['grammar_score']}% | Verdict: {res['verdict']}")
        if res['issues_found']:
            print(f"   Issues: {res['issues_found']}")
            print(f"   Suggestions: {res['suggestions']}")
        print()

def run_api_endpoint_verifications():
    print("==================================================")
    print("4. VERIFYING ALL BACKEND FASTAPI ENDPOINTS")
    print("==================================================")
    endpoints = [
        ("GET", "/health", None),
        ("GET", "/critique_profiles", None),
        ("GET", "/tts/voices", None),
        ("POST", "/critique_pose", {"landmarks": np.random.randn(21, 3).tolist(), "target_sign": "PEACE"}),
        ("POST", "/critique_sequence", {"sequence": [np.random.randn(21, 3).tolist() for _ in range(5)], "target_sign": "PEACE"}),
        ("POST", "/critique_gloss", {"glosses": ["YOU", "NAME", "WHAT"]}),
        ("POST", "/translate_text_to_gloss", {"text": "What is your name?"}),
        ("POST", "/translate_gloss_to_english", {"glosses": ["YOU", "NAME", "WHAT"]}),
        ("POST", "/tts/elevenlabs", {"text": "Hello welcome to SignFlow Studio"}),
        ("POST", "/multicam_triangulate", {"cam1_pts": np.random.randn(21, 2).tolist(), "cam2_pts": np.random.randn(21, 2).tolist()}),
        ("GET", "/multilingual_languages", None),
        ("POST", "/translate_multilingual_sign", {"sequence": [np.random.randn(225).tolist() for _ in range(15)], "target_lang": "es"}),
        ("POST", "/predict_easymocap", {"frames": [{"people": [{"pose3d": np.random.randn(21, 3).tolist()}]} for _ in range(10)]})
    ]

    for method, path, payload in endpoints:
        if method == "GET":
            resp = client.get(path)
        else:
            resp = client.post(path, json=payload)
        status = "✅ OK" if resp.status_code == 200 else f"❌ FAIL ({resp.status_code})"
        print(f"{method:<5} {path:<30} -> Status: {status} ({resp.status_code})")

if __name__ == "__main__":
    run_pose_profile_critiques()
    run_sequence_critiques()
    run_gloss_grammar_critiques()
    run_api_endpoint_verifications()
