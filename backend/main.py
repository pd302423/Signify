"""
FastAPI Backend Server & Static Host for SignFlow Sign Language Recognition.

Includes:
- EasyMocap Multi-Camera 3D Mocap Adapter (/predict_easymocap)
- Multi-Camera Stereo Triangulation & MANO 3D Hand Mesh Fitting (/multicam_triangulate)
- Local IP camera CORS proxy endpoint (/proxy_frame)
- Real-time gesture sequence inference (/predict)
- Static host serving index.html on port 8000
"""

import os
import json
import urllib.request
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Response, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from inference import SignInferenceEngine
from cslr_inference import CSLRInferenceEngine, ContinuousSentenceStreamer
from critiquing_agent import UniversalCritiquingAgent, TARGET_POSE_PROFILES
from tts_elevenlabs import ElevenLabsTTSService, DEFAULT_VOICES
from video_transcriber import VideoTranscriber

from multicam_smplx_hand import MultiCameraStereoTriangulator, MANOParametricHandModel
from easymocap_integration import EasyMocapAdapter

app = FastAPI(
    title="SignFlow Multi-Camera & EasyMocap SLR API",
    description="Multi-Camera Stereo 3D Hand Mesh (EasyMocap / MANO / SMPL-X) & Real-Time SLR API with Universal Critiquing Agent and ElevenLabs TTS",
    version="2.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engines
inference_engine = SignInferenceEngine()
cslr_engine = CSLRInferenceEngine()
sentence_streamer = ContinuousSentenceStreamer(cslr_engine)
critiquing_agent = UniversalCritiquingAgent()
elevenlabs_service = ElevenLabsTTSService()
multicam_triangulator = MultiCameraStereoTriangulator()
easymocap_adapter = EasyMocapAdapter()
video_transcriber = VideoTranscriber(cslr_engine)



@app.get("/health")
def health_check():
    return {
        "status": "online",
        "system": "SignFlow Multi-Camera & Universal Critiquing Engine",
        "version": "2.3.0",
        "features": [
            "ElevenLabs Neural Text-to-Speech",
            "Real-Time Continuous Sentence Streamer",
            "Universal AI Critiquing Agent",
            "EasyMocap 3D Mocap Adapter",
            "Multi-Camera DLT Triangulation",
            "MANO 3D Hand Mesh",
            "PyTorch BiLSTM CTC",
            "WebSocket Duplex Stream"
        ]
    }


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Full-duplex WebSocket endpoint for continuous landmark streaming & real-time inference.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            msg_type = payload.get("type", "landmarks")
            
            if msg_type == "landmarks":
                sequence = np.array(payload.get("landmarks", []), dtype=np.float32)
                if sequence.size > 0:
                    res = inference_engine.predict_sequence(sequence)
                    await websocket.send_json({"type": "prediction", "data": res})
            elif msg_type == "cslr":
                sequence = np.array(payload.get("landmarks", []), dtype=np.float32)
                if sequence.size > 0:
                    res = cslr_engine.decode_continuous_sequence(sequence)
                    await websocket.send_json({"type": "cslr_result", "data": res})
            elif msg_type == "critique":
                landmarks = np.array(payload.get("landmarks", []), dtype=np.float32)
                target_sign = payload.get("target_sign", "PEACE")
                if landmarks.size > 0:
                    res = critiquing_agent.critique_pose(landmarks, target_sign=target_sign)
                    await websocket.send_json({"type": "critique_result", "data": res})
            elif msg_type == "stream_sentence":
                frame_landmarks = np.array(payload.get("landmarks", []), dtype=np.float32)
                if frame_landmarks.size > 0:
                    res = sentence_streamer.process_frame(frame_landmarks)
                    await websocket.send_json({"type": "sentence_stream", "data": res})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


# ElevenLabs Neural TTS Endpoints
class ElevenLabsTTSRequest(BaseModel):
    text: str = Field(..., description="Text sentence to synthesize with ElevenLabs AI voice")
    voice_id: str = Field("21m00Tcm4TlvDq8ikWAM", description="ElevenLabs voice ID (e.g. Rachel, Adam, Antoni)")
    api_key: Optional[str] = Field(None, description="Optional ElevenLabs API key")


@app.post("/tts/elevenlabs")
def elevenlabs_tts_endpoint(request: ElevenLabsTTSRequest):
    """
    Synthesizes sign translation text into human-like audio using ElevenLabs REST API.
    """
    return elevenlabs_service.synthesize_speech(
        text=request.text,
        voice_id=request.voice_id,
        api_key=request.api_key
    )


@app.get("/tts/voices")
def list_elevenlabs_voices():
    """
    Returns default ElevenLabs voices available for sign dictation speech synthesis.
    """
    return {
        "status": "success",
        "voices": DEFAULT_VOICES
    }


class TextToGlossRequest(BaseModel):
    text: str = Field(..., description="English text sentence to translate into ASL Gloss sequence")


@app.post("/translate_text_to_gloss")
def translate_text_to_asl_gloss_route(request: TextToGlossRequest):
    """
    Translates input spoken/written English text into an ordered ASL Gloss sequence for 3D Avatar animation.
    """
    glosses = cslr_engine.translate_text_to_asl_gloss(request.text)
    return {
        "status": "success",
        "input_text": request.text,
        "asl_glosses": glosses
    }


class GlossToEnglishRequest(BaseModel):
    glosses: List[str] = Field(..., description="List of ASL Gloss strings to translate into natural English")


@app.post("/translate_gloss_to_english")
def translate_gloss_to_english_route(request: GlossToEnglishRequest):
    """
    Translates an ASL Gloss sequence into a grammatically natural English sentence.
    """
    sentence = cslr_engine.translate_gloss_to_english(request.glosses)
    return {
        "status": "success",
        "input_glosses": request.glosses,
        "sentence": sentence
    }


class MultiLingualSignTranslationRequest(BaseModel):
    sequence: List[Any] = Field(..., description="Continuous landmark sequence matrix [Time, 225] or [Time, 1629]")
    target_lang: str = Field("en", description="Target spoken language code (en, es, hi, zh, de)")


@app.post("/translate_multilingual_sign")
def translate_multilingual_sign_route(request: MultiLingualSignTranslationRequest):
    """
    Translates continuous landmark streams into multi-lingual text (EN, ES, HI, ZH, DE)
    and automatically classifies source sign language (ASL, BSL, ISL, CSL, DGS).
    """
    seq_np = np.array(request.sequence, dtype=np.float32)
    res = cslr_engine.translate_multilingual_sign_sequence(seq_np, target_lang=request.target_lang)
    return {
        "status": "success",
        "result": res
    }


@app.post("/transcribe_video")
async def transcribe_video_endpoint(
    file: UploadFile = File(...),
    target_lang: str = Query("en", description="Target spoken language code (en, es, hi, zh, de, fr)")
):
    """
    Ingests recorded or uploaded video files (.mp4, .webm, .mov, .avi),
    extracts frame MediaPipe landmark sequences, and returns AI continuous sentence transcription.
    """
    contents = await file.read()
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    res = video_transcriber.transcribe_video_bytes(contents, file_extension=ext, target_lang=target_lang)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@app.get("/multilingual_languages")

def get_supported_multilingual_languages():
    """
    Returns supported sign languages (ASL, BSL, ISL, CSL, DGS) and target spoken languages (en, es, hi, zh, de).
    """
    from multilingual_sign_transformer import SUPPORTED_SIGN_LANGUAGES, SUPPORTED_TARGET_LANGUAGES
    return {
        "status": "success",
        "supported_sign_languages": SUPPORTED_SIGN_LANGUAGES,
        "supported_target_languages": SUPPORTED_TARGET_LANGUAGES
    }



# Universal Critiquing Agent Endpoints
class PoseCritiqueRequest(BaseModel):
    landmarks: List[Any] = Field(..., description="21 Hand landmarks list [21, 2] or [21, 3] or flattened [42]/[63]")
    target_sign: str = Field("PEACE", description="Target sign name to critique posture against")


@app.post("/critique_pose")
def critique_pose_endpoint(request: PoseCritiqueRequest):
    """
    Evaluates 2D/3D hand landmarks against a target ASL sign profile, returning detailed finger flexions, spatial scores, and coaching tips.
    """
    landmarks_np = np.array(request.landmarks, dtype=np.float32)
    return critiquing_agent.critique_pose(landmarks_np, target_sign=request.target_sign)


class SequenceCritiqueRequest(BaseModel):
    sequence: List[Any] = Field(..., description="List of frame landmarks matrices")
    target_sign: str = Field("PEACE", description="Target sign name")


@app.post("/critique_sequence")
def critique_sequence_endpoint(request: SequenceCritiqueRequest):
    """
    Critiques continuous frame sequence for velocity, fluidity, and peak pose accuracy.
    """
    sequence_np = np.array(request.sequence, dtype=np.float32)
    return critiquing_agent.critique_sequence(sequence_np, target_sign=request.target_sign)


class GlossCritiqueRequest(BaseModel):
    glosses: List[str] = Field(..., description="List of ASL Gloss strings")


@app.post("/critique_gloss")
def critique_gloss_endpoint(request: GlossCritiqueRequest):
    """
    Critiques ASL Gloss sequence grammar, word order, and WH-word placement.
    """
    return critiquing_agent.critique_gloss_grammar(request.glosses)


@app.get("/critique_profiles")
def list_critique_profiles():
    """
    Returns available target ASL pose profiles supported by the Universal Critiquing Agent.
    """
    return {
        "status": "success",
        "profiles": TARGET_POSE_PROFILES
    }





class EasyMocapInferenceRequest(BaseModel):
    frames: List[Dict[str, Any]] = Field(..., description="List of EasyMocap 3D output JSON frame dictionaries")


@app.post("/predict_easymocap")
def predict_easymocap_sequence(request: EasyMocapInferenceRequest):
    """
    Ingests multi-camera 3D motion capture outputs from EasyMocap,
    extracts 123-dim 3D joint & bone vectors, and runs PyTorch BiLSTM CTC sentence decoding.
    """
    result = easymocap_adapter.predict_easymocap_sequence(request.frames)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


class MultiCamTriangulationRequest(BaseModel):
    cam1_pts: List[List[float]] = Field(..., description="2D keypoints [21, 2] from Camera 1 (Laptop)")
    cam2_pts: List[List[float]] = Field(..., description="2D keypoints [21, 2] from Camera 2 (Phone)")


@app.post("/multicam_triangulate")
def process_multicam_triangulation(request: MultiCamTriangulationRequest):
    c1 = np.array(request.cam1_pts, dtype=np.float64)
    c2 = np.array(request.cam2_pts, dtype=np.float64)

    if c1.shape != (21, 2) or c2.shape != (21, 2):
        raise HTTPException(status_code=400, detail="Expected 2D keypoints shape [21, 2] for both cameras.")

    pts_3d = multicam_triangulator.triangulate_points(c1, c2)

    return {
        "status": "success",
        "metric_3d_keypoints": pts_3d.tolist(),
        "wrist_center_3d": pts_3d[0].tolist(),
        "hand_scale_mm": float(np.linalg.norm(pts_3d[9] - pts_3d[0]) * 1000.0)
    }


@app.get("/proxy_frame")
def proxy_phone_frame(target: str = Query(..., description="Phone IP Camera URL")):
    try:
        if not target.startswith("http://") and not target.startswith("https://"):
            target = "http://" + target
        
        snapshot_url = target.replace("/video", "/shot.jpg")
        if not snapshot_url.endswith("/shot.jpg") and not snapshot_url.endswith(".jpg"):
            snapshot_url = snapshot_url.rstrip("/") + "/shot.jpg"

        req = urllib.request.Request(snapshot_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=2) as resp:
            frame_bytes = resp.read()
            return Response(content=frame_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch phone frame from {target}: {str(e)}")


workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
app.mount("/", StaticFiles(directory=workspace_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
