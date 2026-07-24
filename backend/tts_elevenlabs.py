"""
ElevenLabs Neural Text-to-Speech (TTS) Integration Service.

Generates ultra-realistic human speech audio from sign language text/sentence predictions
using ElevenLabs REST API endpoints.
"""

import os
import json
import urllib.request
import urllib.error
import base64
from typing import Dict, Any, Optional

DEFAULT_VOICES = {
    "Rachel": "21m00Tcm4TlvDq8ikWAM",
    "Adam": "pNInz6obpgDQGcFmaJgB",
    "Antoni": "ErXwobaYiN019PkySvjV",
    "Bella": "EXAVITQu4vr4xnSDxMaL",
    "Domi": "AZnzlk1XvdvUeBnXmlld",
    "Elli": "MF3mGyEYCl7XYWbV9V6O",
    "Josh": "TxGEqnHWrfWFTfGW9XjX"
}


class ElevenLabsTTSService:
    """Handles speech generation via ElevenLabs API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")

    def synthesize_speech(
        self,
        text: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        api_key: Optional[str] = None,
        model_id: str = "eleven_monolingual_v1"
    ) -> Dict[str, Any]:
        """
        Synthesizes text into MP3 audio via ElevenLabs REST API.
        Returns base64 encoded audio string or error status.
        """
        key = api_key or self.api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        
        if not text or not text.strip():
            return {"status": "error", "message": "No text provided for speech synthesis."}

        if not key:
            return {
                "status": "fallback",
                "message": "No ElevenLabs API key provided. Using browser Web Speech TTS fallback.",
                "text": text
            }

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": key
        }
        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        try:
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as response:
                audio_bytes = response.read()
                b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
                return {
                    "status": "success",
                    "format": "audio/mpeg",
                    "audio_base64": f"data:audio/mpeg;base64,{b64_audio}",
                    "text": text,
                    "voice_id": voice_id
                }
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8") if e.fp else str(e)
            return {"status": "error", "message": f"ElevenLabs API HTTP {e.code}: {err_msg}"}
        except Exception as e:
            return {"status": "error", "message": f"ElevenLabs request failed: {str(e)}"}
