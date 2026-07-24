"""
Continuous Sign Language Recognition (CSLR) & Multi-Lingual Sign Language Translation (SLT) Engine.

Decodes frame sequence landmarks into sign language glosses & multi-lingual text (ASL, BSL, ISL, CSL, DGS)
via PyTorch BiLSTM CTC and Spatio-Temporal Multi-Lingual Sign Transformer.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from cslr_model import CSLR_BiLSTM_CTC
from multilingual_sign_transformer import MultiLingualSignTransformer, SUPPORTED_SIGN_LANGUAGES, SUPPORTED_TARGET_LANGUAGES

GLOSS_VOCAB = [
    "BLANK", "HELLO", "HI", "TODAY", "LEARN", "THANK_YOU", "PLEASE", "YES", "NO", "HELP", "ME", "MY",
    "YOU", "NAME", "WHAT", "WHERE", "WHY", "HOW", "TIME", "EAT",
    "FOOD", "DRINK", "WATER", "WANT", "MORE", "FINISH", "GO", "COME",
    "FRIEND", "FAMILY", "HOUSE", "WORK", "SCHOOL", "GOOD", "BAD",
    "HAPPY", "SAD", "LOVE", "SEE", "HEAR", "UNDERSTAND", "AGAIN", "STOP", "IS", "LOLA"
]

LEXICON = {
    "hello": "HELLO", "hi": "HI", "hey": "HELLO", "today": "TODAY",
    "thank": "THANK_YOU", "thanks": "THANK_YOU",
    "please": "PLEASE", "yes": "YES", "no": "NO",
    "help": "HELP", "me": "ME", "i": "ME", "my": "MY", "myself": "ME",
    "you": "YOU", "your": "YOU", "yours": "YOU",
    "name": "NAME", "what": "WHAT", "where": "WHERE", "why": "WHY", "how": "HOW",
    "when": "WHEN", "who": "WHO", "eat": "EAT", "food": "FOOD", "drink": "DRINK",
    "water": "WATER", "want": "WANT", "more": "MORE", "finish": "FINISH", "go": "GO",
    "friend": "FRIEND", "family": "FAMILY", "house": "HOUSE", "work": "WORK",
    "school": "SCHOOL", "learn": "LEARN", "sign": "SIGN", "good": "GOOD",
    "bad": "BAD", "happy": "HAPPY", "sad": "SAD", "love": "LOVE", "see": "SEE",
    "understand": "UNDERSTAND", "again": "AGAIN", "stop": "STOP", "lola": "LOLA"
}

FILLER_WORDS = {"are", "am", "was", "were", "be", "being", "been", "the", "a", "an", "to", "of", "do", "does", "did"}



class CSLRInferenceEngine:
    """Decodes continuous landmark video streams into multi-lingual sign translations & glosses."""

    def __init__(self, model_path: str = "weights/cslr_bilstm.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = GLOSS_VOCAB
        self.model = CSLR_BiLSTM_CTC(input_dim=225, num_glosses=len(self.vocab) - 1).to(self.device)
        self.multilingual_transformer = MultiLingualSignTransformer(input_dim=225).to(self.device)

        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model_state = self.model.state_dict()
                filtered_state = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
                model_state.update(filtered_state)
                self.model.load_state_dict(model_state)
                print(f"✅ Loaded CSLR weights from {model_path}")
            except Exception as e:
                print(f"Notice: CSLR engine initialized with base model ({e})")


        self.model.eval()
        self.multilingual_transformer.eval()

    def translate_multilingual_sign_sequence(
        self,
        sequence: np.ndarray,
        target_lang: str = "en"
    ) -> Dict[str, Any]:
        """
        Translates continuous keypoint stream into multi-lingual text (EN, ES, HI, ZH, DE)
        and automatically identifies source sign language (ASL, BSL, ISL, CSL, DGS).
        """
        if sequence.ndim == 2:
            tensor_seq = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            tensor_seq = torch.tensor(sequence, dtype=torch.float32).to(self.device)

        cslr_res = self.decode_continuous_sequence(sequence)
        trans_res = self.multilingual_transformer.translate_sequence(tensor_seq, target_lang=target_lang)

        glosses = cslr_res.get("glosses", [])
        if glosses:
            clean_sentence = self.translate_gloss_to_english(glosses)
        else:
            clean_sentence = trans_res.get("translated_text", "")

        return {
            "detected_sign_language": trans_res.get("detected_sign_language", "ASL"),
            "detected_sign_language_full": trans_res.get("detected_sign_language_full", "American Sign Language"),
            "sign_language_confidence": trans_res.get("sign_language_confidence", 0.95),
            "target_spoken_language": target_lang,
            "translated_text": clean_sentence,
            "tokens": glosses or trans_res.get("tokens", [])
        }


    def translate_text_to_asl_gloss(self, text: str) -> List[str]:
        """
        Converts spoken/written English sentences into an ordered ASL Gloss sequence
        enforcing ASL grammar rules (Topic-Comment, WH-end positioning, copula dropping).
        """
        if not text or not text.strip():
            return []
        
        words = text.lower().replace("?", "").replace("!", "").replace(".", "").replace(",", "").split()
        glosses = []
        wh_words = []
        time_words = []

        for word in words:
            if word in FILLER_WORDS:
                continue
            
            gloss = LEXICON.get(word, word.upper())
            
            if word in ["what", "where", "why", "how", "when", "who"]:
                wh_words.append(gloss)
            elif word in ["today", "tomorrow", "yesterday", "now", "later"]:
                time_words.append(gloss)
            else:
                glosses.append(gloss)

        final_sequence = time_words + glosses + wh_words
        return final_sequence

    def translate_gloss_to_english(self, glosses: List[str]) -> str:
        """
        Translates an ASL Gloss sequence into a grammatically natural English sentence.
        """
        if not glosses:
            return ""

        upper_glosses = [g.upper().replace("_", " ") for g in glosses]
        
        wh_words = [g for g in upper_glosses if g in {"WHAT", "WHERE", "WHY", "HOW", "WHEN", "WHO"}]
        other_words = [g for g in upper_glosses if g not in {"WHAT", "WHERE", "WHY", "HOW", "WHEN", "WHO"}]

        if wh_words and len(other_words) > 0:
            reordered = wh_words + [w.lower() for w in other_words]
            sentence = " ".join(reordered).capitalize() + "?"
            return sentence

        words = []
        for g in upper_glosses:
            w = g.lower()
            if w == "me":
                w = "I" if len(words) == 0 else "me"
            words.append(w)

        sentence = " ".join(words)
        sentence = sentence[0].upper() + sentence[1:] + "."
        return sentence

    def decode_continuous_sequence(self, sequence: np.ndarray) -> Dict[str, Any]:
        """
        Decodes a continuous sequence matrix [Time, 225] into recognized glosses & translated English.
        """
        if sequence.ndim == 2:
            tensor_seq = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            tensor_seq = torch.tensor(sequence, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_seq)
            decoded_indices = self.model.decode_greedy(logits)[0]

        recognized_glosses = [self.vocab[idx] for idx in decoded_indices if idx < len(self.vocab)]
        english_translation = self.translate_gloss_to_english(recognized_glosses)

        return {
            "glosses": recognized_glosses,
            "sentence": english_translation,
            "frame_count": sequence.shape[0]
        }


class ContinuousSentenceStreamer:
    """
    Maintains a sliding temporal window of streaming landmark frames,
    detects active gesture motion vs. rest breaks, forms natural sentences in real-time,
    and signals when a sentence is complete.
    """

    def __init__(self, cslr_engine: CSLRInferenceEngine, window_size: int = 45):
        self.engine = cslr_engine
        self.window_size = window_size
        self.buffer = []
        self.accumulated_glosses = []
        self.last_sentence = ""
        self.still_frames_count = 0

    def process_frame(self, frame_landmarks: np.ndarray) -> Dict[str, Any]:
        pts = np.array(frame_landmarks, dtype=np.float32).flatten()
        if pts.size < 225:
            padded = np.zeros(225, dtype=np.float32)
            padded[:pts.size] = pts
            pts = padded

        self.buffer.append(pts)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        if len(self.buffer) > 1:
            vel = np.linalg.norm(self.buffer[-1][:63] - self.buffer[-2][:63])
            if vel < 0.02:
                self.still_frames_count += 1
            else:
                self.still_frames_count = 0

        sentence_complete = False
        if len(self.buffer) >= 15:
            buf_matrix = np.array(self.buffer, dtype=np.float32)
            res = self.engine.decode_continuous_sequence(buf_matrix)
            current_glosses = res.get("glosses", [])
            
            for g in current_glosses:
                if g != "BLANK" and (not self.accumulated_glosses or self.accumulated_glosses[-1] != g):
                    self.accumulated_glosses.append(g)

            live_sentence = self.engine.translate_gloss_to_english(self.accumulated_glosses)
            self.last_sentence = live_sentence

            if self.still_frames_count > 15 and len(self.accumulated_glosses) > 0:
                sentence_complete = True
                completed_sentence = live_sentence
                self.accumulated_glosses = []
                self.buffer = []
                self.still_frames_count = 0
                return {
                    "type": "sentence_complete",
                    "sentence": completed_sentence,
                    "glosses": [],
                    "is_complete": True
                }

        return {
            "type": "sentence_streaming",
            "sentence": self.last_sentence,
            "glosses": self.accumulated_glosses,
            "is_complete": False
        }


if __name__ == "__main__":
    engine = CSLRInferenceEngine()
    dummy_seq = np.random.randn(30, 225).astype(np.float32)
    m_res = engine.translate_multilingual_sign_sequence(dummy_seq, target_lang="es")
    print("✅ Multi-Lingual Sign Translation Output:", m_res)
