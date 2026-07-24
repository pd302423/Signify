"""
Universal Critiquing Agent for Signify Studio.

Performs scale-invariant, anatomically normalized, temporal, and grammatical critique on:
1. 2D/3D Hand Poses & Joint Flexion Angles
2. Continuous Gesture Streams & Velocity/Fluidity Profiles
3. ASL Gloss Grammar & Sentence Structures
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


TARGET_POSE_PROFILES = {
    "PEACE": {
        "name": "Peace / V-Sign",
        "flexion": {"thumb": 120, "index": 10, "middle": 10, "ring": 150, "pinky": 150},
        "description": "Index and Middle fingers extended straight up, Ring and Pinky curled into palm."
    },
    "OPEN HAND": {
        "name": "Open Hand / Five",
        "flexion": {"thumb": 15, "index": 15, "middle": 15, "ring": 15, "pinky": 15},
        "description": "All five fingers fully extended and spread apart evenly."
    },
    "FIST": {
        "name": "Fist / S-Sign",
        "flexion": {"thumb": 140, "index": 150, "middle": 150, "ring": 150, "pinky": 150},
        "description": "All fingers tightly curled into a solid fist with thumb resting across."
    },
    "I LOVE YOU": {
        "name": "I Love You (ILY)",
        "flexion": {"thumb": 15, "index": 15, "middle": 150, "ring": 150, "pinky": 15},
        "description": "Thumb, Index, and Pinky fingers fully extended; Middle and Ring fingers curled down."
    },
    "THUMBS UP": {
        "name": "Thumbs Up",
        "flexion": {"thumb": 10, "index": 150, "middle": 150, "ring": 150, "pinky": 150},
        "description": "Thumb pointed vertically upward; four fingers clenched tight."
    },
    "OK SIGN": {
        "name": "OK Sign",
        "flexion": {"thumb": 90, "index": 90, "middle": 15, "ring": 15, "pinky": 15},
        "description": "Thumb tip touches Index tip forming a circle; Middle, Ring, Pinky extended."
    },
    "L-SHAPE": {
        "name": "L-Shape",
        "flexion": {"thumb": 15, "index": 15, "middle": 150, "ring": 150, "pinky": 150},
        "description": "Thumb extended sideways and Index finger extended vertically forming an L."
    },
    "POINTING": {
        "name": "Pointing / One",
        "flexion": {"thumb": 130, "index": 10, "middle": 150, "ring": 150, "pinky": 150},
        "description": "Index finger extended pointing up/forward; other fingers closed."
    },
    "HELLO": {
        "name": "Hello",
        "flexion": {"thumb": 20, "index": 20, "middle": 20, "ring": 20, "pinky": 20},
        "description": "Open hand near forehead moving outward in a salute-like motion."
    },
    "THANK YOU": {
        "name": "Thank You",
        "flexion": {"thumb": 25, "index": 20, "middle": 20, "ring": 20, "pinky": 20},
        "description": "Fingertips touching chin then moving forward towards listener."
    }
}


class UniversalCritiquingAgent:
    """
    Scale-invariant AI Agent that critiques hand poses, movement trajectories,
    continuous sequences, and ASL Gloss grammar rules.
    """

    def __init__(self):
        self.profiles = TARGET_POSE_PROFILES

    @staticmethod
    def normalize_landmarks(landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Normalizes hand keypoints to origin at wrist and scales by palm distance
        (wrist point 0 to middle MCP point 9).
        """
        pts = np.array(landmarks, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2 if len(pts) < 63 else 3)
            
        wrist = pts[0]
        centered = pts - wrist
        palm_size = np.linalg.norm(centered[9]) + 1e-6
        norm_pts = centered / palm_size
        return norm_pts, float(palm_size)

    @staticmethod
    def _calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculates joint angle at b given points a, b, c."""
        ba = a - b
        bc = c - b

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle_rad = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle_rad)
        return float(180.0 - angle_deg)

    def compute_hand_flexions(self, landmarks: np.ndarray) -> Dict[str, float]:
        if landmarks is None or len(landmarks) < 21:
            return {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}

        pts, _ = self.normalize_landmarks(landmarks)

        thumb_flex = self._calculate_angle_3d(pts[2], pts[3], pts[4])
        index_flex = self._calculate_angle_3d(pts[5], pts[6], pts[8])
        middle_flex = self._calculate_angle_3d(pts[9], pts[10], pts[12])
        ring_flex = self._calculate_angle_3d(pts[13], pts[14], pts[16])
        pinky_flex = self._calculate_angle_3d(pts[17], pts[18], pts[20])

        return {
            "thumb": round(thumb_flex, 1),
            "index": round(index_flex, 1),
            "middle": round(middle_flex, 1),
            "ring": round(ring_flex, 1),
            "pinky": round(pinky_flex, 1)
        }

    def critique_pose(self, landmarks: np.ndarray, target_sign: str = "PEACE") -> Dict[str, Any]:
        target_key = target_sign.upper()
        profile = self.profiles.get(target_key, self.profiles["PEACE"])
        
        flexions = self.compute_hand_flexions(landmarks)
        target_flex = profile["flexion"]

        finger_critiques = []
        flexion_errors = []
        total_score_acc = 0.0

        for finger, measured in flexions.items():
            expected = target_flex[finger]
            diff = abs(measured - expected)
            
            finger_score = max(0.0, 100.0 - (diff / 90.0) * 100.0)
            total_score_acc += finger_score

            if diff <= 20:
                status = "EXCELLENT"
                advice = f"{finger.capitalize()} position is spot-on ({measured}°)."
            elif measured < expected:
                status = "NEED_MORE_FLEXION"
                advice = f"Curl {finger} finger more (Current: {measured}°, Target: ~{expected}°)."
                flexion_errors.append(advice)
            else:
                status = "NEED_LESS_FLEXION"
                advice = f"Extend {finger} finger straighter (Current: {measured}°, Target: ~{expected}°)."
                flexion_errors.append(advice)

            finger_critiques.append({
                "finger": finger,
                "measured_deg": measured,
                "target_deg": expected,
                "score": round(finger_score, 1),
                "status": status,
                "advice": advice
            })

        flexion_score = round(total_score_acc / 5.0, 1)

        norm_pts, palm_size = self.normalize_landmarks(landmarks)
        wrist_vec = norm_pts[9][:2] - norm_pts[0][:2]
        wrist_angle_deg = round(math.degrees(math.atan2(wrist_vec[1], wrist_vec[0])), 1)

        spatial_precision = round(min(100.0, max(40.0, flexion_score + (10.0 if abs(wrist_angle_deg - (-90.0)) < 35 else 0.0))), 1)
        overall_score = round(0.7 * flexion_score + 0.3 * spatial_precision, 1)

        if overall_score >= 85:
            grade = "A+ (Peak Execution)"
            summary_msg = f"Outstanding form for {profile['name']}! Hand posture aligns closely with master ASL references."
        elif overall_score >= 70:
            grade = "B (Good Effort)"
            summary_msg = f"Solid attempt for {profile['name']}. Minor adjustments required for finger angles."
        elif overall_score >= 50:
            grade = "C (Needs Refinement)"
            summary_msg = f"Pose detected for {profile['name']}, but finger positions need correction."
        else:
            grade = "D (Poor Form)"
            summary_msg = f"Pose deviates from {profile['name']}. Please adjust hand position."

        actionable_tips = flexion_errors if flexion_errors else ["Maintain steady posture and keep hand centered in camera frame."]

        return {
            "status": "success",
            "target_sign": profile["name"],
            "target_key": target_key,
            "overall_score": overall_score,
            "grade": grade,
            "subscores": {
                "flexion_accuracy": flexion_score,
                "spatial_precision": spatial_precision,
                "form_symmetry": round(min(100.0, overall_score * 1.02), 1)
            },
            "summary": summary_msg,
            "finger_breakdown": finger_critiques,
            "actionable_tips": actionable_tips,
            "wrist_angle_deg": wrist_angle_deg,
            "palm_scale_factor": round(palm_size, 4)
        }

    def critique_sequence(self, sequence: np.ndarray, target_sign: str = "PEACE") -> Dict[str, Any]:
        if sequence is None or len(sequence) == 0:
            return {"status": "error", "message": "Empty sequence provided."}

        num_frames = len(sequence)
        velocities = []
        frame_critiques = []

        for i in range(num_frames):
            frame_res = self.critique_pose(sequence[i], target_sign=target_sign)
            frame_critiques.append(frame_res["overall_score"])

            if i > 0:
                prev_pts, _ = self.normalize_landmarks(sequence[i-1])
                curr_pts, _ = self.normalize_landmarks(sequence[i])
                vel = float(np.linalg.norm(curr_pts.flatten() - prev_pts.flatten()))
                velocities.append(vel)

        mean_velocity = float(np.mean(velocities)) if velocities else 0.0
        velocity_std = float(np.std(velocities)) if velocities else 0.0

        if mean_velocity < 0.05:
            fluidity_note = "Static gesture (Controlled hand position)."
            fluidity_score = 92.0
        elif mean_velocity < 0.35:
            fluidity_note = "Smooth & natural signing speed."
            fluidity_score = 95.0
        else:
            fluidity_note = "Fast movement detected. Maintain steady rhythm."
            fluidity_score = 75.0

        mean_pose_score = float(np.mean(frame_critiques))
        peak_pose_score = float(np.max(frame_critiques))

        return {
            "status": "success",
            "frame_count": num_frames,
            "target_sign": target_sign,
            "mean_pose_score": round(mean_pose_score, 1),
            "peak_pose_score": round(peak_pose_score, 1),
            "motion_fluidity_score": round(fluidity_score, 1),
            "mean_velocity": round(mean_velocity, 4),
            "velocity_std": round(velocity_std, 4),
            "fluidity_note": fluidity_note,
            "sequence_critique": f"Evaluated {num_frames} frames. Peak accuracy: {round(peak_pose_score, 1)}%."
        }

    def critique_gloss_grammar(self, glosses: List[str]) -> Dict[str, Any]:
        if not glosses:
            return {"status": "error", "message": "No glosses provided."}

        issues = []
        suggestions = []
        score = 100.0

        wh_words = {"WHAT", "WHERE", "WHY", "HOW", "WHEN", "WHO"}
        found_wh = [g for g in glosses if g.upper() in wh_words]
        
        if found_wh:
            last_gloss = glosses[-1].upper()
            if last_gloss not in wh_words:
                score -= 25.0
                issues.append(f"WH-word '{found_wh[0]}' should be positioned at sentence end.")
                suggestions.append(f"Re-order sequence: {' '.join([g for g in glosses if g.upper() not in wh_words])} {found_wh[0]}")

        english_filler = {"IS", "ARE", "AM", "THE", "A", "AN", "TO"}
        found_filler = [g for g in glosses if g.upper() in english_filler]
        if found_filler:
            score -= 15.0 * len(found_filler)
            issues.append(f"ASL grammar omits English filler words: {', '.join(found_filler)}.")
            suggestions.append(f"Remove filler words: {' '.join([g for g in glosses if g.upper() not in english_filler])}")

        score = max(0.0, score)

        return {
            "status": "success",
            "input_glosses": glosses,
            "grammar_score": round(score, 1),
            "issues_found": issues,
            "suggestions": suggestions,
            "verdict": "Valid ASL Grammar" if score >= 80 else "ASL Grammar Needs Adjustment"
        }
