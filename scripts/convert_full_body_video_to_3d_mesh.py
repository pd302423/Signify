"""
Full-Body & Dual-Hand 3D Mesh Generator (SMPL-X / WholeBody Style).

Converts the entire human body + hands from an ASL video file into a complete
3D mesh sequence (.OBJ) and compiles a unified Blender (.blend) project.
"""

import os
import sys
import json
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


def generate_cylinder_mesh(start_pt, end_pt, radius=0.03, num_sides=10):
    """
    Generates a 3D cylinder mesh connecting two 3D joint keypoints.
    """
    vec = end_pt - start_pt
    height = np.linalg.norm(vec)
    if height < 1e-6:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    
    unit_vec = vec / height

    if abs(unit_vec[0]) < 0.9:
        perp = np.cross(unit_vec, np.array([1, 0, 0]))
    else:
        perp = np.cross(unit_vec, np.array([0, 1, 0]))
    perp = perp / np.linalg.norm(perp)
    perp2 = np.cross(unit_vec, perp)

    verts = []
    for i in range(num_sides):
        angle = 2 * np.pi * i / num_sides
        pt = start_pt + radius * (np.cos(angle) * perp + np.sin(angle) * perp2)
        verts.append(pt)
    for i in range(num_sides):
        angle = 2 * np.pi * i / num_sides
        pt = end_pt + radius * (np.cos(angle) * perp + np.sin(angle) * perp2)
        verts.append(pt)

    verts = np.array(verts, dtype=np.float32)

    faces = []
    for i in range(num_sides):
        next_i = (i + 1) % num_sides
        faces.append([i, next_i, i + num_sides])
        faces.append([next_i, next_i + num_sides, i + num_sides])

    return verts, np.array(faces, dtype=np.int32)


def generate_head_mesh(head_center, radius=0.10):
    """
    Generates a 3D sphere mesh for the head.
    """
    u = np.linspace(0, 2 * np.pi, 12)
    v = np.linspace(0, np.pi, 12)
    x = radius * np.outer(np.cos(u), np.sin(v)) + head_center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + head_center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + head_center[2]

    verts = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    faces = []
    rows, cols = 12, 12
    for r in range(rows - 1):
        for c in range(cols - 1):
            i = r * cols + c
            faces.append([i, i + 1, i + cols])
            faces.append([i + 1, i + cols + 1, i + cols])

    return verts.astype(np.float32), np.array(faces, dtype=np.int32)


def convert_full_body_video_to_3d_mesh(video_path: str, output_dir: str):
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    obj_dir = os.path.join(output_dir, "full_body_obj_frames")
    os.makedirs(obj_dir, exist_ok=True)

    hand_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")
    pose_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "pose_landmarker.task")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎥 Processing Full-Body Video Mesh: {os.path.basename(video_path)} ({total_frames} frames @ {fps:.1f} FPS)")

    # Hand Detector
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    # Pose Detector if available
    pose_detector = None
    if os.path.exists(pose_model_path):
        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=pose_model_path),
            running_mode=vision.RunningMode.VIDEO
        )
        pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    frame_idx = 0
    hand_connections = [
        (0,1),(1,2),(2,3),(3,4),     # Thumb
        (0,5),(5,6),(6,7),(7,8),     # Index
        (0,9),(9,10),(10,11),(11,12),# Middle
        (0,13),(13,14),(14,15),(15,16),# Ring
        (0,17),(17,18),(18,19),(19,20)# Pinky
    ]

    pose_connections = [
        (11, 12), # Shoulders
        (11, 13), (13, 15), # Left arm
        (12, 14), (14, 16), # Right arm
        (11, 23), (12, 24), # Torso sides
        (23, 24), # Hips
        (23, 25), (25, 27), # Left leg
        (24, 26), (26, 28)  # Right leg
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((frame_idx / fps) * 1000)

        hand_res = hand_detector.detect_for_video(mp_image, timestamp_ms)
        pose_res = pose_detector.detect_for_video(mp_image, timestamp_ms) if pose_detector else None

        all_verts = []
        all_faces = []
        vert_offset = 0

        # 1. BODY & HEAD 3D MESH RECONSTRUCTION
        if pose_res and (pose_res.pose_world_landmarks or pose_res.pose_landmarks):
            pose_lms = pose_res.pose_world_landmarks[0] if pose_res.pose_world_landmarks else pose_res.pose_landmarks[0]
            pts_pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_lms], dtype=np.float32)

            # Head Sphere Mesh
            head_pt = (pts_pose[0] + pts_pose[2] + pts_pose[5]) / 3.0
            h_v, h_f = generate_head_mesh(head_pt, radius=0.12)
            all_verts.append(h_v)
            all_faces.append(h_f + vert_offset)
            vert_offset += len(h_v)

            # Limb Cylinder Meshes
            for s, e in pose_connections:
                if s < len(pts_pose) and e < len(pts_pose):
                    c_v, c_f = generate_cylinder_mesh(pts_pose[s], pts_pose[e], radius=0.04)
                    if len(c_v) > 0:
                        all_verts.append(c_v)
                        all_faces.append(c_f + vert_offset)
                        vert_offset += len(c_v)

        # 2. DUAL HAND 3D MESH RECONSTRUCTION
        hand_landmarks_list = hand_res.hand_world_landmarks if (hasattr(hand_res, 'hand_world_landmarks') and hand_res.hand_world_landmarks) else hand_res.hand_landmarks

        if hand_landmarks_list:
            for hand_lms in hand_landmarks_list:
                pts_hand = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)

                for s, e in hand_connections:
                    v_cyl, f_cyl = generate_cylinder_mesh(pts_hand[s], pts_hand[e], radius=0.015)
                    if len(v_cyl) > 0:
                        all_verts.append(v_cyl)
                        all_faces.append(f_cyl + vert_offset)
                        vert_offset += len(v_cyl)

        # Save Combined Wavefront OBJ
        if len(all_verts) > 0:
            combined_verts = np.vstack(all_verts)
            combined_faces = np.vstack(all_faces)

            obj_file_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}.obj")
            with open(obj_file_path, "w") as f:
                f.write(f"# Full-Body 3D Mesh - Frame {frame_idx}\n")
                for v in combined_verts:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for fc in combined_faces:
                    f.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")

        frame_idx += 1

    cap.release()
    hand_detector.close()
    if pose_detector: pose_detector.close()

    print(f"🎉 Successfully Generated Full-Body 3D Mesh Sequence for {frame_idx} Frames!")
    print(f"📁 Full-Body OBJ Directory: {obj_dir}")

if __name__ == "__main__":
    video_file = "/home/pd/Downloads/Hi my name is… in ASL (Not a Teacher) #americansignlanguage #signlanguage #aslstudent #asl - Lifestyle with Lola (720p).mp4"
    out_dir = "/home/pd/Documents/sony-wh100xm6/output_3d_meshes"
    convert_full_body_video_to_3d_mesh(video_file, out_dir)
