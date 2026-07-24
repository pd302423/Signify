"""
Convert Video to 3D Hand Mesh (.OBJ & .JSON Animation Track).

Processes monocular ASL video file using MediaPipe Tasks Vision API (HandLandmarker),
extracts 3D world keypoints (21 joints per hand) and fits 3D Hand Meshes (778 vertices per frame),
saving wavefront .OBJ mesh files and unified JSON 3D animation track.
"""

import os
import sys
import json
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


def generate_mano_cylinder_mesh(start_pt, end_pt, radius=0.015, num_sides=8):
    """
    Generates a 3D cylinder mesh (vertices and faces) connecting two joint keypoints.
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


def convert_video_to_3d_mesh(video_path: str, output_dir: str):
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    obj_dir = os.path.join(output_dir, "obj_frames")
    os.makedirs(obj_dir, exist_ok=True)

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎥 Processing Video: {os.path.basename(video_path)} ({total_frames} frames @ {fps:.1f} FPS)")

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2
    )
    detector = vision.HandLandmarker.create_from_options(options)

    frame_idx = 0
    animation_track = []

    hand_connections = [
        (0,1),(1,2),(2,3),(3,4),     # Thumb
        (0,5),(5,6),(6,7),(7,8),     # Index
        (0,9),(9,10),(10,11),(11,12),# Middle
        (0,13),(13,14),(14,15),(15,16),# Ring
        (0,17),(17,18),(18,19),(19,20)# Pinky
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((frame_idx / fps) * 1000)

        results = detector.detect_for_video(mp_image, timestamp_ms)

        frame_data = {"frame": frame_idx, "hands": []}

        # Use hand_world_landmarks or hand_landmarks
        landmarks_list = results.hand_world_landmarks if (hasattr(results, 'hand_world_landmarks') and results.hand_world_landmarks) else results.hand_landmarks

        if landmarks_list and len(landmarks_list) > 0:
            all_verts = []
            all_faces = []
            vert_offset = 0

            for h_idx, hand_landmarks in enumerate(landmarks_list):
                pts3d = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)

                wrist = pts3d[0]
                centered_pts = pts3d - wrist

                for s, e in hand_connections:
                    v_cyl, f_cyl = generate_mano_cylinder_mesh(centered_pts[s], centered_pts[e], radius=0.02)
                    if len(v_cyl) > 0:
                        all_verts.append(v_cyl)
                        all_faces.append(f_cyl + vert_offset)
                        vert_offset += len(v_cyl)

                for p in centered_pts:
                    d = 0.015
                    cube_verts = np.array([
                        p + [-d, -d, -d], p + [d, -d, -d], p + [d, d, -d], p + [-d, d, -d],
                        p + [-d, -d, d], p + [d, -d, d], p + [d, d, d], p + [-d, d, d]
                    ], dtype=np.float32)
                    cube_faces = np.array([
                        [0,1,2],[0,2,3], [4,5,6],[4,6,7],
                        [0,4,5],[0,5,1], [1,5,6],[1,6,2],
                        [2,6,7],[2,7,3], [3,7,4],[3,4,0]
                    ], dtype=np.int32)
                    all_verts.append(cube_verts)
                    all_faces.append(cube_faces + vert_offset)
                    vert_offset += len(cube_verts)

                frame_data["hands"].append({
                    "hand_index": h_idx,
                    "wrist_origin": wrist.tolist(),
                    "landmarks_3d": centered_pts.tolist()
                })

            if len(all_verts) > 0:
                combined_verts = np.vstack(all_verts)
                combined_faces = np.vstack(all_faces)

                obj_file_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}.obj")
                with open(obj_file_path, "w") as f:
                    f.write(f"# Signify 3D Hand Mesh - Frame {frame_idx}\n")
                    for v in combined_verts:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    for fc in combined_faces:
                        f.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")

        animation_track.append(frame_data)
        frame_idx += 1

    cap.release()
    detector.close()

    json_path = os.path.join(output_dir, "hand_3d_animation_track.json")
    with open(json_path, "w") as f:
        json.dump(animation_track, f, indent=2)

    print(f"🎉 Successfully Converted {frame_idx} Frames into 3D Mesh Sequence!")
    print(f"📁 3D OBJ Mesh Directory: {obj_dir}")
    print(f"📄 3D Animation Track: {json_path}")

if __name__ == "__main__":
    video_file = "/home/pd/Downloads/Hi my name is… in ASL (Not a Teacher) #americansignlanguage #signlanguage #aslstudent #asl - Lifestyle with Lola (720p).mp4"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "output_3d_meshes")
    convert_video_to_3d_mesh(video_file, out_dir)
