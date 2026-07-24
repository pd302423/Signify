"""
Blender Converter Script: Full-Body OBJ Sequence -> .blend Scene File.

Imports full-body + dual-hand 3D OBJ mesh frames from output_3d_meshes/full_body_obj_frames,
sets up animated frame-by-frame visibility keyframes in Blender (bpy),
applies a smooth 3D material, and saves as output_3d_meshes/full_body_3d_animation.blend.
"""

import os
import glob
import bpy

def convert_full_body_obj_to_blend():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    obj_dir = os.path.join(repo_root, "output_3d_meshes", "full_body_obj_frames")
    blend_output_path = os.path.join(repo_root, "output_3d_meshes", "full_body_3d_animation.blend")

    if not os.path.exists(obj_dir):
        print(f"❌ OBJ directory not found: {obj_dir}")
        return

    bpy.ops.wm.read_factory_settings(use_empty=True)

    obj_files = sorted(glob.glob(os.path.join(obj_dir, "*.obj")))
    if not obj_files:
        print(f"❌ No .obj files found in {obj_dir}")
        return

    print(f"🎬 Importing {len(obj_files)} Full-Body OBJ Mesh Frames into Blender...")

    # Create Body Material
    mat = bpy.data.materials.new(name="FullBody3DMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.22, 0.74, 0.97, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.3

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(obj_files)

    for i, file_path in enumerate(obj_files):
        frame_num = i + 1
        
        try:
            bpy.ops.wm.obj_import(filepath=file_path)
        except Exception:
            bpy.ops.import_scene.obj(filepath=file_path)

        imported_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        if not imported_objs:
            continue

        imported_mesh = imported_objs[0]
        imported_mesh.name = f"FullBody_Frame_{i:04d}"
        
        if imported_mesh.data.materials:
            imported_mesh.data.materials[0] = mat
        else:
            imported_mesh.data.materials.append(mat)

        imported_mesh.hide_viewport = True
        imported_mesh.hide_render = True
        
        if frame_num > 1:
            scene.frame_set(frame_num - 1)
            imported_mesh.keyframe_insert(data_path="hide_viewport")
            imported_mesh.keyframe_insert(data_path="hide_render")

        scene.frame_set(frame_num)
        imported_mesh.hide_viewport = False
        imported_mesh.hide_render = False
        imported_mesh.keyframe_insert(data_path="hide_viewport")
        imported_mesh.keyframe_insert(data_path="hide_render")

        if frame_num < len(obj_files):
            scene.frame_set(frame_num + 1)
            imported_mesh.hide_viewport = True
            imported_mesh.hide_render = True
            imported_mesh.keyframe_insert(data_path="hide_viewport")
            imported_mesh.keyframe_insert(data_path="hide_render")

    bpy.ops.wm.save_as_mainfile(filepath=blend_output_path)
    print(f"🎉 Successfully Generated Full-Body Blender File: {blend_output_path}")

if __name__ == "__main__":
    convert_full_body_obj_to_blend()
