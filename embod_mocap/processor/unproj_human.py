import cv2
import os
import json
import numpy as np
import argparse
import open3d as o3d
import copy
from embod_mocap.processor.unproj_scene import unproj_depth
from embod_mocap.processor.base import combine_RT, convert_world_cam
from embod_mocap.human.utils.mesh_utils import slice_mesh_o3d
from embod_mocap.config_paths import PATHS


def select_frames_spatially_uniform(T, n):
    N = T.shape[0]
    if n > N:
        raise ValueError("n cannot be greater than the number of trajectory frames N")
    
    selected_indices = [0]
    remaining_indices = list(range(1, N))

    for _ in range(n - 1):
        last_selected = selected_indices[-1]
        last_point = T[last_selected]

        distances = np.linalg.norm(T[remaining_indices] - last_point, axis=1)

        farthest_idx = remaining_indices[np.argmax(distances)]

        selected_indices.append(farthest_idx)
        remaining_indices.remove(farthest_idx)
    
    return sorted(selected_indices)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Unproject point clouds from depth images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_folder",
        type=str,
        default="",
        help="Path to the input sequence folder.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Scale factor for the input images",
    )
    parser.add_argument(
        "--depth_refine",
        action="store_true",
        help="Whether to use depth refinement",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--depth_trunc",
        type=float,
        default=4.0,
        help="Maximum depth value to consider",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.01,
        help="Voxel size for TSDF volume",
    )
    parser.add_argument(
        "--correct_convention",
        action="store_true",
        help="Whether to correct the camera convention",
    )
    parser.add_argument(
        "--vggt_refine",
        action="store_true",
        help="Whether to use VGGT for depth refinement",
    )
    parser.add_argument(
        "--sdf_trunc",
        type=float,
        default=0.1,
        help="SDF truncation value",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=30,
        help="Stride for selecting frames",
    )
    parser.add_argument(
        "--proc_v1",
        action="store_true",
        help="Whether to process v1",
    )
    parser.add_argument(
        "--proc_v2",
        action="store_true",
        help="Whether to process v2",
    )
    parser.add_argument(
        "--use_keyframe_depths",
        action="store_true",
        help="Use depths_keyframe_refined/ instead of depths/.",
    )
    parser.add_argument(
        "--use_keyframe_masks",
        action="store_true",
        help="Use masks_keyframe/ instead of masks/.",
    )
    
    
    args = parser.parse_args()
    W = 1440 / 2
    H = 1920 / 2
    
    # Use transformed camera file
    camera_file = "cameras_sai_transformed.npz"
    
    ##############################################################
    if args.proc_v1:
        args1 = copy.deepcopy(args)
        args1.mask_out = True
        args1.input_folder = os.path.join(args1.input_folder, "v1")
        args1.mask_dir = "masks_keyframe" if args.use_keyframe_masks else "masks"
        depth_dir = "depths_keyframe_refined" if args.use_keyframe_depths else "depths"
        args1.seg = False

        cameras_sai = np.load(os.path.join(args.input_folder, "v1", camera_file))
        R1 = cameras_sai["R"]
        T1 = cameras_sai["T"]
        K1 = cameras_sai["K"][:3, :3].astype(np.int32) 
        # scale1 = cameras_sai["scale"]
        frame_ids = list(range(0, len(R1)))
        transforms = dict()
        transforms['cx'] = float(K1[0, 2]) 
        transforms['cy'] = float(K1[1, 2]) 
        transforms['fl_x'] = float(K1[0, 0]) 
        transforms['fl_y'] = float(K1[1, 1]) 

        transforms['w'] = W
        transforms['h'] = H
        transforms['frames'] = []
        keyframes_json = os.path.join(args.input_folder, "keyframes.json")
        if os.path.exists(keyframes_json):
            with open(keyframes_json, "r", encoding="utf-8") as f:
                kf_data = json.load(f)
            valid_ids = sorted([i for i in kf_data.get("unproj", []) if i < len(frame_ids)])
            print(f"v1 using keyframes.json unproj field: {len(valid_ids)} frames")
        else:
            valid_ids = list(range(0, len(frame_ids), args.stride))

        P1 = combine_RT(R1, T1)
        
        for i in valid_ids:
            frame_id = frame_ids[i]
            transforms['frames'].append({
                'transform_matrix': P1[i].tolist(),
                'file_path': os.path.join("images", f"v1_{frame_id:04d}.jpg"),
                'depth_file_path': os.path.join(depth_dir, f"v1_{frame_id:04d}.png"),
            })
        with open(os.path.join(args.input_folder, "v1", "transforms.json"), "w") as f:
            json.dump(transforms, f, indent=4)
        mesh = unproj_depth(args1)[0]
        mesh = mesh.simplify_vertex_clustering(0.01)
        vertices = np.asarray(mesh.vertices)
        if vertices.size == 0:
            print("v1 mesh has 0 vertices; skip top-slice.")
        else:
            highest = vertices[:, 2].max()
            invalid_indices = np.where(vertices[:, 2] > (highest - 0.5))[0]
            mesh = slice_mesh_o3d(mesh, invalid_indices)

        o3d.io.write_triangle_mesh(os.path.join(args.input_folder, "v1", "mesh.ply"), mesh)
        pointcloud1 = o3d.geometry.PointCloud()
        pointcloud1.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        pointcloud1.colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
        pointcloud1.normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
        o3d.io.write_point_cloud(os.path.join(args.input_folder, "v1", "pointcloud.ply"), pointcloud1)
        print(f"v1 pointcloud saved to {os.path.join(args.input_folder, 'v1', 'pointcloud.ply')}")
    else:
        print(f"Skip unproj human for v1")  
    ###############################################################
    if args.proc_v2:
        args2 = copy.deepcopy(args)
        args2.mask_out = True
        args2.input_folder = os.path.join(args.input_folder, "v2")
        args2.mask_dir = "masks_keyframe" if args.use_keyframe_masks else "masks"
        depth_dir = "depths_keyframe_refined" if args.use_keyframe_depths else "depths"
        args2.seg = False

        cameras_sai = np.load(os.path.join(args.input_folder, "v2", camera_file))
        R2 = cameras_sai["R"]
        T2 = cameras_sai["T"]
        K2 = cameras_sai["K"][:3, :3].astype(np.int32)
        # scale2 = cameras_sai["scale"]
        frame_ids = list(range(0, len(R2)))
        transforms = dict()
        transforms['cx'] = float(K2[0, 2])
        transforms['cy'] = float(K2[1, 2])
        transforms['fl_x'] = float(K2[0, 0])
        transforms['fl_y'] = float(K2[1, 1])
        transforms['w'] = W
        transforms['h'] = H
        transforms['frames'] = []
        keyframes_json = os.path.join(args.input_folder, "keyframes.json")
        if os.path.exists(keyframes_json):
            with open(keyframes_json, "r", encoding="utf-8") as f:
                kf_data = json.load(f)
            valid_ids = sorted([i for i in kf_data.get("unproj", []) if i < len(frame_ids)])
            print(f"v2 using keyframes.json unproj field: {len(valid_ids)} frames")
        else:
            valid_ids = list(range(0, len(frame_ids), args.stride))
        P2 = combine_RT(R2, T2)
        for i in valid_ids:
            frame_id = frame_ids[i]
            transforms['frames'].append({
                'transform_matrix': P2[i].tolist(),
                'file_path': os.path.join("images", f"v2_{frame_id:04d}.jpg"),
                'depth_file_path': os.path.join(depth_dir, f"v2_{frame_id:04d}.png"),
            })
        with open(os.path.join(args.input_folder, "v2", "transforms.json"), "w") as f:
            json.dump(transforms, f, indent=4)
        # args2.scale = scale2
        mesh = unproj_depth(args2)[0]
        mesh = mesh.simplify_vertex_clustering(0.01)
        vertices = np.asarray(mesh.vertices)
        if vertices.size == 0:
            print("v2 mesh has 0 vertices; skip top-slice.")
        else:
            highest = vertices[:, 2].max()
            invalid_indices = np.where(vertices[:, 2] > (highest - 0.5))[0]
            mesh = slice_mesh_o3d(mesh, invalid_indices)

        o3d.io.write_triangle_mesh(os.path.join(args.input_folder, "v2", "mesh.ply"), mesh)
        pointcloud2 = o3d.geometry.PointCloud()
        pointcloud2.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        pointcloud2.colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
        pointcloud2.normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
        o3d.io.write_point_cloud(os.path.join(args.input_folder, "v2", "pointcloud.ply"), pointcloud2)
        print(f"v2 pointcloud saved to {os.path.join(args.input_folder, 'v2', 'pointcloud.ply')}")
    else:
        print(f"Skip unproj human for v2")  
