import os
import cv2
import re
import random
import json
import math
import numpy as np
import open3d as o3d
import argparse
from embod_mocap.config_paths import PATHS
from embod_mocap.processor.base import CAM_CONVENTION_CHANGE, load_transform_json, export_cameras_to_ply, run_cmd, combine_RT, write_warning_to_log
from scipy.spatial.transform import Rotation as R


def check_points_in_mask(points, mask):
    """
    Check whether input points fall on True values in a boolean mask.

    Args:
        points (numpy.ndarray): point array with shape (N, 2), each row is (x, y).
        mask (numpy.ndarray): boolean mask with shape (H, W).

    Returns:
        numpy.ndarray: boolean array of shape (N,) indicating whether each point lies on True mask values.
    """
    H, W = mask.shape
    
    x, y = points[:, 0].astype(np.int32), points[:, 1].astype(np.int32)
    
    in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    
    result = np.zeros(len(points), dtype=bool)
    valid_indices = np.where(in_bounds)[0]
    result[valid_indices] = mask[y[valid_indices], x[valid_indices]]

    return result



def parse_colmap_files(folder):
    """
    Parse COLMAP output files: images.txt, points3D.txt, cameras.txt.

    Returns:
        cameras (dict): Dictionary of camera intrinsics.
        images (list): List of images with their poses and 2D-3D correspondences.
        points3D (dict): Dictionary of 3D points and their coordinates.
    """
    # Parse cameras.txt
    images_path, points3D_path, cameras_path = f"{folder}/images.txt", f"{folder}/points3D.txt", f"{folder}/cameras.txt"
    cameras = {}
    with open(cameras_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = np.array(list(map(float, elems[4:])))
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params  # Typically [fx, fy, cx, cy]
            }

    # Parse images.txt
    images = []
    with open(images_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('#') or not line.strip():
                i += 1
                continue
            # Image data
            elems = line.split()
            image_id = int(elems[0])
            qw, qx, qy, qz = map(float, elems[1:5])  # Quaternion
            tx, ty, tz = map(float, elems[5:8])  # Translation
            camera_id = int(elems[8])
            image_name = elems[9]
            # 2D-3D correspondences
            i += 1
            points_line = lines[i].strip()
            points = list(map(float, points_line.split()))
            points = np.array(points).reshape(-1, 3)  # [x, y, 3D_id]
            points = points[points[:, 2] > 0]  # Filter out invalid points
            if image_name.startswith("v"):
                images.append({
                    'id': image_id,
                    'qvec': np.array([qw, qx, qy, qz]),
                    'tvec': np.array([tx, ty, tz]),
                    'camera_id': camera_id,
                    'image_name': image_name,
                    'points': points,
                })
            i += 1

    # Parse points3D.txt
    points3D = {}
    with open(points3D_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            elems = line.split()
            point_id = int(elems[0])
            xyz = np.array(list(map(float, elems[1:4])))
            rgb = list(map(int, elems[4:7]))
            error = float(elems[7])
            track = list(map(int, elems[8:]))
            points3D[point_id] = {
                'xyz': xyz,
                'rgb': rgb,
                'error': error,
                'track': track
            }

    return cameras, images, points3D

def read_text(file_path):
    view_lines = []
    pattern = re.compile(r'v.*\.jpg')
    quat = []
    transl = []
    fnames = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if pattern.search(line):
                view_lines.append(line.strip())
                quat.append(view_lines[-1].split(' ')[1:5])
                transl.append(view_lines[-1].split(' ')[5:8])
                fnames.append(view_lines[-1].split(' ')[-1])
    quat = np.array(quat, dtype=np.float32)
    transl = np.array(transl, dtype=np.float32)
    return quat, transl, fnames

def parse_camera(view_folder):
    quat, transl, fnames = read_text(os.path.join(view_folder, "colmap", "images.txt"))
    quat = np.vstack([quat[:, 0], -quat[:, 3], quat[:, 2], quat[:, 1]]).T
    rotmat = R.from_quat(quat).as_matrix()
    transl[:, 0] *= -1
    transl = rotmat @ transl[..., None]
    P = np.eye(4, 4)[None].repeat(len(quat), axis=0)
    P[:, :3, :3] = rotmat
    P[:, :3, 3:4] = transl
    P = P @ CAM_CONVENTION_CHANGE[None]
    Rotmat = P[:, :3, :3]
    T = P[:, :3, 3:4]
    return Rotmat, T, fnames

def rotation_matrix_to_quaternion(matrix):

    rotation = R.from_matrix(matrix) 
    quat = rotation.as_quat() 
    return [quat[0], quat[3], quat[2], -quat[1]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calibrate the human views to the scene")
    parser.add_argument(
        "input_folder",
        type=str,
        default="",
        help="Path to the sequence folder.",
    )
    parser.add_argument(
        "--colmap_num",
        type=int,
        default=500,
        help="Scale factor for matching images.",
    )
    parser.add_argument(
        "--vertical",
        action="store_true",
        help="Vertical camera.",
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
        "--keyframe_mask",
        action="store_true",
        help="Skip mask filtering when only keyframe masks are available.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="",
        help="Path to the log file.",
    )
    parser.add_argument(
        "--min_valid_ratio",
        type=float,
        default=0.1,
        help="Minimum ratio of valid registered frames to trigger retry.",
    )
    args = parser.parse_args()
    scene_path = os.path.dirname(args.input_folder)
    
    w = 960
    h = 720

    if args.proc_v1:
        v1_path = os.path.join(args.input_folder, "v1")
        image_names1 = sorted(os.listdir(os.path.join(v1_path, "images")))
        num_frames1 = len(image_names1)
        with open(f"{os.path.join(args.input_folder, 'raw1')}/calibration.json", "r", encoding="utf-8") as f:
            calibration = json.load(f)
        focal = calibration['cameras'][0]['focalLengthX'] / 2
        cx = calibration['cameras'][0]['principalPointX'] / 2
        cy = calibration['cameras'][0]['principalPointY'] / 2
    
        K = np.array([[focal, 0, cx, 0],
                        [0, focal, cy, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        if args.vertical:
            cx, cy = h - cy, cx
            K = np.array([[focal, 0, cx, 0],
                            [0, focal, cy, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        images_dump = dict()
        points3D_dump = dict()

        def sample_and_register(view_path, image_names, colmap_num, scene_path, focal, cx, cy, attempt="uniform"):
            if len(image_names) > colmap_num:
                if attempt == "uniform":
                    stride = max(1, len(image_names) // colmap_num)
                    sampled = image_names[::stride][:colmap_num]
                else:
                    sampled = sorted(random.sample(image_names, colmap_num))
            else:
                sampled = image_names
            print(f"Processing {len(sampled)} images ({attempt} sampling, total {len(image_names)})")
            with open(os.path.join(view_path, "image-list.txt"), "w", encoding="utf-8") as f:
                for line in sampled:
                    f.write(line + "\n")
            cmd = f"bash processor/regist_seq.sh {scene_path} {view_path} {focal} {cx} {cy} {PATHS.colmap_vocab_tree_path}"
            run_cmd(cmd)

        def run_registration(view_path, view_prefix, image_names, num_frames, attempt="uniform"):
            sample_and_register(view_path, image_names, args.colmap_num, scene_path, focal, cx, cy, attempt)
            cameras_parsed, images_parsed, points3D_parsed = parse_colmap_files(f'{view_path}/colmap')
            imgs_dump = dict()
            for image in images_parsed:
                im_name = image['image_name']
                if args.keyframe_mask:
                    imgs_dump[im_name] = image['points']
                else:
                    mask = cv2.imread(os.path.join(view_path, "masks", im_name.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED) > 127
                    point_mask = check_points_in_mask(image['points'][:, :2], mask)
                    valid = np.where(point_mask==False)[0]
                    imgs_dump[im_name] = image['points'][valid]
            R_out, T_out, fnames_out = parse_camera(view_path)
            
            frame_idx = []
            fname_to_colmap_idx = {}
            for colmap_idx, fname in enumerate(fnames_out):
                fname_to_colmap_idx[fname] = colmap_idx
            
            for i in range(num_frames):
                fname = f"{view_prefix}_{i:04d}.jpg"
                if fname in fname_to_colmap_idx:
                    frame_idx.append(i)
            
            reorder_indices = [fname_to_colmap_idx[f"{view_prefix}_{fid:04d}.jpg"] for fid in frame_idx]
            R_out = R_out[reorder_indices]
            T_out = T_out[reorder_indices]
            
            return R_out, T_out, fnames_out, frame_idx, imgs_dump, points3D_parsed

        try:
            R1, T1, fnames1, source_frame_idx, images_dump, points3D_result = run_registration(
                v1_path, "v1", image_names1, num_frames1, "uniform")

            valid_ratio = len(source_frame_idx) / args.colmap_num if len(image_names1) > args.colmap_num else len(source_frame_idx) / len(image_names1)
            if valid_ratio < args.min_valid_ratio and len(image_names1) > args.colmap_num:
                print(f"v1: valid ratio {valid_ratio:.2f} < {args.min_valid_ratio}, retrying with random sampling...")
                R1, T1, fnames1, source_frame_idx, images_dump, points3D_result = run_registration(
                    v1_path, "v1", image_names1, num_frames1, "random")
                print(f"v1 retry: {len(source_frame_idx)} valid frames")

            points3D_dump = dict()
            points_ids = np.array(list(points3D_result.keys()))
            points_xyz = [points3D_result[pid]['xyz'] for pid in points3D_result]
            points_rgb = [points3D_result[pid]['rgb'] for pid in points3D_result]
            points3D_dump['point_ids'] = points_ids
            points3D_dump['points'] = np.array(points_xyz, dtype=np.float32)
            points3D_dump['colors'] = np.array(points_rgb, dtype=np.uint8)

            np.savez_compressed(os.path.join(args.input_folder, "v1", "points3D.npz"), **points3D_dump)
            np.savez_compressed(os.path.join(args.input_folder, "v1", "points2D.npz"), **images_dump)

            cameras = dict(K=K, R=R1, T=T1, valid_ids=source_frame_idx)
            RT1 = combine_RT(R1, T1)
            export_cameras_to_ply(RT1, os.path.join(args.input_folder, "v1", "cameras_colmap.ply"))
            np.savez(os.path.join(args.input_folder, "v1", "cameras_colmap.npz"), **cameras)

        except Exception as e:
            print(f"Error parsing colmap camera for {args.input_folder} v1: {e}")
            write_warning_to_log(args.log_file, f"Colmap regist camera for {args.input_folder} v1 failed")

    else:
        print(f"Skip colmap register for human cam v1")
    ##############################################################################
    if args.proc_v2:
        v2_path = os.path.join(args.input_folder, "v2")
        image_names2 = sorted(os.listdir(os.path.join(v2_path, "images")))
        num_frames2 = len(image_names2)
        with open(f"{os.path.join(args.input_folder, 'raw2')}/calibration.json", "r", encoding="utf-8") as f:
            calibration = json.load(f)
        focal = calibration['cameras'][0]['focalLengthX'] / 2
        cx = calibration['cameras'][0]['principalPointX'] / 2
        cy = calibration['cameras'][0]['principalPointY'] / 2

        K = np.array([[focal, 0, cx, 0],
                        [0, focal, cy, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        if args.vertical:
            cx, cy = h - cy, cx
            K = np.array([[focal, 0, cx, 0],
                            [0, focal, cy, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        try:
            R2, T2, fnames2, source_frame_idx, images_dump, points3D_result = run_registration(
                v2_path, "v2", image_names2, num_frames2, "uniform")

            valid_ratio = len(source_frame_idx) / args.colmap_num if len(image_names2) > args.colmap_num else len(source_frame_idx) / len(image_names2)
            if valid_ratio < args.min_valid_ratio and len(image_names2) > args.colmap_num:
                print(f"v2: valid ratio {valid_ratio:.2f} < {args.min_valid_ratio}, retrying with random sampling...")
                R2, T2, fnames2, source_frame_idx, images_dump, points3D_result = run_registration(
                    v2_path, "v2", image_names2, num_frames2, "random")
                print(f"v2 retry: {len(source_frame_idx)} valid frames")

            points3D_dump = dict()
            points_ids = np.array(list(points3D_result.keys()))
            points_xyz = [points3D_result[pid]['xyz'] for pid in points3D_result]
            points_rgb = [points3D_result[pid]['rgb'] for pid in points3D_result]
            points3D_dump['point_ids'] = points_ids
            points3D_dump['points'] = np.array(points_xyz, dtype=np.float32)
            points3D_dump['colors'] = np.array(points_rgb, dtype=np.uint8)

            np.savez_compressed(os.path.join(args.input_folder, "v2", "points3D.npz"), **points3D_dump)
            np.savez_compressed(os.path.join(args.input_folder, "v2", "points2D.npz"), **images_dump)

            cameras = dict(K=K, R=R2, T=T2, valid_ids=source_frame_idx)
            RT2 = combine_RT(R2, T2)
            export_cameras_to_ply(RT2, os.path.join(args.input_folder, "v2", "cameras_colmap.ply"))
            np.savez(os.path.join(args.input_folder, "v2", "cameras_colmap.npz"), **cameras)
        except Exception as e:
            print(f"Error parsing colmap camera for {args.input_folder} v2: {e}")
            write_warning_to_log(args.log_file, f"Colmap regist camera for {args.input_folder} v2 failed")

    else:
        print(f"Skip colmap register for human cam v2")
