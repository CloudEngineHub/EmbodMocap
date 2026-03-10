import sys
import json
import os
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from embod_mocap.processor.base import export_cameras_to_ply, run_cmd, write_warning_to_log
from embod_mocap.human.utils.transforms import interpolate_RT


def compute_extrinsic_matrix(position, orientation):
    """
    Compute the camera extrinsic matrix from position and quaternion orientation.

    Parameters:
    - position: dict with keys "x", "y", "z" representing the translation vector.
    - orientation: dict with keys "w", "x", "y", "z" representing the quaternion.

    Returns:
    - extrinsic_matrix: A 4x4 numpy array representing the camera extrinsic matrix.
    """
    # Extract quaternion in (x, y, z, w) order as required by scipy
    quaternion = [
        orientation["x"],
        orientation["y"],
        orientation["z"],
        orientation["w"],
    ]
    
    # Convert quaternion to a 3x3 rotation matrix
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()  # Get the 3x3 rotation matrix

    # Extract translation vector from position
    translation_vector = np.array([
        position["x"],
        position["y"],
        position["z"],
    ])

    # Construct the 4x4 extrinsic matrix
    extrinsic_matrix = np.eye(4)  # Start with an identity matrix
    extrinsic_matrix[:3, :3] = rotation_matrix  # Set the rotation matrix
    extrinsic_matrix[:3, 3] = translation_vector  # Set the translation vector

    return extrinsic_matrix

def read_jsonl_to_numpy(file_path):
    data = {
        "timestamps": [],
        "frame_id": []
    }

    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            if not 'frames' in record:
                continue
            data["timestamps"].append(record["time"])
            data["frame_id"].append(record["number"])
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice views from two videos.")
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the sequence folder containing two videos.",
    )
    parser.add_argument(
        "--down_scale",
        type=int,
        default=2,
        help="Downscale factor for the input images",
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
        "--log_file",
        type=str,
        default=None,
        help="Path to the log file",
    )
    args = parser.parse_args()
    seq_folder = args.input_folder

    ######################################################
    if args.proc_v1:
        frame_info1 = read_jsonl_to_numpy(os.path.join(args.input_folder, "raw1", "data.jsonl"))

        with open(f"{os.path.join(args.input_folder, 'raw1')}/calibration.json", "r", encoding="utf-8") as f:
            calibration = json.load(f)
        focal = calibration['cameras'][0]['focalLengthX'] / args.down_scale
        cx = calibration['cameras'][0]['principalPointX'] / args.down_scale
        cy = calibration['cameras'][0]['principalPointY'] / args.down_scale
        K1 = np.array([[focal, 0, cx, 0],
                        [0, focal, cy, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        
        cmd = f"sai-cli smooth {seq_folder}/raw1/ {seq_folder}/raw1/cameras_sai.jsonl"
        run_cmd(cmd)
        data1 = []
        json_file1 = f"{seq_folder}/raw1/cameras_sai.jsonl"
        if not os.path.exists(json_file1):
            print(f"Smoothing camera for {seq_folder} v1 failed")
            write_warning_to_log(args.log_file, f"Smoothing camera for {seq_folder} v1 failed")
            exit()

        with open(json_file1, "r") as file:
            for line in file:
                data1.append(json.loads(line.strip()))
        P1 = []
        timestamps1 = []
        frame_ids = []
        for i in range(len(data1)):
            P1.append(compute_extrinsic_matrix(data1[i]['position'], data1[i]['orientation']))
            timestamps1.append(data1[i]['time'])
            frame_idx = frame_info1['timestamps'].index(timestamps1[-1])
            frame_ids.append(frame_info1['frame_id'][frame_idx])
        timestamps1 = np.array(timestamps1)
        P1 = np.stack(P1, axis=0)
        np.savez(f"{seq_folder}/raw1/cameras_sai.npz", K=K1, timestamps=timestamps1, frame_ids=frame_ids, R=P1[:, :3, :3], T=P1[:, :3, 3])
        export_cameras_to_ply(P1, f"{seq_folder}/raw1/cameras_sai.ply")
    else:
        print(f"Skip smoothing camera for raw1")

    ######################################################
    if args.proc_v2:
        frame_info2 = read_jsonl_to_numpy(os.path.join(args.input_folder, "raw2", "data.jsonl"))

        with open(f"{os.path.join(args.input_folder, 'raw2')}/calibration.json", "r", encoding="utf-8") as f:
            calibration = json.load(f)
        focal = calibration['cameras'][0]['focalLengthX'] / args.down_scale
        cx = calibration['cameras'][0]['principalPointX'] / args.down_scale
        cy = calibration['cameras'][0]['principalPointY'] / args.down_scale
        K2 = np.array([[focal, 0, cx, 0],
                        [0, focal, cy, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        cmd = f"sai-cli smooth {seq_folder}/raw2/ {seq_folder}/raw2/cameras_sai.jsonl"
        run_cmd(cmd)

        data2 = []
        json_file2 = f"{seq_folder}/raw2/cameras_sai.jsonl"
        if not os.path.exists(json_file2):
            print(f"Smoothing camera for {seq_folder} v2 failed")
            write_warning_to_log(args.log_file, f"Smoothing camera for {seq_folder} v2 failed")
            exit()

        with open(json_file2, "r") as file:
            for line in file:
                data2.append(json.loads(line.strip()))
        P2 = []
        timestamps2 = []
        frame_ids = []
        for i in range(len(data2)):
            P2.append(compute_extrinsic_matrix(data2[i]['position'], data2[i]['orientation']))
            timestamps2.append(data2[i]['time'])
            frame_idx = frame_info2['timestamps'].index(timestamps2[-1])
            frame_ids.append(frame_info2['frame_id'][frame_idx])
        timestamps2 = np.array(timestamps2)
        P2 = np.stack(P2, axis=0)
        np.savez(f"{seq_folder}/raw2/cameras_sai.npz", K=K2, timestamps=timestamps2, frame_ids=frame_ids, R=P2[:, :3, :3], T=P2[:, :3, 3])
        export_cameras_to_ply(P2, f"{seq_folder}/raw2/cameras_sai.ply")
    else:
        print(f"Skip smoothing camera for raw2")
