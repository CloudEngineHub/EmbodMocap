import os
import argparse
import shutil    
import subprocess
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from embod_mocap.processor.base import export_cameras_to_ply, combine_RT, run_cmd, write_warning_to_log, rotate_R_around_z_axis
from tqdm import trange, tqdm
from embod_mocap.human.utils.transforms import interpolate_RT


def run_ffmpeg(cmd):
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def batch_run(tasks, max_workers=8, desc=""):
    """Run tasks in parallel; each task is a (func, args) tuple."""
    if not tasks:
        return
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(func, *a) for func, a in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=desc):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice views from two videos.")
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the sequence folder containing two videos.",
    )
    parser.add_argument(
        "--v1_start",
        type=int,
        default=0,
        help="Start frame for video 1.",
    )
    parser.add_argument(
        "--v2_start",
        type=int,
        default=0,
        help="Start frame for video 2.",
    )
    parser.add_argument(
        "--vertical",
        action="store_true",
        help="Whether to stack the images vertically.",
    )
    parser.add_argument(
        "--v1_fps",
        type=int,
        default=30,
        help="FPS for view 1 (used for depth index mapping).",
    )
    parser.add_argument(
        "--v2_fps",
        type=int,
        default=30,
        help="FPS for view 2 (used for depth index mapping).",
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
        "--remove_raw_images",
        action="store_true",
        help="Deprecated: raw data cleanup is now handled by --clean_cache. This flag is kept for backward compatibility.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=5,
        help="JPEG quality for exported images (1-31, lower is higher quality).",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Log file.",
    )
    args = parser.parse_args()

    v1_start = args.v1_start
    v2_start = args.v2_start
    v1_frames_dir = os.path.join(args.input_folder, "raw1", "images")
    v2_frames_dir = os.path.join(args.input_folder, "raw2", "images")
    v1_depths_dir = os.path.join(args.input_folder, "raw1", "frames2")
    v2_depths_dir = os.path.join(args.input_folder, "raw2", "frames2")

    cameras1 = np.load(os.path.join(args.input_folder, "raw1", "cameras_sai.npz"))
    cameras2 = np.load(os.path.join(args.input_folder, "raw2", "cameras_sai.npz"))

    v1_end = cameras1['frame_ids'][-1]
    v2_end = cameras2['frame_ids'][-1]

    offset = v2_start - v1_start
    if v1_start < cameras1['frame_ids'][0]:
        v1_start = cameras1['frame_ids'][0]
        v2_start = v1_start + offset
        if v2_start < cameras2['frame_ids'][0]:
            v2_start = cameras2['frame_ids'][0]
            v1_start = v2_start - offset
    else:
        v2_start = v1_start + offset
        if v2_start < cameras2['frame_ids'][0]:
            v2_start = cameras2['frame_ids'][0]
            v1_start = v2_start - offset
            
    length = min(v1_end - v1_start, v2_end - v2_start)
    v1_end = v1_start + length
    v2_end = v2_start + length

    outfolder1 = os.path.join(args.input_folder, "v1")
    os.makedirs(outfolder1, exist_ok=True)
    os.makedirs(os.path.join(outfolder1, "images"), exist_ok=True)
    os.makedirs(os.path.join(outfolder1, "depths"), exist_ok=True)

    outfolder2 = os.path.join(args.input_folder, "v2")
    os.makedirs(outfolder2, exist_ok=True)
    os.makedirs(os.path.join(outfolder2, "images"), exist_ok=True)
    os.makedirs(os.path.join(outfolder2, "depths"), exist_ok=True)

    assert v1_end - v1_start == v2_end - v2_start, "the number of images in the two folders must be the same!"
    print("number of v1 frames: ", v1_end - v1_start, "start: ", v1_start, "end: ", v1_end)
    print("number of v2 frames: ", v2_end - v2_start, "start: ", v2_start, "end: ", v2_end)

    cameras1_sliced = dict()
    cameras1_sliced["K"] = cameras1["K"].astype(np.float32)
    w = 960
    h = 720

    frame_ids = cameras1['frame_ids']
    P1 = combine_RT(cameras1["R"], cameras1["T"])
    if len(P1) != frame_ids[-1] - frame_ids[0] + 1:
        P1 = interpolate_RT(P1, frame_ids, list(range(frame_ids[0], frame_ids[-1]+1)))
        warning_message = f"Warning: missing frames in the smoothed camera in {args.input_folder}/raw1/, so interpolated."
        print(warning_message)
        if args.log_file is not None:
            write_warning_to_log(args.log_file, warning_message)
        frame_ids = list(range(frame_ids[0], frame_ids[-1]+1))
        cameras1["R"] = P1[:, :3, :3]
        cameras1["T"] = P1[:, :3, 3]
        cameras1["frame_ids"] = frame_ids

    v1_start_sai = cameras1['frame_ids'].tolist().index(v1_start)
    v1_end_sai = cameras1['frame_ids'].tolist().index(v1_end)
    cameras1_sliced["R"] = cameras1["R"][v1_start_sai:v1_end_sai]
    cameras1_sliced["T"] = cameras1["T"][v1_start_sai:v1_end_sai].squeeze()

    assert cameras1_sliced["R"].shape[0] == cameras1_sliced["T"].shape[0], "R and T should have the same number of frames"
    assert cameras1_sliced["R"].shape[0] == length, "R and T should have the same number of frames"

    target_fps = 30
    fps_scale1 = int(args.v1_fps / target_fps)
    fps_scale2 = int(args.v2_fps / target_fps)

    # Apply camera rotation if vertical
    if args.vertical:
        # Rotate R matrix
        cameras1_sliced["R"] = rotate_R_around_z_axis(cameras1_sliced["R"], -np.pi/2)
        
        # Rotate intrinsics K (swap cx, cy with adjustment)
        H, W = 720, 960
        K1 = cameras1_sliced["K"].copy()
        cx, cy = K1[0, 2], K1[1, 2]
        K1[0, 2] = H - cy
        K1[1, 2] = cx
        cameras1_sliced["K"] = K1

    np.savez(
        os.path.join(outfolder1, "cameras_sai_sliced.npz"),
        **cameras1_sliced,
    )
    export_cameras_to_ply(
        combine_RT(cameras1_sliced["R"], cameras1_sliced["T"]),
        os.path.join(outfolder1, "cameras_sai_sliced.ply"),
    )

    if not os.path.exists(v1_frames_dir):
        if args.log_file is not None:
            write_warning_to_log(args.log_file, f"Warning: {v1_frames_dir} not found")
    if not os.path.exists(v1_depths_dir):
        if args.log_file is not None:
            write_warning_to_log(args.log_file, f"Warning: {v1_depths_dir} not found")

    new_i = 0
    rgb_tasks_v1 = []
    depth_tasks_v1 = []
    for i in range(v1_start, v1_end):
        v1_frame = os.path.join(v1_frames_dir, f"raw1_{i:04d}.jpg")
        depth_in = os.path.join(v1_depths_dir, f"{(i * fps_scale1):08d}.png")

        rotate_cmd_depth = '-vf "transpose=1"' if args.vertical else ""
        if os.path.exists(v1_frame):
            out_img = os.path.join(outfolder1, "images", f"v1_{new_i:04d}.jpg")
            rgb_tasks_v1.append((shutil.copy2, (v1_frame, out_img)))

        depth_out = os.path.join(outfolder1, "depths", f"v1_{new_i:04d}.png")
        if os.path.exists(depth_in):
            depth_tasks_v1.append((run_ffmpeg, (f'ffmpeg -i {depth_in} {rotate_cmd_depth} {depth_out} -y -loglevel quiet',)))

        new_i += 1

    batch_run(rgb_tasks_v1, max_workers=8, desc="v1 RGB")
    batch_run(depth_tasks_v1, max_workers=8, desc="v1 depth")

    ##################################################################################
    frame_ids = cameras2['frame_ids']
    P2 = combine_RT(cameras2["R"], cameras2["T"])
    if len(P2) != frame_ids[-1] - frame_ids[0] + 1:
        P2 = interpolate_RT(P2, frame_ids, list(range(frame_ids[0], frame_ids[-1]+1)))
        warning_message = f"Warning: missing frames in the smoothed camera in {args.input_folder}/raw2/, so interpolated."
        print(warning_message)
        if args.log_file is not None:
            write_warning_to_log(args.log_file, warning_message)
        frame_ids = list(range(frame_ids[0], frame_ids[-1]+1))
        cameras2["R"] = P2[:, :3, :3]
        cameras2["T"] = P2[:, :3, 3]
        cameras2["frame_ids"] = frame_ids

    cameras2_sliced = dict()
    cameras2_sliced["K"] = cameras2["K"].astype(np.float32)
    v2_start_sai = cameras2['frame_ids'].tolist().index(v2_start)
    v2_end_sai = cameras2['frame_ids'].tolist().index(v2_end)
    cameras2_sliced["R"] = cameras2["R"][v2_start_sai:v2_end_sai]
    cameras2_sliced["T"] = cameras2["T"][v2_start_sai:v2_end_sai].squeeze()
    assert cameras2_sliced["R"].shape[0] == cameras2_sliced["T"].shape[0], "R and T should have the same number of frames"
    assert cameras2_sliced["R"].shape[0] == length, "R and T should have the same number of frames"
    
    # Apply camera rotation if vertical
    if args.vertical:
        # Rotate R matrix
        cameras2_sliced["R"] = rotate_R_around_z_axis(cameras2_sliced["R"], -np.pi/2)
        
        # Rotate intrinsics K (swap cx, cy with adjustment)
        H, W = 720, 960
        K2 = cameras2_sliced["K"].copy()
        cx, cy = K2[0, 2], K2[1, 2]
        K2[0, 2] = H - cy
        K2[1, 2] = cx
        cameras2_sliced["K"] = K2
    
    np.savez(
        os.path.join(outfolder2, "cameras_sai_sliced.npz"),
        **cameras2_sliced,
    )
    export_cameras_to_ply(
        combine_RT(cameras2_sliced["R"], cameras2_sliced["T"]),
        os.path.join(outfolder2, "cameras_sai_sliced.ply"),
    )

    if not os.path.exists(v2_frames_dir):
        if args.log_file is not None:
            write_warning_to_log(args.log_file, f"Warning: {v2_frames_dir} not found")
    if not os.path.exists(v2_depths_dir):
        if args.log_file is not None:
            write_warning_to_log(args.log_file, f"Warning: {v2_depths_dir} not found")

    new_i = 0
    rgb_tasks_v2 = []
    depth_tasks_v2 = []
    for i in range(v2_start, v2_end):
        v2_frame = os.path.join(v2_frames_dir, f"raw2_{i:04d}.jpg")
        depth_in = os.path.join(v2_depths_dir, f"{(i * fps_scale2):08d}.png")

        rotate_cmd_depth = '-vf "transpose=1"' if args.vertical else ""
        if os.path.exists(v2_frame):
            out_img = os.path.join(outfolder2, "images", f"v2_{new_i:04d}.jpg")
            rgb_tasks_v2.append((shutil.copy2, (v2_frame, out_img)))

        depth_out = os.path.join(outfolder2, "depths", f"v2_{new_i:04d}.png")
        if os.path.exists(depth_in):
            depth_tasks_v2.append((run_ffmpeg, (f'ffmpeg -i {depth_in} {rotate_cmd_depth} {depth_out} -y -loglevel quiet',)))

        new_i += 1

    batch_run(rgb_tasks_v2, max_workers=8, desc="v2 RGB")
    batch_run(depth_tasks_v2, max_workers=8, desc="v2 depth")
