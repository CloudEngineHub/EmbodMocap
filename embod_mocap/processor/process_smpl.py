import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from embod_mocap.processor.process_frames import (
    inference_human,
    check_existing_outputs,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process SMPL parameters from video frames (bbox + keypoints + VIMO)."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the sequence folder.",
    )
    parser.add_argument(
        "--v1_fps",
        default=30,
        type=int,
    )
    parser.add_argument(
        "--v2_fps",
        default=30,
        type=int,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--vimo_stride",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="overwrite",
        choices=["overwrite", "skip"],
    )
    args = parser.parse_args()

    v1_folder = f"{args.folder}/v1"
    v2_folder = f"{args.folder}/v2"
    v1_frames_dir = f"{v1_folder}/images"
    v2_frames_dir = f"{v2_folder}/images"

    ########## v1
    print("Processing v1 SMPL params")
    raw1_image_list = []
    image_names = sorted([f for f in os.listdir(v1_frames_dir) if f.startswith("v1_") and f.endswith(".jpg")])
    for name in image_names:
        raw1_image_list.append(os.path.join(v1_frames_dir, name))
    num_v1_images = len(raw1_image_list)

    v1_skip_smpl = False
    if args.mode == "skip":
        _, _, v1_skip_smpl = check_existing_outputs(
            v1_folder, "v1", num_v1_images, [], False, False
        )

    if not v1_skip_smpl:
        with torch.autocast('cuda', dtype=torch.float32):
            smpl_params, _ = inference_human(
                raw1_image_list,
                device=args.device,
                vimo_stride=args.vimo_stride,
                skip_masks=True,
            )
        np.savez(os.path.join(v1_folder, "smpl_params.npz"), **smpl_params)
    else:
        print("v1 smpl_params.npz already exists, skip")

    print("processing v1 SMPL done.")

    ########## v2
    print("Processing v2 SMPL params")
    raw2_image_list = []
    image_names = sorted([f for f in os.listdir(v2_frames_dir) if f.startswith("v2_") and f.endswith(".jpg")])
    for name in image_names:
        raw2_image_list.append(os.path.join(v2_frames_dir, name))
    num_v2_images = len(raw2_image_list)

    v2_skip_smpl = False
    if args.mode == "skip":
        _, _, v2_skip_smpl = check_existing_outputs(
            v2_folder, "v2", num_v2_images, [], False, False
        )

    if not v2_skip_smpl:
        with torch.autocast('cuda', dtype=torch.float32):
            smpl_params, _ = inference_human(
                raw2_image_list,
                device=args.device,
                vimo_stride=args.vimo_stride,
                skip_masks=True,
            )
        np.savez(os.path.join(v2_folder, "smpl_params.npz"), **smpl_params)
    else:
        print("v2 smpl_params.npz already exists, skip")

    print("processing v2 SMPL done.")
