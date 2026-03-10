import argparse
import json
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from embod_mocap.config_paths import PATHS
from embod_mocap.processor.base import lingbotdepth_refine_batch, MDMModel, load_image_rotate
from embod_mocap.human.utils.lang_sam_utils import lang_sam_forward
from embod_mocap.processor.process_frames import process_depth_batch
from embod_mocap.processor.colmap_human_cam import check_points_in_mask
from PIL import Image


def load_keyframes_json(folder):
    path = os.path.join(folder, "keyframes.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_masks(image_list, frame_ids, output_dir, prefix,
                   lang_sam_chunk_size=5,
                   lang_sam_sam_type="sam2.1_hiera_small",
                   lang_sam_sam_ckpt_path=None):
    os.makedirs(output_dir, exist_ok=True)
    text_prompt = "person."
    selected_imgs = []
    selected_ids = []
    for fid in frame_ids:
        if 0 <= fid < len(image_list):
            selected_imgs.append(Image.open(image_list[fid]).convert("RGB"))
            selected_ids.append(fid)

    if not selected_imgs:
        return

    results = lang_sam_forward(
        selected_imgs,
        text_prompt,
        chunk_size=lang_sam_chunk_size,
        sam_type=lang_sam_sam_type,
        sam_ckpt_path=lang_sam_sam_ckpt_path,
    )
    for img, fid, res in zip(selected_imgs, selected_ids, results):
        out_path = os.path.join(output_dir, f"{prefix}_{fid:04d}.png")
        if res['masks'] is not None and len(res['masks']) > 0:
            mask = res['masks'][0] * 255
            cv2.imwrite(out_path, mask)
        else:
            # Keep one mask per requested frame so downstream consumers do not crash
            # when detector misses a person on some keyframes.
            w, h = img.size
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(out_path, empty_mask)


def filter_points2d_with_masks(points2d_path, mask_dir, prefix, output_path):
    data = np.load(points2d_path, allow_pickle=True)
    filtered = {}
    for key in data.files:
        points = data[key]
        mask_name = key.replace(".jpg", ".png")
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) > 127
            point_mask = check_points_in_mask(points[:, :2], mask)
            valid = np.where(point_mask == False)[0]
            filtered[key] = points[valid]
        else:
            filtered[key] = points
    np.savez_compressed(output_path, **filtered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process depth refinement, mask generation, and points2D filtering."
    )
    parser.add_argument("folder", type=str, help="Path to the sequence folder.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--need_all_depth_mask", action="store_true", 
                        help="Process all frames (depth refine + mask). If False, only process keyframes.")
    parser.add_argument("--depth_refine_chunk_size", type=int, default=1)
    parser.add_argument("--depth_refine_max_size", type=int, default=540)
    parser.add_argument("--skip_masks", action="store_true")
    parser.add_argument("--lang_sam_chunk_size", type=int, default=5)
    parser.add_argument("--lang_sam_sam_type", type=str, default="sam2.1_hiera_small")
    parser.add_argument("--lang_sam_sam_ckpt_path", type=str, default=None)
    parser.add_argument("--use_vggt", action="store_true", help="Include vggt keyframes.")
    parser.add_argument("--use_unproj", action="store_true", help="Include unproj keyframes.")
    parser.add_argument("--use_p2p", action="store_true", help="Include p2p keyframes.")
    parser.add_argument("--mode", type=str, default="overwrite", choices=["overwrite", "skip"])
    args = parser.parse_args()
    
    v1_folder = f"{args.folder}/v1"
    v2_folder = f"{args.folder}/v2"
    v1_frames_dir = f"{v1_folder}/images"
    v2_frames_dir = f"{v2_folder}/images"
    device = torch.device(args.device)

    kf_data = load_keyframes_json(args.folder)
    if kf_data is None:
        print(f"Warning: keyframes.json not found in {args.folder}, skip")
        exit(0)

    raw1_image_list = sorted([
        os.path.join(v1_frames_dir, f)
        for f in os.listdir(v1_frames_dir)
        if f.startswith("v1_") and f.endswith(".jpg")
    ])
    raw2_image_list = sorted([
        os.path.join(v2_frames_dir, f)
        for f in os.listdir(v2_frames_dir)
        if f.startswith("v2_") and f.endswith(".jpg")
    ])
    num_v1 = len(raw1_image_list)
    num_v2 = len(raw2_image_list)

    # 确定要处理的帧
    if args.need_all_depth_mask:
        # 处理所有帧（不需要读取关键帧列表）
        v1_keyframe_ids = None
        v2_keyframe_ids = None
        print(f"Processing ALL frames for depth refine and mask generation")
        print(f"Total frames: v1={num_v1}, v2={num_v2}")
    else:
        # 只处理关键帧（需要读取关键帧列表）
        v1_kf_set = set()
        v2_kf_set = set()
        if args.use_vggt:
            v1_kf_set.update(kf_data.get("vggt", []))
            v2_kf_set.update(kf_data.get("vggt", []))
        if args.use_unproj:
            v1_kf_set.update(kf_data.get("unproj", []))
            v2_kf_set.update(kf_data.get("unproj", []))
        if args.use_p2p:
            v1_kf_set.update(kf_data.get("p2p_v1", []))
            v2_kf_set.update(kf_data.get("p2p_v2", []))

        v1_keyframe_ids = sorted(v1_kf_set)
        v2_keyframe_ids = sorted(v2_kf_set)
        
        if not v1_keyframe_ids and not v2_keyframe_ids:
            print("Warning: No keyframes selected. Use --use_vggt, --use_unproj, or --use_p2p to select keyframes.")
            exit(0)
        
        print(f"Processing keyframes only: v1={len(v1_keyframe_ids)}, v2={len(v2_keyframe_ids)}")

    # 根据模式设置输出目录
    if args.need_all_depth_mask:
        # 处理所有帧：输出到 depths_refined/ 和 masks/
        depth_frames_outdir1 = f"{v1_folder}/depths_refined"
        depth_frames_outdir2 = f"{v2_folder}/depths_refined"
        mask_outdir1 = f"{v1_folder}/masks"
        mask_outdir2 = f"{v2_folder}/masks"
    else:
        # 只处理关键帧：输出到 depths_keyframe_refined/ 和 masks_keyframe/
        depth_frames_outdir1 = f"{v1_folder}/depths_keyframe_refined"
        depth_frames_outdir2 = f"{v2_folder}/depths_keyframe_refined"
        mask_outdir1 = f"{v1_folder}/masks_keyframe"
        mask_outdir2 = f"{v2_folder}/masks_keyframe"
    
    depth_frames_dir1 = f"{v1_folder}/depths"
    depth_frames_dir2 = f"{v2_folder}/depths"
    os.makedirs(depth_frames_outdir1, exist_ok=True)
    os.makedirs(depth_frames_outdir2, exist_ok=True)
    os.makedirs(mask_outdir1, exist_ok=True)
    os.makedirs(mask_outdir2, exist_ok=True)

    ########## Depth Refine ##########
    depth_refine_model = MDMModel.from_pretrained(PATHS.lingbotdepth_ckpt).to(device)
    depth_refine_model.eval()

    for prefix, frames_dir, depth_in, depth_out, kf_ids in [
        ("v1", v1_frames_dir, depth_frames_dir1, depth_frames_outdir1, v1_keyframe_ids),
        ("v2", v2_frames_dir, depth_frames_dir2, depth_frames_outdir2, v2_keyframe_ids),
    ]:
        if kf_ids is not None and not kf_ids:
            print(f"No keyframes for {prefix}, skip depth refine")
            continue
        if kf_ids is None:
            print(f"Processing {prefix} depth refine (ALL frames)")
        else:
            print(f"Processing {prefix} depth refine ({len(kf_ids)} keyframes)")
        process_depth_batch(
            depth_refine_model, device, frames_dir, depth_in, depth_out,
            max_size=args.depth_refine_max_size,
            chunk_size=args.depth_refine_chunk_size,
            prefix=prefix,
            skip_depth_refine=False,
            frame_ids=kf_ids,
            image_ext=".jpg",
            overwrite=(args.mode == "overwrite"),
        )

    if depth_refine_model is not None:
        del depth_refine_model
        torch.cuda.empty_cache()

    ########## Mask Generation ##########
    if not args.skip_masks:
        # 根据 need_all_depth_mask 决定处理哪些帧
        if args.need_all_depth_mask:
            v1_mask_ids = list(range(num_v1))
            v2_mask_ids = list(range(num_v2))
            print(f"Generating masks for ALL frames: v1={num_v1}, v2={num_v2}")
        else:
            v1_mask_ids = v1_keyframe_ids
            v2_mask_ids = v2_keyframe_ids
            print(f"Generating masks for keyframes only: v1={len(v1_keyframe_ids)}, v2={len(v2_keyframe_ids)}")
        
        generate_masks(
            raw1_image_list, v1_mask_ids, mask_outdir1, "v1",
            lang_sam_chunk_size=args.lang_sam_chunk_size,
            lang_sam_sam_type=args.lang_sam_sam_type,
            lang_sam_sam_ckpt_path=args.lang_sam_sam_ckpt_path,
        )
        generate_masks(
            raw2_image_list, v2_mask_ids, mask_outdir2, "v2",
            lang_sam_chunk_size=args.lang_sam_chunk_size,
            lang_sam_sam_type=args.lang_sam_sam_type,
            lang_sam_sam_ckpt_path=args.lang_sam_sam_ckpt_path,
        )

    ########## Copy keyframes to keyframe folders (if processing all frames) ##########
    if args.need_all_depth_mask:
        import shutil
        print("Copying keyframes to keyframe folders for downstream steps...")
        
        # 需要重新读取关键帧列表
        v1_kf_set = set()
        v2_kf_set = set()
        if args.use_vggt:
            v1_kf_set.update(kf_data.get("vggt", []))
            v2_kf_set.update(kf_data.get("vggt", []))
        if args.use_unproj:
            v1_kf_set.update(kf_data.get("unproj", []))
            v2_kf_set.update(kf_data.get("unproj", []))
        if args.use_p2p:
            v1_kf_set.update(kf_data.get("p2p_v1", []))
            v2_kf_set.update(kf_data.get("p2p_v2", []))
        
        v1_kf_list = sorted(v1_kf_set)
        v2_kf_list = sorted(v2_kf_set)
        
        # 创建关键帧文件夹
        depth_kf_dir1 = f"{v1_folder}/depths_keyframe_refined"
        depth_kf_dir2 = f"{v2_folder}/depths_keyframe_refined"
        mask_kf_dir1 = f"{v1_folder}/masks_keyframe"
        mask_kf_dir2 = f"{v2_folder}/masks_keyframe"
        os.makedirs(depth_kf_dir1, exist_ok=True)
        os.makedirs(depth_kf_dir2, exist_ok=True)
        os.makedirs(mask_kf_dir1, exist_ok=True)
        os.makedirs(mask_kf_dir2, exist_ok=True)
        
        # 复制 v1 关键帧
        for frame_id in v1_kf_list:
            depth_src = os.path.join(depth_frames_outdir1, f"v1_{frame_id:04d}.png")
            depth_dst = os.path.join(depth_kf_dir1, f"v1_{frame_id:04d}.png")
            if os.path.exists(depth_src):
                shutil.copy2(depth_src, depth_dst)
            
            if not args.skip_masks:
                mask_src = os.path.join(mask_outdir1, f"v1_{frame_id:04d}.png")
                mask_dst = os.path.join(mask_kf_dir1, f"v1_{frame_id:04d}.png")
                if os.path.exists(mask_src):
                    shutil.copy2(mask_src, mask_dst)
        
        # 复制 v2 关键帧
        for frame_id in v2_kf_list:
            depth_src = os.path.join(depth_frames_outdir2, f"v2_{frame_id:04d}.png")
            depth_dst = os.path.join(depth_kf_dir2, f"v2_{frame_id:04d}.png")
            if os.path.exists(depth_src):
                shutil.copy2(depth_src, depth_dst)
            
            if not args.skip_masks:
                mask_src = os.path.join(mask_outdir2, f"v2_{frame_id:04d}.png")
                mask_dst = os.path.join(mask_kf_dir2, f"v2_{frame_id:04d}.png")
                if os.path.exists(mask_src):
                    shutil.copy2(mask_src, mask_dst)
        
        print(f"Copied keyframes: v1={len(v1_kf_list)}, v2={len(v2_kf_list)}")

    ########## Filter points2D with masks ##########
    if not args.skip_masks:
        # Filter points2D 总是使用关键帧的 mask
        mask_kf_dir1 = f"{v1_folder}/masks_keyframe"
        mask_kf_dir2 = f"{v2_folder}/masks_keyframe"
        
        for prefix, v_folder, mask_dir in [
            ("v1", v1_folder, mask_kf_dir1),
            ("v2", v2_folder, mask_kf_dir2),
        ]:
            points2d_path = os.path.join(v_folder, "points2D.npz")
            if os.path.exists(points2d_path):
                print(f"Filtering {prefix} points2D with keyframe masks")
                filter_points2d_with_masks(
                    points2d_path, mask_dir, prefix, points2d_path,
                )
            else:
                print(f"Warning: {points2d_path} not found, skip points2D filtering")

    print("process_depth_mask done.")
