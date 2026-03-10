import argparse
import os
import shutil
import cv2
from PIL import Image
import numpy as np

import torch
import math
from yacs.config import CfgNode as CN
from tqdm import tqdm, trange
from embod_mocap.human.detector import DetectionModel
from embod_mocap.config_paths import PATHS
from embod_mocap.human.utils.camera_utils import pred_cam_to_full_cam
from embod_mocap.human.utils.lang_sam_utils import lang_sam_forward
from embod_mocap.human.backbone.vimo import HMR_VIMO
from embod_mocap.human.utils.bbox_utils import kp2d_to_bbox
from embod_mocap.processor.base import lingbotdepth_refine_batch, MDMModel, load_image_rotate


def check_existing_outputs(v_folder, prefix, num_images, keyframe_ids, keyframe_depth, keyframe_mask):
    """
    Check existing outputs and decide which sub-steps can be skipped.
    Args:
        v_folder: v1 or v2 folder path
        prefix: "v1" or "v2"
        num_images: number of images in the images directory
        keyframe_ids: keyframe ID list
        keyframe_depth: whether to process keyframe depth only
        keyframe_mask: whether to process keyframe masks only
    Returns:
        (skip_depth, skip_mask, skip_smpl): three booleans indicating whether to skip depth/mask/SMPL steps
    """
    skip_depth = False
    if keyframe_depth:
        depth_dir = os.path.join(v_folder, "depths_keyframe_refined")
        expected_depth_count = len(keyframe_ids) if keyframe_ids else 0
    else:
        depth_dir = os.path.join(v_folder, "depths")
        expected_depth_count = num_images
    
    if os.path.exists(depth_dir) and expected_depth_count > 0:
        actual_depth_count = len([f for f in os.listdir(depth_dir) if f.startswith(f"{prefix}_") and f.endswith(".png")])
        if actual_depth_count >= expected_depth_count:
            skip_depth = True
            print(f"[{prefix}] Depth maps already done ({actual_depth_count}/{expected_depth_count}), skipping depth processing")
    
    skip_mask = False
    if keyframe_mask:
        mask_dir = os.path.join(v_folder, "masks_keyframe")
        expected_mask_count = len(keyframe_ids) if keyframe_ids else 0
    else:
        mask_dir = os.path.join(v_folder, "masks")
        expected_mask_count = num_images
    
    if os.path.exists(mask_dir) and expected_mask_count > 0:
        actual_mask_count = len([f for f in os.listdir(mask_dir) if f.startswith(f"{prefix}_") and f.endswith(".png")])
        if actual_mask_count >= expected_mask_count:
            skip_mask = True
            print(f"[{prefix}] masks already done ({actual_mask_count}/{expected_mask_count}), skipping mask processing")
    
    skip_smpl = False
    smpl_file = os.path.join(v_folder, "smpl_params.npz")
    if os.path.exists(smpl_file):
        skip_smpl = True
        print(f"[{prefix}] smpl_params.npz already exists, skipping SMPL processing")
    
    return skip_depth, skip_mask, skip_smpl


def get_default_config():
    cfg_file = os.path.join(
        os.path.dirname(__file__),
        'config_vimo.yaml'
        )

    cfg = CN()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_file)
    return cfg

def get_hmr_vimo(checkpoint=None, device='cuda', inference_stride=None):
    cfg = get_default_config()
    cfg.device = device
    
    # Override inference stride if provided
    if inference_stride is not None:
        cfg.INFERENCE_STRIDE = inference_stride
    
    model = HMR_VIMO(cfg)

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location='cpu')
        _ = model.load_state_dict(ckpt['model'], strict=False)

    model = model.to(device)
    _ = model.eval()

    return model

def fill_missing_tracks(tracks_dict, total_frames):
    if 0 not in tracks_dict:
        raise ValueError("Tracking dictionary must contain data for id=0")

    id0_data = tracks_dict[0]
    id0_frames = set(id0_data['frame_id'])

    all_frames = set(range(total_frames))

    missing_frames = sorted(all_frames - id0_frames)

    if not missing_frames:
        return tracks_dict

    for frame in missing_frames:
        filled = False
        for other_id, other_data in tracks_dict.items():
            if other_id == 0:
                continue 
            
            if frame in other_data['frame_id']:
                frame_idx = other_data['frame_id'].index(frame)

                bbox = other_data['bbox'][frame_idx]
                keypoints = other_data['keypoints'][frame_idx]

                id0_data['frame_id'].append(frame)
                id0_data['bbox'].append(bbox)
                id0_data['keypoints'].append(keypoints)
                filled = True
                break 

        if not filled:
            last_idx = len(id0_data['frame_id']) - 1
            bbox = id0_data['bbox'][last_idx]
            keypoints = id0_data['keypoints'][last_idx]

            id0_data['frame_id'].append(frame)
            id0_data['bbox'].append(bbox)
            id0_data['keypoints'].append(keypoints)

    sorted_indices = sorted(range(len(id0_data['frame_id'])), key=lambda i: id0_data['frame_id'][i])
    id0_data['frame_id'] = [id0_data['frame_id'][i] for i in sorted_indices]
    id0_data['bbox'] = [id0_data['bbox'][i] for i in sorted_indices]
    id0_data['keypoints'] = [id0_data['keypoints'][i] for i in sorted_indices]

    return tracks_dict


def inference_human(
    img_paths,
    device='cuda',
    fps=30,
    vimo_stride=None,
    skip_masks=False,
    mask_frame_ids=None,
    lang_sam_chunk_size=5,
    lang_sam_sam_type="sam2.1_hiera_small",
    lang_sam_sam_ckpt_path=None,
):
    img_paths = img_paths
    img_pils = []
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        # img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv2.INTER_CUBIC)
        img_pils.append(Image.fromarray(img))
        imgs.append(img)
    detector = DetectionModel(bbox_model_ckpt=PATHS.bbox_model_ckpt, pose_model_ckpt=PATHS.pose_model_ckpt, vit_cfg=PATHS.vit_cfg, device=device)
    pose_estimator = get_hmr_vimo(checkpoint=PATHS.vimo_ckpt, device=device, inference_stride=vimo_stride)
    with torch.no_grad():
        print(f'>> Inference detection model on {len(imgs)} images')
        for img in tqdm(imgs):
            # if single_person:
            #     detector.detect(img, fps)
            # else:
            detector.track(img, fps)
            torch.cuda.empty_cache()
    tracking_results = detector.process(fps)
    del detector
    torch.cuda.empty_cache()

    # vis kps
    # im_kp2d = draw_kps(imgs[i], tracking_results[0]['keypoints'][i], get_coco_joint_names(), get_coco_skeleton())
    h, w = imgs[0].shape[:2]
    img_focal = math.sqrt(h**2 + w**2)
    img_center = np.array([w/2., h/2.])
    print(f'>> Inference pose estimator on {len(imgs)} images')
    if not len(tracking_results[0]['keypoints']) == len(tracking_results[0]['frame_id']) == len(tracking_results[0]['bbox']) == len(imgs):
        tracking_results = merge_tracks_by_bbox(tracking_results, len(imgs))
    tracking_results = {0: tracking_results[0]}

    assert len(tracking_results[0]['keypoints']) == len(tracking_results[0]['frame_id']) == len(imgs) == len(tracking_results[0]['bbox']), "length mismatch"

    print("Extracting human pose from images")
    for _id, val in tracking_results.items():
        frame = val['frame_id']
        valid = np.ones(len(frame), dtype=bool)
        c = val['bbox'][:, :2]
        s = val['bbox'][:, 2:3] * 200
        # boxes: nx5 = x1, y1, x2, y2, score
        boxes_infer = np.concatenate([c - s / 2, c + s / 2], axis=1) 
        boxes_infer = np.clip(boxes_infer, 0, None)
        boxes_infer = np.concatenate([boxes_infer, np.ones((len(boxes_infer), 1))], axis=1)
        with torch.no_grad():
            results = pose_estimator.inference_perfect_det(imgs, boxes_infer, valid=valid, frame=frame,
                                    img_focal=img_focal, img_center=img_center)

        tracking_results[_id]['global_orient'] = results['pred_rotmat'][:, :1]
        tracking_results[_id]['body_pose'] = results['pred_rotmat'][:, 1:]
        tracking_results[_id]['betas'] = results['pred_shape']
        tracking_results[_id]['pred_cam'] = results['pred_cam']

    del pose_estimator
    torch.cuda.empty_cache()
    
    for _id in tracking_results:
        pred_cam = tracking_results[_id]['pred_cam']
        smpl_frame = results['frame'].tolist()
        track_frame = tracking_results[_id]['frame_id']
        mapping = [track_frame.tolist().index(i) for i in smpl_frame]
        bbox = tracking_results[_id]['bbox'][mapping]
        c = torch.from_numpy(bbox[:, :2]).float()
        s = torch.from_numpy(bbox[:, 2:3]).float() * 200
        full_cam = pred_cam_to_full_cam(pred_cam.cpu(), c, s, torch.Tensor([h, w])[None]).to(device)
        tracking_results[_id]['pred_cam'] = full_cam
        tracking_results[_id]['betas'] = tracking_results[_id]['betas'].mean(0)[None]
        # smpl_rotmats = torch.cat([tracking_results[_id]['global_orient'], tracking_results[_id]['body_pose']], 1)
        # smpl_rotmats, full_cam = interpolate_smpl_rotmat_camera(smpl_rotmats, full_cam, smpl_frame, list(range(len(imgs))))
        # tracking_results[_id]['global_orient'] = smpl_rotmats[:, :1]
        # tracking_results[_id]['body_pose'] = smpl_rotmats[:, 1:]
        # tracking_results[_id]['pred_cam'] = full_cam
        assert len(tracking_results[_id]['global_orient']) == len(imgs)
        assert len(tracking_results[_id]['body_pose']) == len(imgs)
        assert len(tracking_results[_id]['pred_cam']) == len(imgs)

    c = tracking_results[0]['bbox'][:, :2]
    s = tracking_results[0]['bbox'][:, 2:3] * 200
    bbox_xyxy_ = np.concatenate([c - s / 2, c + s / 2], axis=1)
    bbox_xyxy = np.zeros((len(imgs), 4))
    bbox_xyxy[tracking_results[0]['frame_id']] = bbox_xyxy_
    
    # Generate masks only if not skipping
    masks = [None] * len(imgs)
    if not skip_masks:
        text_prompt = "person."
        if mask_frame_ids is None:
            mask_frame_ids = list(range(len(imgs)))
        else:
            mask_frame_ids = sorted({i for i in mask_frame_ids if 0 <= i < len(imgs)})
        if mask_frame_ids:
            selected_imgs = [img_pils[i] for i in mask_frame_ids]
            results = lang_sam_forward(
                selected_imgs,
                text_prompt,
                chunk_size=lang_sam_chunk_size,
                sam_type=lang_sam_sam_type,
                sam_ckpt_path=lang_sam_sam_ckpt_path
            )
            for idx, res in zip(mask_frame_ids, results):
                masks[idx] = res['masks']

    tracking_results = tracking_results[0]
    tracking_results['global_orient'] = tracking_results['global_orient'].view(-1, 1, 3, 3)
    tracking_results['body_pose'] = tracking_results['body_pose'].view(-1, 23, 3, 3)
    tracking_results['betas'] = tracking_results['betas'].view(-1, 10)


    tracking_results['bbox'] = tracking_results['bbox']
    # tracking_results['keypoints'][:, :, :2]
    bbox_xyxy = kp2d_to_bbox(tracking_results['keypoints'][..., :2], 1.2)
    tracking_results['bbox_xyxy'] = bbox_xyxy

    for k in tracking_results:
        if isinstance(tracking_results[k], torch.Tensor):
            tracking_results[k] = tracking_results[k].cpu().numpy()
    torch.cuda.empty_cache()
    return tracking_results, masks


def process_depth_batch(depth_refine_model, device, frames_dir, depth_frames_dir, depth_frames_outdir, max_size=1008, chunk_size=4, prefix="v1", skip_depth_refine=False, frame_ids=None, image_ext=".jpg", overwrite=False):
    """Optimized batch processing for depth refinement"""
    if frame_ids is None:
        image_names = [f for f in os.listdir(frames_dir) if f.startswith(f"{prefix}_") and f.endswith(image_ext)]
        num_frames = len(image_names)
        frame_ids = list(range(num_frames))
    # Keep output size aligned with the original depth resolution (per-frame).
    
    # Pre-compute all paths
    frame_data = []
    for i in frame_ids:
        im_path = os.path.join(frames_dir, f"{prefix}_{i:04d}{image_ext}")
        src_depth = os.path.join(depth_frames_dir, f"{prefix}_{i:04d}.png")
        dst_depth = os.path.join(depth_frames_outdir, f"{prefix}_{i:04d}.png")

        if not os.path.exists(im_path) or not os.path.exists(src_depth):
            continue
        
        # Skip if already processed
        if os.path.exists(dst_depth) and not overwrite:
            continue
            
        frame_data.append((im_path, src_depth, dst_depth))
    
    if not frame_data:
        print(f"All {prefix} depths already processed")
        return
    
    # Fast mode: save depths at original size without depth refinement
    # Note: These depths won't be used in downstream steps:
    #   - Camera alignment uses depths_keyframe_refined
    #   - Unprojection (step 7) is skipped when skip_depth_refine=True
    # Original size is kept to preserve depth accuracy for potential future use
    if skip_depth_refine:
        for im_path, src_depth, dst_path in tqdm(frame_data, desc=f"Saving {prefix} depths resized to RGB (no depth refine)"):
            rgb = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(src_depth, cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None:
                continue
            rgb_h, rgb_w = rgb.shape[:2]
            if depth.shape[:2] != (rgb_h, rgb_w):
                depth = cv2.resize(depth, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            cv2.imwrite(dst_path, depth, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        return
    
    if depth_refine_model is None:
        raise ValueError("Depth refine model is None but skip_depth_refine=False. Check model loading.")

    # Process in optimized chunks with depth refinement
    for i in tqdm(range(0, len(frame_data), chunk_size), desc=f"Processing {prefix} depths with depth refine"):
        batch_data = frame_data[i:i+chunk_size]
        
        # Pre-load batch data
        images = []
        prompt_depths = []
        target_sizes = []
        dst_paths = []
        
        for im_path, src_depth, dst_path in batch_data:
            # Load and preprocess image

            rgb_raw = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
            if rgb_raw is None:
                continue
            target_h, target_w = rgb_raw.shape[:2]
            image = load_image_rotate(im_path, max_size=max_size)
            images.append(image)
            # Load and preprocess depth
            prompt_depth = cv2.imread(src_depth, cv2.IMREAD_UNCHANGED) / 1000.0
            if prompt_depth is None:
                continue
            prompt_depth = torch.tensor(prompt_depth, dtype=torch.float32)[None, None]
            prompt_depths.append(prompt_depth)
            target_sizes.append((target_w, target_h))
            
            dst_paths.append(dst_path)
        
        if not images:
            continue
            
        # Batch inference
        with torch.cuda.amp.autocast():
            images_batch = torch.cat(images, dim=0)
            prompt_depths_batch = torch.cat(prompt_depths, dim=0)
            refined_depths = lingbotdepth_refine_batch(
                model=depth_refine_model, device=device,
                images=images_batch, prompt_depths=prompt_depths_batch
            )
        
        # Post-process and save
        for j, (refined_depth, dst_path) in enumerate(zip(refined_depths, dst_paths)):
            target_w, target_h = target_sizes[j]
            depth = cv2.resize(refined_depth, (target_w, target_h), interpolation=cv2.INTER_CUBIC) * 1000
            depth = np.clip(depth, 0, 65535).astype(np.uint16)
            cv2.imwrite(dst_path, depth, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
        torch.cuda.empty_cache()

def merge_tracks_by_bbox(tracks_dict, total_frames, iou_threshold=0.5):
    """
    Merge tracking results: prefer id=0, otherwise use the most similar bbox to keep id=0 track complete.
    Args:
        tracks_dict: {id: {'frame_id': [...], 'bbox': [...], 'keypoints': [...]}}
        total_frames: total frame count
        iou_threshold: bbox similarity threshold (tunable)
    Returns:
        new_tracks_dict: complete track containing only id=0
    """
    def bbox_iou(box1, box2):
        # box: [x, y, w, h] or [x1, y1, w, h]
        if len(box1) == 3:
            c = box1[:2]
            s = box1[2] * 200
            box1 = np.concatenate([c - s / 2, c + s / 2], axis=0)
        if len(box2) == 3:
            c = box2[:2]
            s = box2[2] * 200
            box2 = np.concatenate([c - s / 2, c + s / 2], axis=0)
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    frame_to_candidates = {i: [] for i in range(total_frames)}
    for tid, data in tracks_dict.items():
        for idx, f in enumerate(data['frame_id']):
            frame_to_candidates[f].append((tid, idx))

    new_track = {'frame_id': [], 'bbox': [], 'keypoints': []}
    for f in range(total_frames):
        if 0 in tracks_dict and f in tracks_dict[0]['frame_id']:
            idx = tracks_dict[0]['frame_id'].tolist().index(f)
            new_track['frame_id'].append(f)
            new_track['bbox'].append(tracks_dict[0]['bbox'][idx])
            new_track['keypoints'].append(tracks_dict[0]['keypoints'][idx])
        else:
            best_tid, best_idx, best_score = None, None, -1
            ref_bbox = new_track['bbox'][-1] if new_track['bbox'] else None
            for tid, idx in frame_to_candidates[f]:
                bbox = tracks_dict[tid]['bbox'][idx]
                if ref_bbox is not None:
                    score = bbox_iou(ref_bbox, bbox)
                else:
                    score = 0
                if score > best_score:
                    best_tid, best_idx, best_score = tid, idx, score
            if best_tid is not None:
                new_track['frame_id'].append(f)
                new_track['bbox'].append(tracks_dict[best_tid]['bbox'][best_idx])
                new_track['keypoints'].append(tracks_dict[best_tid]['keypoints'][best_idx])
            else:
                if new_track['bbox']:
                    new_track['frame_id'].append(f)
                    new_track['bbox'].append(new_track['bbox'][-1])
                    new_track['keypoints'].append(new_track['keypoints'][-1])
                    new_track['keypoints'][-1][:, 2] = 0
                else:
                    raise ValueError(f"Frame {f} has no track and no previous bbox to fill.")

    new_track['bbox'] = np.array(new_track['bbox'])
    new_track['keypoints'] = np.array(new_track['keypoints'])
    new_track['frame_id'] = np.array(new_track['frame_id'])
    return {0: new_track}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get frames from a video file and save them as images."
    )
    parser.add_argument(
        "folder",
        type=str,
        default="",
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--v1_fps",
        default=30,
        type=int,
        help="FPS for the first video.",
    )
    parser.add_argument(
        "--v2_fps",
        default=30,
        type=int,
        help="FPS for the second video.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--depth_refine_chunk_size",
        type=int,
        default=1,
        help="Chunk size for depth refinement",
    )
    parser.add_argument(
        "--depth_refine_max_size",
        type=int,
        default=540,
        help="Max size for depth refinement",
    )
    parser.add_argument(
        "--lang_sam_chunk_size",
        type=int,
        default=5,
        help="Chunk size for LangSAM mask generation",
    )
    parser.add_argument(
        "--lang_sam_sam_type",
        type=str,
        default="sam2.1_hiera_small",
        help="LangSAM SAM model type",
    )
    parser.add_argument(
        "--lang_sam_sam_ckpt_path",
        type=str,
        default=None,
        help="Path to SAM checkpoint for LangSAM",
    )
    parser.add_argument(
        "--skip_depth_refine",
        action="store_true",
        help="Skip depth refinement and just resize depths (fast mode)",
    )
    parser.add_argument(
        "--skip_masks",
        action="store_true",
        help="Skip LangSAM mask generation (fast mode)",
    )
    parser.add_argument(
        "--keyframe_depth",
        action="store_true",
        help="Only process keyframe depths (use v1/v2/depths_keyframe -> outputs depths_keyframe_refined).",
    )
    parser.add_argument(
        "--keyframe_mask",
        action="store_true",
        help="Only generate keyframe masks (use v1/v2/masks_keyframe).",
    )
    parser.add_argument(
        "--vimo_stride",
        type=int,
        default=None,
        help="VIMO inference stride (default: from config or 16)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="overwrite",
        choices=["overwrite", "skip"],
        help="overwrite: force reprocessing; skip: detect existing outputs and skip completed sub-steps",
    )
    args = parser.parse_args()
    v1_folder = f"{args.folder}/v1"
    v2_folder = f"{args.folder}/v2"
    v1_frames_dir = f"{v1_folder}/images"
    v2_frames_dir = f"{v2_folder}/images"

    keyframe_ids = []
    keyframe_set = set()
    skip_depth_processing = False
    if args.keyframe_depth or args.keyframe_mask:
        keyframe_file = os.path.join(args.folder, "keyframes.txt")
        if os.path.exists(keyframe_file):
            with open(keyframe_file, "r", encoding="utf-8") as f:
                keyframe_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
            keyframe_set = set(keyframe_ids)
        else:
            print(f"Warning: keyframes.txt not found in {args.folder}")
        if args.keyframe_depth and not keyframe_ids:
            print("Warning: keyframe_depth enabled but no keyframes found; skip depth processing.")
            skip_depth_processing = True
    effective_skip_depth_refine = args.skip_depth_refine

    if args.keyframe_depth:
        depth_frames_dir1 = f"{v1_folder}/depths_keyframe"
        depth_frames_outdir1 = f"{v1_folder}/depths_keyframe_refined"
        depth_frames_dir2 = f"{v2_folder}/depths_keyframe"
        depth_frames_outdir2 = f"{v2_folder}/depths_keyframe_refined"
    else:
        depth_frames_dir1 = f"{v1_folder}/depths"
        depth_frames_outdir1 = depth_frames_dir1
        depth_frames_dir2 = f"{v2_folder}/depths"
        depth_frames_outdir2 = depth_frames_dir2
    os.makedirs(depth_frames_outdir1, exist_ok=True)
    os.makedirs(depth_frames_outdir2, exist_ok=True)

    if args.keyframe_mask:
        mask_frames_outdir1 = f"{v1_folder}/masks_keyframe"
        mask_frames_outdir2 = f"{v2_folder}/masks_keyframe"
    else:
        mask_frames_outdir1 = f"{v1_folder}/masks"
        mask_frames_outdir2 = f"{v2_folder}/masks"
    os.makedirs(mask_frames_outdir1, exist_ok=True)
    os.makedirs(mask_frames_outdir2, exist_ok=True)
    device = torch.device(args.device)
    depth_refine_model = None

    ########## v1
    print("Processing v1 frames with pretrained models")
    
    raw1_image_list = []
    image_names = [f for f in os.listdir(v1_frames_dir) if f.startswith("v1_") and f.endswith(".jpg")]
    image_names.sort()
    for name in image_names:
        raw1_image_list.append(os.path.join(v1_frames_dir, name))
    num_v1_images = len(raw1_image_list)
    
    if args.mode == "skip":
        v1_skip_depth, v1_skip_mask, v1_skip_smpl = check_existing_outputs(
            v1_folder, "v1", num_v1_images, keyframe_ids, args.keyframe_depth, args.keyframe_mask
        )
    else:
        v1_skip_depth, v1_skip_mask, v1_skip_smpl = False, False, False
    
    if not v1_skip_depth:
        if not effective_skip_depth_refine:
            depth_refine_model = MDMModel.from_pretrained(PATHS.lingbotdepth_ckpt).to(device)
            depth_refine_model.eval()
        else:
            print("Skipping depth refine for v1 (fast mode)")
        # Use optimized batch processing
        if not skip_depth_processing and ((not args.keyframe_depth) or keyframe_ids):
            process_depth_batch(
                depth_refine_model, device, v1_frames_dir, depth_frames_dir1,
                depth_frames_outdir1,
                max_size=args.depth_refine_max_size,
                chunk_size=args.depth_refine_chunk_size, prefix="v1",
                skip_depth_refine=effective_skip_depth_refine,
                frame_ids=keyframe_ids if args.keyframe_depth else None,
                image_ext=".jpg",
                overwrite=True,
            )

    if not v1_skip_smpl or not v1_skip_mask:
        with torch.autocast('cuda', dtype=torch.float32):
            smpl_params, raw1_human_masks = inference_human(
                raw1_image_list,
                device=args.device,
                vimo_stride=args.vimo_stride,
                skip_masks=args.skip_masks or v1_skip_mask,
                mask_frame_ids=keyframe_ids if args.keyframe_mask else None,
                lang_sam_chunk_size=args.lang_sam_chunk_size,
                lang_sam_sam_type=args.lang_sam_sam_type,
                lang_sam_sam_ckpt_path=args.lang_sam_sam_ckpt_path)
        
        if not v1_skip_smpl:
            np.savez(os.path.join(v1_folder, "smpl_params.npz"), **smpl_params)

        if not v1_skip_mask and not args.skip_masks:
            for i in range(len(raw1_human_masks)):
                if args.keyframe_mask and i not in keyframe_set:
                    continue
                if raw1_human_masks[i] is None:
                    continue
                mask = raw1_human_masks[i][0] * 255
                mask = cv2.resize(mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(mask_frames_outdir1, f"v1_{i:04d}.png"), mask)
    else:
        print("v1 SMPL and masks are already complete, skipping inference_human")
    
    print("processing v1 done.")

    ########## v2
    print("Processing v2 frames with pretrained models")
    
    raw2_image_list = []
    image_names = [f for f in os.listdir(v2_frames_dir) if f.startswith("v2_") and f.endswith(".jpg")]
    image_names.sort()
    for name in image_names:
        raw2_image_list.append(os.path.join(v2_frames_dir, name))
    num_v2_images = len(raw2_image_list)
    
    if args.mode == "skip":
        v2_skip_depth, v2_skip_mask, v2_skip_smpl = check_existing_outputs(
            v2_folder, "v2", num_v2_images, keyframe_ids, args.keyframe_depth, args.keyframe_mask
        )
    else:
        v2_skip_depth, v2_skip_mask, v2_skip_smpl = False, False, False
    
    if not v2_skip_depth:
        if not effective_skip_depth_refine and depth_refine_model is None:
            depth_refine_model = MDMModel.from_pretrained(PATHS.lingbotdepth_ckpt).to(device)
            depth_refine_model.eval()
        elif effective_skip_depth_refine:
            print("Skipping depth refine for v2 (fast mode)")

        # Use optimized batch processing
        if not skip_depth_processing and ((not args.keyframe_depth) or keyframe_ids):
            process_depth_batch(
                depth_refine_model, device, v2_frames_dir, depth_frames_dir2,
                depth_frames_outdir2,
                max_size=args.depth_refine_max_size,
                chunk_size=args.depth_refine_chunk_size, prefix="v2",
                skip_depth_refine=effective_skip_depth_refine,
                frame_ids=keyframe_ids if args.keyframe_depth else None,
                image_ext=".jpg",
                overwrite=True,
            )

    if not v2_skip_smpl or not v2_skip_mask:
        with torch.autocast('cuda', dtype=torch.float32):
            smpl_params, raw2_human_masks = inference_human(
                raw2_image_list,
                device=args.device,
                vimo_stride=args.vimo_stride,
                skip_masks=args.skip_masks or v2_skip_mask,
                mask_frame_ids=keyframe_ids if args.keyframe_mask else None,
                lang_sam_chunk_size=args.lang_sam_chunk_size,
                lang_sam_sam_type=args.lang_sam_sam_type,
                lang_sam_sam_ckpt_path=args.lang_sam_sam_ckpt_path,
            )
        
        if not v2_skip_smpl:
            np.savez(os.path.join(v2_folder, "smpl_params.npz"), **smpl_params)

        if not v2_skip_mask and not args.skip_masks:
            for i in range(len(raw2_human_masks)):
                if args.keyframe_mask and i not in keyframe_set:
                    continue
                if raw2_human_masks[i] is None:
                    continue
                mask = raw2_human_masks[i][0] * 255
                mask = cv2.resize(mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(mask_frames_outdir2, f"v2_{i:04d}.png"), mask)
    else:
        print("v2 SMPL and masks are already complete, skipping inference_human")
    
    print("processing v2 done.")
