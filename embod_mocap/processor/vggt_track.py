import torch
import numpy as np
import time
import cv2
import os
import argparse
import trimesh
from tqdm import tqdm
from embod_mocap.vggt.vggt.models.vggt import VGGT
from embod_mocap.vggt.vggt.utils.load_fn import load_and_preprocess_images
from embod_mocap.processor.base import write_warning_to_log, run_cmd
from embod_mocap.config_paths import PATHS
from embod_mocap.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from embod_mocap.vggt.vggt.utils.geometry import unproject_depth_map_to_point_map
from embod_mocap.vggt.vggt.utils.visual_track import visualize_tracks_on_images
import matplotlib.cm as cm


def preprocess_depth_like_vggt(depth_path, target_width=518):
    """Apply the same preprocessing to depth as vggt does to images"""
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        return None

    # Convert to meters first
    depth_img = depth_img.astype(np.float32) / 1000.0

    height, width = depth_img.shape
    new_width = target_width

    # Calculate height maintaining aspect ratio, divisible by 14
    new_height = round(height * (new_width / width) / 14) * 14

    # Resize with new dimensions
    depth_resized = cv2.resize(depth_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Center crop height if it's larger than 518
    if new_height > 518:
        start_y = (new_height - 518) // 2
        depth_resized = depth_resized[start_y:start_y + 518, :]

    return depth_resized


def transform_back_to_original_coords(coords, original_width, original_height, new_width=518):
    aspect_ratio_height = original_height * (new_width / original_width)
    new_height = round(aspect_ratio_height / 14) * 14

    if new_height > 518:
        start_y = (new_height - 518) // 2
    else:
        start_y = 0

    x = coords[:, 0]
    y = coords[:, 1]

    y_resized = y + start_y
    x_resized = x

    x_original = x_resized * (original_width / new_width)
    y_original = y_resized * (original_height / new_height)

    original_coords = torch.stack([x_original, y_original], dim=1)

    return original_coords

def vggt_track_pair(image_pair_names, mask_pair_names, depth_pair_names=None, num_sample=100, return_pointcloud=False, expand_mask=False):
    images = load_and_preprocess_images(image_pair_names).to(device)
    masks = load_and_preprocess_images(mask_pair_names).to(device)
    masks = masks[:, 0]
    if expand_mask:
        # Use depth-based mask expansion
        for i in range(masks.shape[0]):
            # Preprocess depth using the same logic as vggt images
            depth_processed = preprocess_depth_like_vggt(depth_pair_names[i])
            if depth_processed is not None:
                # Calculate depth threshold from human mask region
                human_mask = masks[i].cpu().numpy() > 0.5

                # Pad depth to match processed mask size if needed
                h, w = masks[i].shape
                dh, dw = depth_processed.shape
                if dh != h or dw != w:
                    # Apply same padding logic as vggt to depth
                    h_padding = h - dh
                    w_padding = w - dw
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left
                    depth_processed = np.pad(depth_processed, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

                # Extract depth values from human mask region
                human_depths = depth_processed[human_mask]
                human_depths = human_depths[human_depths > 0]  # Remove invalid depth values

                if len(human_depths) > 0:
                    # Use average depth of human as threshold
                    depth_threshold = human_depths.min()
                    # Create mask for pixels with depth < threshold
                    depth_mask = depth_processed > 3.0
                    # depth_mask = depth_mask | human_mask
                    masks[i] = torch.from_numpy(depth_mask.astype(np.float32)).to(device)
                else:
                    # Fallback to original mask if no valid depth found
                    pass  # Keep original mask

    masks = masks > 0
    images_raw = images.clone()
    with torch.no_grad():
        
        with torch.amp.autocast('cuda', dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        if return_pointcloud:                        
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

            # point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
                
            point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                        extrinsic.squeeze(0), 
                                                                        intrinsic.squeeze(0))
        indices = torch.nonzero(masks[0], as_tuple=False)
        query_points = indices[:, [1, 0]] 
        indices = torch.randperm(len(query_points))[:num_sample*10]
        query_points = query_points[indices]
        track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])

    track = track_list[-1]
    # visualize_tracks_on_images(images, track, (conf_score>0.2) & (vis_score>0.2), out_dir="track_visuals")

    new_w = images.shape[3]
    orig_h, orig_w = cv2.imread(image_names1[0]).shape[:2]
    threshold = 0.2
    mask1 = (conf_score[0, 0] > threshold) * (vis_score[0, 0] > threshold)
    mask2 = (conf_score[0, 1] > threshold) * (vis_score[0, 1] > threshold)
    valid_mask = (mask1 & mask2)

    indices = torch.where(valid_mask)[0]
    
    # Get ALL candidate tracks first (before sampling)
    track_v1_all = track[0, 0, indices]
    track_v2_all = track[0, 1, indices]

    # Transform to original coordinates
    track_v1_all = transform_back_to_original_coords(track_v1_all, orig_w, orig_h, new_width=new_w)
    track_v2_all = transform_back_to_original_coords(track_v2_all, orig_w, orig_h, new_width=new_w)

    # Filter 1: Check coordinate bounds
    coord_mask_v1 = (track_v1_all[:, 0] >= 0) & (track_v1_all[:, 0] < orig_w) & (track_v1_all[:, 1] >= 0) & (track_v1_all[:, 1] < orig_h)
    coord_mask_v2 = (track_v2_all[:, 0] >= 0) & (track_v2_all[:, 0] < orig_w) & (track_v2_all[:, 1] >= 0) & (track_v2_all[:, 1] < orig_h)
    coord_valid_mask = coord_mask_v1 & coord_mask_v2

    # Filter 2: Check if v2 points fall within the human mask
    mask2_original = cv2.imread(mask_pair_names[1], cv2.IMREAD_GRAYSCALE)
    if mask2_original is not None:
        mask_valid_v2 = []
        for i in range(len(track_v2_all)):
            x, y = int(track_v2_all[i, 0].item()), int(track_v2_all[i, 1].item())
            if 0 <= x < orig_w and 0 <= y < orig_h:
                mask_valid_v2.append(mask2_original[y, x] > 127)
            else:
                mask_valid_v2.append(False)
        
        mask_valid_v2 = torch.tensor(mask_valid_v2, dtype=torch.bool, device=track_v2_all.device)
        # Combine coordinate validity and mask validity
        final_valid_mask = coord_valid_mask & mask_valid_v2
    else:
        final_valid_mask = coord_valid_mask
    
    # Apply filtering to get valid tracks
    track_v1_valid = track_v1_all[final_valid_mask]
    track_v2_valid = track_v2_all[final_valid_mask]
    
    # Now sample num_sample points from valid tracks
    num_valid = len(track_v1_valid)
    if num_valid > num_sample:
        # Randomly sample num_sample from valid tracks
        sample_indices = torch.randperm(num_valid)[:num_sample]
        track_v1 = track_v1_valid[sample_indices]
        track_v2 = track_v2_valid[sample_indices]
    else:
        # If we have fewer valid tracks than requested, use all valid tracks
        track_v1 = track_v1_valid
        track_v2 = track_v2_valid

    if return_pointcloud:
        images = images_raw.squeeze().cpu().numpy().transpose(0, 2, 3, 1) 
        point_maps = point_map_by_unprojection.reshape(-1, 3)
        colors = images# now (S, H, W, 3)
        colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
        num_exports = len(point_maps) // 5
        random_idx = np.random.choice(len(point_maps), num_exports, replace=False)
        pointcloud = trimesh.PointCloud(point_maps[random_idx], colors=colors_flat[random_idx])
    else:
        pointcloud = None
    return track_v1, track_v2, pointcloud


def visualize_vggt_tracks(seq_path, output_folder="vggt_vis", num_tracks=20):
    """
    Visualize VGGT tracking results by drawing correspondence lines between v1 and v2 views
    
    Args:
        seq_path: Path to sequence folder
        output_folder: Output folder for visualization images
        num_tracks: Number of tracking correspondences to visualize per frame (default: 20)
    """
    print(f"Visualizing VGGT tracking for sequence: {seq_path}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Read VGGT tracking data
    vggt_tracks_path = os.path.join(seq_path, "vggt_tracks.npz")
    
    if not os.path.exists(vggt_tracks_path):
        print(f"Error: VGGT tracks not found: {vggt_tracks_path}")
        return
    
    # Check if file is empty (created by touch command when no tracks found)
    if os.path.getsize(vggt_tracks_path) == 0:
        print(f"Error: VGGT tracks file is empty: {vggt_tracks_path}")
        print("This usually means no valid tracking data was generated.")
        return
    
    try:
        vggt_data = np.load(vggt_tracks_path)
    except Exception as e:
        print(f"Error loading VGGT tracks: {e}")
        return
    
    # Check VGGT data format
    if 'frame_ids' not in vggt_data or len(vggt_data['frame_ids']) == 0:
        print(f"Error: No valid VGGT tracking data found in {vggt_tracks_path}")
        return
    
    if 'track_v1' not in vggt_data or 'track_v2' not in vggt_data:
        print(f"Error: Invalid VGGT data format. Expected keys: track_v1, track_v2, frame_ids")
        print(f"Available keys: {list(vggt_data.keys())}")
        return
    
    # Get available frame IDs
    frame_ids = vggt_data['frame_ids']
    track_v1_all = vggt_data['track_v1']  # Shape: (N_frames, N_samples, 2)
    track_v2_all = vggt_data['track_v2']  # Shape: (N_frames, N_samples, 2)
    
    print(f"Found VGGT tracking data for {len(frame_ids)} frames")
    print(f"Track v1 shape: {track_v1_all.shape}")
    print(f"Track v2 shape: {track_v2_all.shape}")
    print(f"Frame IDs: {frame_ids}")
    
    # Use rainbow colormap for correspondence lines
    rainbow_cmap = cm.rainbow
    
    # Process each frame
    for frame_idx_in_array, frame_id in enumerate(tqdm(frame_ids, desc="Visualizing tracks")):
        # Read v1 and v2 images
        v1_img_path = os.path.join(seq_path, "v1", "images", f"v1_{frame_id:04d}.jpg")
        v2_img_path = os.path.join(seq_path, "v2", "images", f"v2_{frame_id:04d}.jpg")
        
        if not os.path.exists(v1_img_path) or not os.path.exists(v2_img_path):
            print(f"Warning: Images not found for frame {frame_id}")
            continue
        
        v1_img = cv2.imread(v1_img_path)
        v2_img = cv2.imread(v2_img_path)
        
        if v1_img is None or v2_img is None:
            print(f"Warning: Failed to load images for frame {frame_id}")
            continue
        
        # Get image dimensions
        h, w = v1_img.shape[:2]
        v1_width = w
        
        # Concatenate v1 and v2 horizontally
        combined_img = np.hstack([v1_img, v2_img])
        
        # Get tracking data for this frame
        track_v1 = track_v1_all[frame_idx_in_array]  # (N_samples, 2)
        track_v2 = track_v2_all[frame_idx_in_array]  # (N_samples, 2)
        
        # Select a subset of tracks to visualize
        total_tracks = len(track_v1)
        num_tracks_to_show = min(num_tracks, total_tracks)
        
        if total_tracks > num_tracks_to_show:
            # Randomly select tracks to visualize
            indices = np.random.RandomState(seed=frame_id).choice(
                total_tracks, num_tracks_to_show, replace=False
            )
            track_v1_selected = track_v1[indices]
            track_v2_selected = track_v2[indices]
        else:
            track_v1_selected = track_v1
            track_v2_selected = track_v2
        
        # Draw tracking correspondences
        for i in range(len(track_v1_selected)):
            # v1 pixel coordinates (x, y)
            pt1 = (int(track_v1_selected[i][0]), int(track_v1_selected[i][1]))
            # v2 pixel coordinates need to be offset by v1_width (x + v1_width, y)
            pt2 = (int(track_v2_selected[i][0] + v1_width), int(track_v2_selected[i][1]))
            
            # Choose color based on pixel height (y coordinate) for better visualization
            y_normalized = pt1[1] / h  # Normalize to 0-1
            color_rgba = rainbow_cmap(y_normalized)  # Get RGBA color
            color_bgr = (
                int(color_rgba[2] * 255), 
                int(color_rgba[1] * 255), 
                int(color_rgba[0] * 255)
            )  # Convert to BGR format for OpenCV
            
            # Draw correspondence points and connection line
            cv2.circle(combined_img, pt1, 5, color_bgr, -1)  # v1 point
            cv2.circle(combined_img, pt2, 5, color_bgr, -1)  # v2 point
            cv2.line(combined_img, pt1, pt2, color_bgr, 2)   # Connection line
        
        # Add frame info text
        text = f"Frame {frame_id} | Showing {len(track_v1_selected)}/{total_tracks} tracks"
        cv2.putText(
            combined_img, text, (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        # Save visualization
        output_path = os.path.join(output_folder, f"vggt_tracks_frame_{frame_id:04d}.jpg")
        cv2.imwrite(output_path, combined_img)
    
    print(f"\nVisualization complete! Saved {len(frame_ids)} images to {output_folder}")
    print(f"Showing {num_tracks_to_show} tracks per frame (out of {total_tracks} total)")


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
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to the log file.",
    )
    parser.add_argument(
        "--vggt_track_samples",
        type=int,
        default=200,
        help="Number of sample points for VGGT tracking.",
    )
    parser.add_argument(
        "--expand_mask",
        action="store_true",
        help="Expand mask using depth threshold.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize VGGT tracking results after generation.",
    )
    parser.add_argument(
        "--vis_output",
        type=str,
        default="vggt_vis",
        help="Output folder for visualization images (default: vggt_vis in seq folder).",
    )
    parser.add_argument(
        "--vis_num_tracks",
        type=int,
        default=20,
        help="Number of tracks to visualize per frame (default: 20).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=100,
        help="Stride for keyframe sampling (default: 100).",
    )
    parser.add_argument(
        "--use_keyframe_masks",
        action="store_true",
        help="Use keyframe masks from masks_keyframe/ instead of regular masks/.",
    )
    parser.add_argument(
        "--use_keyframe_depths",
        action="store_true",
        help="Use keyframe depths from depths_keyframe_refined/ instead of regular depths/.",
    )
    args = parser.parse_args()
    image_root1 = f"{args.input_folder}/v1/images"
    image_root2 = f"{args.input_folder}/v2/images"
    
    # Keyframe directories (for fast mode)
    mask_keyframe_root1 = f"{args.input_folder}/v1/masks_keyframe"
    mask_keyframe_root2 = f"{args.input_folder}/v2/masks_keyframe"
    depth_keyframe_root1 = f"{args.input_folder}/v1/depths_keyframe_refined"
    depth_keyframe_root2 = f"{args.input_folder}/v2/depths_keyframe_refined"
    
    # Regular directories (fallback)
    mask_root1 = f"{args.input_folder}/v1/masks"
    mask_root2 = f"{args.input_folder}/v2/masks"
    depth_root1 = f"{args.input_folder}/v1/depths"
    depth_root2 = f"{args.input_folder}/v2/depths"
    
    # Use command line arguments directly (controlled by yaml config)
    # When skip_masks=True in yaml: use_keyframe_masks=True (only keyframe masks generated)
    # When skip_masks=False in yaml: use_keyframe_masks=False (full masks generated)
    use_keyframe_masks = args.use_keyframe_masks
    use_keyframe_depths = args.use_keyframe_depths
    
    print(f"Data sources:")
    print(f"  Masks:  {'keyframe (masks_keyframe/)' if use_keyframe_masks else 'regular (masks/)'}")
    if use_keyframe_depths:
        print("  Depths: keyframe (depths_keyframe_refined/)")
    else:
        print("  Depths: regular (depths/)")
    
    num_frames = len([name for name in os.listdir(image_root1) if name.endswith(".jpg")])

    device = args.device
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT()
    model.load_state_dict(torch.load(PATHS.vggt_ckpt))
    model = model.to(device)
    
    # Load and preprocess example images
    image_names1 = [f"{image_root1}/v1_{i:04d}.jpg" for i in range(num_frames)]
    image_names2 = [f"{image_root2}/v2_{i:04d}.jpg" for i in range(num_frames)]

    keyframes_json = os.path.join(args.input_folder, "keyframes.json")
    if os.path.exists(keyframes_json):
        import json
        with open(keyframes_json, "r", encoding="utf-8") as f:
            kf_data = json.load(f)
        valid_ids = sorted(kf_data.get("vggt", []))
        print(f"Using keyframes.json vggt field: {len(valid_ids)} frames")
    else:
        valid_ids = list(range(0, num_frames, args.stride))
        print(f"keyframes.json not found, fallback to stride={args.stride}: {len(valid_ids)} frames")

    num_sample = args.vggt_track_samples
    tracks = dict(track_v1=[], track_v2=[], frame_ids=[])
    expand_mask = args.expand_mask
    
    for frame_id in tqdm(valid_ids, desc="VGGT tracking"):
        image_pair_names = [image_names1[frame_id], image_names2[frame_id]]
        
        # Separately choose mask and depth sources based on availability
        # All files use frame_id for naming consistency
        if use_keyframe_masks:
            mask_pair_names = [f"{mask_keyframe_root1}/v1_{frame_id:04d}.png",
                              f"{mask_keyframe_root2}/v2_{frame_id:04d}.png"]
        else:
            mask_pair_names = [f"{mask_root1}/v1_{frame_id:04d}.png",
                              f"{mask_root2}/v2_{frame_id:04d}.png"]
        
        if use_keyframe_depths:
            depth_pair_names = [f"{depth_keyframe_root1}/v1_{frame_id:04d}.png",
                               f"{depth_keyframe_root2}/v2_{frame_id:04d}.png"]
        else:
            depth_pair_names = [f"{depth_root1}/v1_{frame_id:04d}.png",
                               f"{depth_root2}/v2_{frame_id:04d}.png"]

        track_v1, track_v2, _ = vggt_track_pair(image_pair_names, mask_pair_names, depth_pair_names, num_sample=num_sample, return_pointcloud=False, expand_mask=expand_mask)
        # if i == 0:
        #     if not expand_mask and (len(track_v1) == 0 or len(track_v2) == 0):
        #         expand_mask = True
        #         print(f"Expand mask for {args.input_folder} for no valid tracks on human surface")
        #         track_v1, track_v2, _ = vggt_track_pair(image_pair_names, mask_pair_names, depth_pair_names, num_sample=num_sample, return_pointcloud=False, expand_mask=expand_mask)
        if len(track_v1) == num_sample and len(track_v2) == num_sample:
            tracks['track_v1'].append(track_v1)
            tracks['track_v2'].append(track_v2)
            tracks['frame_ids'].append(frame_id)
    
    if len(tracks['track_v1']) > 0:

        tracks['track_v1'] = torch.stack(tracks['track_v1'], dim=0).cpu().numpy()
        tracks['track_v2'] = torch.stack(tracks['track_v2'], dim=0).cpu().numpy()
        tracks['frame_ids'] = np.array(tracks['frame_ids'])
        tracks['expand_mask'] = expand_mask
        np.savez(f"{args.input_folder}/vggt_tracks.npz", **tracks)
        
        # Visualize if requested
        if args.visualize:
            vis_output_path = os.path.join(args.input_folder, args.vis_output)
            print(f"\nVisualizing VGGT tracks to {vis_output_path}...")
            visualize_vggt_tracks(
                seq_path=args.input_folder,
                output_folder=vis_output_path,
                num_tracks=args.vis_num_tracks
            )
    else:
        print(f"No vggt tracks found for {args.input_folder}, make an empty file")
        run_cmd(f"touch {args.input_folder}/vggt_tracks.npz")
        warning_message = f"No vggt tracks found for {args.input_folder}, make an empty file"
        write_warning_to_log(args.log_file, warning_message)
