import argparse
import json
import os
import cv2
import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm, trange
from PIL import Image
from embod_mocap.config_paths import PATHS
from embod_mocap.processor.base import CAM_CONVENTION_CHANGE, lingbotdepth_refine_batch, MDMModel, load_image_rotate, expand_to_rectangle
from embod_mocap.human.utils.mesh_utils import filter_mesh


def tsdf_clean(rgb_images, depth_images, w, h, K, RT, invalid_masks=None, voxel_size=0.005,  sdf_trunc=0.1, depth_trunc=3.0):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc, 
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8  
    )

    w = int(w)
    h = int(h)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h,  
        fx=fx, fy=fy, 
        cx=cx, cy=cy  
    )
    if invalid_masks is not None:
        for i in range(len(rgb_images)):
            invalid_mask = invalid_masks[i]
            depth_images[i] = np.where(invalid_mask > 0.5, 0, depth_images[i])

    for i, pose, color, depth in zip(range(len(rgb_images)), RT, rgb_images, depth_images):
        color = o3d.geometry.Image(color)
        depth = o3d.geometry.Image(depth)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1.0,  
            depth_trunc=depth_trunc,     
            convert_rgb_to_intensity=False 
        )
        
        tsdf_volume.integrate(
            rgbd_image,
            intrinsic,
            np.linalg.inv(np.array(pose)) 
        )

    point_cloud = tsdf_volume.extract_point_cloud()
    mesh = tsdf_volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    return point_cloud, mesh



def unproj_depth(args):
    root = args.input_folder
    device = torch.device(args.device)
    # scale = args.scale
    with open(f"{root}/transforms.json", "r") as f:
        transforms = json.load(f)
    cx = transforms['cx']
    cy = transforms['cy']
    fx = transforms['fl_x']
    fy = transforms['fl_y']
    w = transforms['w']
    h = transforms['h']
    RT = []
    K = np.array([[fx, 0, cx, 0],
                  [0, fy, cy, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    if args.depth_refine:
        depth_refine_model = MDMModel.from_pretrained(PATHS.lingbotdepth_ckpt).to(device)
        depth_refine_model.eval()
    else:
        depth_refine_model = None

    if args.vggt_refine:
        # Use config for chunk size
        vggt_mask = vggt_depth_predict(args, chunk_size=args.vggt_depth_chunk)

    rgb_paths = []
    depth_paths = []
    full_depth_images = []
    full_rgb_images = []
    if args.mask_out:
        invalid_masks = []
    else:
        invalid_masks = None

    # New batch processing implementation
    for frame_id, frame_info in enumerate(transforms['frames']):
        if args.correct_convention:
            RT.append(frame_info['transform_matrix'] @ CAM_CONVENTION_CHANGE)
        else:
            RT.append(frame_info['transform_matrix'])
        rgb_paths.append(os.path.join(root, frame_info['file_path']))
        depth_paths.append(os.path.join(root, frame_info['depth_file_path']))

    # Load all images and depths
    for frame_id, frame_info in enumerate(transforms['frames']):
        rgb_image = Image.open(rgb_paths[frame_id]).convert("RGB")
        rgb_image = np.array(rgb_image)

        full_rgb_images.append(rgb_image)
        
        depth_image = cv2.imread(depth_paths[frame_id], cv2.IMREAD_UNCHANGED)
        depth_image = depth_image.astype(np.float32) / 1000.0

        
        if args.mask_out:
            mask_dir = args.mask_dir if hasattr(args, "mask_dir") else "masks"
            mask_path = os.path.join(root, mask_dir, os.path.basename(frame_info['file_path']).replace("jpg", "png"))
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            mask = expand_to_rectangle(mask, filling=255, padding=0)
            mask = mask > 127
            invalid_masks.append(mask)
            
        full_depth_images.append(depth_image)

    # Batch depth refinement
    if args.depth_refine:
        chunk_size = args.depth_refine_scene_chunk
        refined_depths = []
        
        for i in tqdm(range(0, len(full_rgb_images), chunk_size), desc="Refining depths"):
            end_idx = min(i + chunk_size, len(full_rgb_images))
            batch_rgb = full_rgb_images[i:end_idx]
            batch_depth = full_depth_images[i:end_idx]
            
            # Prepare batch tensors
            rgb_tensors = []
            depth_tensors = []
            for j in range(len(batch_rgb)):
                rgb_tensor = load_image_rotate(batch_rgb[j], to_tensor=True)
                depth_tensor = torch.Tensor(batch_depth[j])[None][None]
                rgb_tensors.append(rgb_tensor)
                depth_tensors.append(depth_tensor)
            rgb_tensors = torch.cat(rgb_tensors, dim=0)
            depth_tensors = torch.cat(depth_tensors, dim=0)
            # Batch refinement
            refined_batch = lingbotdepth_refine_batch(model=depth_refine_model, device=device,
                                                images=rgb_tensors, prompt_depths=depth_tensors)
            
            # Process and store results
            for j, refined_depth in enumerate(refined_batch):
                resized_depth = cv2.resize(refined_depth, (w, h), interpolation=cv2.INTER_CUBIC)
                refined_depths.append(resized_depth)
        
        full_depth_images = refined_depths
    full_rgb_images = np.array(full_rgb_images)
    full_depth_images = np.array(full_depth_images)

    RT = np.array(RT)

    if args.vggt_refine:
        if invalid_masks is None:
            invalid_masks = ~vggt_mask
        else:
            invalid_masks *= ~vggt_mask

    cleaned_ply, mesh = tsdf_clean(full_rgb_images, full_depth_images, w, h, K, RT, invalid_masks, voxel_size=args.voxel_size, sdf_trunc=args.sdf_trunc, depth_trunc=args.depth_trunc)
               
    return mesh, cleaned_ply


def compute_scale_batch(depth1, depth2, mask=None):
    B, H, W = depth1.shape
    depth1_flat = depth1.reshape(B, -1)  # (B, H*W)
    depth2_flat = depth2.reshape(B, -1)  # (B, H*W)
    
    if mask is not None:
        mask_flat = mask.reshape(B, -1)  # (B, H*W)
        depth1_flat = depth1_flat * mask_flat
        depth2_flat = depth2_flat * mask_flat

    numerator = np.sum(depth2_flat * depth1_flat, axis=1)  # ∑(depth2 * depth1) for each batch
    denominator = np.sum(depth2_flat ** 2, axis=1)         # ∑(depth2^2) for each batch
    scales = numerator / denominator  # (B,)

    return scales


def vggt_depth_predict(args, chunk_size=30):
    # predict the depths using vggt, use a chunk size
    from embod_mocap.vggt.vggt.models.vggt import VGGT
    from embod_mocap.vggt.vggt.utils.load_fn import load_and_preprocess_images
    root = args.input_folder
    with open(f"{root}/transforms.json", "r") as f:
        transforms = json.load(f)
    vggt_depths = []
    vggt_depth_conf = []
    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT()
    model.load_state_dict(torch.load(PATHS.vggt_ckpt))
    model = model.to(device)

    image_names = []
    sensor_depths = []
    for frame_info in transforms['frames']:
        image_names.append(os.path.join(root, frame_info['file_path']))
        sensor_depths.append(cv2.imread(os.path.join(root, frame_info['depth_file_path']), cv2.IMREAD_UNCHANGED))
    sensor_depths = np.array(sensor_depths)
    sensor_depths = sensor_depths.astype(np.float32) / 1000.0
    raw_h = sensor_depths.shape[1]
    raw_w = sensor_depths.shape[2]

    images = load_and_preprocess_images(image_names).to(device)

    for chunk_id in trange(0, len(images), chunk_size, desc="Predicting vggt depths and confs"):
        chunk_images = images[chunk_id:chunk_id + chunk_size]
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                chunk_images = chunk_images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = model.aggregator(chunk_images)
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, chunk_images, ps_idx)
        depth_map = depth_map[0]
        depth_conf = depth_conf[0]
        vggt_depths.append(depth_map.cpu().numpy())
        vggt_depth_conf.append(depth_conf.cpu().numpy())
        torch.cuda.empty_cache()
    del model

    vggt_depths = np.concatenate(vggt_depths, axis=0)
    vggt_depth_conf = np.concatenate(vggt_depth_conf, axis=0)
    vggt_depths_resized = []
    vggt_depth_conf_resized = []
    for depth, depth_conf in zip(vggt_depths, vggt_depth_conf):
        depth = cv2.resize(depth, (raw_w, raw_h), interpolation=cv2.INTER_CUBIC)
        vggt_depths_resized.append(depth)
        depth_conf = cv2.resize(depth_conf, (raw_w, raw_h), interpolation=cv2.INTER_CUBIC)
        vggt_depth_conf_resized.append(depth_conf)
    # vggt_depths_resized = np.array(vggt_depths_resized)
    vggt_depth_conf_resized = np.array(vggt_depth_conf_resized)
    mask = (vggt_depth_conf_resized > 3) * (sensor_depths < args.depth_trunc)
    # scale = compute_scale_batch(sensor_depths, vggt_depths_resized, mask)
    # vggt_depths_resized = vggt_depths_resized * scale[:, None, None]
    # vggt_depths_resized = vggt_depths_resized * 1000
    # vggt_depths_resized = np.clip(vggt_depths_resized, 0, 65535).astype(np.uint16)
    return mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Unproject point clouds from depth images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_folder",
        type=str,
        default="",
        help="Path to the input folder containing depth images",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Scale factor for the input images",
    )
    parser.add_argument(
        "--depth_trunc",
        type=float,
        default=5.0,
        help="Maximum depth value to consider",
    )
    parser.add_argument(
        "--depth_refine",
        action="store_true",
        help="Whether to use depth refinement",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.01,
        help="Voxel size for TSDF volume",
    )
    parser.add_argument(
        "--depth_refine_scene_chunk",
        type=int,
        default=5,
        help="Chunk size for depth refinement scene processing",
    )
    parser.add_argument(
        "--vggt_depth_chunk",
        type=int,
        default=30,
        help="Chunk size for VGGT depth prediction",
    )
    parser.add_argument(
        "--mask_out",
        action="store_true",
        help="Whether to mask out the depth images",
    )
    parser.add_argument(
        "--vggt_refine",
        action="store_true",
        help="Whether to use VGG-T for depth prediction",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="masks",
        help="Mask directory relative to input folder.",
    )
    parser.add_argument(
        "--correct_convention",
        action="store_true",
        help="Whether to correct the camera convention",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--sdf_trunc",
        type=float,
        default=0.1,
        help="SDF truncation value",
    )

    args = parser.parse_args()
    mesh, cleaned_ply = unproj_depth(args)

    mesh = filter_mesh(mesh)

    mesh = mesh.simplify_vertex_clustering(0.01)
    
    mesh_raw_path = os.path.join(args.input_folder, "mesh_raw.ply")
    o3d.io.write_triangle_mesh(mesh_raw_path, mesh)
    print(f"Saved mesh to {mesh_raw_path}")

    # Also save a lighter mesh for visualization/export workflows.
    mesh_simplified = mesh.simplify_vertex_clustering(voxel_size=0.12)
    mesh_simplified.remove_unreferenced_vertices()
    mesh_simplified_path = os.path.join(args.input_folder, "mesh_simplified.ply")
    o3d.io.write_triangle_mesh(mesh_simplified_path, mesh_simplified)
    print(f"Saved simplified mesh to {mesh_simplified_path}")
