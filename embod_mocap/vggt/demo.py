import torch
import numpy as np
import time
import cv2
import trimesh
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

image_root = '/home/wwj/programs/colmap/projects/P0_09/images'
mask_root = '/home/wwj/programs/colmap/projects/P0_09/masks'

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
# model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

def expand_mask_to_rectangle_torch(mask):
    if mask.dim() != 2:
        raise ValueError("Input mask must be a 2D tensor (single channel)!")
    
    nonzero_coords = torch.nonzero(mask, as_tuple=False)
    
    ymin = torch.min(nonzero_coords[:, 0]).item()
    ymax = torch.max(nonzero_coords[:, 0]).item()
    xmin = torch.min(nonzero_coords[:, 1]).item()
    xmax = torch.max(nonzero_coords[:, 1]).item()
    
    rect_mask = torch.zeros_like(mask)
    
    rect_mask[ymin:ymax+1, xmin:xmax+1] = 1
    
    return rect_mask

model = VGGT()
model.load_state_dict(torch.load('../../checkpoints/vggt.pt'))
model = model.to(device)
# Load and preprocess example images (replace with your own image paths)
image_names = [f"{image_root}/{i:05d}.jpg" for i in range(400, 500, 3)]
mask_names = [f"{mask_root}/{i:05d}.jpg" for i in range(400, 500, 3)]
images = load_and_preprocess_images(image_names).to(device)
masks = load_and_preprocess_images(mask_names).to(device)[:, 2:3]#.repeat(1, 3, 1, 1)

masks = masks > 0.5
masks_rect = masks.clone()
for i in range(len(masks)):
    masks_rect[i] = expand_mask_to_rectangle_torch(masks_rect[i, 0])[None]
masks = ~ masks
masks_rect = ~ masks_rect
# images = images * masks
# with torch.no_grad():
#     with torch.cuda.amp.autocast(dtype=dtype):
#         # Predict attributes including cameras, depth maps, and point maps.
#         predictions = model(images)
images_raw = images.clone()
# images = images * masks_rect
crop = True
start = time.time()
with torch.no_grad():
    
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
                
    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    # Predict Point Maps
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
        
    # Construct 3D Points from Depth Maps and Cameras
    # which usually leads to more accurate 3D points than point map branch
    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                                intrinsic.squeeze(0))

    # Predict Tracks
    # choose your own points to track, with shape (N, 2) for one scene
    query_points = torch.FloatTensor([[100.0, 200.0], 
                                        [60.72, 259.94]]).to(device)
    track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])

print(f"Time taken: {time.time() - start:.2f}s")
images = images_raw.squeeze().cpu().numpy().transpose(0, 2, 3, 1) 

if not crop:
    point_maps = point_map_by_unprojection.reshape(-1, 3)

    colors = images# now (S, H, W, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
else:

    masks = masks.cpu().numpy()
    point_maps = [point_map_by_unprojection[i][mask.squeeze(0)] for i, mask in enumerate(masks)]
    point_maps = np.concatenate(point_maps, axis=0)
    colors = [images[i][mask.squeeze(0)] for i, mask in enumerate(masks)]
    colors = np.concatenate(colors, axis=0)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)

num_exports = len(point_maps) // 5


random_idx = np.random.choice(len(point_maps), num_exports, replace=False)

pointcloud = trimesh.PointCloud(point_maps[random_idx], colors=colors_flat[random_idx])
pointcloud.export("pointcloud.ply")
