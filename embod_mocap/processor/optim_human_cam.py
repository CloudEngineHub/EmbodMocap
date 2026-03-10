import os
import argparse
import numpy as np
import open3d as o3d
import copy
import torch
import cv2
import torch

from embod_mocap.processor.base import combine_RT, project_3d_to_2d, batch_depthmap_to_pts3d_numpy, export_cameras_to_ply
from t3drender.transforms import rotmat_to_rot6d, rot6d_to_rotmat
import torch.nn.functional as F
import numpy as np
from pytorch3d.loss import chamfer_distance
from embod_mocap.human.utils.mesh_utils import slice_pointcloud_o3d


def apply_rigid_transform_points(points, R, T, scale=1.0):
    return scale * (points @ R.T) + T


def optim_cam(scene, v1_info, v2_info, vggt_track, chamfer, dba, p2p, device, lr=1e-3, max_iters=100, log_interval=10, fix_relative=False, z_rot_only=False):
    R_sai_v1 = torch.from_numpy(v1_info['R']).to(device).float()
    T_sai_v1 = torch.from_numpy(v1_info['T']).to(device).float()
    points2D_v1 = v1_info['points2D']
    points3D_v1 = v1_info['points3D']
    point_ids_v1 = v1_info['point_ids']
    K1 = v1_info['K']
    K1 = torch.from_numpy(K1).to(device).float()
    H1 = v1_info['H']
    W1 = v1_info['W']
    human_pc1 = v1_info['human_pc']
    scene_v1 = v1_info['scene']
    
    R_sai_v2 = torch.from_numpy(v2_info['R']).to(device).float()    
    T_sai_v2 = torch.from_numpy(v2_info['T']).to(device).float()
    points2D_v2 = v2_info['points2D']
    points3D_v2 = v2_info['points3D']
    point_ids_v2 = v2_info['point_ids']
    K2 = v2_info['K']
    K2 = torch.from_numpy(K2).to(device).float()
    H2 = v2_info['H']
    W2 = v2_info['W']
    human_pc2 = v2_info['human_pc']
    scene_v2 = v2_info['scene']

    if p2p:
        p2p_depth_pts_v1 = v1_info['p2p_depth_pts']
        p2p_colmap_pts_v1 = v1_info['p2p_colmap_pts']
        p2p_depth_pts_v2 = v2_info['p2p_depth_pts']
        p2p_colmap_pts_v2 = v2_info['p2p_colmap_pts']

    if z_rot_only:
        # Only optimize z-axis rotation using angle parameters
        z_angle1 = torch.zeros(1, device=device).float().requires_grad_(True)
        z_angle2 = torch.zeros(1, device=device).float().requires_grad_(True)
        R1_offset = None  # Will be computed from z_angle1
        R2_offset = None  # Will be computed from z_angle2
    else:
        # Original 6DOF rotation optimization
        R1_offset = torch.eye(3, 3, device=device).float()
        R1_offset = rotmat_to_rot6d(R1_offset[None]).clone().requires_grad_(True)   
        R2_offset = torch.eye(3, 3, device=device).float()
        R2_offset = rotmat_to_rot6d(R2_offset[None]).clone().requires_grad_(True)
        z_angle1 = None
        z_angle2 = None

    T1_offset = torch.zeros(1, 3, device=device).float().requires_grad_(True)
    T2_offset = torch.zeros(1, 3, device=device).float().requires_grad_(True)

    def apply_rigid_transform_torch(pc, R, T, scale=1.0):
        return scale * torch.bmm(pc, R.transpose(1, 2)) + T
    
    def z_angle_to_rotmat(angle):
        """Convert z-axis rotation angle to rotation matrix"""
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        zeros = torch.zeros_like(angle)
        ones = torch.ones_like(angle)
        
        R = torch.stack([
            torch.stack([cos_a, -sin_a, zeros], dim=-1),
            torch.stack([sin_a, cos_a, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1)
        ], dim=-2)
        return R

    if z_rot_only:
        if fix_relative:
            optimizer = torch.optim.Adam([z_angle1, T1_offset], lr=lr)
        else:
            optimizer = torch.optim.Adam([z_angle1, z_angle2, T1_offset, T2_offset], lr=lr)
    else:
        if fix_relative:
            optimizer = torch.optim.Adam([R1_offset, T1_offset], lr=lr)
        else:
            optimizer = torch.optim.Adam([R1_offset, R2_offset, T1_offset, T2_offset], lr=lr)

    # Preprocess points for bundle adjustment
    def preprocess_bundle_adjustment(points2D, points3D, point_ids, H, W):
        cam_ids = []
        p2ds_all = []
        p3ds_all = []
        for im_name in points2D:
            frame_id = int(im_name.split('_')[1].replace('.jpg', ''))
            point2D = points2D[im_name]
            for point in point2D:
                pid = int(point[2])
                p2d = point[:2]
                index = point_ids.index(pid)
                cam_ids.append(frame_id)
                p2ds_all.append(torch.from_numpy(p2d).to(device).float())
                p3ds_all.append(torch.from_numpy(points3D[index]).to(device).float())

        p2ds_all = torch.stack(p2ds_all).to(device)  # (N, 2)

        p3ds_all = torch.stack(p3ds_all).to(device)  # (N, 3)
        p2ds_all[:, 0] = p2ds_all[:, 0] / W
        p2ds_all[:, 1] = p2ds_all[:, 1] / H

        return cam_ids, p2ds_all, p3ds_all

    # Prepare bundle adjustment data for both views
    if dba:
        cam_ids_v1, p2ds_v1, p3ds_v1 = preprocess_bundle_adjustment(points2D_v1, points3D_v1, point_ids_v1, H1, W1)
        cam_ids_v2, p2ds_v2, p3ds_v2 = preprocess_bundle_adjustment(points2D_v2, points3D_v2, point_ids_v2, H2, W2)

    def dense_ba_loss_fn(R, T, cam_ids, p2ds_all, p3ds_all, K, H, W):
        R = R.clone()
        T = T.clone()
        R = torch.permute(R, (0, 2, 1))
        T = - torch.einsum('nij,nj->ni', R, T)
        with torch.cuda.amp.autocast(enabled=False):
            projected = project_3d_to_2d(K, R[cam_ids], T[cam_ids], p3ds_all)
        projected[:, 0] = projected[:, 0] / W
        projected[:, 1] = projected[:, 1] / H
        ba_loss = F.mse_loss(projected, p2ds_all)  # (N, 2)
        ba_loss = torch.mean(ba_loss) * 1e3
        return ba_loss

    for itr in range(max_iters):
        optimizer.zero_grad()

        # Convert rotation offsets back to rotation matrices
        if z_rot_only:
            R1_offset_rotmat = z_angle_to_rotmat(z_angle1)
            R2_offset_rotmat = z_angle_to_rotmat(z_angle2)
        else:
            R1_offset_rotmat = rot6d_to_rotmat(R1_offset)
            R2_offset_rotmat = rot6d_to_rotmat(R2_offset)
        
        if fix_relative:
            R2_offset_rotmat = R1_offset_rotmat
            T2_offset = T1_offset

        if chamfer:
            # Transform the scene and human point clouds
            scene_v1_transformed = apply_rigid_transform_torch(scene_v1, R1_offset_rotmat, T1_offset, 1.0)
            scene_v2_transformed = apply_rigid_transform_torch(scene_v2, R2_offset_rotmat, T2_offset, 1.0)

            # Chamfer distance loss
            chamfer_loss1 = chamfer_distance(scene_v1_transformed, scene)[0] * 0.1
            chamfer_loss2 = chamfer_distance(scene_v2_transformed, scene)[0] * 0.1
        else:
            chamfer_loss1 = 0
            chamfer_loss2 = 0

        if vggt_track:
            human_pc1_transformed = apply_rigid_transform_torch(human_pc1, R1_offset_rotmat, T1_offset, 1.0)
            human_pc2_transformed = apply_rigid_transform_torch(human_pc2, R2_offset_rotmat, T2_offset, 1.0)
            # MSE loss for human point clouds
            human_pc_loss = F.mse_loss(human_pc1_transformed, human_pc2_transformed) * 10
        else:
            human_pc_loss = 0

        # Bundle adjustment losses for both views
        if dba:
            R1_transformed, T1_transformed = apply_rigid_to_RT_torch(R_sai_v1, T_sai_v1, R1_offset_rotmat[0], T1_offset, 1.0)
            R2_transformed, T2_transformed = apply_rigid_to_RT_torch(R_sai_v2, T_sai_v2, R2_offset_rotmat[0], T2_offset, 1.0)

            ba_loss_v1 = dense_ba_loss_fn(
                R1_transformed, T1_transformed, cam_ids_v1, p2ds_v1, p3ds_v1, 
                K1, H1, W1
            )
            ba_loss_v2 = dense_ba_loss_fn(
                R2_transformed, T2_transformed, cam_ids_v2, p2ds_v2, p3ds_v2, 
                K2, H2, W2
            )
        else:
            ba_loss_v1 = 0
            ba_loss_v2 = 0

        if p2p:
            p2p_v1_transformed = apply_rigid_transform_torch(p2p_depth_pts_v1, R1_offset_rotmat, T1_offset, 1.0)
            p2p_v2_transformed = apply_rigid_transform_torch(p2p_depth_pts_v2, R2_offset_rotmat, T2_offset, 1.0)
            p2p_v1_loss = F.mse_loss(p2p_v1_transformed, p2p_colmap_pts_v1) * 1e5 / p2p_colmap_pts_v1.shape[1]
            p2p_v2_loss = F.mse_loss(p2p_v2_transformed, p2p_colmap_pts_v2) * 1e5 / p2p_colmap_pts_v2.shape[1]
        else:
            p2p_v1_loss = 0
            p2p_v2_loss = 0
      
        # Combine all losses into the total loss
        total_loss = human_pc_loss * 2 + chamfer_loss1 + chamfer_loss2 + ba_loss_v1 + ba_loss_v2 + p2p_v1_loss + p2p_v2_loss

        # Backpropagation and optimization step
        total_loss.backward()
        optimizer.step()
        if itr % log_interval == 0 or itr == max_iters - 1:
            loss_info = f"Iteration {itr}/{max_iters}, Total Loss: {total_loss.item():04f}, "

            if chamfer:
                loss_info += f"Chamfer1: {chamfer_loss1.item():04f}, Chamfer2: {chamfer_loss2.item():04f}, "
            if vggt_track:
                loss_info += f"VGGT Track: {human_pc_loss.item():04f}"
            if dba:
                loss_info += f"Bundle Adjustment v1: {ba_loss_v1.item():04f}, Bundle Adjustment v2: {ba_loss_v2.item():04f}"
            if p2p:
                loss_info += f", P2P v1: {p2p_v1_loss.item():04f}, P2P v2: {p2p_v2_loss.item():04f}"
            print(loss_info)

    # Extract final results
    R1_final = R1_offset_rotmat.detach().cpu().numpy()[0]
    T1_final = T1_offset.detach().cpu().numpy()[0]
    R2_final = R2_offset_rotmat.detach().cpu().numpy()[0]
    T2_final = T2_offset.detach().cpu().numpy()[0]
    if vggt_track:
        human_pc1_transformed = human_pc1_transformed.detach().cpu().numpy()[0]
        human_pc2_transformed = human_pc2_transformed.detach().cpu().numpy()[0]
    else:
        human_pc1_transformed = None
        human_pc2_transformed = None

    return R1_final, T1_final, R2_final, T2_final, human_pc1_transformed, human_pc2_transformed


def random_downsample_point_cloud(pcd, ratio):
    num_points = np.asarray(pcd.points).shape[0]
    
    sampled_indices = np.random.choice(num_points, size=int(num_points * ratio), replace=False)
    
    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[sampled_indices])
    
    if pcd.has_colors():
        downsampled_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[sampled_indices])
    
    if pcd.has_normals():
        downsampled_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[sampled_indices])
    return downsampled_pcd


# Apply alignment to all RTs (R, T)
def apply_rigid_to_RT(R, T, R_offset, T_offset, scale=1.0):
    R_new = np.einsum('ij,njk->nik', R_offset, R)
    T_new = scale * (R_offset @ T.T).T + T_offset
    return R_new, T_new

def apply_rigid_to_RT_torch(R, T, R_offset, T_offset, scale=1.0):
    R_new = torch.einsum('ij,njk->nik', R_offset, R)
    T_new = scale * (R_offset @ T.T).T + T_offset
    return R_new, T_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calibrate the human views to the scene")
    parser.add_argument(
        "input_folder",
        type=str,
        default="",
        help="Path to the input folder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for computation.",
    )
    parser.add_argument(
        "--vggt_track",
        action="store_true",
        help="Whether to use VGGT tracks on human body.",
    )
    parser.add_argument(
        "--chamfer",
        action="store_true",
        help="Whether to use Chamfer loss.",
    )
    parser.add_argument(    
        "--dba",
        action="store_true",
        help="Whether to use Dense Bundle Adjustment.",
    )
    parser.add_argument(
        "--p2p",
        action="store_true",
        help="Whether to use point-to-point loss (depth-unprojected 2D colmap points vs colmap 3D points).",
    )
    parser.add_argument(
        "--z_rot_only",
        action="store_true",
        help="Whether to optimize only z-axis rotation (gravity-aligned).",
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
    parser.add_argument(
        "--fix_scale",
        action="store_true",
        help="Whether to fix the scale.",
    )
    args = parser.parse_args()
    #############################################################
    # load data
    fps = 30
    H = 960
    W = 720
    device = torch.device(args.device)
    scene_folder = os.path.dirname(os.path.abspath(args.input_folder))
    pc_scene_mesh = o3d.io.read_triangle_mesh(os.path.join(scene_folder, "mesh_raw.ply"))
    pc_scene = o3d.geometry.PointCloud()
    pc_scene.points = pc_scene_mesh.vertices
    pc_scene.colors = pc_scene_mesh.vertex_colors
    pc_scene = random_downsample_point_cloud(pc_scene, 0.1)

    use_vggt_track = False
    if args.vggt_track:
        if os.path.exists(os.path.join(args.input_folder, "vggt_tracks.npz")):
            try:
                vggt_tracks = np.load(os.path.join(args.input_folder, "vggt_tracks.npz"), allow_pickle=True)
                track_v1 = vggt_tracks['track_v1']
                track_v2 = vggt_tracks['track_v2']
                track_frame_ids = vggt_tracks['frame_ids']

                track_v1 = track_v1.astype(np.int32)
                track_v2 = track_v2.astype(np.int32)
                track_v1[:, :, 0] = np.clip(track_v1[:, :, 0], 0, W - 1)
                track_v1[:, :, 1] = np.clip(track_v1[:, :, 1], 0, H - 1)
                track_v2[:, :, 0] = np.clip(track_v2[:, :, 0], 0, W - 1)
                track_v2[:, :, 1] = np.clip(track_v2[:, :, 1], 0, H - 1)
                use_vggt_track = True
            except:
                print(f"No vggt tracks found for {args.input_folder}")
        else:
            print(f"No vggt tracks found for {args.input_folder}")
        
    points2dv1 = np.load(os.path.join(args.input_folder, "v1", "points2D.npz"))
    points3dv1 = np.load(os.path.join(args.input_folder, "v1", "points3D.npz"))
    point_ids_v1 = points3dv1['point_ids']
    points_xyz_v1 = points3dv1['points']

    points2dv2 = np.load(os.path.join(args.input_folder, "v2", "points2D.npz"))
    points3dv2 = np.load(os.path.join(args.input_folder, "v2", "points3D.npz"))
    point_ids_v2 = points3dv2['point_ids']
    points_xyz_v2 = points3dv2['points']

    #############################################################
    # Load pre-aligned cameras (should be generated by align_cameras.py first)
    cameras_aligned1 = np.load(os.path.join(args.input_folder, "v1", "cameras_sai_transformed.npz"))
    R1_sai_aligned = cameras_aligned1["R"]
    T1_sai_aligned = cameras_aligned1["T"]
    K1 = cameras_aligned1["K"][:3, :3].astype(np.int32)  # Update K1 with aligned version
    
    cameras_aligned2 = np.load(os.path.join(args.input_folder, "v2", "cameras_sai_transformed.npz"))
    R2_sai_aligned = cameras_aligned2["R"]
    T2_sai_aligned = cameras_aligned2["T"]
    K2 = cameras_aligned2["K"][:3, :3].astype(np.int32)  # Update K2 with aligned version
    
    RT1_sai_aligned = combine_RT(R1_sai_aligned, T1_sai_aligned)
    RT2_sai_aligned = combine_RT(R2_sai_aligned, T2_sai_aligned)
    
    if args.chamfer:
        pc_scene_human1_o3d = o3d.io.read_point_cloud(os.path.join(args.input_folder, "v1", "pointcloud.ply"))
        pc_scene_human2_o3d = o3d.io.read_point_cloud(os.path.join(args.input_folder, "v2", "pointcloud.ply"))

    #############################################################
    if use_vggt_track:
        depths1 = []
        masks1 = []
        depth_dir = "depths_keyframe_refined" if args.use_keyframe_depths else "depths"
        v1_depth_path = os.path.join(args.input_folder, "v1", depth_dir)
        v1_mask_path = os.path.join(args.input_folder, "v1", "masks_keyframe" if args.use_keyframe_masks else "masks")

        # for i in range(len(os.listdir(v1_depth_path)))
        for i in track_frame_ids:
            depth = cv2.imread(os.path.join(v1_depth_path, f"v1_{i:04d}.png"), cv2.IMREAD_UNCHANGED)
            depth = depth / 1000
            mask = cv2.imread(os.path.join(v1_mask_path, f"v1_{i:04d}.png"), cv2.IMREAD_UNCHANGED)

            mask = mask > 127
            depths1.append(depth)
            masks1.append(mask)
        depths1 = np.stack(depths1, axis=0)
        masks1 = np.stack(masks1, axis=0)
        pc1 = batch_depthmap_to_pts3d_numpy(depths1, K1, R1_sai_aligned[track_frame_ids], T1_sai_aligned[track_frame_ids, :, None], H, W, 1)

        depths2 = []
        masks2 = []
        depth_dir = "depths_keyframe_refined" if args.use_keyframe_depths else "depths"
        v2_depth_path = os.path.join(args.input_folder, "v2", depth_dir)
        v2_mask_path = os.path.join(args.input_folder, "v2", "masks_keyframe" if args.use_keyframe_masks else "masks")
        # for i in range(len(os.listdir(v2_depth_path))):
        for i in track_frame_ids:
            depth = cv2.imread(os.path.join(v2_depth_path, f"v2_{i:04d}.png"), cv2.IMREAD_UNCHANGED)
            depth = depth / 1000
            mask = cv2.imread(os.path.join(v2_mask_path, f"v2_{i:04d}.png"), cv2.IMREAD_UNCHANGED)

            mask = mask > 127
            depths2.append(depth)
            masks2.append(mask)
        depths2 = np.stack(depths2, axis=0)
        masks2 = np.stack(masks2, axis=0)
        pc2 = batch_depthmap_to_pts3d_numpy(depths2, K2, R2_sai_aligned[track_frame_ids], T2_sai_aligned[track_frame_ids, :, None], H, W, 1)

        human_pc1_full = [pc1[i][masks1[i]] for i in range(len(pc1))]
        human_pc2_full = [pc2[i][masks2[i]] for i in range(len(pc2))]
        human_pc1_full = np.concatenate(human_pc1_full, axis=0)
        human_pc2_full = np.concatenate(human_pc2_full, axis=0)

        #############################################################
        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(human_pc1_full)
        # pcd1.colors = o3d.utility.Vector3dVector(np.ones_like(human_pc1_full) * np.array([0, 0, 1]))
        # o3d.io.write_point_cloud(os.path.join(args.input_folder, "pc1.ply"), pcd1)

        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(human_pc2_full)
        # pcd2.colors = o3d.utility.Vector3dVector(np.ones_like(human_pc2_full) * np.array([1, 1, 0]))
        # o3d.io.write_point_cloud(os.path.join(args.input_folder, "pc2.ply"), pcd2)
        #############################################################
        human_pc1 = [pc1[i][track_v1[i, :, 1], track_v1[i, :, 0]] for i in range(len(pc1))]
        human_pc2 = [pc2[i][track_v2[i, :, 1], track_v2[i, :, 0]] for i in range(len(pc2))]
        human_pc1 = np.concatenate(human_pc1, axis=0)
        human_pc2 = np.concatenate(human_pc2, axis=0)
        human_pc1 = torch.from_numpy(human_pc1).to(device).float()
        human_pc2 = torch.from_numpy(human_pc2).to(device).float()
        human_pc1 = human_pc1[None]
        human_pc2 = human_pc2[None]

    else:
        human_pc1 = None
        human_pc2 = None

    if args.chamfer:
        height = 2.0
        highest = np.asarray(pc_scene.points)[:, 2].min() + height
        invalid_indices = np.where(np.asanyarray(pc_scene.points)[:, 2] > highest)[0]
        pc_scene = slice_pointcloud_o3d(pc_scene, invalid_indices)

        highest = np.asarray(pc_scene_human1_o3d.points)[:, 2].min() + height
        invalid_indices = np.where(np.asanyarray(pc_scene_human1_o3d.points)[:, 2] > highest)[0]
        pc_scene_human1_o3d = slice_pointcloud_o3d(pc_scene_human1_o3d, invalid_indices)

        highest = np.asarray(pc_scene_human2_o3d.points)[:, 2].min() + height
        invalid_indices = np.where(np.asanyarray(pc_scene_human2_o3d.points)[:, 2] > highest)[0]
        pc_scene_human2_o3d = slice_pointcloud_o3d(pc_scene_human2_o3d, invalid_indices)

        pc_scene = torch.from_numpy(np.asarray(pc_scene.points)).to(device).float()
        pc_scene_human1 = torch.from_numpy(np.asarray(pc_scene_human1_o3d.points)).to(device).float()
        pc_scene_human2 = torch.from_numpy(np.asarray(pc_scene_human2_o3d.points)).to(device).float()
        
        pc_scene = pc_scene[::10][None]
        pc_scene_human1 = pc_scene_human1[::10][None]
        pc_scene_human2 = pc_scene_human2[::10][None]
    else:
        pc_scene_human1 = None
        pc_scene_human2 = None

    p2p_depth_pts_v1 = None
    p2p_colmap_pts_v1 = None
    p2p_depth_pts_v2 = None
    p2p_colmap_pts_v2 = None
    if args.p2p:
        def compute_p2p_data(view, points2D, points3D, point_ids, R_aligned, T_aligned, K_mat, H, W, use_keyframe_depths, input_folder):
            depth_dir = "depths_keyframe_refined" if use_keyframe_depths else "depths"
            depth_path = os.path.join(input_folder, view, depth_dir)
            
            depth_pts_list = []
            colmap_pts_list = []
            point_ids_list = point_ids.tolist()
            
            for im_name in points2D:
                frame_id = int(im_name.split('_')[1].replace('.jpg', ''))
                depth_file = os.path.join(depth_path, f"{view}_{frame_id:04d}.png")
                if not os.path.exists(depth_file):
                    continue
                depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                
                point2D = points2D[im_name]
                R_frame = R_aligned[frame_id]
                T_frame = T_aligned[frame_id]
                
                fx, fy = K_mat[0, 0], K_mat[1, 1]
                cx, cy = K_mat[0, 2], K_mat[1, 2]
                
                for point in point2D:
                    pid = int(point[2])
                    if pid not in point_ids_list:
                        continue
                    x, y = point[0], point[1]
                    xi, yi = int(round(x)), int(round(y))
                    if xi < 0 or xi >= W or yi < 0 or yi >= H:
                        continue
                    d = depth[yi, xi]
                    if d <= 0:
                        continue
                    
                    cam_pt = np.array([d * (x - cx) / fx, d * (y - cy) / fy, d])
                    world_pt = R_frame @ cam_pt + T_frame
                    
                    idx = point_ids_list.index(pid)
                    colmap_pt = points3D[idx]
                    
                    depth_pts_list.append(world_pt)
                    colmap_pts_list.append(colmap_pt)
            
            if len(depth_pts_list) == 0:
                return None, None
            depth_pts = torch.from_numpy(np.array(depth_pts_list)).to(device).float()[None]
            colmap_pts = torch.from_numpy(np.array(colmap_pts_list)).to(device).float()[None]
            return depth_pts, colmap_pts
        
        p2p_depth_pts_v1, p2p_colmap_pts_v1 = compute_p2p_data(
            "v1", points2dv1, points_xyz_v1, point_ids_v1,
            R1_sai_aligned, T1_sai_aligned, K1, H, W,
            args.use_keyframe_depths, args.input_folder
        )
        p2p_depth_pts_v2, p2p_colmap_pts_v2 = compute_p2p_data(
            "v2", points2dv2, points_xyz_v2, point_ids_v2,
            R2_sai_aligned, T2_sai_aligned, K2, H, W,
            args.use_keyframe_depths, args.input_folder
        )
        if p2p_depth_pts_v1 is not None and p2p_depth_pts_v2 is not None:
            print(f"[p2p] v1: {p2p_depth_pts_v1.shape[1]} correspondences, v2: {p2p_depth_pts_v2.shape[1]} correspondences")
        else:
            print("[p2p] Warning: not enough p2p correspondences, disabling p2p loss")
            args.p2p = False

    v1_info = dict()
    v1_info['R'] = R1_sai_aligned
    v1_info['T'] = T1_sai_aligned
    v1_info['points2D'] = points2dv1
    v1_info['points3D'] = points3dv1['points']
    v1_info['point_ids'] = point_ids_v1.tolist()
    v1_info['K'] = K1
    v1_info['H'] = H
    v1_info['W'] = W
    v1_info['human_pc'] = human_pc1
    v1_info['scene'] = pc_scene_human1
    v1_info['p2p_depth_pts'] = p2p_depth_pts_v1
    v1_info['p2p_colmap_pts'] = p2p_colmap_pts_v1

    v2_info = dict()
    v2_info['R'] = R2_sai_aligned
    v2_info['T'] = T2_sai_aligned
    v2_info['points2D'] = points2dv2
    v2_info['points3D'] = points3dv2['points']
    v2_info['point_ids'] = point_ids_v2.tolist()
    v2_info['K'] = K2
    v2_info['H'] = H
    v2_info['W'] = W
    v2_info['human_pc'] = human_pc2
    v2_info['scene'] = pc_scene_human2
    v2_info['p2p_depth_pts'] = p2p_depth_pts_v2
    v2_info['p2p_colmap_pts'] = p2p_colmap_pts_v2
    
    R1_offset_optim, T1_offset_optim, R2_offset_optim, T2_offset_optim, human_pc1_, human_pc2_ = optim_cam(pc_scene,
            v1_info, v2_info, use_vggt_track, args.chamfer, args.dba, args.p2p, device,
            lr=1e-2, max_iters=200, log_interval=10, z_rot_only=args.z_rot_only,
        )
    P_offset1 = combine_RT(R1_offset_optim[None], T1_offset_optim[None])
    P_offset2 = combine_RT(R2_offset_optim[None], T2_offset_optim[None])

    if args.chamfer:
        pc_scene_human1_o3d = pc_scene_human1_o3d.transform(P_offset1[0])
        pc_scene_human1_o3d = random_downsample_point_cloud(pc_scene_human1_o3d, 0.1)

        o3d.io.write_point_cloud(os.path.join(args.input_folder, "v1", "pointcloud_aligned.ply"), pc_scene_human1_o3d)

        pc_scene_human2_o3d = pc_scene_human2_o3d.transform(P_offset2[0])
        pc_scene_human2_o3d = random_downsample_point_cloud(pc_scene_human2_o3d, 0.1)
        o3d.io.write_point_cloud(os.path.join(args.input_folder, "v2", "pointcloud_aligned.ply"), pc_scene_human2_o3d)

    R1_final, T1_final = apply_rigid_to_RT(R1_sai_aligned, T1_sai_aligned, R1_offset_optim, T1_offset_optim, 1.0)
    R2_final, T2_final = apply_rigid_to_RT(R2_sai_aligned, T2_sai_aligned, R2_offset_optim, T2_offset_optim, 1.0)

    #############################################################
    # pcd1 = o3d.geometry.PointCloud()
    # human_pc1_full_t = apply_rigid_transform_points(human_pc1_full, R1_offset2.squeeze(), T1_offset2.squeeze())
    # pcd1.points = o3d.utility.Vector3dVector(human_pc1_full_t)
    # pcd1.colors = o3d.utility.Vector3dVector(np.ones_like(human_pc1_full_t) * np.array([0, 0, 1]))
    # o3d.io.write_point_cloud(os.path.join(args.input_folder, "pc1_.ply"), pcd1)

    # pcd2 = o3d.geometry.PointCloud()
    # human_pc2_full_t = apply_rigid_transform_points(human_pc2_full, R2_offset2.squeeze(), T2_offset2.squeeze())
    # pcd2.points = o3d.utility.Vector3dVector(human_pc2_full_t)
    # pcd2.colors = o3d.utility.Vector3dVector(np.ones_like(human_pc2_full_t) * np.array([1, 1, 0]))
    # o3d.io.write_point_cloud(os.path.join(args.input_folder, "pc2_.ply"), pcd2)
    #############################################################

    RT1 = combine_RT(R1_final, T1_final)
    cameras = dict()
    cameras["R"] = R1_final.astype(np.float32)
    cameras["T"] = T1_final.astype(np.float32)
    cameras["K"] = cameras_aligned1["K"].astype(np.float32)

    np.savez(os.path.join(args.input_folder, "v1", "cameras.npz"), **cameras)
    export_cameras_to_ply(RT1, os.path.join(args.input_folder, "v1", "cameras.ply"),)

    RT2 = combine_RT(R2_final, T2_final)
    cameras = dict()
    cameras["R"] = R2_final.astype(np.float32)
    cameras["T"] = T2_final.astype(np.float32)
    cameras["K"] = cameras_aligned2["K"].astype(np.float32)

    np.savez(os.path.join(args.input_folder, "v2", "cameras.npz"), **cameras)
    export_cameras_to_ply(RT2, os.path.join(args.input_folder, "v2", "cameras.ply"),)
    print(f"export cameras to f'{args.input_folder}, v1, v2, cameras.ply'")
