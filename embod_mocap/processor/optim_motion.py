"""work flow: triangulate kp2d -> kp3d, 
transform smpl params to world coordinate, 
transform depth to world coordinate, 
use loss function to optimize smpl params: [diffrenttiable rendering, smplify, pointcloud]"""
import torch
import numpy as np
import cv2
import os
import copy
import argparse

from embod_mocap.human.utils.kp_utils import create_skeleton_mesh, get_coco_body_skeleton, get_coco_bone_skeleton, get_coco_skeleton, smooth_and_interpolate, triangulate_sequence, visualize_kp3d_to_video
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.ndimage import gaussian_filter1d

from embod_mocap.human.configs import BMODEL
from embod_mocap.human.smpl import SMPL, SMPL_JOINT_NAMES

from embod_mocap.human.utils.transforms import eliminate_RT
from embod_mocap.processor.base import project_kp3d_to_2d, batch_depthmap_to_pts3d_numpy, convert_world_cam, batch_ortho_rotmat
from embod_mocap.human.utils.tensor_utils import list_to_padded
from embod_mocap.human.utils.mesh_utils import filter_and_sample_points
from t3drender.transforms import aa_to_rotmat, rotmat_to_aa


def detect_local_stationary_joints(kp3d, 
                                   velocity_threshold=0.1,
                                   min_segment_length=5):
    """
    Detect local stationary segments for each end effector.
    
    Args:
        kp3d: [T, 17, 3] COCO format 3D keypoints
        velocity_threshold: velocity threshold
        min_segment_length: minimum stationary segment length
    
    Returns:
        joint_stationary_masks: dict of per-joint stationary masks [T,]
    """
    end_effectors = {
        'left_wrist': 9,
        'right_wrist': 10, 
        'left_ankle': 15,
        'right_ankle': 16,
        'pelvis': None
    }
    
    joint_stationary_masks = {}
    
    for joint_name, joint_idx in end_effectors.items():
        if joint_name == 'pelvis':
            left_hip = kp3d[:, 11]  # [T, 3]
            right_hip = kp3d[:, 12]  # [T, 3]
            joint_points = (left_hip + right_hip) / 2
        else:
            joint_points = kp3d[:, joint_idx]  # [T, 3]
        
        velocities = np.diff(joint_points, axis=0)  # [T-1, 3]
        speeds = np.linalg.norm(velocities, axis=1)  # [T-1,]
        speeds = np.concatenate([[0], speeds])  # [T,]
        
        is_stationary = speeds < velocity_threshold
        
        filtered_stationary = np.zeros_like(is_stationary)
        start_idx = None
        for i, stationary in enumerate(is_stationary):
            if stationary and start_idx is None:
                start_idx = i
            elif not stationary and start_idx is not None:
                if i - start_idx >= min_segment_length:
                    filtered_stationary[start_idx:i] = True
                start_idx = None
        if start_idx is not None and len(is_stationary) - start_idx >= min_segment_length:
            filtered_stationary[start_idx:] = True
        
        joint_stationary_masks[joint_name] = filtered_stationary
        
        stationary_count = np.sum(filtered_stationary)
        print(f"{joint_name}: {stationary_count}/{len(filtered_stationary)} frames stationary")
    
    return joint_stationary_masks


def kp3d_smoothing(kp3d, bones, kp2d1, kp2d1_conf, K1, R1, T1, kp2d2, kp2d2_conf, K2, R2, T2,  H, W, num_iters=100, lr=1e-2, 
                   use_local_stationary=False, stationary_velocity_threshold=0.1, stationary_min_length=5, stationary_smooth_weight=10.0):
    """
    Optimize 3D keypoints with PyTorch and bone-length smoothness constraints.
    
    Args:
        kp3d (torch.Tensor): initial 3D keypoints with shape (N, J, 3), where N is timestep count and J is joint count.
        bones (list): bone edge list, each element is (start_idx, end_idx).
        lambda_smooth (float): weight of bone-length smoothness loss.
        lambda_const (float): weight of bone-length consistency loss.
        num_iters (int): number of optimization iterations.
        lr (float): learning rate.
    
    Returns:
        optimized_kp3d (torch.Tensor): optimized 3D keypoints with shape (N, J, 3).
    """
    joint_stationary_masks = None
    if use_local_stationary:
        joint_stationary_masks = detect_local_stationary_joints(
            kp3d, 
            velocity_threshold=stationary_velocity_threshold,
            min_segment_length=stationary_min_length
        )
    
    kp3d = torch.from_numpy(kp3d)
    kp2d1 = torch.from_numpy(kp2d1)
    kp2d1_conf = torch.from_numpy(kp2d1_conf)
    K1 = torch.from_numpy(K1).float()
    R1 = torch.from_numpy(R1)
    T1 = torch.from_numpy(T1)

    kp2d2 = torch.from_numpy(kp2d2)
    kp2d2_conf = torch.from_numpy(kp2d2_conf)
    K2 = torch.from_numpy(K2).float()
    R2 = torch.from_numpy(R2)
    T2 = torch.from_numpy(T2)

    kp3d = kp3d.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([kp3d], lr=lr)

    kp2d_1[..., 0] /= W
    kp2d_1[..., 1] /= H

    kp2d_2[..., 0] /= W
    kp2d_2[..., 1] /= H

    for iter in range(num_iters):
        optimizer.zero_grad()

        bone_lengths = []
        for start, end in bones:
            vec = kp3d[:, start, :] - kp3d[:, end, :]
            length = torch.norm(vec, dim=-1)
            bone_lengths.append(length)
        bone_lengths = torch.stack(bone_lengths, dim=0)
        bone_lengths = bone_lengths.transpose(1, 0)

        conf_ids = (kp2d1_conf * kp2d2_conf).mean(1).squeeze() >0.8
        bone_diff = bone_lengths - bone_lengths[conf_ids].mean(dim=0, keepdim=True)
        bone_smooth_loss = torch.sum(bone_diff ** 2)

        reproj_kp2d_1 = project_kp3d_to_2d(K1, R1, T1, kp3d)
        reproj_kp2d_2 = project_kp3d_to_2d(K2, R2, T2, kp3d)
        reproj_kp2d_1[..., 0] /= W
        reproj_kp2d_1[..., 1] /= H
        reproj_kp2d_2[..., 0] /= W
        reproj_kp2d_2[..., 1] /= H

        reproj_loss_1 = torch.mean(((reproj_kp2d_1 - kp2d1) * kp2d1_conf) ** 2)
        reproj_loss_2 = torch.mean(((reproj_kp2d_2 - kp2d2) * kp2d2_conf) ** 2)
        kp3d_smooth_loss = torch.mean(torch.norm(kp3d[:-1] - kp3d[1:], dim=-1))
        lhip = kp3d[:, 11, :]
        rhip = kp3d[:, 12, :]
        pelvis = (lhip + rhip) / 2
        pelvis_smooth_loss = torch.mean(torch.norm(pelvis[1:] - pelvis[:-1], dim=-1))
        
        local_stationary_loss = torch.tensor(0.0)
        if use_local_stationary and joint_stationary_masks is not None:
            coco_indices = {
                'left_wrist': 9,
                'right_wrist': 10, 
                'left_ankle': 15,
                'right_ankle': 16
            }
            
            for joint_name, stationary_mask in joint_stationary_masks.items():
                if joint_name == 'pelvis':
                    # pelvis = (left_hip + right_hip) / 2
                    joint_3d = pelvis
                elif joint_name in coco_indices:
                    joint_3d = kp3d[:, coco_indices[joint_name]]
                else:
                    continue
                
                stationary_tensor = torch.from_numpy(stationary_mask).float()
                
                if len(joint_3d) > 1:
                    joint_diff = torch.norm(joint_3d[1:] - joint_3d[:-1], dim=-1) ** 2
                    weights = stationary_tensor[1:] * stationary_smooth_weight + 1.0
                    weighted_smooth = joint_diff * weights
                    local_stationary_loss += weighted_smooth.mean()
        
        total_loss = 10 * reproj_loss_1 + 10 * reproj_loss_2 #+ 0.01 * kp3d_smooth_loss + 0.01 * pelvis_smooth_loss + 0.01 * local_stationary_loss

        total_loss.backward()
        optimizer.step()    

        if iter % 10 == 0 or iter == num_iters - 1:
            log_str = f"Iteration {iter + 1}/{num_iters}, Loss: {total_loss.item():.6f}, "
            log_str += f"Reproj Loss: {reproj_loss_1.item() + reproj_loss_2.item():.6f}, "
            log_str += f"kp3d Smooth Loss: {kp3d_smooth_loss.item():.6f}, "
            log_str += f"pelvis Smooth Loss: {pelvis_smooth_loss.item():.6f}"
            if use_local_stationary and joint_stationary_masks is not None:
                log_str += f", Local Stationary Loss: {local_stationary_loss.item():.6f}"
            print(log_str)

    return kp3d.detach()


def smplify_optimization(
    smpl_model,  
    smpl_params, 
    kp3d,  
    kp2d1,
    kp2d2,
    kp2d1_conf,
    kp2d2_conf, 
    kp3d_conf,  
    R1, 
    T1,  
    K1,
    R2,
    T2,
    K2,  
    H,
    W,
    device,
    stages=[],
):
    smpl_params = {k: v.to(device) for k, v in smpl_params.items()}

    losses = []     

    smpl_params['global_orient'] = rotmat_to_aa(smpl_params['global_orient'])
    smpl_params['body_pose'] = rotmat_to_aa(smpl_params['body_pose'])
    init_body_pose = smpl_params['body_pose'].clone()

    for key in ['global_orient', 'body_pose', 'betas', 'transl']:
        if key in smpl_params:
            smpl_params[key] = smpl_params[key].to(device)
        else:
            raise ValueError(f"Key {key} not found in smpl_params.")

    kp3d = torch.from_numpy(kp3d).to(device)
    kp3d_conf = torch.from_numpy(kp3d_conf).to(device)
    kp2d1 = torch.from_numpy(kp2d1).to(device)
    kp2d1_conf = torch.from_numpy(kp2d1_conf).to(device)
    kp2d2 = torch.from_numpy(kp2d2).to(device)
    kp2d2_conf = torch.from_numpy(kp2d2_conf).to(device)
    K1 = torch.from_numpy(K1).to(device)
    K2 = torch.from_numpy(K2).to(device)

    for stage in stages:
        reproj_loss_weight1 = stage.get('reproj_loss_weight1', .0)
        reproj_loss_weight2 = stage.get('reproj_loss_weight2', .0)
        kp3d_loss_weight = stage.get('kp3d_loss_weight', .0)
        kp3d_smooth_loss_weight = stage.get('kp3d_smooth_loss_weight', .0)
        verts_smooth_loss_weight = stage.get('verts_smooth_loss_weight', .0)
        motion_smooth_loss_weight = stage.get('motion_smooth_loss_weight', .0)
        transl_smooth_loss_weight = stage.get('transl_smooth_loss_weight', .0)
        regularization_loss_weight = stage.get('regularization_loss_weight', .0)
        motion_accel_loss_weight = stage.get('motion_accel_loss_weight', .0)
        kp3d_accel_loss_weight = stage.get('kp3d_accel_loss_weight', .0)
        transl_accel_loss_weight = stage.get('transl_accel_loss_weight', .0)
        num_iters = stage.get('num_iters', 100)
        lr = stage.get('lr', 1e-2)

        for key in ['global_orient', 'body_pose', 'betas', 'transl']:
            if key in stage['active_params']:
                smpl_params[key].requires_grad_(True)
            else:
                smpl_params[key].requires_grad_(False)

        optimizer = torch.optim.Adam(
            [smpl_params['global_orient'], smpl_params['body_pose'], smpl_params['betas'], smpl_params['transl']],
            lr=lr,
        )

        for iter in range(num_iters):
            optimizer.zero_grad()

            smpl_out = smpl_model(betas=smpl_params['betas'], body_pose=aa_to_rotmat(smpl_params['body_pose']), transl=smpl_params['transl'], global_orient=aa_to_rotmat(smpl_params['global_orient']), pose2rot=False)

            smpl_joints = smpl_out.joints
            smpl_verts = smpl_out.vertices
            pred_kp3d = smpl_joints[:, BMODEL.SMPL54_to_COCO]

            if reproj_loss_weight1 > 0:
                reproj_kp2d_1 = project_kp3d_to_2d(K1, R1, T1, pred_kp3d)

                reproj_kp2d_1[..., 0] /= W
                reproj_kp2d_1[..., 1] /= H
                reproj_loss_1 = torch.mean(((reproj_kp2d_1 - kp2d1) * kp2d1_conf) ** 2)
                reproj_loss1 = reproj_loss_1.mean() * reproj_loss_weight1
            else:
                reproj_loss1 = torch.tensor(0.0).to(device)
            
            if reproj_loss_weight2 > 0:
                reproj_kp2d_2 = project_kp3d_to_2d(K2, R2, T2, pred_kp3d)
                reproj_kp2d_2[..., 0] /= W
                reproj_kp2d_2[..., 1] /= H
                reproj_loss_2 = torch.mean(((reproj_kp2d_2 - kp2d2) * kp2d2_conf) ** 2)
                reproj_loss2 = reproj_loss_2.mean() * reproj_loss_weight2
            else:
                reproj_loss2 = torch.tensor(0.0).to(device)

            if kp3d_loss_weight > 0:
                kp3d_loss = ((pred_kp3d - kp3d) ** 2).sum(dim=-1) 
                kp3d_loss = (kp3d_loss * kp3d_conf.squeeze()).mean()
                kp3d_loss = kp3d_loss.mean() * kp3d_loss_weight
            else:
                kp3d_loss = torch.tensor(0.0).to(device)
        
            
            if kp3d_smooth_loss_weight > 0:
                kp3d_smooth_loss = (smpl_joints[1:] - smpl_joints[:-1]) ** 2
                kp3d_smooth_loss = kp3d_smooth_loss.mean() * kp3d_smooth_loss_weight
            else:
                kp3d_smooth_loss = torch.tensor(0.0).to(device)

            if verts_smooth_loss_weight > 0:
                verts_smooth_loss = (smpl_verts[1:] - smpl_verts[:-1]) ** 2
                verts_smooth_loss = verts_smooth_loss.mean() * verts_smooth_loss_weight
            else:
                verts_smooth_loss = torch.tensor(0.0).to(device)

            if motion_smooth_loss_weight > 0:
                body_pose_smooth = ((smpl_params['body_pose'][1:] - smpl_params['body_pose'][:-1]) ** 2).mean()
                motion_smooth_loss = body_pose_smooth * motion_smooth_loss_weight
            else:
                motion_smooth_loss = torch.tensor(0.0).to(device)

            if transl_smooth_loss_weight > 0:
                transl_smooth_loss = ((smpl_params['transl'][1:] - smpl_params['transl'][:-1]) ** 2).mean() * transl_smooth_loss_weight
            else:
                transl_smooth_loss = torch.tensor(0.0).to(device)

            if regularization_loss_weight > 0:
                regularization_loss = (smpl_params['body_pose'] - init_body_pose) ** 2
                regularization_loss = regularization_loss.mean() * regularization_loss_weight
            else:
                regularization_loss = torch.tensor(0.0).to(device)

            if motion_accel_loss_weight > 0 and smpl_joints.shape[0] > 2:
                body_pose_accel = smpl_params['body_pose'][2:] - 2*smpl_params['body_pose'][1:-1] + smpl_params['body_pose'][:-2]
                motion_accel_loss = (body_pose_accel ** 2).mean() 
                motion_accel_loss = motion_accel_loss * motion_accel_loss_weight
            else:
                motion_accel_loss = torch.tensor(0.0).to(device)
            
            if transl_accel_loss_weight > 0:
                transl_accel_loss = ((smpl_params['transl'][2:] - smpl_params['transl'][:-2]) ** 2).mean() * transl_accel_loss_weight
            else:
                transl_accel_loss = torch.tensor(0.0).to(device)

            if kp3d_accel_loss_weight > 0 and pred_kp3d.shape[0] > 2:
                kp3d_accel = pred_kp3d[2:] - 2*pred_kp3d[1:-1] + pred_kp3d[:-2]
                kp3d_accel_loss = (kp3d_accel ** 2).mean() * kp3d_accel_loss_weight
            else:
                kp3d_accel_loss = torch.tensor(0.0).to(device)

            total_loss = reproj_loss1 + reproj_loss2 + kp3d_loss + kp3d_smooth_loss + verts_smooth_loss + motion_smooth_loss + transl_smooth_loss + regularization_loss + motion_accel_loss + kp3d_accel_loss + transl_accel_loss
            losses.append(total_loss.item())

            total_loss.backward()
            optimizer.step()

            if (iter + 1) % 10 == 0 or iter == 0:
                log_str = f"Stage {stages.index(stage) + 1}, Iter {iter + 1}/{num_iters}, "
                log_str += f"Total Loss: {total_loss.item():.5f}, "
                if reproj_loss_weight1 > 0:
                    log_str += f"Reproj Loss: {reproj_loss1.item():.5f}, "
                if reproj_loss_weight2 > 0:
                    log_str += f"Reproj Loss: {reproj_loss2.item():.5f}, "
                if kp3d_loss_weight > 0:
                    log_str += f"kp3d Loss: {kp3d_loss.item():.5f}, "
                if kp3d_smooth_loss_weight > 0:
                    log_str += f"kp3d Smooth Loss: {kp3d_smooth_loss.item():.5f}, "
                if verts_smooth_loss_weight > 0:
                    log_str += f"Verts Smooth Loss: {verts_smooth_loss.item():.5f}, "
                if motion_smooth_loss_weight > 0:
                    log_str += f"Motion Smooth Loss: {motion_smooth_loss.item():.5f}, "
                if transl_smooth_loss_weight > 0:
                    log_str += f"Transl Smooth Loss: {transl_smooth_loss.item():.5f}, "
                if regularization_loss_weight > 0:
                    log_str += f"Regular Loss: {regularization_loss.item():.5f}, "
                if motion_accel_loss_weight > 0:
                    log_str += f"Motion Accel Loss: {motion_accel_loss.item():.5f}, "
                if kp3d_accel_loss_weight > 0:
                    log_str += f"KP3D Accel Loss: {kp3d_accel_loss.item():.5f}, "
                if transl_accel_loss_weight > 0:
                    log_str += f"Transl Accel Loss: {transl_accel_loss.item():.5f}, "
                print(log_str)
    smpl_params['smpl_joints'] = smpl_joints
    smpl_params['kp3d'] = kp3d
    return smpl_params, losses

def post_process_transl_with_stationary_detection(transl_tensor, 
                                                   velocity_threshold=0.3, 
                                                   min_segment_length=5,
                                                   num_iters=100,
                                                   lr=1e-2,
                                                   origin_loss_weight=1.0,
                                                   smooth_loss_weight=10.0):
    """
    Optimize transl with stationary-segment detection and segment-aware losses.
    For stationary segments: replace origin targets with segment mean and keep smooth loss.
    For moving segments: use both origin and smooth losses.
    Args:
        transl_tensor: [T, 3] torch tensor
        velocity_threshold: velocity threshold，below this value is considered stationary
        min_segment_length: minimum stationary segment length
        num_iters: number of optimization iterations
        lr: learning rate
        origin_loss_weight: origin transl similarity loss weight
        smooth_loss_weight: smoothing loss weight
    """
    device = transl_tensor.device
    
    transl_np = transl_tensor.detach().cpu().numpy()
    velocities = np.diff(transl_np, axis=0)  # [T-1, 3]
    speeds = np.linalg.norm(velocities, axis=1)  # [T-1,]
    speeds = np.concatenate([[0], speeds])  # [T,]

    is_stationary = (speeds * 30 < velocity_threshold) & (transl_np[:, 2] < (transl_np[:, 2].max()-0.4))
    
    stationary_segments = []
    start_idx = None
    for i, stationary in enumerate(is_stationary):
        if stationary and start_idx is None:
            start_idx = i
        elif not stationary and start_idx is not None:
            if i - start_idx >= min_segment_length:
                stationary_segments.append((start_idx, i-1))
            start_idx = None
    if start_idx is not None and len(is_stationary) - start_idx >= min_segment_length:
        stationary_segments.append((start_idx, len(is_stationary)-1))
    
    print(f"Detected {len(stationary_segments)} stationary segments")
    
    if not stationary_segments:
        return transl_tensor
    
    origin_transl = transl_tensor.clone().detach()
    for start_idx, end_idx in stationary_segments:
        segment_mean = origin_transl[start_idx:end_idx + 1].mean(dim=0, keepdim=True)
        origin_transl[start_idx:end_idx + 1] = segment_mean
    
    transl_optim = transl_tensor.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([transl_optim], lr=lr)
    
    for iter in range(num_iters):
        optimizer.zero_grad()
        
        origin_loss = torch.mean((transl_optim - origin_transl) ** 2) * origin_loss_weight
        
        smooth_loss = torch.mean((transl_optim[1:] - transl_optim[:-1]) ** 2) * smooth_loss_weight
        
        total_loss = origin_loss + smooth_loss
        
        total_loss.backward()
        optimizer.step()
        
        if iter % 20 == 0 or iter == num_iters - 1:
            print(f"  Iter {iter + 1}/{num_iters}, Loss: {total_loss.item():.6f}, "
                  f"Origin Loss: {origin_loss.item():.6f}, Smooth Loss: {smooth_loss.item():.6f}")
    
    return transl_optim.detach()


def post_process_body_pose_smoothing(body_pose_tensor, method='slerp', **kwargs):
    """
    Smooth body-pose sequences (supports slerp and Gaussian filtering).
    Args:
        body_pose_tensor: [T, N, 3, 3] rotation-matrix tensor
        method: 'slerp', 'gaussian', etc.
        kwargs: other parameters
    Returns:
        smoothed_body_pose: [T, N, 3, 3] smoothed rotation-matrix tensor
    """
    T, N = body_pose_tensor.shape[:2]
    device = body_pose_tensor.device
    body_pose_np = body_pose_tensor.detach().cpu().numpy()  # (T, N, 3, 3)

    smoothed_pose = np.zeros_like(body_pose_np)

    if method == 'slerp':
        window_size = kwargs.get('window_size', 5)
        for j in range(N):
            rotations = R.from_matrix(body_pose_np[:, j])  # (T,)
            smoothed_rotations = []
            for i in range(T):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(T, i + window_size // 2 + 1)
                if end_idx - start_idx < 2:
                    smoothed_rotations.append(rotations[i])
                    continue
                times = np.linspace(0, 1, end_idx - start_idx)
                slerp = Slerp(times, rotations[start_idx:end_idx])
                center_time = (i - start_idx) / (end_idx - start_idx - 1) if end_idx - start_idx > 1 else 0.5
                smoothed_rot = slerp(center_time)
                smoothed_rotations.append(smoothed_rot)
            smoothed_rotmats = R.from_quat([r.as_quat() for r in smoothed_rotations]).as_matrix()
            smoothed_pose[:, j] = smoothed_rotmats  # (T, 3, 3)
    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        for j in range(N):
            rotations = R.from_matrix(body_pose_np[:, j])
            quats = rotations.as_quat()  # (T, 4)
            smoothed_quats = np.zeros_like(quats)
            for k in range(4):
                smoothed_quats[:, k] = gaussian_filter1d(quats[:, k], sigma=sigma)
            smoothed_quats = smoothed_quats / np.linalg.norm(smoothed_quats, axis=1, keepdims=True)
            smoothed_rotmats = R.from_quat(smoothed_quats).as_matrix()
            smoothed_pose[:, j] = smoothed_rotmats
    else:
        smoothed_pose = body_pose_np

    return torch.from_numpy(smoothed_pose).to(device).float()


def post_process_global_orient_smoothing(global_orient_tensor, method='slerp', **kwargs):
    """
    Post-process smoothing for global_orient (e.g., slerp).
    Args:
        global_orient_tensor: [T, 3, 3] rotation-matrix tensor
        method: 'slerp', 'gaussian', etc.
    """

    rotmats = global_orient_tensor.detach().cpu().numpy()
    rotations = R.from_matrix(rotmats)
    
    if method == 'slerp':
        window_size = kwargs.get('window_size', 5)
        smoothed_rotations = []
        
        for i in range(len(rotations)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(rotations), i + window_size // 2 + 1)
            
            if end_idx - start_idx < 2:
                smoothed_rotations.append(rotations[i])
                continue
                
            times = np.linspace(0, 1, end_idx - start_idx)
            slerp = Slerp(times, rotations[start_idx:end_idx])
            
            center_time = (i - start_idx) / (end_idx - start_idx - 1) if end_idx - start_idx > 1 else 0.5
            smoothed_rot = slerp(center_time)
            smoothed_rotations.append(smoothed_rot)
            
        smoothed_rotations = R.from_quat([r.as_quat() for r in smoothed_rotations])
        
    elif method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        sigma = kwargs.get('sigma', 1.0)
        
        quats = rotations.as_quat()  # [T, 4]
        smoothed_quats = np.zeros_like(quats)
        
        for i in range(4):
            smoothed_quats[:, i] = gaussian_filter1d(quats[:, i], sigma=sigma)
        
        smoothed_quats = smoothed_quats / np.linalg.norm(smoothed_quats, axis=1, keepdims=True)
        smoothed_rotations = R.from_quat(smoothed_quats)
    
    else:
        smoothed_rotations = rotations
    
    smoothed_rotmats = smoothed_rotations.as_matrix()
    return torch.from_numpy(smoothed_rotmats).to(global_orient_tensor.device).float()

# Removed get_visible_points function as pointcloud loss is no longer used

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize processed frames and concatenate videos."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        default="",
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--optim_kp3d",
        action="store_true",
        help="Whether to use skeleton to optimized camera."
    )
    parser.add_argument(
        "--smooth_kp2d",
        action="store_true",
        help="Whether to smooth 3d keypoints."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for processing.",
    )
    parser.add_argument(
        "--pcscale",
        type=int,
        default=4,
        help="Scale for point cloud.",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=["male", "female", "neutral"],
    )

    parser.add_argument(
        "--post_smooth",
        action="store_true", 
        help="Apply post-processing smoothing after optimization.",
    )
    parser.add_argument(
        "--reproj",
        action="store_true",
        help="Whether to use reprojection loss.",
    )
    parser.add_argument(
        "--use_prior",
        action="store_true",
        help="Whether to use prior loss.",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Whether to use acceleration smoothing.",
    )
    parser.add_argument(
        "--use_kp3d",
        action="store_true",
        help="Whether to use kp3d loss.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    cameras1 = np.load(os.path.join(args.input_folder, "v1", "cameras.npz"))
    cameras2 = np.load(os.path.join(args.input_folder, "v2", "cameras.npz"))

    K1 = cameras1['K'][:3, :3]
    K2 = cameras2['K'][:3, :3]

    R1, T1 = cameras1['R'], cameras1['T']
    R1_w2c, T1_w2c = convert_world_cam(R1, T1)
    R2, T2 = cameras2['R'], cameras2['T']
    R2_w2c, T2_w2c = convert_world_cam(R2, T2)

    T1_w2c = T1_w2c.squeeze()
    T2_w2c = T2_w2c.squeeze()

    ######################################################
    smpl_params1 = np.load(os.path.join(args.input_folder, "v1", "smpl_params.npz"), allow_pickle=True)
    smpl_params2 = np.load(os.path.join(args.input_folder, "v2", "smpl_params.npz"), allow_pickle=True)
    smpl_params1 = dict(smpl_params1)
    smpl_params2 = dict(smpl_params2)
    smpl_params1['keypoints2d'] = smpl_params1['keypoints']
    smpl_params2['keypoints2d'] = smpl_params2['keypoints']

    kp2d_1 = smpl_params1['keypoints2d'][..., :2].copy()
    kp2d_2 = smpl_params2['keypoints2d'][..., :2].copy()
    h = 960
    w = 720

    if args.smooth_kp2d:
        kp2d_1 = smooth_and_interpolate(kp2d_1, smpl_params1['keypoints2d'][..., 2], confidence_threshold=0.5, sigma=2)
        kp2d_2 = smooth_and_interpolate(kp2d_2, smpl_params2['keypoints2d'][..., 2], confidence_threshold=0.5, sigma=2)
    kp3d = triangulate_sequence(K1, K2, R1_w2c, T1_w2c, R2_w2c, T2_w2c, kp2d_1, kp2d_2)

    kp2d1_conf = smpl_params1['keypoints2d'][..., 2:3]
    kp2d2_conf = smpl_params2['keypoints2d'][..., 2:3]  
    kp2d1_conf = (kp2d1_conf > 0.6) * kp2d1_conf
    kp2d2_conf = (kp2d2_conf > 0.6) * kp2d2_conf
    if args.optim_kp3d:
        kp3d = kp3d_smoothing(kp3d, get_coco_bone_skeleton(), kp2d_1, kp2d1_conf, K1, R1_w2c, T1_w2c, kp2d_2, kp2d2_conf, K2, R2_w2c, T2_w2c, h, w, num_iters=100, lr=1e-2).cpu().numpy()
    kp3d_conf = (kp2d1_conf * kp2d2_conf)
    kp3d_conf = (kp3d_conf > 0.36) * kp3d_conf

    kp3d_mesh = create_skeleton_mesh(kp3d[::100], get_coco_body_skeleton(), cylinder_radius=0.005)
    kp3d_mesh.export(os.path.join(args.input_folder, "kp3d.ply"))
    # visualize_kp3d_to_video(kp3d, get_coco_skeleton(), video_path=f"{args.input_folder}/kp3d.mp4", image_size=(480, 480), fps=30, title="3D Keypoints Animation")

    body_model = SMPL(
            model_path=BMODEL.FLDR,
            gender=args.gender,
            extra_joints_regressor=BMODEL.JOINTS_REGRESSOR_EXTRA,
            create_transl=False).to(device)
    
    smpl_params1 = dict(smpl_params1)
    smpl_params_init = copy.deepcopy(smpl_params1)

    pred_cam = smpl_params_init['pred_cam'].reshape(-1, 3)

    s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
    f = K1[0, 0] / (1920 / 2)
    tz = f / s
    transl = np.stack([tx, ty, tz], axis=1)
    global_orient = smpl_params_init['global_orient']

    smpl_params_init['global_orient'] = global_orient[:, None]
    smpl_params_init['transl'] = transl
    
    for k in smpl_params_init:
        smpl_params_init[k] = torch.from_numpy(smpl_params_init[k]).to(device)

    smpl_params_init['global_orient'] = smpl_params_init['global_orient'].view(-1, 1, 3, 3)
    smpl_params_init['transl'] = smpl_params_init['transl'].view(-1, 3)
    smpl_params_init['body_pose'] = smpl_params_init['body_pose'].view(-1, 23, 3, 3)

    R1_w2c = torch.from_numpy(R1_w2c).to(device)
    T1_w2c = torch.from_numpy(T1_w2c).to(device)
    R2_w2c = torch.from_numpy(R2_w2c).to(device)
    T2_w2c = torch.from_numpy(T2_w2c).to(device)
    global_orient_world, transl_world = eliminate_RT(torch.from_numpy(R1).to(device), torch.from_numpy(T1).to(device), body_model, pose2rot=False, **smpl_params_init)
    smpl_params_init['global_orient'] = global_orient_world
    smpl_params_init['transl'] = transl_world

    # Note: Removed mask and pointcloud processing as pointcloud_loss is not used in optimization
    # Masks and point clouds were only used for visualization/debugging

    
    stages = [
        {'active_params': ['global_orient', 'transl'],
         'kp3d_loss_weight': 1.0 if args.use_kp3d else 0.0,
         'kp3d_smooth_loss_weight': 1.0 if args.smooth else 0.0,
         'num_iters': 200,
         'lr': 1e-2},
        {'active_params': ['global_orient', 'betas', 'body_pose', 'transl'],
         'kp3d_loss_weight': 2.0 if args.use_kp3d else 0.0,
         'reproj_loss_weight1': 2.0 if args.reproj else 0.0,
         'reproj_loss_weight2': 2.0 if args.reproj else 0.0,
         'kp3d_smooth_loss_weight': 1.0 if args.smooth else 0.0,
         'verts_smooth_loss_weight': 1.0 if args.smooth else 0.0,
         'motion_smooth_loss_weight': 1.0 if args.smooth else 0.0,
         'transl_smooth_loss_weight': 1.0 if args.smooth else 0.0,
         'regularization_loss_weight': 10.0 if args.use_prior else 0.0,
         'motion_accel_loss_weight': 5 if args.smooth else 0,
         'kp3d_accel_loss_weight': 5 if args.smooth else 0,
         'num_iters': 300,
         'lr': 1e-2},
        # {'active_params': ['global_orient', 'transl'],
        #  'kp3d_loss_weight': 5.0 if args.use_kp3d else 0.0,
        #  'reproj_loss_weight1': 1.0 if args.reproj else 0.0,
        #  'reproj_loss_weight2': 1.0 if args.reproj else 0.0,
        #  'verts_smooth_loss_weight': 5.0 if args.smooth else 0,
        #  'kp3d_smooth_loss_weight': 10.0 if args.smooth else 0,
        #  'transl_smooth_loss_weight': 5.0 if args.smooth else 0,
        #  'num_iters': 100,
        #  'lr': 1e-2},
    ]

    with torch.amp.autocast('cuda', enabled=False): 
        smpl_params_optim, optim_losses = smplify_optimization(
            smpl_model=body_model,
            smpl_params=smpl_params_init,
            kp3d=kp3d,
            kp2d1=kp2d_1,
            kp2d2=kp2d_2,
            kp2d1_conf=kp2d1_conf,
            kp2d2_conf=kp2d2_conf,
            kp3d_conf=kp3d_conf,
            R1=R1_w2c,
            T1=T1_w2c,
            K1=K1,
            R2=R2_w2c,
            T2=T2_w2c,
            K2=K2,
            H=h,
            W=w,
            device=device,
            stages=stages)

    smpl_params_optim['global_orient'] = aa_to_rotmat(smpl_params_optim['global_orient'])
    smpl_params_optim['body_pose'] = aa_to_rotmat(smpl_params_optim['body_pose'])

    if args.post_smooth:
        print("Applying post-processing smoothing...")
        smpl_params_optim['transl'] = post_process_transl_with_stationary_detection(
            smpl_params_optim['transl'],
            velocity_threshold=0.2,
            min_segment_length=10,
            lr=1e-2,
            num_iters=100,
            origin_loss_weight=1.0,
            smooth_loss_weight=10.0
        )
        
        smpl_params_optim['global_orient'] = post_process_global_orient_smoothing(
            smpl_params_optim['global_orient'][:, 0],
            method='slerp',
            window_size=9
        ).unsqueeze(1)

        smpl_params_optim['body_pose'] = post_process_body_pose_smoothing(
            smpl_params_optim['body_pose'],
            method='slerp',
            window_size=9
        )


    # part_indices = [SMPL_JOINT_NAMES[1:].index(name) for name in ['left_ankle', 'right_ankle', 'left_foot', 'right_foot']]
    # smpl_params_optim['body_pose'][:, part_indices] = smpl_params_init['body_pose'][:, part_indices]

    smpl_params_optim['global_orient'] = batch_ortho_rotmat(smpl_params_optim['global_orient'])
    smpl_params_optim['body_pose'] = batch_ortho_rotmat(smpl_params_optim['body_pose'])

    smpl_params_optim['keypoints3d'] = kp3d
    smpl_params_optim['keypoints3d_conf'] = kp3d_conf
    smpl_params_optim['keypoints2d_v1'] = smpl_params1['keypoints2d'][..., :2]
    smpl_params_optim['keypoints2d_v2'] = smpl_params2['keypoints2d'][..., :2]
    smpl_params_optim['keypoints2d_v1_conf'] = kp2d1_conf
    smpl_params_optim['keypoints2d_v2_conf'] = kp2d2_conf
    smpl_params_optim['bbox_xyxy1'] = smpl_params1['bbox_xyxy']
    smpl_params_optim['bbox_xyxy2'] = smpl_params2['bbox_xyxy']
    smpl_params_optim['K1'] = K1
    smpl_params_optim['K2'] = K2
    smpl_params_optim['R1'] = R1
    smpl_params_optim['R2'] = R2
    smpl_params_optim['T1'] = T1
    smpl_params_optim['T2'] = T2
    if len(optim_losses) > 0:
        smpl_params_optim['optim_total_loss'] = np.float32(optim_losses[-1])
        print(f"[optim_motion] Final total loss: {optim_losses[-1]:.6f}")
    else:
        smpl_params_optim['optim_total_loss'] = np.float32(np.nan)
        print("[optim_motion] Warning: No optimization loss recorded.")
    smpl_params_optim.pop('vertices', None)

    for k in smpl_params_optim:
        if isinstance(smpl_params_optim[k], torch.Tensor):
            smpl_params_optim[k] = smpl_params_optim[k].detach().cpu().numpy()

    np.savez(os.path.join(args.input_folder, "optim_params.npz"), **smpl_params_optim)
