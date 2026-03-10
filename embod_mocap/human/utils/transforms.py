import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R, Slerp
from t3drender.transforms import aa_to_rotmat, rotmat_to_aa, quat_to_rotmat, rotmat_to_quat
from t3drender.cameras.convert_convention import convert_world_view
from typing import Union, Optional
from torch.nn.functional import normalize


def combine_RT(R, T):
    if R.ndim == 2:
        R = R[None]

    batch_size = R.shape[0]
    T = T.view(batch_size, 3, 1)
    RT = torch.zeros(batch_size, 4, 4).to(R.device)
    RT[:, 3, 3] = 1
    RT[:, :3, :3] = R
    RT[:, :3, 3:] = T
    return RT

def matrix_tranform(matrix, points):
    # points: NxVx3
    # matrix: Nx4x4
    ones = torch.ones(points.shape[0], points.shape[1], 1, device=points.device)
    points_homogeneous = torch.cat([points, ones], dim=-1)  # NxVx4
    points = torch.bmm(points_homogeneous, matrix.transpose(1, 2))
    points = points[..., :3] / points[..., 3:4]
    return points

def transform_RT(R, T, points):
    # points: NxVx3
    # R: Nx3x3 (rotation matrices)
    # T: Nx3 (translation vectors)
    ones = torch.ones(points.shape[0], points.shape[1], 1, device=points.device)  # NxVx1
    points_homogeneous = torch.cat([points, ones], dim=-1)  # NxVx4
    T_homogeneous = torch.eye(4, device=points.device).expand(points.shape[0], -1, -1)  # Nx4x4
    T_homogeneous[:, :3, :3] = R  # Set rotation matrix
    T_homogeneous[:, :3, 3] = T  # Set translation vector
    transformed_points = torch.bmm(points_homogeneous, T_homogeneous.transpose(1, 2))  # NxVx4
    transformed_points = transformed_points[..., :3] / transformed_points[..., 3:4]  # Divide by the homogeneous coordinate
    return transformed_points

def quaternion_extrapolate(q1, q2, factor):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    q_extrapolated = q1 + factor * (q2 - q1)
    q_extrapolated = q_extrapolated / np.linalg.norm(q_extrapolated)
    return q_extrapolated


def interpolate_RT(RT, source_frame_idx, target_frame_idx):
    is_torch = isinstance(RT, torch.Tensor)
    device = RT.device if is_torch else None
    if is_torch:
        RT = RT.detach().cpu().numpy()
    
    # Extract rotation matrices and translation vectors
    R_matrices = RT[:, :3, :3]
    T_vectors = RT[:, :3, 3]
    
    # Convert rotation matrices to quaternions
    quaternions = R.from_matrix(R_matrices).as_quat()
    
    # Ensure quaternion consistency (same sign) for interpolation
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[i], quaternions[i - 1]) < 0:
            quaternions[i] = -quaternions[i]
    
    # Create SLERP object for interpolation
    slerp = Slerp(source_frame_idx, R.from_quat(quaternions))
    
    # Prepare output container
    target_times = np.array(target_frame_idx)
    interpolated_quaternions = []
    interpolated_translations = np.empty((len(target_frame_idx), 3))
    
    # Check for overlapping indices and handle interpolation
    for i, t in enumerate(target_times):
        if t in source_frame_idx:  # Directly use source value if index overlaps
            idx = np.where(source_frame_idx == t)[0][0]
            interpolated_quaternions.append(quaternions[idx])
            interpolated_translations[i] = T_vectors[idx]
        elif t < source_frame_idx[0]:  # Extrapolate before the first frame
            t1, t2 = source_frame_idx[0], source_frame_idx[1]
            q1, q2 = quaternions[0], quaternions[1]
            factor = (t - t1) / (t2 - t1)
            interpolated_quaternions.append(quaternion_extrapolate(q1, q2, factor))
            v1, v2 = T_vectors[0], T_vectors[1]
            interpolated_translations[i] = v1 + factor * (v2 - v1)
        elif t > source_frame_idx[-1]:  # Extrapolate after the last frame
            t1, t2 = source_frame_idx[-2], source_frame_idx[-1]
            q1, q2 = quaternions[-2], quaternions[-1]
            factor = (t - t2) / (t2 - t1)
            interpolated_quaternions.append(quaternion_extrapolate(q2, q1, factor))
            v1, v2 = T_vectors[-2], T_vectors[-1]
            interpolated_translations[i] = v2 + factor * (v2 - v1)
        else:  # Interpolate within the range
            interpolated_quaternions.append(slerp(t).as_quat())
            for j in range(3):
                interpolated_translations[i, j] = np.interp(
                    t, source_frame_idx, T_vectors[:, j]
                )
    
    # Combine interpolated rotations and translations into RT matrices
    interpolated_RT = []
    for i in range(len(target_frame_idx)):
        R_interp = R.from_quat(interpolated_quaternions[i]).as_matrix()
        T_interp = interpolated_translations[i]
        RT_interp = np.eye(4).astype(np.float32)
        RT_interp[:3, :3] = R_interp
        RT_interp[:3, 3] = T_interp
        interpolated_RT.append(RT_interp)
    interpolated_RT = np.stack(interpolated_RT, axis=0)
    
    if is_torch:
        interpolated_RT = torch.from_numpy(interpolated_RT).to(device)

    return interpolated_RT

def quaternion_extrapolate(q1, q2, factor):
    """
    Extrapolate quaternion q1 towards q2 by a given factor.
    """
    q1 = np.array(q1)
    q2 = np.array(q2)
    delta_q = q2 - q1
    extrapolated_q = q1 + factor * delta_q
    return extrapolated_q / np.linalg.norm(extrapolated_q)
# def interpolate_RT(RT, source_frame_idx, target_frame_idx):
#     is_torch = isinstance(RT, torch.Tensor)
#     device = RT.device if is_torch else None
#     if is_torch:
#         RT = RT.detach().cpu().numpy()
    
#     R_matrices = [RT[i, :3, :3] for i in range(len(RT))]
#     T_vectors = [RT[i, :3, 3] for i in range(len(RT))]
    
#     quaternions = R.from_matrix(R_matrices).as_quat()  

#     slerp = Slerp(source_frame_idx, R.from_quat(quaternions))
    
#     target_times = np.array(target_frame_idx)
#     interpolated_quaternions = []
    
#     for t in target_times:
#         if t < source_frame_idx[0]:
#             t1, t2 = source_frame_idx[0], source_frame_idx[1]
#             q1, q2 = quaternions[0], quaternions[1]
#             factor = (t - t1) / (t2 - t1)
#             q_extrapolated = quaternion_extrapolate(q1, q2, factor)
#             interpolated_quaternions.append(q_extrapolated)
#         elif t > source_frame_idx[-1]:
#             t1, t2 = source_frame_idx[-2], source_frame_idx[-1]
#             q1, q2 = quaternions[-2], quaternions[-1]
#             factor = (t - t2) / (t2 - t1)
#             q_extrapolated = quaternion_extrapolate(q2, q1, factor)
#             interpolated_quaternions.append(q_extrapolated)
#         else:
#             interpolated_quaternions.append(slerp(t).as_quat())
    
#     T_vectors = np.array(T_vectors)
#     interpolated_translations = np.empty((len(target_frame_idx), 3))
#     for i in range(3):
#         interpolated_translations[:, i] = np.interp(
#             target_times,
#             source_frame_idx,
#             T_vectors[:, i]
#         )
    
#     for j, t in enumerate(target_times):
#         if t < source_frame_idx[0]: 
#             t1, t2 = source_frame_idx[0], source_frame_idx[1]
#             v1, v2 = T_vectors[0], T_vectors[1]
#             interpolated_translations[j] = v1 + (t - t1) / (t2 - t1) * (v2 - v1)
#         elif t > source_frame_idx[-1]:
#             t1, t2 = source_frame_idx[-2], source_frame_idx[-1]
#             v1, v2 = T_vectors[-2], T_vectors[-1]
#             interpolated_translations[j] = v2 + (t - t2) / (t2 - t1) * (v2 - v1)
    
#     interpolated_RT = []
#     for i in range(len(target_frame_idx)):
#         R_interp = R.from_quat(interpolated_quaternions[i]).as_matrix()
#         T_interp = interpolated_translations[i]
#         RT_interp = np.eye(4).astype(np.float32)
#         RT_interp[:3, :3] = R_interp
#         RT_interp[:3, 3] = T_interp
#         interpolated_RT.append(RT_interp)
#     interpolated_RT = np.stack(interpolated_RT, axis=0)
    
#     if is_torch:
#         interpolated_RT = torch.from_numpy(interpolated_RT).to(device)

#     return interpolated_RT


def slerp(
    quat_a: torch.Tensor, 
    quat_b: torch.Tensor, 
    t: torch.Tensor
) -> torch.Tensor:
    """
    Performs spherical linear interpolation (SLERP) between two quaternions.

    Args:
        quat_a: A tensor of shape (N, 4) or (1, 4), representing the first quaternion.
        quat_b: A tensor of shape (N, 4) or (1, 4), representing the second quaternion.
        t:      A tensor of shape (N, 1) or (1, 1), representing the interpolation factor (ranging from 0.0 to 1.0).

    Returns:
        A tensor of shape (N, 4) or (1, 4), representing the interpolated quaternion.
    """
    # Normalize the input quaternions
    quat_a = quat_a / (quat_a.norm(dim=-1, keepdim=True) + 1e-8)
    quat_b = quat_b / (quat_b.norm(dim=-1, keepdim=True) + 1e-8)

    # Compute the dot product
    dot = (quat_a * quat_b).sum(dim=-1, keepdim=True)

    # If dot < 0, flip one of the quaternions
    neg_mask = (dot < 0).expand_as(quat_b)
    quat_b = torch.where(neg_mask, -quat_b, quat_b)
    dot = torch.abs(dot)

    # Clamp dot to prevent numerical issues
    dot = torch.clamp(dot, max=1.0)
    omega = torch.acos(dot)  # angle between the quaternions

    eps = 1e-6
    sin_omega = torch.sqrt(1.0 - dot * dot)
    is_small_angle = sin_omega < eps

    # SLERP formula
    alpha = torch.sin((1.0 - t) * omega) / (sin_omega + eps)
    beta = torch.sin(t * omega) / (sin_omega + eps)

    # For very small angles, fall back to linear interpolation
    alpha_lin = 1.0 - t
    beta_lin = t

    alpha = torch.where(is_small_angle, alpha_lin, alpha)
    beta = torch.where(is_small_angle, beta_lin, beta)

    quat_out = alpha * quat_a + beta * quat_b
    quat_out = quat_out / (quat_out.norm(dim=-1, keepdim=True) + 1e-8)
    return quat_out


def get_extrinsic_at_frame(
    tf: float,
    sorted_vals: torch.Tensor,
    quat_list: torch.Tensor,
    T_list: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Computes a 4x4 camera extrinsic matrix by interpolating between
    predefined keyframe indices, given a target frame index tf.

    Args:
        tf:          Target frame index (float or int).
        sorted_vals: A tensor of shape (N,) containing sorted keyframe indices.
        quat_list:   A tensor of shape (N, 4) containing the quaternions associated with the keyframes.
        T_list:      A tensor of shape (N, 3) containing the translation vectors associated with the keyframes.
        device:      A PyTorch device.
        dtype:       A PyTorch dtype.

    Returns:
        A 4x4 tensor representing the interpolated extrinsic matrix.
    """
    # Number of keyframes
    n = sorted_vals.shape[0]

    # Determine the interval in which tf resides
    if tf <= sorted_vals[0]:
        idx0, idx1 = 0, 1
    elif tf >= sorted_vals[-1]:
        idx0, idx1 = n - 2, n - 1
    else:
        # Find sorted_vals[i] <= tf < sorted_vals[i+1]
        idx1 = torch.where(sorted_vals > tf)[0][0]
        idx0 = idx1 - 1

    f0 = sorted_vals[idx0].item()
    f1 = sorted_vals[idx1].item()
    alpha = (tf - f0) / (f1 - f0 + 1e-8)
    alpha = torch.tensor(alpha, device=device, dtype=dtype)

    # SLERP interpolation for rotation
    qa = quat_list[idx0:idx0+1]  # shape: (1, 4)
    qb = quat_list[idx1:idx1+1]  # shape: (1, 4)
    q_interp = slerp(qa, qb, alpha[None])  # shape: (1, 4)

    # Linear interpolation for translation
    Ta = T_list[idx0]  # shape: (3,)
    Tb = T_list[idx1]  # shape: (3,)
    T_interp = Ta * (1 - alpha) + Tb * alpha

    # Convert quaternion to rotation matrix
    R_interp = quat_to_rotmat(q_interp)[0]  # shape: (3, 3)

    extrinsic = torch.eye(4, device=device, dtype=dtype)
    extrinsic[:3, :3] = R_interp
    extrinsic[:3, 3] = T_interp
    return extrinsic


def interpolate_RT_tensor(
    RT: torch.Tensor,
    source_frame_idx: torch.Tensor,
    target_frame_idx: torch.Tensor
) -> torch.Tensor:
    """
    Given a set of camera extrinsics and corresponding source frame indices,
    interpolate the extrinsic matrices for a set of target frame indices.

    Args:
        RT:              A tensor of shape (N, 4, 4), representing the extrinsics for N keyframes.
        source_frame_idx: A tensor (or list) of shape (N,), containing keyframe indices.
        target_frame_idx: A tensor (or list) of shape (M,), containing target frame indices to be interpolated.

    Returns:
        A tensor of shape (M, 4, 4), representing the interpolated extrinsic matrices for each target frame.
    """

    # Convert list inputs to tensors and validate shapes
    if isinstance(source_frame_idx, list):
        source_frame_idx = torch.tensor(source_frame_idx, device=RT.device, dtype=RT.dtype)
    if isinstance(target_frame_idx, list):
        target_frame_idx = torch.tensor(target_frame_idx, device=RT.device, dtype=RT.dtype)

    assert source_frame_idx.shape[0] == RT.shape[0], "Mismatch between source_frame_idx and RT length."
    assert RT.shape[1:] == (4, 4), "RT must be of shape [N, 4, 4]."

    # Sort the keyframes by their indices
    sorted_vals, sorted_idx = torch.sort(source_frame_idx)
    RT_sorted = RT[sorted_idx]

    # Extract rotation matrices and translation vectors
    R_list = RT_sorted[:, :3, :3]  # shape: (N, 3, 3)
    T_list = RT_sorted[:, :3, 3]   # shape: (N, 3)

    # Convert rotation matrices to quaternions
    quat_list = rotmat_to_quat(R_list)  # shape: (N, 4)

    # Prepare output
    M = target_frame_idx.shape[0]
    out = torch.zeros((M, 4, 4), device=RT.device, dtype=RT.dtype)
    out[:, 3, 3] = 1.0

    # Sort target frame indices
    target_sorted_vals, target_sorted_idx = torch.sort(target_frame_idx)
    out_sorted = out.clone()

    # Interpolate for each target index
    for i in range(M):
        tf = target_sorted_vals[i].item()
        out_sorted[i] = get_extrinsic_at_frame(
            tf=tf,
            sorted_vals=sorted_vals,
            quat_list=quat_list,
            T_list=T_list,
            device=RT.device,
            dtype=RT.dtype
        )

    # Place interpolated results back in the original order
    out[target_sorted_idx] = out_sorted

    return out

def interpolate_smpl_rotmat_camera(
    smpl_rotmats: torch.Tensor,
    pred_cam: torch.Tensor,
    source_frame_idx: torch.Tensor,
    target_frame_idx: torch.Tensor
):
    """
    Interpolates SMPL body rotations (NxJx3x3) using SLERP and camera parameters (Nx3) using
    linear interpolation, given source and target frame indices of arbitrary length.

    Args:
        smpl_rotmats:      A tensor of shape (N, J, 3, 3), representing SMPL body part rotations.
        pred_cam:          A tensor of shape (N, 3), representing camera parameters (e.g., orthographic scale, tx, ty).
        source_frame_idx:  A tensor (or list) of shape (N, ), representing source frame indices.
        target_frame_idx:  A tensor (or list) of shape (M, ), representing target frame indices to be interpolated.

    Returns:
        out_rotmats: A tensor of shape (M, J, 3, 3) with SLERP-interpolated SMPL rotations.
        out_cam:     A tensor of shape (M, 3), with linearly interpolated camera parameters.
    """
    device = smpl_rotmats.device
    dtype = smpl_rotmats.dtype

    # Convert list inputs to tensors and validate shapes
    if isinstance(source_frame_idx, list):
        source_frame_idx = torch.tensor(source_frame_idx, device=device, dtype=dtype)
    if isinstance(target_frame_idx, list):
        target_frame_idx = torch.tensor(target_frame_idx, device=device, dtype=dtype)

    N, J, _, _ = smpl_rotmats.shape
    assert source_frame_idx.shape[0] == N, "Mismatch between source_frame_idx and smpl_rotmats length."
    assert pred_cam.shape[0] == N,        "Mismatch between pred_cam and smpl_rotmats length."

    # Sort source frames
    sorted_vals, sorted_idx = torch.sort(source_frame_idx)
    smpl_sorted = smpl_rotmats[sorted_idx]  # shape (N, J, 3, 3)
    cam_sorted = pred_cam[sorted_idx]       # shape (N, 3)

    # Convert rotation matrices to quaternions for each joint
    # We can flatten (N, J, 3, 3) into (N*J, 3, 3) to convert all at once
    smpl_sorted_flat = smpl_sorted.reshape(-1, 3, 3)  # shape (N*J, 3, 3)
    quat_sorted_flat = rotmat_to_quat(smpl_sorted_flat)  # shape (N*J, 4)

    # Prepare output
    M = target_frame_idx.shape[0]
    out_rotmats = torch.zeros((M, J, 3, 3), device=device, dtype=dtype)
    out_cam = torch.zeros((M, 3), device=device, dtype=dtype)

    # Sort target frames
    target_sorted_vals, target_sorted_idx = torch.sort(target_frame_idx)

    # Helper function to get SMPL rotation & camera at a given frame
    def get_smpl_at_frame(tf: float):
        """
        Interpolates SMPL rotations (J joints) and camera parameters at a given frame tf.
        Uses SLERP for rotation, linear interpolation for camera.

        Args:
            tf: Target frame index (float or int).

        Returns:
            A tuple (rot3x3, cam) where:
              - rot3x3 is a tensor of shape (J, 3, 3)
              - cam is a tensor of shape (3,)
        """
        # Determine interval [idx0, idx1]
        if tf <= sorted_vals[0]:
            idx0, idx1 = 0, 1
        elif tf >= sorted_vals[-1]:
            idx0, idx1 = N - 2, N - 1
        else:
            idx1 = torch.where(sorted_vals > tf)[0][0]
            idx0 = idx1 - 1

        f0 = sorted_vals[idx0].item()
        f1 = sorted_vals[idx1].item()
        alpha = (tf - f0) / (f1 - f0 + 1e-8)
        alpha_t = torch.tensor(alpha, device=device, dtype=dtype)

        # For rotations: we have J quaternions at frame idx0, idx1
        # Each block of size J in quat_sorted_flat corresponds to a single frame
        start0 = idx0 * J
        start1 = idx1 * J
        qa = quat_sorted_flat[start0 : start0 + J]  # shape (J, 4)
        qb = quat_sorted_flat[start1 : start1 + J]  # shape (J, 4)

        # Broadcast alpha_t for SLERP: shape (J, 1)
        alpha_broadcast = alpha_t.view(1, 1).expand(J, 1)
        q_interp = slerp(qa, qb, alpha_broadcast)  # shape (J, 4)

        # Convert interpolated quaternions back to rotation matrices
        rot_interp = quat_to_rotmat(q_interp)  # shape (J, 3, 3)

        # Linear interpolation for camera
        Ta = cam_sorted[idx0]  # shape (3,)
        Tb = cam_sorted[idx1]  # shape (3,)
        cam_interp = Ta * (1 - alpha_t) + Tb * alpha_t

        return rot_interp, cam_interp

    # Compute interpolation for each target index
    out_rotmats_sorted = torch.zeros_like(out_rotmats)
    out_cam_sorted = torch.zeros_like(out_cam)
    for i in range(M):
        tf = target_sorted_vals[i].item()
        rot_i, cam_i = get_smpl_at_frame(tf)
        out_rotmats_sorted[i] = rot_i
        out_cam_sorted[i] = cam_i

    # Reorder back to original target_frame_idx order
    out_rotmats[target_sorted_idx] = out_rotmats_sorted
    out_cam[target_sorted_idx] = out_cam_sorted

    return out_rotmats, out_cam

# def interpolate_RT_tensor(
#     RT: torch.Tensor,
#     source_frame_idx: torch.Tensor,
#     target_frame_idx: torch.Tensor
# ) -> torch.Tensor:
#     if isinstance(source_frame_idx, list):
#         source_frame_idx = torch.tensor(source_frame_idx, device=RT.device, dtype=RT.dtype)
#     if isinstance(target_frame_idx, list):
#         target_frame_idx = torch.tensor(target_frame_idx, device=RT.device, dtype=RT.dtype)
#     assert source_frame_idx.shape[0] == RT.shape[0], ""
#     assert RT.shape[1:] == (4, 4), ""
#     sorted_vals, sorted_idx = torch.sort(source_frame_idx)
#     RT_sorted = RT[sorted_idx]

#     # R: (N, 3, 3), T: (N, 3)
#     R_list = RT_sorted[:, :3, :3]
#     T_list = RT_sorted[:, :3, 3]

#     quat_list = rotmat_to_quat(R_list)  # (N, 4)

#     M = target_frame_idx.shape[0]
#     out = torch.zeros((M, 4, 4), device=RT.device, dtype=RT.dtype)
#     out[:, 3, 3] = 1.0 

#     target_sorted_vals, target_sorted_idx = torch.sort(target_frame_idx)
#     out_sorted = out.clone()

#     n = sorted_vals.shape[0]

#     def slerp(quat_a: torch.Tensor, quat_b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         quat_a = quat_a / (quat_a.norm(dim=-1, keepdim=True) + 1e-8)
#         quat_b = quat_b / (quat_b.norm(dim=-1, keepdim=True) + 1e-8)

#         dot = (quat_a * quat_b).sum(dim=-1, keepdim=True)

#         neg_mask = (dot < 0).expand_as(quat_b)
#         quat_b = torch.where(neg_mask, -quat_b, quat_b)
#         dot = torch.abs(dot)

#         # cos(omega)
#         dot = torch.clamp(dot, max=1.0)
#         omega = torch.acos(dot)

#         eps = 1e-6
#         sin_omega = torch.sqrt(1 - dot * dot)
#         is_small_angle = sin_omega < eps

#         # slerp = (sin((1 - t) * omega) / sin(omega)) * quat_a + (sin(t * omega) / sin(omega)) * quat_b
#         alpha = torch.sin((1.0 - t) * omega) / (sin_omega + eps)
#         beta = torch.sin(t * omega) / (sin_omega + eps)
        
#         alpha_lin = 1.0 - t
#         beta_lin = t

#         alpha = torch.where(is_small_angle, alpha_lin, alpha)
#         beta = torch.where(is_small_angle, beta_lin, beta)

#         quat_out = alpha * quat_a + beta * quat_b
#         quat_out = quat_out / (quat_out.norm(dim=-1, keepdim=True) + 1e-8)
#         return quat_out
    
#     def get_extrinsic_at_frame(tf: float):
#         if tf <= sorted_vals[0]:
#             idx0, idx1 = 0, 1
#         elif tf >= sorted_vals[-1]:
#             idx0, idx1 = n - 2, n - 1
#         else:
#             # find source_frame_idx[i] <= tf < source_frame_idx[i+1]
#             idx1 = torch.where(sorted_vals > tf)[0][0]
#             idx0 = idx1 - 1

#         f0 = sorted_vals[idx0].item()
#         f1 = sorted_vals[idx1].item()
#         alpha = (tf - f0) / (f1 - f0 + 1e-8)
#         alpha = torch.tensor(alpha, device=RT.device, dtype=RT.dtype)

#         # slerp for rotation
#         qa = quat_list[idx0:idx0+1]  # (1,4)
#         qb = quat_list[idx1:idx1+1]  # (1,4)
#         q_interp = slerp(qa, qb, alpha[None])  # (1,4)

#         # linear for translation
#         Ta = T_list[idx0]
#         Tb = T_list[idx1]
#         T_interp = Ta * (1 - alpha) + Tb * alpha

#         # (4,4)
#         R_interp = quat_to_rotmat(q_interp)[0]  # (3,3)
#         extrinsic = torch.eye(4, device=RT.device, dtype=RT.dtype)
#         extrinsic[:3, :3] = R_interp
#         extrinsic[:3, 3] = T_interp
#         return extrinsic

#     #  target_sorted_vals 
#     for i in range(M):
#         tf = target_sorted_vals[i].item() 
#         out_sorted[i] = get_extrinsic_at_frame(tf)
#     out[target_sorted_idx] = out_sorted

#     return out


def eliminate_RT(
    R: Union[np.ndarray, torch.Tensor],
    T: Union[np.ndarray, torch.Tensor],
    body_model: nn.Module,
    global_orient: Optional[Union[np.ndarray, torch.Tensor]] = None,
    body_pose: Optional[Union[np.ndarray, torch.Tensor]] = None,
    transl: Optional[Union[np.ndarray, torch.Tensor]] = None,
    betas: Optional[Union[np.ndarray, torch.Tensor]] = None,
    pose2rot: bool = True,
    **kwargs,
):
    device = global_orient.device

    body_model_output = body_model(global_orient=global_orient,
                                   betas=betas,
                                   body_pose=body_pose,
                                   transl=transl,
                                   pose2rot=pose2rot)
    joints = body_model_output.joints
    transl_root = joints[:, 0]
    if pose2rot:
        global_orient_cam = rotmat_to_aa(
            torch.bmm(R.to(device), aa_to_rotmat(global_orient)))
    else:
        global_orient_cam = torch.bmm(R.to(device), global_orient.squeeze(1)).unsqueeze(1)
    RT = combine_RT(R, T).to(device)
    transl_homo = torch.cat([transl_root, torch.ones_like(transl)[:, :1]], 1)
    transl_homo = torch.bmm(RT, transl_homo[..., None])
    transl_cam = transl_homo[:, :3, 0] / transl_homo[:, 3:4, 0]

    params_cam = dict()
    params_cam['betas'] = betas
    params_cam['global_orient'] = global_orient_cam
    params_cam['body_pose'] = body_pose
    body_model_output = body_model(pose2rot=pose2rot, **params_cam)
    pelvis_shift = body_model_output.joints[:, 0]
    transl_cam = transl_cam - pelvis_shift
    return global_orient_cam, transl_cam


def apply_RT(
    R: Union[np.ndarray, torch.Tensor],
    T: Union[np.ndarray, torch.Tensor],
    body_model: nn.Module,
    global_orient: Optional[Union[np.ndarray, torch.Tensor]] = None,
    body_pose: Optional[Union[np.ndarray, torch.Tensor]] = None,
    transl: Optional[Union[np.ndarray, torch.Tensor]] = None,
    betas: Optional[Union[np.ndarray, torch.Tensor]] = None,
    rot_format: str = 'rotmat',
    **kwargs,
):
    device = global_orient.device
    if rot_format == 'rotmat':
        global_orient = global_orient.reshape(-1, 1, 3, 3)
        body_pose = body_pose.reshape(-1, 23, 3, 3)
    elif rot_format == 'aa':
        global_orient = aa_to_rotmat(global_orient)
        body_pose = aa_to_rotmat(body_pose)
    else:
        raise ValueError(f"Invalid rotation format: {rot_format}")
    with torch.cuda.amp.autocast(enabled=False):
        body_model_output = body_model(global_orient=global_orient,
                                    betas=betas,
                                    body_pose=body_pose,
                                    transl=transl,
                                    pose2rot=False)
    joints = body_model_output.joints
    transl_root = joints[:, 0]
    # R, T = convert_world_view(R, T)
    global_orient_world = torch.bmm(R.to(device), global_orient[:, 0]).unsqueeze(1)
    RT = combine_RT(R, T).to(device)
    transl_homo = torch.cat([transl_root, torch.ones_like(transl)[:, :1]], 1)
    transl_homo = torch.bmm(RT, transl_homo[..., None])
    transl_world = transl_homo[:, :3, 0] / transl_homo[:, 3:4, 0]

    params_world = dict()
    params_world['betas'] = betas
    params_world['global_orient'] = global_orient_world
    params_world['body_pose'] = body_pose
    params_world['pose2rot'] = False
    body_model_output = body_model(**params_world)
    pelvis_shift = body_model_output.joints[:, 0]
    transl_world = transl_world - pelvis_shift
    return global_orient_world, transl_world