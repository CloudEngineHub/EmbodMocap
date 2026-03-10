import os
import argparse
import numpy as np

from embod_mocap.processor.base import combine_RT, rotate_R_around_z_axis, export_cameras_to_ply


def align_point_clouds(Y, X, weight=None, fix_scale=False, z_rot_only=False):
    """
    Align point cloud X to point cloud Y using Procrustes analysis.
    Returns scale, rotation matrix R, and translation t such that Y ≈ s * R @ X + t
    
    Args:
        Y: Target point cloud (N, 3)
        X: Source point cloud (N, 3) 
        weight: Optional weights for each point (N,)
        fix_scale: If True, force scale=1.0
        z_rot_only: If True, only allow rotation around z-axis
    
    Returns:
        s: Scale factor
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
    """
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    assert Y.shape == X.shape, f"Point clouds must have same shape, got {Y.shape} vs {X.shape}"
    assert Y.shape[-1] == 3, f"Last dimension must be 3, got {Y.shape[-1]}"
    
    if weight is not None:
        weight = np.asarray(weight, dtype=np.float64)
        weight = weight / weight.sum()
        my = np.sum(Y * weight[:, None], axis=0) 
        mx = np.sum(X * weight[:, None], axis=0)
        N = weight.sum()
    else:
        N = Y.shape[0]  
        my = np.mean(Y, axis=0)  
        mx = np.mean(X, axis=0)
    
    y0 = Y - my[None, :]  # (N, 3)
    x0 = X - mx[None, :]  # (N, 3)
    if weight is not None:
        C = (y0 * weight[:, None]).T @ x0  # (3, 3)
    else:
        C = y0.T @ x0  # (3, 3)
    
    if z_rot_only:
        # Only allow rotation around z-axis
        # Project the correlation matrix to only z-axis rotation
        angle = np.arctan2(C[1, 0] - C[0, 1], C[0, 0] + C[1, 1])
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    else:
        # Original full 3D rotation using SVD
        U, D, Vh = np.linalg.svd(C)
        S = np.eye(3)
        det_sign = np.linalg.det(U) * np.linalg.det(Vh.T)
        if det_sign < 0:
            S[2, 2] = -1
        
        R = U @ S @ Vh  # (3, 3)
    
    if fix_scale:
        s = 1.0
    else:
        # More robust scale calculation using least squares approach
        # Solve for s in: y0 = s * R @ x0 (in least squares sense)
        x0_rotated = (R @ x0.T).T  # Apply rotation to source points
        
        if weight is not None:
            # Weighted least squares: s = (y0^T W x0_rot) / (x0_rot^T W x0_rot)
            numerator = np.sum(np.sum(y0 * x0_rotated, axis=1) * weight)
            denominator = np.sum(np.sum(x0_rotated * x0_rotated, axis=1) * weight)
        else:
            # Unweighted least squares: s = (y0^T x0_rot) / (x0_rot^T x0_rot)
            numerator = np.sum(y0 * x0_rotated)
            denominator = np.sum(x0_rotated * x0_rotated)
        
        s = numerator / denominator if denominator > 1e-10 else 1.0
    
    t = my - s * (R @ mx)
    return s, R, t


def apply_rigid_to_RT(R, T, R_offset, T_offset, scale=1.0):
    """
    Apply rigid transformation to camera rotation and translation matrices.
    
    Args:
        R: Camera rotation matrices (N, 3, 3)
        T: Camera translation vectors (N, 3)
        R_offset: Rotation offset (3, 3)
        T_offset: Translation offset (3,)
        scale: Scale factor
    
    Returns:
        R_new: Transformed rotation matrices (N, 3, 3)
        T_new: Transformed translation vectors (N, 3)
    """
    R_new = np.einsum('ij,njk->nik', R_offset, R)
    T_new = scale * (R_offset @ T.T).T + T_offset
    return R_new, T_new


def align_sai_to_colmap(input_folder, z_rot_only=False, fix_scale=False, ):
    """
    Align SAI cameras to COLMAP cameras using the exact same logic as optim_human_cam.py
    and save the aligned results directly.
    
    Args:
        input_folder: Path to input folder containing v1 and v2 subdirectories
        z_rot_only: Whether to only optimize z-axis rotation
        fix_scale: Whether to fix scale to 1.0
    """
    print(f"Aligning SAI cameras to COLMAP for: {input_folder}")
    
    # Load SAI cameras for v1
    cameras_sai1 = np.load(os.path.join(input_folder, "v1", "cameras_sai_sliced.npz"))
    R1_sai = cameras_sai1["R"]
    T1_sai = cameras_sai1["T"].reshape(-1, 3)
    K1 = cameras_sai1["K"][:3, :3].astype(np.int32)
    
    # Load SAI cameras for v2
    cameras_sai2 = np.load(os.path.join(input_folder, "v2", "cameras_sai_sliced.npz"))
    R2_sai = cameras_sai2["R"]
    T2_sai = cameras_sai2["T"].reshape(-1, 3)
    K2 = cameras_sai2["K"][:3, :3].astype(np.int32)
    
    # Load COLMAP cameras for v1
    cameras_colmap1 = np.load(os.path.join(input_folder, "v1", "cameras_colmap.npz"))
    R1_colmap = cameras_colmap1["R"]
    T1_colmap = cameras_colmap1["T"].reshape(-1, 3)
    colmap_valid_ids_v1 = cameras_colmap1["valid_ids"]
    
    # Load COLMAP cameras for v2
    cameras_colmap2 = np.load(os.path.join(input_folder, "v2", "cameras_colmap.npz"))
    R2_colmap = cameras_colmap2["R"]
    T2_colmap = cameras_colmap2["T"].reshape(-1, 3)
    colmap_valid_ids_v2 = cameras_colmap2["valid_ids"]
    
    print(f"v1: SAI cameras={len(T1_sai)}, COLMAP cameras={len(T1_colmap)}, valid_ids={len(colmap_valid_ids_v1)}")
    print(f"v2: SAI cameras={len(T2_sai)}, COLMAP cameras={len(T2_colmap)}, valid_ids={len(colmap_valid_ids_v2)}")
    
    # EXACTLY the same alignment logic as optim_human_cam.py:
    # Align v1
    scale1, R1_offset_init, T1_offset_init = align_point_clouds(T1_colmap, T1_sai[colmap_valid_ids_v1], fix_scale=fix_scale, z_rot_only=z_rot_only)
    # Align v2
    scale2, R2_offset_init, T2_offset_init = align_point_clouds(T2_colmap, T2_sai[colmap_valid_ids_v2], fix_scale=fix_scale, z_rot_only=z_rot_only)
    
    print(f"v1 alignment: scale={scale1:.6f}")
    print(f"v2 alignment: scale={scale2:.6f}")
    
    # Apply alignment to all SAI cameras - EXACTLY like optim_human_cam.py
    R1_sai_aligned, T1_sai_aligned = apply_rigid_to_RT(R1_sai, T1_sai, R1_offset_init, T1_offset_init, scale1)
    R2_sai_aligned, T2_sai_aligned = apply_rigid_to_RT(R2_sai, T2_sai, R2_offset_init, T2_offset_init, scale2)
    
    # Save aligned cameras for v1
    cameras_aligned1 = {
        "R": R1_sai_aligned.astype(np.float32),
        "T": T1_sai_aligned.astype(np.float32),
        "K": K1.astype(np.float32),
        "scale": scale1,
    }
    np.savez(os.path.join(input_folder, "v1", "cameras_sai_transformed.npz"), **cameras_aligned1)
    RT1_aligned = combine_RT(R1_sai_aligned, T1_sai_aligned)
    export_cameras_to_ply(RT1_aligned, os.path.join(input_folder, "v1", "cameras_sai_transformed.ply"),)
    
    # Save aligned cameras for v2
    cameras_aligned2 = {
        "R": R2_sai_aligned.astype(np.float32),
        "T": T2_sai_aligned.astype(np.float32),
        "K": K2.astype(np.float32),
        "scale": scale2,
    }
    np.savez(os.path.join(input_folder, "v2", "cameras_sai_transformed.npz"), **cameras_aligned2)
    RT2_aligned = combine_RT(R2_sai_aligned, T2_sai_aligned)
    export_cameras_to_ply(RT2_aligned, os.path.join(input_folder, "v2", "cameras_sai_transformed.ply"),)
    
    # Print alignment quality metrics (T2_aligned_subset is the transformed SAI subset for RMSE calculation)
    T1_sai_subset = T1_sai[colmap_valid_ids_v1]
    T1_aligned_subset = R1_offset_init @ (scale1 * T1_sai_subset.T) + T1_offset_init[:, None]
    T1_aligned_subset = T1_aligned_subset.T  # This is T1_aligned_subset
    rmse1 = np.sqrt(np.mean((T1_aligned_subset - T1_colmap)**2))
    
    T2_sai_subset = T2_sai[colmap_valid_ids_v2]
    T2_aligned_subset = R2_offset_init @ (scale2 * T2_sai_subset.T) + T2_offset_init[:, None]
    T2_aligned_subset = T2_aligned_subset.T  # This is T2_aligned_subset
    rmse2 = np.sqrt(np.mean((T2_aligned_subset - T2_colmap)**2))
    
    print(f"Alignment RMSE: v1={rmse1:.6f}, v2={rmse2:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Align SAI cameras to COLMAP cameras")
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the input folder containing v1 and v2 subdirectories.",
    )
    parser.add_argument(
        "--z_rot_only",
        action="store_true",
        help="Whether to optimize only z-axis rotation (gravity-aligned).",
    )
    parser.add_argument(
        "--fix_scale",
        action="store_true",
        help="Whether to fix scale to 1.0 (no scaling).",
    )

    args = parser.parse_args()
    if args.fix_scale:
        print(f"Fixed scale to 1.0")
    else:
        print(f"Warning: Optimizing scale")

    align_sai_to_colmap(
        args.input_folder,
        z_rot_only=args.z_rot_only,
        fix_scale=args.fix_scale,
    )
