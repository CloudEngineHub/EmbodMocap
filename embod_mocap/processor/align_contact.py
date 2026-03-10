"""Contact-based alignment optimization: read contacts from xlsx, optimize human and camera poses to align with scene.
Scene remains fixed, human and camera share global transformation (z-rotation only + xy translation)"""

import numpy as np
import torch
import os
import argparse
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pytorch3d.loss import chamfer_distance
import open3d as o3d

from embod_mocap.human.configs import BMODEL
from embod_mocap.human.smpl import SMPL
from embod_mocap.processor.base import export_cameras_to_ply, combine_RT, write_warning_to_log, project_kp3d_to_2d
import torch.nn.functional as F


def read_scene_contacts_from_xlsx(xlsx_path, scene_name):
    """Read scene-level contact coordinates from the xlsx 'contact' sheet
    
    Args:
        xlsx_path: xlsx file path
        scene_name: scene name
    
    Returns:
        scene_contacts: list of [x, y, z] coordinates, or empty list if not found
    """
    if not os.path.exists(xlsx_path):
        print(f"Warning: {xlsx_path} not found")
        return []
    
    try:
        df_contact = pd.read_excel(xlsx_path, sheet_name='contact')
        
        matching_rows = df_contact[df_contact['scene'] == scene_name]
        
        if len(matching_rows) == 0:
            print(f"Info: No contact info found for scene {scene_name} in contact sheet")
            return []
        
        row = matching_rows.iloc[0]
        contacts_str = str(row.get('contacts', ''))
        
        if contacts_str == '' or contacts_str == 'nan':
            print(f"Info: No contacts found for scene {scene_name}")
            return []
        
        contacts = eval(contacts_str)
        print(f"Loaded {len(contacts)} scene-level contacts for {scene_name}")
        for i, coord in enumerate(contacts):
            print(f"  Contact[{i}]: {coord}")
        return contacts
        
    except ValueError:
        print(f"Info: No 'contact' sheet found in {xlsx_path}")
        return []
    except Exception as e:
        print(f"Warning: Error reading scene contacts: {e}")
        return []


def read_contacts_from_xlsx(xlsx_path, scene_folder, seq_name, scene_contacts=None):
    """Read contact data from xlsx, supporting multiple formats.
    
    Args:
        xlsx_path: xlsx file path
        scene_folder: scene folder path
        seq_name: sequence name
        scene_contacts: optional scene-level contact coordinate list
    
    Returns:
        contacts: list of [frame_idx, contact_coordinates]
        where contact_coordinates is a 3D point [x, y, z]
        
    Supported formats:
    1. Legacy format: [frame_idx, [x, y, z]] - inline coordinates
    2. New format: [frame_idx, contact_id] - references scene-level contact by integer ID
    """
    if not os.path.exists(xlsx_path):
        print(f"Warning: {xlsx_path} not found")
        return []
    
    df = pd.read_excel(xlsx_path)
    
    matching_rows = df[(df['scene_folder'] == scene_folder) & (df['seq_name'] == seq_name)]
    
    if len(matching_rows) == 0:
        print(f"Warning: No matching row found for {scene_folder}/{seq_name}")
        return []
    
    row = matching_rows.iloc[0]
    contacts_str = str(row.get('contacts', ''))
    
    if contacts_str == '' or contacts_str == 'nan':
        print(f"No contacts found for {scene_folder}/{seq_name}")
        return []
    
    try:
        contacts_raw = eval(contacts_str)
        print(f"Raw contacts data: {contacts_raw}")
        
        contacts = []
        for item in contacts_raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                print(f"Warning: Invalid contact format (expected length 2): {item}")
                continue
            
            frame_idx = item[0]
            second_element = item[1]
            
            if isinstance(second_element, int):
                contact_id = second_element
                
                if scene_contacts is None or len(scene_contacts) == 0:
                    print(f"Warning: contact_id={contact_id} format used but no scene_contacts available")
                    continue
                
                if contact_id < 0 or contact_id >= len(scene_contacts):
                    print(f"Warning: Invalid contact_id={contact_id}, scene has {len(scene_contacts)} contacts")
                    continue
                
                contact_coords = scene_contacts[contact_id]
                contacts.append([frame_idx, contact_coords])
                print(f"  Frame {frame_idx}: using scene contact[{contact_id}] = {contact_coords}")
                
            elif isinstance(second_element, (list, tuple)):
                if len(second_element) != 3:
                    print(f"Warning: Invalid contact coordinates (expected length 3): {second_element}")
                    continue
                
                contact_coords = list(second_element)
                contacts.append([frame_idx, contact_coords])
                print(f"  Frame {frame_idx}: using inline contact = {contact_coords}")
                
            else:
                print(f"Warning: Invalid second element type in contact: {type(second_element)}, value: {second_element}")
                continue
        
        print(f"Found {len(contacts)} valid contact points")
        return contacts
        
    except Exception as e:
        print(f"Error parsing contacts: {e}")
        import traceback
        traceback.print_exc()
        return []


def apply_transformation_to_smpl_and_cameras(smpl_params, cameras1, cameras2, z_rot, xy_trans, device='cuda'):
    """Apply a transformation to SMPL parameters and camera parameters.
    
    Args:
        smpl_params: SMPL parameter dictionary
        cameras1, cameras2: camera parameters
        z_rot: z-axis rotation angle
        xy_trans: xy translation
        device: torch device
    
    Returns:
        aligned_smpl_params, aligned_cameras1, aligned_cameras2
    """
    import torch
    from embod_mocap.human.utils.transforms import apply_RT
    from embod_mocap.human.configs import BMODEL
    from embod_mocap.human.smpl import SMPL
    
    cos_z, sin_z = np.cos(z_rot), np.sin(z_rot)
    rotation_matrix = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])
    translation = np.array([xy_trans[0], xy_trans[1], 0])
    
    device = torch.device(device)
    smpl_model = SMPL(
        model_path=BMODEL.FLDR,
        gender="neutral",
        extra_joints_regressor=BMODEL.JOINTS_REGRESSOR_EXTRA,
        create_transl=False
    ).to(device)
    
    global_orient = torch.from_numpy(smpl_params['global_orient']).float().to(device)
    body_pose = torch.from_numpy(smpl_params['body_pose']).float().to(device)
    transl = torch.from_numpy(smpl_params['transl']).float().to(device)
    betas = torch.from_numpy(smpl_params['betas']).float().to(device)
    
    R_transform = torch.from_numpy(rotation_matrix).float().to(device).unsqueeze(0).expand(len(global_orient), -1, -1)
    T_transform = torch.from_numpy(translation).float().to(device).unsqueeze(0).expand(len(global_orient), -1)
    
    transformed_global_orient, transformed_transl = apply_RT(
        R=R_transform,
        T=T_transform,
        body_model=smpl_model,
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        betas=betas,
        rot_format='rotmat'
    )
    
    aligned_smpl = smpl_params.copy()
    aligned_smpl['global_orient'] = transformed_global_orient.detach().cpu().numpy()
    aligned_smpl['transl'] = transformed_transl.detach().cpu().numpy()
    
    aligned_cameras1 = cameras1.copy()
    aligned_cameras2 = cameras2.copy()
    
    for cameras in [aligned_cameras1, aligned_cameras2]:
        if 'R' in cameras and 'T' in cameras:
            R_cam = cameras['R']  # [N, 3, 3]
            T_cam = cameras['T']  # [N, 3, 1] or [N, 3]
            
            if T_cam.ndim == 3:
                T_cam = T_cam.squeeze(-1)
            
            # R_world, T_world = convert_world_view(R_cam, T_cam)
            
            # transformed_R_world = rotation_matrix @ R_world  # [N, 3, 3]
            # transformed_T_world = (rotation_matrix @ T_world.T).T + translation  # [N, 3]
            
            # transformed_R, transformed_T = convert_world_view(transformed_R_world, transformed_T_world)
            
            transformed_R = rotation_matrix @ R_cam  # [N, 3, 3]
            transformed_T = (rotation_matrix @ T_cam.T).T + translation  # [N, 3]
            cameras['R'] = transformed_R
            cameras['T'] = transformed_T
    transform_matrix = combine_RT(rotation_matrix[None], translation[None])
    return aligned_smpl, aligned_cameras1, aligned_cameras2, transform_matrix


def optimize_contact_alignment(contact_pelvis, contact_points, device, 
                                   num_iters=200, lr=1e-2):
    """Optimize contact alignment with a PyTorch optimizer.
    
    Args:
        contact_pelvis: torch tensor [N, 3] pelvis positions at contact frames
        contact_points: numpy array [N, 3] contact point cloud
        device: torch device
        num_iters: number of optimization iterations
        lr: learning rate
    
    Returns:
        optimal_params: dict with optimized parameters
    """
    
    contact_points_tensor = torch.from_numpy(contact_points).float().to(device)
    
    if len(contact_pelvis) == 0:
        print("No contact pelvis points for optimization")
        return None
    
    z_rot = torch.tensor(0.0, requires_grad=True, device=device)
    xy_trans = torch.zeros(2, requires_grad=True, device=device)
    
    optimizer = torch.optim.Adam([z_rot, xy_trans], lr=lr)
    
    print(f"Starting PyTorch optimization with {len(contact_pelvis)} contact pelvis points...")
    
    def z_rot_to_rotmat(z_rot):
        cos_z, sin_z = torch.cos(z_rot), torch.sin(z_rot)
        return torch.stack([
            torch.stack([cos_z, -sin_z, torch.zeros_like(z_rot)]),
            torch.stack([sin_z, cos_z, torch.zeros_like(z_rot)]),
            torch.stack([torch.zeros_like(z_rot), torch.zeros_like(z_rot), torch.ones_like(z_rot)])
        ]).to(device)
    for iter_idx in range(num_iters):
        optimizer.zero_grad()
        
        rotation_matrix = z_rot_to_rotmat(z_rot)
        
        transformed_pelvis = contact_pelvis @ rotation_matrix.T
        
        transformed_pelvis = transformed_pelvis.clone()
        transformed_pelvis[:, :2] += xy_trans.unsqueeze(0)
        
        pelvis_xy = transformed_pelvis[:, :2]  # [N, 2]
        scene_xy = contact_points_tensor[:, :2]  # [N, 2]
        
        distances = (pelvis_xy - scene_xy) ** 2
        
        contact_loss = torch.mean(distances)
        
        # reg_loss = 0.1 * (z_rot ** 2) + 0.01 * torch.sum(xy_trans ** 2)
        
        total_loss = contact_loss # + reg_loss

        total_loss.backward()
        optimizer.step()
        
        if iter_idx % 20 == 0 or iter_idx == num_iters - 1:
            print(f"Iter {iter_idx + 1}/{num_iters}, Loss: {total_loss.item():.6f}, "
                  f"Contact Loss: {contact_loss.item():.6f}, "
                  f"z_rot: {z_rot.item():.3f}, xy_trans: [{xy_trans[0].item():.3f}, {xy_trans[1].item():.3f}]")
    
    return z_rot, xy_trans, total_loss


def optimize_z_translation(smpl_model, smpl_params, frame_ids, contact_points, 
                           kp2d_v1, kp2d_v2, kp2d_conf_v1, kp2d_conf_v2,
                           R1, T1, K1, R2, T2, K2, H, W, device,
                           num_iters=200, lr=1e-2, 
                           contact_weight=100.0, reproj_weight=10.0):
    """Optimize z-translation using contact-height and kp2d reprojection losses.
    
    Args:
        smpl_model: SMPL model
        smpl_params: SMPL parameter dictionary (already on device)
        frame_ids: index list of contact frames
        contact_points: numpy array [N, 3] contact point coordinates
        kp2d_v1, kp2d_v2: 2D keypoints [N_frames, N_joints, 2]
        kp2d_conf_v1, kp2d_conf_v2: 2D keypoint confidence [N_frames, N_joints, 1]
        R1, T1, K1: camera 1 parameters
        R2, T2, K2: camera 2 parameters
        H, W: image height and width
        device: torch device
        num_iters: number of optimization iterations
        lr: learning rate
        contact_weight: contact constraint weight
        reproj_weight: reprojection loss weight
    
    Returns:
        z_trans: optimized z-axis translation
    """
    contact_points_tensor = torch.from_numpy(contact_points).float().to(device)
    contact_z = contact_points_tensor[:, 2]  # [N_contacts]
    
    kp2d_v1 = torch.from_numpy(kp2d_v1).float().to(device)
    kp2d_v2 = torch.from_numpy(kp2d_v2).float().to(device)
    kp2d_conf_v1 = torch.from_numpy(kp2d_conf_v1).float().to(device)
    kp2d_conf_v2 = torch.from_numpy(kp2d_conf_v2).float().to(device)
    
    kp2d_v1_norm = kp2d_v1.clone()
    kp2d_v1_norm[..., 0] /= W
    kp2d_v1_norm[..., 1] /= H
    
    kp2d_v2_norm = kp2d_v2.clone()
    kp2d_v2_norm[..., 0] /= W
    kp2d_v2_norm[..., 1] /= H
    
    z_trans = torch.zeros(1, requires_grad=True, device=device)
    
    optimizer = torch.optim.Adam([z_trans], lr=lr)
    
    print(f"\nStarting Z-axis optimization with {len(frame_ids)} contact frames...")
    
    for iter_idx in range(num_iters):
        optimizer.zero_grad()
        
        transl_updated = smpl_params['transl'].clone()
        transl_updated[:, 2] += z_trans
        
        smpl_output = smpl_model(
            betas=smpl_params['betas'],
            body_pose=smpl_params['body_pose'],
            global_orient=smpl_params['global_orient'],
            transl=transl_updated,
            pose2rot=False
        )
        
        smpl_vertices = smpl_output.vertices
        smpl_joints = smpl_output.joints
        
        contact_vertices = smpl_vertices[frame_ids]  # [N_contacts, N_verts, 3]
        lowest_z = torch.min(contact_vertices[:, :, 2], dim=1)[0]  # [N_contacts]
        contact_loss = F.mse_loss(lowest_z, contact_z) * contact_weight
        
        pred_kp3d = smpl_joints[:, BMODEL.SMPL54_to_COCO]  # [N_frames, N_joints, 3]
        
        reproj_kp2d_v1 = project_kp3d_to_2d(K1, R1, T1, pred_kp3d)
        reproj_kp2d_v1[..., 0] /= W
        reproj_kp2d_v1[..., 1] /= H
        reproj_loss_v1 = torch.mean(((reproj_kp2d_v1 - kp2d_v1_norm) * kp2d_conf_v1) ** 2)
        
        reproj_kp2d_v2 = project_kp3d_to_2d(K2, R2, T2, pred_kp3d)
        reproj_kp2d_v2[..., 0] /= W
        reproj_kp2d_v2[..., 1] /= H
        reproj_loss_v2 = torch.mean(((reproj_kp2d_v2 - kp2d_v2_norm) * kp2d_conf_v2) ** 2)
        
        reproj_loss = (reproj_loss_v1 + reproj_loss_v2) * reproj_weight
        
        total_loss = contact_loss + reproj_loss
        
        total_loss.backward()
        optimizer.step()
        
        if iter_idx % 20 == 0 or iter_idx == num_iters - 1:
            print(f"Iter {iter_idx + 1}/{num_iters}, Total Loss: {total_loss.item():.6f}, "
                  f"Contact Loss: {contact_loss.item():.6f}, "
                  f"Reproj Loss: {reproj_loss.item():.6f}, "
                  f"z_trans: {z_trans.item():.4f}")
    
    return z_trans


def main():
    parser = argparse.ArgumentParser(description="Contact-based alignment optimization")
    parser.add_argument("input_folder", type=str, help="Path to sequence folder")
    parser.add_argument("--xlsx_path", type=str, required=True, help="Path to xlsx file with contact information")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num_iters", type=int, default=200, help="Number of optimization iterations")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--log_file", type=str, default=None, help="Path to the log file")
    parser.add_argument("--loss_threshold", type=float, default=0.03, help="Loss threshold")

    args = parser.parse_args()
    
    device = torch.device(args.device)
    seq_path = args.input_folder
    
    seq_name = os.path.basename(seq_path)
    scene_folder = os.path.dirname(seq_path)
    scene_name = os.path.basename(scene_folder)
    
    print(f"Processing: {scene_folder}/{seq_name}")
    print(f"Scene name: {scene_name}")
    
    scene_contacts = read_scene_contacts_from_xlsx(args.xlsx_path, scene_name)
    
    contacts = read_contacts_from_xlsx(args.xlsx_path, scene_folder, seq_name, scene_contacts)
    
    if not contacts:
        msg = "[align_contact] No contacts found, skipping alignment."
        print(msg)
        if args.log_file:
            write_warning_to_log(args.log_file, msg)
        return
    
    frame_ids = []
    all_contact_points = []
    for frame_id, contact_points in contacts:
        frame_ids.append(frame_id)
        all_contact_points.append(contact_points)
    
    all_contact_points = np.stack(all_contact_points, axis=0)
    
    smpl_model = SMPL(
        model_path=BMODEL.FLDR,
        gender="neutral",
        extra_joints_regressor=BMODEL.JOINTS_REGRESSOR_EXTRA,
        create_transl=False
    ).to(device)
    
    # File paths
    optim_params_path = os.path.join(seq_path, "optim_params.npz")
    cameras1_path = os.path.join(seq_path, "v1", "cameras.npz")
    cameras2_path = os.path.join(seq_path, "v2", "cameras.npz")
    output_smpl = "optim_params_aligned.npz"
    output_cameras1 = "cameras_aligned.npz"
    output_cameras2 = "cameras_aligned.npz"
    output_kp3d = "kp3d_aligned.ply"
    
    if not os.path.exists(optim_params_path):
        msg = f"[align_contact] {optim_params_path} not found. Please run step 13 first."
        print(msg)
        if args.log_file:
            write_warning_to_log(args.log_file, msg)
        return
    
    smpl_params = dict(np.load(optim_params_path, allow_pickle=True))
    for k, v in smpl_params.items():
        if isinstance(v, np.ndarray):
            smpl_params[k] = torch.from_numpy(v).float().to(device)
    smpl_params.pop('vertices', None)
    
    global_orient = smpl_params['global_orient'][frame_ids]
    transl = smpl_params['transl'][frame_ids]
    betas = smpl_params['betas']
    body_pose = smpl_params['body_pose'][frame_ids]
    
    smpl_output = smpl_model(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        pose2rot=False
    )
    smpl_joints = smpl_output.joints
    contact_pelvis = smpl_joints[:, 0] 
    print(f"Contact pelvis: {contact_pelvis}")

    z_rot, xy_trans, total_loss = optimize_contact_alignment(
        contact_pelvis, all_contact_points, device, 
        num_iters=args.num_iters, lr=args.lr
    )

    cameras1 = np.load(cameras1_path)
    cameras1 = dict(cameras1)
    cameras2 = np.load(cameras2_path)
    cameras2 = dict(cameras2)
    for k, v in smpl_params.items():
        if isinstance(v, torch.Tensor):
            smpl_params[k] = v.cpu().numpy()
    if total_loss.item() > args.loss_threshold:
        with open("align_contact_loss_too_high.txt", "a") as f:
            f.write(f"{scene_folder} {seq_name} {total_loss.item():.6f}\n")
        warn_msg = f"[align_contact] Total loss too high ({total_loss.item():.6f}); set z_rot and xy_trans to 0."
        print(warn_msg)
        if args.log_file:
            write_warning_to_log(args.log_file, warn_msg)
        z_rot = z_rot * 0
        xy_trans = xy_trans * 0
    
    z_rot = z_rot.detach().cpu().numpy()
    xy_trans = xy_trans.detach().cpu().numpy()
    aligned_smpl, aligned_cameras1, aligned_cameras2, offset_matrix = apply_transformation_to_smpl_and_cameras(
        smpl_params, cameras1, cameras2, z_rot, xy_trans, device=device
    )
    
    # Stage 2: Optimize Z translation using contact height and kp2d reprojection
    # print("\n" + "="*60)
    # print("Stage 2: Optimizing Z translation")
    # print("="*60)
    
    # Load keypoints2d data
    # smpl_params_v1_path = os.path.join(seq_path, "v1", "smpl_params.npz")
    # smpl_params_v2_path = os.path.join(seq_path, "v2", "smpl_params.npz")
    
    # if os.path.exists(smpl_params_v1_path) and os.path.exists(smpl_params_v2_path):
    #     smpl_params_v1 = np.load(smpl_params_v1_path, allow_pickle=True)
    #     smpl_params_v2 = np.load(smpl_params_v2_path, allow_pickle=True)
        
    #     # Extract kp2d and confidence
    #     kp2d_v1 = smpl_params_v1['keypoints2d'][..., :2]  # [N, J, 2]
    #     kp2d_v2 = smpl_params_v2['keypoints2d'][..., :2]  # [N, J, 2]
    #     kp2d_conf_v1 = smpl_params_v1['keypoints2d'][..., 2:3]  # [N, J, 1]
    #     kp2d_conf_v2 = smpl_params_v2['keypoints2d'][..., 2:3]  # [N, J, 1]
        
    #     # Convert aligned_smpl back to torch tensors for optimization
    #     aligned_smpl_torch = {}
    #     for k, v in aligned_smpl.items():
    #         if isinstance(v, np.ndarray):
    #             aligned_smpl_torch[k] = torch.from_numpy(v).float().to(device)
    #         else:
    #             aligned_smpl_torch[k] = v
        
    #     # Load camera parameters
    #     R1 = torch.from_numpy(aligned_cameras1['R']).float().to(device)
    #     T1 = torch.from_numpy(aligned_cameras1['T']).float().to(device)
    #     K1 = torch.from_numpy(aligned_cameras1['K'][:3, :3]).float().to(device)
        
    #     R2 = torch.from_numpy(aligned_cameras2['R']).float().to(device)
    #     T2 = torch.from_numpy(aligned_cameras2['T']).float().to(device)
    #     K2 = torch.from_numpy(aligned_cameras2['K'][:3, :3]).float().to(device)
        
        # H, W = 960, 720
        
        # # Run z optimization
        # z_trans = optimize_z_translation(
        #     smpl_model=smpl_model,
        #     smpl_params=aligned_smpl_torch,
        #     frame_ids=frame_ids,
        #     contact_points=all_contact_points,
        #     kp2d_v1=kp2d_v1,
        #     kp2d_v2=kp2d_v2,
        #     kp2d_conf_v1=kp2d_conf_v1,
        #     kp2d_conf_v2=kp2d_conf_v2,
        #     R1=R1, T1=T1, K1=K1,
        #     R2=R2, T2=T2, K2=K2,
        #     H=H, W=W,
        #     device=device,
        #     num_iters=args.num_iters,
        #     lr=args.lr,
        #     contact_weight=100.0,
        #     reproj_weight=10.0
        # )
        
        # Apply z translation to aligned SMPL and cameras
        # z_trans_np = z_trans.detach().cpu().numpy().item()
        # aligned_smpl['transl'][:, 2] += z_trans_np
        # aligned_cameras1['T'][:, 2] += z_trans_np
        # aligned_cameras2['T'][:, 2] += z_trans_np
        
        # print(f"\nApplied Z translation: {z_trans_np:.4f}")
    # else:
    #     print(f"Warning: Could not find keypoints2d data, skipping Z optimization")
    #     print(f"  Expected: {smpl_params_v1_path} and {smpl_params_v2_path}")
    
    output_path = os.path.join(seq_path, output_smpl)
    np.savez(output_path, **aligned_smpl)
    print(f"Saved aligned SMPL parameters to {output_path}")
    
    np.savez(os.path.join(seq_path, "v1", output_cameras1), **aligned_cameras1)
    RT1 = combine_RT(aligned_cameras1['R'], aligned_cameras1['T'])
    export_cameras_to_ply(RT1, os.path.join(seq_path, "v1", output_cameras1.replace('.npz', '.ply')))

    np.savez(os.path.join(seq_path, "v2", output_cameras2), **aligned_cameras2)
    RT2 = combine_RT(aligned_cameras2['R'], aligned_cameras2['T'])
    export_cameras_to_ply(RT2, os.path.join(seq_path, "v2", output_cameras2.replace('.npz', '.ply')))
    print("Saved aligned camera parameters")
    
    # Transform kp3d if it exists
    kp3d_path = os.path.join(seq_path, "kp3d.ply")
    
    if os.path.exists(kp3d_path):
        kp3d = o3d.io.read_triangle_mesh(kp3d_path)
        transformed_kp3d = kp3d.transform(offset_matrix[0])
        o3d.io.write_triangle_mesh(os.path.join(seq_path, output_kp3d), transformed_kp3d)

    contact_spheres = o3d.geometry.TriangleMesh()
    sphere_radius = 0.05  #
    
    for i, contact_point in enumerate(all_contact_points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(contact_point)
        
        colors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        sphere_color = colors[i % 3]
        sphere.paint_uniform_color(sphere_color)
        
        contact_spheres += sphere
    
    # contact_ply_path = os.path.join(seq_path, "contact.ply")
    # o3d.io.write_triangle_mesh(contact_ply_path, contact_spheres)
    # print(f"Saved contact points spheres to {contact_ply_path}")

if __name__ == "__main__":
    main()
