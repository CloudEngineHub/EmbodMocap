import json
import subprocess
import numpy as np
import torch

import cv2
import imageio
from embod_mocap.thirdparty.lingbot_depth.mdm.model.v2 import MDMModel


def to_tensor_func(arr):
    """Convert a numpy array to a tensor with BCHW layout."""
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


CAM_CONVENTION_CHANGE = np.array([
    [1, 0, 0, 0],
    [0,-1, 0, 0],
    [0, 0,-1, 0],
    [0, 0, 0, 1]
])

INV_CAM_CONVENTION_CHANGE = CAM_CONVENTION_CHANGE 


def expand_to_rectangle(mask, filling=255, padding=0):
    coords = np.column_stack(np.where(mask > 0))
    
    if coords.size == 0:
        return np.zeros_like(mask)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(mask.shape[0] - 1, x_max + padding)
    y_max = min(mask.shape[1] - 1, y_max + padding)
    
    rect_mask = np.zeros_like(mask, dtype=np.uint8)
    rect_mask[x_min:x_max + 1, y_min:y_max + 1] = filling
    return rect_mask

def write_warning_to_log(log_file_path, warning_message):
    try:
        with open(log_file_path, 'r') as log_file:
            log_content = log_file.read()
            if warning_message in log_content:
                return
    except FileNotFoundError:
        pass

    with open(log_file_path, 'a') as log_file:
        log_file.write(warning_message + '\n')


def rotate_R_around_z_axis(R, theta):
    """
    Rotate extrinsic rotation matrices R (nx3x3) around the z axis by theta.
    :param R: Extrinsic rotation matrices with shape (n, 3, 3)
    :param theta: Rotation angle around z axis in radians
    :return: Rotated R and T with shapes (n, 3, 3) and (n, 3)
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    Rz = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0,          0,         1]
    ])
    
    n = R.shape[0]
    R_new = np.zeros_like(R)  # (n, 3, 3)
    
    for i in range(n):
        R_new[i] = R[i] @ Rz 
        
    return R_new

def convert_world_cam(R, T):
    if isinstance(R, np.ndarray):
        R = R.transpose(0, 2, 1)
    elif isinstance(R, torch.Tensor):
        R = R.transpose(1, 2)
    if T.ndim == 2:
        T = T[..., None]
    T = -R @ T
    return R, T.squeeze()

def batch_depthmap_to_pts3d_numpy(depth, K, R, T, H, W, scale):
    # R = np.transpose(R, (0, 2, 1))  # Change shape from (N, 3, 3) to (N, 3, 3)
    # T = - R @ T  # Change shape from (N, 3, 1) to (N, 3, 1)
    scale = int(scale)
    N = depth.shape[0]  # Batch size and depth map dimensions

    # Extract camera intrinsic matrix parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create pixel grid (u, v)
    u = np.arange(0, W, scale)  # Horizontal direction
    v = np.arange(0, H, scale)  # Vertical direction
    u, v = np.meshgrid(u, v)  # Create the grid
    pixel_grid = np.stack((u, v), axis=-1)  # (H, W, 2)
    pixel_grid = np.expand_dims(pixel_grid, axis=0).repeat(N, axis=0)  # (N, H, W, 2)
    # pixel_grid = pixel_grid[:, ::scale, ::scale]
    # Add an additional dimension to the depth map, shape becomes (N, H, W, 1)
    depth = np.expand_dims(depth, axis=-1)

    # Compute 3D points in the camera coordinate system
    pp = np.array([cx, cy]).reshape(1, 1, 2)  # Principal point (1, 1, 2)
    focal = np.array([fx, fy]).reshape(1, 1, 2)  # Focal lengths (1, 1, 2)
    points_camera = np.concatenate((depth * (pixel_grid - pp) / focal, depth), axis=-1)  # (N, H, W, 3)

    # Convert to the world coordinate system
    points_camera_flat = points_camera.reshape(N, -1, 3, 1)  # Flatten to (N, H*W, 3)
    points_world = np.einsum('nij,nmjk->nmik', R, points_camera_flat).squeeze() + T.reshape(-1, 1, 3)  # World coordinate transformation
    points_world = points_world.reshape(N, H//scale, W//scale, 3)  # Reshape back to (N, H, W, 3)

    return points_world

def project_kp3d_to_2d(K, R, T, kp3d):
    N, J, _ = kp3d.shape 
    kp3d_homo = torch.cat([kp3d, torch.ones((N, J, 1), device=kp3d.device)], dim=-1)

    extrinsics = torch.zeros((N, 3, 4), device=kp3d.device, dtype=kp3d.dtype)
    extrinsics[:, :, :3] = R
    extrinsics[:, :, 3] = T

    projection_matrices = torch.matmul(K, extrinsics)

    kp2d_homo = torch.einsum('nij,njk->nik', kp3d_homo, projection_matrices.transpose(1, 2))

    kp2d = kp2d_homo[..., :2] / kp2d_homo[..., 2:3]

    return kp2d

def project_3d_to_2d(K, R, T, kp3d):
    N, _ = kp3d.shape 
    kp3d_homo = torch.cat([kp3d, torch.ones((N, 1), device=kp3d.device)], dim=-1)

    extrinsics = torch.zeros((N, 3, 4), device=kp3d.device, dtype=kp3d.dtype)
    extrinsics[:, :, :3] = R
    extrinsics[:, :, 3] = T

    projection_matrices = torch.matmul(K, extrinsics)

    kp2d_homo = torch.einsum('nj,njk->nk', kp3d_homo, projection_matrices.transpose(1, 2))

    kp2d = kp2d_homo[..., :2] / kp2d_homo[..., 2:3]

    return kp2d


def read_jsonl_to_numpy(file_path):
    data = {
        "gyroscope": [],
        "magnetometer": [],
        "accelerometer": [],
    }

    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            if not 'sensor' in record:
                continue
            sensor_type = record["sensor"]["type"]
            values = record["sensor"]["values"]
            time = record["time"]

            if sensor_type in data:
                data[sensor_type].append(values + [time])

    for sensor_type in data:
        data[sensor_type] = np.array(data[sensor_type])

    return data

def combine_RT(R, T):
    num_frames = R.shape[0]
    if T.ndim == 2:
        T = T[:, :, np.newaxis]
    assert R.shape[1:] == (3, 3), "R should be of shape (num_frames, 3, 3)"
    assert T.shape[1:] == (3, 1), "T should be of shape (num_frames, 3, 1)"
    assert T.shape[0] == num_frames, "T should have the same number of frames as R"
    RT = np.zeros((num_frames, 4, 4))
    RT[:, :3, :3] = R
    RT[:, :3, 3:4] = T
    RT[:, 3, 3] = 1
    RT[:, 3, :3] = 0
    return RT


def run_cmd(cmd):
    try:
        subprocess.run(f"bash -c '{cmd}'", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"{e.returncode}")

def lingbotdepth_refine_batch(model, device, image_paths=None, images=None, prompt_depth_paths=None, prompt_depths=None):
    if images is None:
        images = torch.stack([load_image(path).to(device) for path in image_paths])
    else:
        images = images.to(device)

    if prompt_depths is None:
        prompt_depths = torch.stack([load_depth(path).to(device) for path in prompt_depth_paths])
    else:
        prompt_depths = prompt_depths.to(device)
    output = model.infer(
        images,
        depth_in=prompt_depths,
        intrinsics=None)
    return output['depth'].cpu().numpy()

def load_image_rotate(image_path_or_numpy, vertical=False, to_tensor=True, max_size=1008, multiple_of=14):
    '''
    Load image from path and convert to tensor.
    Ensure output height and width are multiples of multiple_of and long side <= max_size.
    '''
    if isinstance(image_path_or_numpy, str):
        image = np.asarray(imageio.imread(image_path_or_numpy)).astype(np.float32)
    else:
        image = image_path_or_numpy.astype(np.float32)
        
    if vertical:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = image / 255.

    h, w = image.shape[:2]
    max_size = max_size // multiple_of * multiple_of
    scale = min(1.0, max_size / max(h, w))
    tar_h = int(np.round(h * scale))
    tar_w = int(np.round(w * scale))
    tar_h = int(np.ceil(tar_h / multiple_of) * multiple_of)
    tar_w = int(np.ceil(tar_w / multiple_of) * multiple_of)
    # 3. resize
    image = cv2.resize(image, (tar_w, tar_h), interpolation=cv2.INTER_AREA)
    if to_tensor:
        return to_tensor_func(image)
    return image


def load_transform_json(json_path):
    with open(json_path, "r") as f:
        transforms = json.load(f)
    cx = transforms['cx']
    cy = transforms['cy']
    fx = transforms['fl_x']
    fy = transforms['fl_y']
    aabb_scale = transforms['aabb_scale']
    w = transforms['w']
    h = transforms['h']
    camera_angular_velocity = []
    camera_linear_velocity = []
    RT = []
    K = np.array([[fx, 0, cx, 0],
                  [0, fy, cy, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    for frame_id, frame_info in enumerate(transforms['frames']):
        RT.append(frame_info['transform_matrix'] @ CAM_CONVENTION_CHANGE)
    RT = np.array(RT)
    results = dict(RT=RT, K=K, aabb_scale=aabb_scale, w=w, h=h,
                   camera_angular_velocity=camera_angular_velocity,
                   camera_linear_velocity=camera_linear_velocity)
    return results


def export_cameras_to_ply(camera_matrices, output_file, scale=0.1, frustum_scale=0.05):
    """
    Export camera poses as a .ply file (axes + frustums) for visualization.
    
    Args:
        camera_matrices (numpy.ndarray): camera pose matrices with shape Nx4x4.
        output_file (str): output .ply file path.
        scale (float): axis scale for camera coordinate frames.
        frustum_scale (float): frustum scale controlling frustum size.
    """
    vertices = []
    edges = []

    for i, T in enumerate(camera_matrices):
        camera_center = T[:3, 3]
        R = T[:3, :3]

        x_axis = camera_center + scale * R[:, 0]
        y_axis = camera_center + scale * R[:, 1]
        z_axis = camera_center + scale * R[:, 2]

        frustum_far = camera_center + frustum_scale * R[:, 2]
        frustum_corners = [
            frustum_far + frustum_scale * (-R[:, 0] - R[:, 1]),
            frustum_far + frustum_scale * (R[:, 0] - R[:, 1]),
            frustum_far + frustum_scale * (R[:, 0] + R[:, 1]),
            frustum_far + frustum_scale * (-R[:, 0] + R[:, 1]),
        ]

        vertices.append((*camera_center, 255, 255, 255))
        vertices.append((*x_axis, 255, 255, 0))
        vertices.append((*y_axis, 0, 255, 0))
        vertices.append((*z_axis, 0, 0, 255))

        for corner in frustum_corners:
            vertices.append((*corner, 255, 0, 0))

        base_idx = i * 8
        edges.append((base_idx, base_idx + 1))
        edges.append((base_idx, base_idx + 2))
        edges.append((base_idx, base_idx + 3))

        for j in range(4):
            edges.append((base_idx, base_idx + 4 + j))

        edges.append((base_idx + 4, base_idx + 5))
        edges.append((base_idx + 5, base_idx + 6))
        edges.append((base_idx + 6, base_idx + 7))
        edges.append((base_idx + 7, base_idx + 4))

    with open(output_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for vertex in vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {vertex[3]} {vertex[4]} {vertex[5]}\n")

        for edge in edges:
            v1, v2 = edge
            color = vertices[v2][3:]
            f.write(f"{v1} {v2} {color[0]} {color[1]} {color[2]}\n")



def batch_ortho_rotmat(matrix):
    """
    Orthogonalizes a batch of rotation matrices using Gram-Schmidt.

    Args:
        matrix (torch.Tensor): A tensor of shape (N, 3, 3) representing a batch of 3D rotation matrices.

    Returns:
        torch.Tensor: A tensor of the same shape (N, 3, 3) with orthogonalized rotation matrices.
    """
    # Extract the columns of the matrix
    ori_shape = matrix.shape
    matrix = matrix.reshape(-1, 3, 3)
    x = matrix[:, :, 0]  # First column (N, 3)
    y = matrix[:, :, 1]  # Second column (N, 3)
    z = matrix[:, :, 2]  # Third column (N, 3)

    # Gram-Schmidt process
    x = torch.nn.functional.normalize(x, dim=-1)  # Normalize x (N, 3)
    y = y - (torch.sum(x * y, dim=-1, keepdim=True) * x)  # Make y orthogonal to x
    y = torch.nn.functional.normalize(y, dim=-1)  # Normalize y
    z = z - (torch.sum(x * z, dim=-1, keepdim=True) * x)  # Make z orthogonal to x
    z = z - (torch.sum(y * z, dim=-1, keepdim=True) * y)  # Make z orthogonal to y
    z = torch.nn.functional.normalize(z, dim=-1)  # Normalize z

    # Reconstruct the orthonormal matrix
    orthogonal_matrix = torch.stack((x, y, z), dim=-1)  # Shape: (N, 3, 3)
    return orthogonal_matrix.reshape(ori_shape)
