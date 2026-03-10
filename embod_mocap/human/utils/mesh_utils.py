import torch
import trimesh
import open3d as o3d
import numpy as np
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from t3drender.render.render_functions import render_rgb, render_depth_perspective, render_flow_perspective, render_segmentation
from t3drender.render.shaders import SegmentationShader
from embod_mocap.human.utils.segmentation import body_segmentation
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.ops import knn_points, sample_farthest_points
from t3drender.cameras import PerspectiveCameras
from t3drender.mesh_utils import join_batch_meshes_as_scene
from collections import defaultdict, deque

import cv2

def clip_mesh_z_with_colors(mesh, z_max=2.0):
    """
    Clip vertices in a PyTorch3D Meshes object where z > z_max and return a new mesh.
    Vertex colors are preserved.

    Args:
    - mesh: pytorch3d.structures.Meshes, input Meshes object, must contain vertex colors (TexturesVertex).
    - z_max: float, z-threshold; vertices above this value are removed.

    Returns:
    - new_mesh: pytorch3d.structures.Meshes, clipped Meshes object with vertex colors.
    """
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    if mesh.textures is None or not isinstance(mesh.textures, TexturesVertex):
        raise ValueError("Input mesh must have vertex colors as TexturesVertex.")
    verts_colors = mesh.textures.verts_features_packed()

    valid_mask = verts[:, 2] <= z_max
    valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

    new_verts = verts[valid_mask]
    new_colors = verts_colors[valid_mask]

    index_map = torch.full((verts.shape[0],), -1, dtype=torch.long, device=verts.device)
    index_map[valid_indices] = torch.arange(len(new_verts), device=verts.device)

    valid_faces_mask = (valid_mask[faces[:, 0]] & valid_mask[faces[:, 1]] & valid_mask[faces[:, 2]])  # (F,)
    new_faces = faces[valid_faces_mask]

    new_faces = index_map[new_faces]

    new_textures = TexturesVertex(verts_features=[new_colors])

    new_mesh = Meshes(verts=[new_verts], faces=[new_faces], textures=new_textures)
    
    return new_mesh


def get_valid_faces_with_remapped_indices(faces, verts_valid_ids):
    valid_id_set = set(verts_valid_ids)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(verts_valid_ids)}
    valid_faces = []
    for face in faces:
        if all(v in valid_id_set for v in face): 
            remapped_face = [old_to_new[v] for v in face]
            valid_faces.append(remapped_face)
    return torch.tensor(valid_faces).long()


def vis_smpl_cam(verts, body_model, pred_cam, device, batch_size=30, resolution=(512, 512), verbose=False, image_paths=None, bg_images=None, alpha=0.7):
    if image_paths is not None and bg_images is not None:
        assert len(image_paths) == len(bg_images)
    s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
    f = 50 # ndc focal length
    tz = f / s
    transl = torch.stack([tx, ty, tz], dim=1)
    verts = verts + transl[:, None]
    meshes = Meshes(verts=torch.Tensor(verts).to(device), faces=torch.Tensor(body_model.faces).to(device)[None].repeat_interleave(len(verts), dim=0))
    meshes.textures = TexturesVertex(torch.ones_like(meshes.verts_padded()))
    focal_length = f * max(resolution) / 2
    image_tensors = render_rgb(meshes, device=device, resolution=resolution, focal_length=focal_length, batch_size=batch_size, verbose=verbose)
    if image_paths is not None:
        for i, (image_path, image_tensor) in enumerate(zip(image_paths, image_tensors)):
            img_smp = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_smp = cv2.cvtColor(img_smp, cv2.COLOR_RGB2BGRA)
            if bg_images is not None:
                bg_image = bg_images[i]
                mask = (img_smp[..., 3] != 0).astype(int)
                im_smpl = img_smp[..., :3] * mask[..., None] * alpha + bg_image * (1 - alpha) * mask[..., None] + bg_image * (1 - mask[..., None])
                img_smp = im_smpl * mask[..., None] + bg_image * (1 - mask[..., None])
            img_smp = img_smp.astype(np.uint8)
            cv2.imwrite(image_path, img_smp)

def vis_smpl(smpl_verts, body_model, device, cameras, batch_size=30, resolution=(512, 512), verbose=False, image_paths=None, bg_images=None, alpha=0.7):
    if image_paths is not None and bg_images is not None:
        assert len(image_paths) == len(bg_images)
    smpl_meshes = Meshes(verts=torch.Tensor(smpl_verts).to(device), faces=torch.Tensor(body_model.faces).to(device)[None].repeat_interleave(len(smpl_verts), dim=0))
    smpl_meshes.textures = TexturesVertex(torch.ones_like(smpl_meshes.verts_padded()))
    image_tensors = render_rgb(smpl_meshes, device=device, resolution=resolution, cameras=cameras, batch_size=batch_size, verbose=verbose)
    
    if image_paths is not None:
        for i, (image_path, image_tensor) in enumerate(zip(image_paths, image_tensors)):
            img_smp = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_smp = cv2.cvtColor(img_smp, cv2.COLOR_RGB2BGRA)
            if bg_images is not None:
                bg_image = bg_images[i]
                mask = (img_smp[..., 3] != 0).astype(int)
                im_smpl = img_smp[..., :3] * mask[..., None] * alpha + bg_image * (1 - alpha) * mask[..., None] + bg_image * (1 - mask[..., None])
                img_smp = im_smpl * mask[..., None] + bg_image * (1 - mask[..., None])
            img_smp = img_smp.astype(np.uint8)
            cv2.imwrite(image_path, img_smp)
    return image_tensors

    
def vis_smpl_scene(smpl_verts, scene_mesh, body_model, device, cameras, lights=None, batch_size=30, resolution=(512, 512), verbose=False, image_paths=None):
    assert len(smpl_verts) == len(image_paths)
    smpl_meshes = Meshes(verts=torch.Tensor(smpl_verts).to(device), faces=torch.Tensor(body_model.faces).to(device)[None].repeat_interleave(len(smpl_verts), dim=0))
    smpl_meshes.textures = TexturesVertex(torch.ones_like(smpl_meshes.verts_padded()))
    scene_mesh = scene_mesh.extend(len(smpl_verts))
    meshes = join_batch_meshes_as_scene([smpl_meshes, scene_mesh])
    image_tensors = render_rgb(meshes, lights=lights, device=device, resolution=resolution, cameras=cameras, batch_size=batch_size, verbose=verbose)
    if image_paths is not None:
        for image_path, image_tensor in zip(image_paths, image_tensors):
            image_tensor = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            image_tensor = cv2.cvtColor(image_tensor, cv2.COLOR_RGB2BGRA)
            cv2.imwrite(image_path, image_tensor)

def render_smpl_depth(verts, body_model, device, f, no_grad=False, resolution=(512, 512)):
    meshes = Meshes(verts=torch.Tensor(verts).to(device), faces=torch.Tensor(body_model.faces).to(device)[None].repeat_interleave(len(verts), dim=0))
    meshes.textures = TexturesVertex(torch.ones_like(meshes.verts_padded()))
    image_tensors = render_depth_perspective(meshes, device=device, resolution=resolution, focal_length=f, batch_size=30, no_grad=no_grad, verbose=False)
    return image_tensors

def render_smpl_depth_parts(verts, body_model, device, f, parts, no_grad=False, resolution=(512, 512)):
    segger = body_segmentation('smpl')
    valid_vert_ids = []
    for pname in parts:
        valid_vert_ids.extend(segger[pname])
    faces_new = get_valid_faces_with_remapped_indices(body_model.faces, valid_vert_ids)
    meshes = Meshes(verts=torch.Tensor(verts[:, valid_vert_ids]).to(device), faces=faces_new.to(device)[None].repeat_interleave(len(verts), dim=0))
    image_tensors = render_depth_perspective(meshes, device=device, resolution=resolution, focal_length=f, batch_size=30, no_grad=no_grad, verbose=False)
    return image_tensors

def render_smpl_flow(verts_source, verts_target, body_model, device, f, no_grad=False, resolution=(512, 512)):
    meshes_source = Meshes(verts=torch.Tensor(verts_source).to(device), faces=torch.Tensor(body_model.faces).to(device)[None].repeat_interleave(len(verts_source), dim=0))
    meshes_target = Meshes(verts=torch.Tensor(verts_target).to(device), faces=torch.Tensor(body_model.faces).to(device)[None].repeat_interleave(len(verts_target), dim=0))
    hfov = 2 * torch.arctan(1/f) * 180 / torch.pi
    image_tensors = render_flow_perspective(meshes_source=meshes_source, meshes_target=meshes_target, device=device, resolution=resolution, hfov=hfov, batch_size=30, no_grad=no_grad, verbose=False)
    return image_tensors

def get_smpl_visbile_verts(verts, body_model, device, focal_length, human_mask, parts=None, resolution=(512, 512)):
    rasterizer = MeshRasterizer(
        raster_settings=RasterizationSettings(
            image_size=resolution,
            bin_size=0,
            blur_radius=0,
            faces_per_pixel=1,
            perspective_correct=False))
    if parts is not None:
        segger = body_segmentation('smpl')
        valid_vert_ids = []
        for pname in parts:
            valid_vert_ids.extend(segger[pname])
        faces_new = get_valid_faces_with_remapped_indices(body_model.faces, valid_vert_ids)
        meshes = Meshes(verts=torch.Tensor(verts[:, valid_vert_ids]), faces=faces_new[None].repeat_interleave(len(verts), dim=0).to(device))
    else:
        meshes = Meshes(verts=torch.Tensor(verts), faces=torch.Tensor(faces)[None].repeat_interleave(len(verts), dim=0))
    meshes = meshes.to(device)
    h, w = resolution
    K = torch.eye(3, 3)[None]
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 0, 2] = w / 2
    K[:, 1, 2] = h / 2
    cameras = PerspectiveCameras(in_ndc=False, K=K, convention='opencv', resolution=resolution).to(device)
    fragments = rasterizer(meshes_world=meshes, cameras=cameras)
    meshes.textures = TexturesVertex(torch.ones_like(meshes.verts_padded()))

    shader = SegmentationShader()
    smpl_valid_mask = shader(fragments=fragments,
                        meshes=meshes,
                        cameras=cameras).squeeze().bool()
    valid_mask = smpl_valid_mask * human_mask
    pix_to_face = fragments.pix_to_face.squeeze()
    pix_to_face = pix_to_face * valid_mask + (-1) * (~valid_mask)
    visible_verts_per_mesh = []
    
    faces = meshes.faces_padded()[0]
    verts_left = meshes.verts_padded()
    face_offsets = meshes.mesh_to_faces_packed_first_idx()
    for batch_idx in range(pix_to_face.shape[0]):
        pix_to_face_batch = pix_to_face[batch_idx] # (H, W, faces_per_pixel)
        visible_faces = pix_to_face_batch[pix_to_face_batch >= 0].unique() - face_offsets[batch_idx] # (N_visible_faces,)
        visible_verts_ids = faces[visible_faces].unique()  # (N_visible_verts,)
        visible_verts_per_mesh.append(verts_left[batch_idx, visible_verts_ids])

    return visible_verts_per_mesh, smpl_valid_mask


def filter_and_sample_points(points, num_points, method="random", k=5, percentage=0.1):
    """
    Remove outliers from a point cloud and retain a specified number of points.
    
    Args:
        points (torch.Tensor): Input point cloud of shape (N, D), where N is the number of points 
                               and D is the dimension (usually 3 for 3D coordinates).
        num_points (int): The desired number of points to retain after outlier removal.
        method (str): Sampling method, supports "fps" (farthest point sampling) or "random" (random sampling).
        k (int): Number of nearest neighbors used for outlier detection (default is 5).
        outlier_threshold (float): Threshold for outlier removal (in units of standard deviation).

    Returns:
        torch.Tensor: The filtered and downsampled point cloud of shape (num_points, D).
    """
    if points.ndimension() != 2:
        raise ValueError("points must be a 2D tensor with shape (N, D).")
    N, D = points.shape
    if N <= num_points: # random sample to num_points
        points = points[torch.randint(0, N, (num_points*1.2,))]
    
    if method not in ["fps", "random"]:
        raise ValueError("method must be one of ['fps', 'random'].")

    point_cloud = points.unsqueeze(0)  # shape: (1, N, 3)
    
    knn_result = knn_points(point_cloud, point_cloud, K=k)
    distances = knn_result.dists.squeeze(0)  # shape: (N, k)
    
    mean_distances = distances[:, 1:].mean(dim=1)  # shape: (N,)
    
    num_points_to_keep = int((1 - percentage) * point_cloud.size(1)) 
    _, sorted_indices = torch.sort(mean_distances)  
    inlier_indices = sorted_indices[:num_points_to_keep]  
    
    points_filtered = point_cloud.squeeze(0)[inlier_indices]
    # Step 2: Downsample the point cloud
    if points_filtered.shape[0] < num_points:
        points_sampled = points_filtered[torch.randint(0, points_filtered.shape[0], (num_points,))]
    else:
        if method == "fps":
            # Farthest point sampling
            points_sampled, _ = sample_farthest_points(points_filtered[None, ...], K=num_points)
            points_sampled = points_sampled[0]
        elif method == "random":
            # Random sampling
            indices = torch.randperm(points_filtered.shape[0])[:num_points]
            points_sampled = points_filtered[indices]
    return points_sampled

def concat_point_clouds(point_clouds):
    all_points = []
    all_colors = []

    for i, pc in enumerate(point_clouds):
        if pc.vertices is not None:
            all_points.append(pc.vertices)
        else:
            raise ValueError(f"PointCloud at index {i} has no vertices!")

        if pc.colors is not None:
            all_colors.append(pc.colors)
        else:
            num_points = len(pc.vertices)
            all_colors.append(np.ones((num_points, 4)))
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    merged_point_cloud = trimesh.points.PointCloud(merged_points, colors=merged_colors)
    return merged_point_cloud


def slice_mesh_o3d(mesh, remove_indices):
    """
    Remove selected vertices and dependent triangles while preserving vertex colors and normals.

    Args:
        mesh: open3d.geometry.TriangleMesh object with vertex colors (vertex_colors) and normals (vertex_normals).
        remove_indices: list or array of vertex indices to remove.

    Returns:
        new_mesh: new open3d.geometry.TriangleMesh with selected vertices and related triangles removed.
    """
    remove_set = set(remove_indices)
    
    new_vertices = []
    new_vertex_colors = []
    new_vertex_normals = []
    mapping = dict()  # old_index -> new_index
    
    vertices = np.asarray(mesh.vertices)
    if mesh.has_vertex_colors():
        vertex_colors = np.asarray(mesh.vertex_colors)
    if mesh.has_vertex_normals():
        vertex_normals = np.asarray(mesh.vertex_normals)
    
    for i, vertex in enumerate(vertices):
        if i in remove_set:
            continue
        new_index = len(new_vertices)
        mapping[i] = new_index
        new_vertices.append(vertex)
        if mesh.has_vertex_colors():
            new_vertex_colors.append(vertex_colors[i])
        if mesh.has_vertex_normals():
            new_vertex_normals.append(vertex_normals[i])
    
    new_triangles = []
    for tri in np.asarray(mesh.triangles):
        if tri[0] in mapping and tri[1] in mapping and tri[2] in mapping:
            new_triangles.append([mapping[tri[0]], mapping[tri[1]], mapping[tri[2]]])
    
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(np.array(new_vertices))
    new_mesh.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))
    
    if len(new_vertex_colors) > 0:
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(new_vertex_colors))
    if len(new_vertex_normals) > 0:
        new_mesh.vertex_normals = o3d.utility.Vector3dVector(np.array(new_vertex_normals))
    
    return new_mesh

def slice_pointcloud_o3d(pointcloud, remove_indices):
    """
    Remove selected points from a point cloud while preserving colors and normals.

    Args:
        pointcloud: open3d.geometry.PointCloud object with point colors and point normals.
        remove_indices: list or array of point indices to remove.

    Returns:
        new_pointcloud: new open3d.geometry.PointCloud with selected points removed.
    """
    remove_set = set(remove_indices)
    
    new_points = []
    new_colors = []
    new_normals = []
    
    points = np.asarray(pointcloud.points)
    if pointcloud.has_colors():
        colors = np.asarray(pointcloud.colors)
    else:
        colors = None
    if pointcloud.has_normals():
        normals = np.asarray(pointcloud.normals)
    else:
        normals = None
    
    for i, point in enumerate(points):
        if i not in remove_set:
            new_points.append(point)
            if colors is not None:
                new_colors.append(colors[i])
            if normals is not None:
                new_normals.append(normals[i])
    
    new_pointcloud = o3d.geometry.PointCloud()
    new_pointcloud.points = o3d.utility.Vector3dVector(np.array(new_points))
    
    if len(new_colors) > 0:
        new_pointcloud.colors = o3d.utility.Vector3dVector(np.array(new_colors))
    if len(new_normals) > 0:
        new_pointcloud.normals = o3d.utility.Vector3dVector(np.array(new_normals))
    
    return new_pointcloud


def filter_mesh(mesh, threshold=10000):
    """
    Filter small disconnected components by vertex-index connectivity.
    
    Args:
        mesh: open3d.geometry.TriangleMesh
            input triangle mesh
        threshold: int
            minimum vertex count threshold for connected components
    
    Returns:
        mesh_filtered: open3d.geometry.TriangleMesh
            filtered triangle mesh
    """
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    adjacency_list = defaultdict(list)
    for tri in triangles:
        adjacency_list[tri[0]].extend([tri[1], tri[2]])
        adjacency_list[tri[1]].extend([tri[0], tri[2]])
        adjacency_list[tri[2]].extend([tri[0], tri[1]])

    visited = set()
    components = []

    def bfs(start):
        queue = deque([start])
        component = []
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                component.append(node)
                queue.extend(adjacency_list[node])
        return component

    for vertex in range(len(vertices)):
        if vertex not in visited:
            component = bfs(vertex)
            components.append(component)

    large_components = [comp for comp in components if len(comp) > threshold]
    large_vertices = set(v for comp in large_components for v in comp)

    mask = np.array([
        not all(v in large_vertices for v in tri)
        for tri in triangles
    ])
    triangles_to_remove = np.where(mask)[0]
    indices_to_remove = np.unique(triangles[triangles_to_remove].flatten())

    all_vertex_indices = np.arange(len(vertices))
    if len(triangles) > 0:
        used_vertices = np.unique(triangles.flatten())
    else:
        used_vertices = np.array([], dtype=int)
    unused_vertices = np.setdiff1d(all_vertex_indices, used_vertices, assume_unique=True)

    indices_to_remove = np.union1d(indices_to_remove, unused_vertices)

    mesh_filtered = slice_mesh_o3d(mesh, indices_to_remove)
    return mesh_filtered
