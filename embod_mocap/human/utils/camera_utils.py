import torch

def pred_cam_to_full_cam(pred_cam,
                            center,
                            scale,
                            full_img_shape,
                            px=None,
                            py=None):
    """convert the camera parameters from the crop camera to the full camera.
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped
       img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, bbox_w = center[:, 0], center[:, 1], scale.squeeze()

    if px is None:
        px = img_w / 2.
    if py is None:
        py = img_h / 2.
    s = pred_cam[:, 0] + 1e-9
    tx = pred_cam[:, 1]
    ty = pred_cam[:, 2]
    trans_x = (2 * (cx - px) / (bbox_w * s)) + tx
    trans_y = (2 * (cy - py) / (bbox_w * s)) + ty
    s_full = s * bbox_w / full_img_shape.max(-1)[0]
    full_cam = torch.stack([s_full, trans_x, trans_y], dim=-1)
    return full_cam


def perspective_projection(points, rotation, translation, focal_length,
                           camera_center):
    """This function computes the perspective projection of a set of points.

    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def project_points_pred_cam(points_3d, camera, focal_length, img_res):
    """Perform orthographic projection of 3D points using the camera
    parameters, return projected 2D points in image plane.

    Notes:
        batch size: B
        point number: N
    Args:
        points_3d (Tensor([B, N, 3])): 3D points.
        camera (Tensor([B, 3])): camera parameters with the
            3 channel as (scale, translation_x, translation_y)
    Returns:
        points_2d (Tensor([B, N, 2])): projected 2D points
            in image space.
    """
    batch_size = points_3d.shape[0]
    device = points_3d.device
    cam_t = torch.stack([
        camera[:, 1], camera[:, 2], 2 * focal_length /
        (img_res * camera[:, 0] + 1e-9)
    ],
                        dim=-1)
    camera_center = torch.ones([batch_size, 2]).to(device) * (img_res / 2)
    rot_t = torch.eye(3, device=device,
                      dtype=points_3d.dtype).unsqueeze(0).expand(
                          batch_size, -1, -1)
    keypoints_2d = perspective_projection(points_3d,
                                          rotation=rot_t,
                                          translation=cam_t,
                                          focal_length=focal_length,
                                          camera_center=camera_center)
    return keypoints_2d


def project_points_focal_length_pixel(points_3d,
                                      focal_length,
                                      translation,
                                      img_res=None,
                                      camera_center=None):
    """Perform orthographic projection of 3D points using the camera
    parameters, return projected 2D points in image plane.

    Notes:
        batch size: B
        point number: N
    Args:
        points_3d (Tensor([B, N, 3])): 3D points.
        camera (Tensor([B, 3])): camera parameters with the
            3 channel as (scale, translation_x, translation_y)
    Returns:
        points_2d (Tensor([B, N, 2])): projected 2D points
            in image space.
    """
    batch_size = points_3d.shape[0]
    device = points_3d.device

    if camera_center is None:
        camera_center = torch.ones([batch_size, 2]).to(device) * (img_res / 2)
    rot_t = torch.eye(3, device=device,
                      dtype=points_3d.dtype).unsqueeze(0).expand(
                          batch_size, -1, -1)
    keypoints_2d = perspective_projection(points_3d,
                                          rotation=rot_t,
                                          translation=translation,
                                          focal_length=focal_length,
                                          camera_center=camera_center)
    return keypoints_2d