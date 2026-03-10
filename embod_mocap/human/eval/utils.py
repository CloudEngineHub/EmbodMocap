import numpy as np
import torch


def as_np_array(d):
    if isinstance(d, torch.Tensor):
        return d.cpu().numpy()
    elif isinstance(d, np.ndarray):
        return d
    else:
        return np.array(d)
    
@torch.no_grad()
def compute_global_metrics(batch, mask=None):
    """Follow WHAM, the input has skipped invalid frames
    Args:
        batch (dict): {
            "pred_j3d_glob": (F, J, 3) tensor
            "target_j3d_glob":
            "pred_verts_glob":
            "target_verts_glob":
        }
    Returns:
        global_metrics (dict): {
            "wa2_mpjpe": (F, ) numpy array
            "waa_mpjpe":
            "rte":
            "jitter":
            "fs":
        }
    """
    # All data is in global coordinates
    pred_j3d_glob = batch["pred_j3d_glob"].cpu()  # (..., J, 3)
    target_j3d_glob = batch["target_j3d_glob"].cpu()
    pred_verts_glob = batch["pred_verts_glob"].cpu()
    target_verts_glob = batch["target_verts_glob"].cpu()
    if mask is not None:
        mask = mask.cpu()
        pred_j3d_glob = pred_j3d_glob[mask].clone()
        target_j3d_glob = target_j3d_glob[mask].clone()
        pred_verts_glob = pred_verts_glob[mask].clone()
        target_verts_glob = target_verts_glob[mask].clone()
    assert "mask" not in batch

    seq_length = pred_j3d_glob.shape[0]

    # Use chunk to compare
    chunk_length = 100
    wa2_mpjpe, waa_mpjpe = [], []
    for start in range(0, seq_length, chunk_length):
        end = min(seq_length, start + chunk_length)

        target_j3d = target_j3d_glob[start:end].clone().cpu()
        pred_j3d = pred_j3d_glob[start:end].clone().cpu()

        w_j3d = first_align_joints(target_j3d, pred_j3d)
        wa_j3d = global_align_joints(target_j3d, pred_j3d)

        if False:
            from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines

            wis3d = make_wis3d(name="debug-metric_utils")
            add_motion_as_lines(target_j3d, wis3d, name="target_j3d")
            add_motion_as_lines(pred_j3d, wis3d, name="pred_j3d")
            add_motion_as_lines(w_j3d, wis3d, name="pred_w2_j3d")
            add_motion_as_lines(wa_j3d, wis3d, name="pred_wa_j3d")

        wa2_mpjpe.append(compute_jpe(target_j3d, w_j3d))
        waa_mpjpe.append(compute_jpe(target_j3d, wa_j3d))

    # Metrics
    m2mm = 1000
    wa2_mpjpe = np.concatenate(wa2_mpjpe) * m2mm
    waa_mpjpe = np.concatenate(waa_mpjpe) * m2mm

    # Additional Metrics
    rte = compute_rte(target_j3d_glob[:, 0].cpu(), pred_j3d_glob[:, 0].cpu()) * 1e2
    jitter = compute_jitter(pred_j3d_glob, fps=30)
    foot_sliding = compute_foot_sliding(target_verts_glob, pred_verts_glob) * m2mm

    global_metrics = {
        "wa2_mpjpe": wa2_mpjpe,
        "waa_mpjpe": waa_mpjpe,
        "rte": rte,
        "jitter": jitter,
        "fs": foot_sliding,
    }
    return global_metrics


def global_align_joints(gt_joints, pred_joints):
    """
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    s_glob, R_glob, t_glob = align_pcl(gt_joints.reshape(-1, 3), pred_joints.reshape(-1, 3))
    pred_glob = s_glob * torch.einsum("ij,tnj->tni", R_glob, pred_joints) + t_glob[None, None]
    return pred_glob


def compute_jpe(S1, S2):
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).mean(dim=-1).numpy()


def compute_perjoint_jpe(S1, S2):
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).numpy()


def first_align_joints(gt_joints, pred_joints):
    """
    align the first two frames
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    # (1, 1), (1, 3, 3), (1, 3)
    s_first, R_first, t_first = align_pcl(gt_joints[:2].reshape(1, -1, 3), pred_joints[:2].reshape(1, -1, 3))
    pred_first = s_first * torch.einsum("tij,tnj->tni", R_first, pred_joints) + t_first[:, None]
    return pred_first


def compute_error_accel(joints_gt, joints_pred, valid_mask=None, fps=None):
    """
    Use [i-1, i, i+1] to compute acc at frame_i. The acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries(-1, 0, +1) in the
    acceleration error will be zero'd out.
    Args:
        joints_gt : (F, J, 3)
        joints_pred : (F, J, 3)
        valid_mask : (F)
    Returns:
        error_accel (F-2) when valid_mask is None, else (F'), F' <= F-2
    """
    # (F, J, 3) -> (F-2) per-joint
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]
    normed = np.linalg.norm(accel_pred - accel_gt, axis=-1).mean(axis=-1)
    if fps is not None:
        normed = normed * fps**2

    if valid_mask is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(valid_mask)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)
        if new_vis.sum() == 0:
            print("Warning!!! no valid acceleration error to compute.")

    return normed[new_vis]


def compute_rte(target_trans, pred_trans):
    # Compute the global alignment
    _, rot, trans = align_pcl(target_trans[None, :], pred_trans[None, :], fix_scale=True)
    pred_trans_hat = (torch.einsum("tij,tnj->tni", rot, pred_trans[None, :]) + trans[None, :])[0]

    # Compute the entire displacement of ground truth trajectory
    disps, disp = [], 0
    for p1, p2 in zip(target_trans, target_trans[1:]):
        delta = (p2 - p1).norm(2, dim=-1)
        disp += delta
        disps.append(disp)

    # Compute absolute root-translation-error (RTE)
    rte = torch.norm(target_trans - pred_trans_hat, 2, dim=-1)

    # Normalize it to the displacement
    return (rte / disp).numpy()


def compute_jitter(joints, fps=30):
    """compute jitter of the motion
    Args:
        joints (N, J, 3).
        fps (float).
    Returns:
        jitter (N-3).
    """
    pred_jitter = torch.norm(
        (joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]) * (fps**3),
        dim=2,
    ).mean(dim=-1)

    return pred_jitter.cpu().numpy() / 10.0


def compute_foot_sliding(target_verts, pred_verts, thr=1e-2):
    """compute foot sliding error
    The foot ground contact label is computed by the threshold of 1 cm/frame
    Args:
        target_verts (N, 6890, 3).
        pred_verts (N, 6890, 3).
    Returns:
        error (N frames in contact).
    """
    assert target_verts.shape == pred_verts.shape
    assert target_verts.shape[-2] == 6890

    # Foot vertices idxs
    foot_idxs = [3216, 3387, 6617, 6787]

    # Compute contact label
    foot_loc = target_verts[:, foot_idxs]
    foot_disp = (foot_loc[1:] - foot_loc[:-1]).norm(2, dim=-1)
    contact = foot_disp[:] < thr

    pred_feet_loc = pred_verts[:, foot_idxs]
    pred_disp = (pred_feet_loc[1:] - pred_feet_loc[:-1]).norm(2, dim=-1)

    error = pred_disp[contact]

    return error.cpu().numpy()


def align_pcl(Y, X, weight=None, fix_scale=False):
    """align similarity transform to align X with Y using umeyama method
    X' = s * R * X + t is aligned with Y
    :param Y (*, N, 3) first trajectory
    :param X (*, N, 3) second trajectory
    :param weight (*, N, 1) optional weight of valid correspondences
    :returns s (*, 1), R (*, 3, 3), t (*, 3)
    """
    *dims, N, _ = Y.shape
    N = torch.ones(*dims, 1, 1) * N

    if weight is not None:
        Y = Y * weight
        X = X * weight
        N = weight.sum(dim=-2, keepdim=True)  # (*, 1, 1)

    # subtract mean
    my = Y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = X.sum(dim=-2) / N[..., 0]
    y0 = Y - my[..., None, :]  # (*, N, 3)
    x0 = X - mx[..., None, :]

    if weight is not None:
        y0 = y0 * weight
        x0 = x0 * weight

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = torch.eye(3).reshape(*(1,) * (len(dims)), 3, 3).repeat(*dims, 1, 1)
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S[neg, 2, 2] = -1

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fix_scale:
        s = torch.ones(*dims, 1, device=Y.device, dtype=torch.float32)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N  # (*, 1, 1)
        s = torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / var[..., 0]  # (*, 1)

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]  # (*, 3)

    return s, R, t
