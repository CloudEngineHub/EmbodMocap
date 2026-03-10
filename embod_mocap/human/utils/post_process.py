import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.ndimage._filters import _gaussian_kernel1d


def is_contact(kp3d, fps=30, velocity_threshold=0.05):
    dt = 1 / fps
    velocities = (kp3d[1:] - kp3d[:-1]) / dt  
    speeds = torch.linalg.norm(velocities, axis=-1) 
    contact = speeds < velocity_threshold
    contact = torch.cat([contact, contact[-1:]])  # (L,)
    return contact

def gaussian_smooth(x, sigma=3, dim=-1):
    kernel_smooth = _gaussian_kernel1d(sigma=sigma, order=0, radius=int(4 * sigma + 0.5))
    kernel_smooth = torch.from_numpy(kernel_smooth).float()[None, None].to(x)  # (1, 1, K)
    rad = kernel_smooth.size(-1) // 2

    x = x.transpose(dim, -1)
    x_shape = x.shape[:-1]
    x = rearrange(x, "... f -> (...) 1 f")  # (NB, 1, f)
    x = F.pad(x[None], (rad, rad, 0, 0), mode="replicate")[0]
    x = F.conv1d(x, kernel_smooth)
    x = x.squeeze(1).reshape(*x_shape, -1)  # (..., f)
    x = x.transpose(-1, dim)
    return x

def pp_static_joint(static_conf_logits, transl, pred_w_j3d):
    static_conf_logits = static_conf_logits[:,:-1]
    # Global FK
    L = pred_w_j3d.shape[1]
    joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    pred_j3d_static = pred_w_j3d.clone()[:, :, joint_ids]  # (B, L, J, 3)

    ######## update overall movement with static info, and make displacement ~[0,0,0]
    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]  # (B, L-1, J, 3)

    static_label_ = static_conf_logits > 0  # (B, L-1, J) # avoid non-contact frame
    static_conf_logits = static_conf_logits.float() - (~static_label_ * 1e6)  # fp16 cannot go through softmax
    is_static = static_label_.sum(dim=-1) > 0  # (B, L-1)

    pred_disp = pred_j_disp * static_conf_logits[..., None].softmax(dim=-2)  # (B, L-1, J, 3)
    pred_disp = pred_disp * is_static[..., None, None]  # (B, L-1, J, 3)
    pred_disp = pred_disp.sum(-2)  # (B, L-1, 3)
    ####################

    # # Overwrite results:
    # if False:  # for-loop
    #     post_w_transl = transl.clone()  # (B, L, 3)
    #     for i in range(1, L):
    #         post_w_transl[:, i:] -= pred_disp[:, i - 1 : i]
    # else:  # vectorized
    pred_w_transl = transl.clone()  # (B, L, 3)
    pred_w_disp = pred_w_transl[:, 1:] - pred_w_transl[:, :-1]  # (B, L-1, 3)
    pred_w_disp_new = pred_w_disp - pred_disp
    post_w_transl = torch.cumsum(torch.cat([pred_w_transl[:, :1], pred_w_disp_new], dim=1), dim=1)
    post_w_transl[..., 0] = gaussian_smooth(post_w_transl[..., 0], dim=-1)
    post_w_transl[..., 1] = gaussian_smooth(post_w_transl[..., 1], dim=-1)

    # Put the sequence on the ground by -min(y), this does not consider foot height, for o3d vis
    post_w_j3d = pred_w_j3d - pred_w_transl.unsqueeze(-2) + post_w_transl.unsqueeze(-2)
    # ground_z = post_w_j3d[..., 2].flatten(-2).min(dim=-1)[0]  # (B,)  Minimum y value
    # post_w_transl[..., 2] -= ground_z

    return post_w_transl