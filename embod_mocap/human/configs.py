import torch

import os.path as osp
root =  osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))


N_JOINTS = 17

    
class KEYPOINTS:
    NUM_JOINTS = N_JOINTS
    H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
    H36M_TO_J14 = H36M_TO_J17[:14]
    J17_TO_H36M = [14, 3, 4, 5, 2, 1, 0, 15, 12, 16, 13, 9, 10, 11, 8, 7, 6]
    TREE = [[5, 6], 0, 0, 1, 2, -1, -1, 5, 6, 7, 8, -1, -1, 11, 12, 13, 14, 15, 15, 15, 16, 16, 16]

    # STD scale for video noise
    S_BIAS = 1e-1
    S_JITTERING = 5e-2
    S_PEAK = 3e-1
    S_PEAK_MASK = 5e-3
    S_MASK = 0.03


class BMODEL:
    MAIN_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]    # reduced_joints
    SMPL54_to_COCO = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 46, 45, 4, 5, 7, 8]
    FLDR = f'{root}/body_models/smpl/'
    MEAN_PARAMS = f'{root}/body_models/smpl/smpl_mean_params.npz'
    JOINTS_REGRESSOR_H36M = f'{root}/body_models/smpl/J_regressor_h36m.npy'
    JOINTS_REGRESSOR_EXTRA = f'{root}/body_models/smpl/J_regressor_extra.npy'
    PARENTS = torch.tensor([
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])

__all__ = ["KEYPOINTS", "BMODEL"]
