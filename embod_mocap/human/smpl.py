import numpy as np
import torch
from smplx.body_models import SMPL as _SMPL
from smplx.lbs import batch_rodrigues, vertices2joints
import inspect

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

SMPL_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand',
    'right_hand'
]

class SMPL(_SMPL):
    def __init__(self, extra_joints_regressor: str = None, **kwargs,) -> None:
        super(SMPL, self).__init__(**kwargs)
        if extra_joints_regressor is not None:
            joints_regressor_extra = torch.tensor(
                np.load(extra_joints_regressor), dtype=torch.float32)
            self.register_buffer('joints_regressor_extra',
                                 joints_regressor_extra)
            
    def forward(self, **kwargs):
        model_out = super(SMPL, self).forward(**kwargs)
        if hasattr(self, 'joints_regressor_extra'):
            joints = model_out.joints
            extra_joints = vertices2joints(self.joints_regressor_extra,
                                           model_out.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
            model_out.joints = joints
        return model_out