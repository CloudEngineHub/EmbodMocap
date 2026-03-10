from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import cv2
import trimesh
import matplotlib.animation as animation

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from trimesh.creation import cylinder, icosphere


def create_skeleton_mesh(kp3d, bones, 
                         sphere_radius=0.01, 
                         cylinder_radius=0.005, 
                         sphere_color=[255, 255, 0],
                         cylinder_color=[128, 128, 128]):
    """
    Create a colored 3D mesh of keypoints and bones.
    
    :param kp3d: (n, J, 3) keypoint coordinates array with shape (n, J, 3), where n is batch size and J is joint count.
    :param bones: bone connection list, e.g. [[0, 1], [1, 2]].
    :param sphere_radius: sphere radius for keypoints.
    :param cylinder_radius: cylinder radius for bone links.
    :param sphere_color: sphere color in RGB, e.g. [255, 255, 0].
    :param cylinder_color: cylinder color in RGB, e.g. [128, 128, 128].
    :return: a merged trimesh Mesh object.
    """
    meshes = []

    for frame_idx in range(kp3d.shape[0]): 
        frame_kps = kp3d[frame_idx]

        for kp in frame_kps:
            sphere_mesh = icosphere(radius=sphere_radius)
            sphere_mesh.apply_translation(kp)
            sphere_mesh.visual.vertex_colors = sphere_color
            meshes.append(sphere_mesh)

        for bone in bones:
            idx1, idx2 = bone
            p1 = frame_kps[idx1]
            p2 = frame_kps[idx2]

            direction = p2 - p1
            length = np.linalg.norm(direction)

            if length > 0:
                cylinder_mesh = cylinder(
                    radius=cylinder_radius, 
                    segment=(p1, p2)
                )
                cylinder_mesh.visual.vertex_colors = cylinder_color
                meshes.append(cylinder_mesh)

    combined_mesh = trimesh.util.concatenate(meshes)

    return combined_mesh


def rotate_kp2d_90_anticlock(kp2d, src_shape):
    if kp2d.ndim != 3 or kp2d.shape[2] != 2:
        raise ValueError("Input kp2d must have shape (N, J, 2).")
    if len(src_shape) != 2:
        raise ValueError("image_shape must be a tuple of (H, W).")

    H, W = src_shape
    rotated_kp2d = np.zeros_like(kp2d)
    
    rotated_kp2d[..., 0] = kp2d[..., 1]         # x' = y
    rotated_kp2d[..., 1] = W - kp2d[..., 0]    # y' = W - x

    return rotated_kp2d


def root_centering(X, joint_type='coco'):
    """Center the root joint to the pelvis."""
    if joint_type != 'common' and X.shape[-2] == 14: return X
    
    conf = None
    if X.shape[-1] == 4:
        conf = X[..., -1:]
        X = X[..., :-1]
        
    if X.shape[-2] == 31:
        X[..., :17, :] = X[..., :17, :] - X[..., [12, 11], :].mean(-2, keepdims=True)
        X[..., 17:, :]  = X[..., 17:, :] - X[..., [19, 20], :].mean(-2, keepdims=True)
        
    elif joint_type == 'coco':
        X = X - X[..., [12, 11], :].mean(-2, keepdims=True)
    
    elif joint_type == 'common':
        X = X - X[..., [2, 3], :].mean(-2, keepdims=True)

    if conf is not None:
        X = torch.cat((X, conf), dim=-1)
    
    return X


def convert_kps(joints2d, src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()

    if isinstance(joints2d, np.ndarray):
        out_joints2d = np.zeros((*joints2d.shape[:-2], len(dst_names), joints2d.shape[-1]))
    else:
        out_joints2d = torch.zeros((*joints2d.shape[:-2], len(dst_names), joints2d.shape[-1]), device=joints2d.device)

    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            out_joints2d[..., idx, :] = joints2d[..., src_names.index(jn), :]

    return out_joints2d

def get_perm_idxs(src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()
    idxs = [src_names.index(h) for h in dst_names if h in src_names]
    return idxs

def get_mpii3d_test_joint_names():
    return [
        'headtop', # 'head_top',
        'neck',
        'rshoulder',# 'right_shoulder',
        'relbow',# 'right_elbow',
        'rwrist',# 'right_wrist',
        'lshoulder',# 'left_shoulder',
        'lelbow', # 'left_elbow',
        'lwrist', # 'left_wrist',
        'rhip', # 'right_hip',
        'rknee', # 'right_knee',
        'rankle',# 'right_ankle',
        'lhip',# 'left_hip',
        'lknee',# 'left_knee',
        'lankle',# 'left_ankle'
        'hip',# 'pelvis',
        'Spine (H36M)',# 'spine',
        'Head (H36M)',# 'head'
    ]

def get_mpii3d_joint_names():
    return [
        'spine3', # 0,
        'spine4', # 1,
        'spine2', # 2,
        'Spine (H36M)', #'spine', # 3,
        'hip', # 'pelvis', # 4,
        'neck', # 5,
        'Head (H36M)', # 'head', # 6,
        "headtop", # 'head_top', # 7,
        'left_clavicle', # 8,
        "lshoulder", # 'left_shoulder', # 9,
        "lelbow", # 'left_elbow',# 10,
        "lwrist", # 'left_wrist',# 11,
        'left_hand',# 12,
        'right_clavicle',# 13,
        'rshoulder',# 'right_shoulder',# 14,
        'relbow',# 'right_elbow',# 15,
        'rwrist',# 'right_wrist',# 16,
        'right_hand',# 17,
        'lhip', # left_hip',# 18,
        'lknee', # 'left_knee',# 19,
        'lankle', #left ankle # 20
        'left_foot', # 21
        'left_toe', # 22
        "rhip", # 'right_hip',# 23
        "rknee", # 'right_knee',# 24
        "rankle", #'right_ankle', # 25
        'right_foot',# 26
        'right_toe' # 27
    ]

def get_insta_joint_names():
    return [
        'OP RHeel',
        'OP RKnee',
        'OP RHip',
        'OP LHip',
        'OP LKnee',
        'OP LHeel',
        'OP RWrist',
        'OP RElbow',
        'OP RShoulder',
        'OP LShoulder',
        'OP LElbow',
        'OP LWrist',
        'OP Neck',
        'headtop',
        'OP Nose',
        'OP LEye',
        'OP REye',
        'OP LEar',
        'OP REar',
        'OP LBigToe',
        'OP RBigToe',
        'OP LSmallToe',
        'OP RSmallToe',
        'OP LAnkle',
        'OP RAnkle',
    ]

def get_insta_skeleton():
    return np.array(
        [
            [0 , 1],
            [1 , 2],
            [2 , 3],
            [3 , 4],
            [4 , 5],
            [6 , 7],
            [7 , 8],
            [8 , 9],
            [9 ,10],
            [2 , 8],
            [3 , 9],
            [10,11],
            [8 ,12],
            [9 ,12],
            [12,13],
            [12,14],
            [14,15],
            [14,16],
            [15,17],
            [16,18],
            [0 ,20],
            [20,22],
            [5 ,19],
            [19,21],
            [5 ,23],
            [0 ,24],
        ])

def get_staf_skeleton():
    return np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 5],
            [5, 6],
            [6, 7],
            [1, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [8, 12],
            [12, 13],
            [13, 14],
            [0, 15],
            [0, 16],
            [15, 17],
            [16, 18],
            [2, 9],
            [5, 12],
            [1, 19],
            [20, 19],
        ]
    )

def get_staf_joint_names():
    return [
        'OP Nose', # 0,
        'OP Neck', # 1,
        'OP RShoulder', # 2,
        'OP RElbow', # 3,
        'OP RWrist', # 4,
        'OP LShoulder', # 5,
        'OP LElbow', # 6,
        'OP LWrist', # 7,
        'OP MidHip', # 8,
        'OP RHip', # 9,
        'OP RKnee', # 10,
        'OP RAnkle', # 11,
        'OP LHip', # 12,
        'OP LKnee', # 13,
        'OP LAnkle', # 14,
        'OP REye', # 15,
        'OP LEye', # 16,
        'OP REar', # 17,
        'OP LEar', # 18,
        'Neck (LSP)', # 19,
        'Top of Head (LSP)', # 20,
    ]

def get_spin_joint_names():
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'OP RHip',        # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'OP LHip',        # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'OP REye',        # 15
        'OP LEye',        # 16
        'OP REar',        # 17
        'OP LEar',        # 18
        'OP LBigToe',     # 19
        'OP LSmallToe',   # 20
        'OP LHeel',       # 21
        'OP RBigToe',     # 22
        'OP RSmallToe',   # 23
        'OP RHeel',       # 24
        'rankle',         # 25
        'rknee',          # 26
        'rhip',           # 27
        'lhip',           # 28
        'lknee',          # 29
        'lankle',         # 30
        'rwrist',         # 31
        'relbow',         # 32
        'rshoulder',      # 33
        'lshoulder',      # 34
        'lelbow',         # 35
        'lwrist',         # 36
        'neck',           # 37
        'headtop',        # 38
        'hip',            # 39 'Pelvis (MPII)', # 39
        'thorax',         # 40 'Thorax (MPII)', # 40
        'Spine (H36M)',   # 41
        'Jaw (H36M)',     # 42
        'Head (H36M)',    # 43
        'nose',           # 44
        'leye',           # 45 'Left Eye', # 45
        'reye',           # 46 'Right Eye', # 46
        'lear',           # 47 'Left Ear', # 47
        'rear',           # 48 'Right Ear', # 48
    ]

def get_h36m_joint_names():
    return [
        'hip',  # 0
        'lhip',  # 1
        'lknee',  # 2
        'lankle',  # 3
        'rhip',  # 4
        'rknee',  # 5
        'rankle',  # 6
        'Spine (H36M)',  # 7
        'neck',  # 8
        'Head (H36M)',  # 9
        'headtop',  # 10
        'lshoulder',  # 11
        'lelbow',  # 12
        'lwrist',  # 13
        'rshoulder',  # 14
        'relbow',  # 15
        'rwrist',  # 16
    ]

'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'

def get_spin_skeleton():
    return np.array(
        [
            [0 , 1],
            [1 , 2],
            [2 , 3],
            [3 , 4],
            [1 , 5],
            [5 , 6],
            [6 , 7],
            [1 , 8],
            [8 , 9],
            [9 ,10],
            [10,11],
            [8 ,12],
            [12,13],
            [13,14],
            [0 ,15],
            [0 ,16],
            [15,17],
            [16,18],
            [21,19],
            [19,20],
            [14,21],
            [11,24],
            [24,22],
            [22,23],
            [0 ,38],
        ]
    )

def get_posetrack_joint_names():
    return [
        "nose",
        "neck",
        "headtop",
        "lear",
        "rear",
        "lshoulder",
        "rshoulder",
        "lelbow",
        "relbow",
        "lwrist",
        "rwrist",
        "lhip",
        "rhip",
        "lknee",
        "rknee",
        "lankle",
        "rankle"
    ]

def get_posetrack_original_kp_names():
    return [
        'nose',
        'head_bottom',
        'head_top',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]

def get_pennaction_joint_names():
   return [
       "headtop",   # 0
       "lshoulder", # 1
       "rshoulder", # 2
       "lelbow",    # 3
       "relbow",    # 4
       "lwrist",    # 5
       "rwrist",    # 6
       "lhip" ,     # 7
       "rhip" ,     # 8
       "lknee",     # 9
       "rknee" ,    # 10
       "lankle",    # 11
       "rankle"     # 12
   ]

def get_common_joint_names():
    return [
        "rankle",    # 0  "lankle",    # 0
        "rknee",     # 1  "lknee",     # 1
        "rhip",      # 2  "lhip",      # 2
        "lhip",      # 3  "rhip",      # 3
        "lknee",     # 4  "rknee",     # 4
        "lankle",    # 5  "rankle",    # 5
        "rwrist",    # 6  "lwrist",    # 6
        "relbow",    # 7  "lelbow",    # 7
        "rshoulder", # 8  "lshoulder", # 8
        "lshoulder", # 9  "rshoulder", # 9
        "lelbow",    # 10  "relbow",    # 10
        "lwrist",    # 11  "rwrist",    # 11
        "neck",      # 12  "neck",      # 12
        "headtop",   # 13  "headtop",   # 13
    ]

def get_coco_common_joint_names():
    return [
        "nose",      # 0
        "leye",      # 1
        "reye",      # 2
        "lear",      # 3
        "rear",      # 4
        "lshoulder", # 5
        "rshoulder", # 6
        "lelbow",    # 7
        "relbow",    # 8
        "lwrist",    # 9
        "rwrist",    # 10
        "lhip",      # 11
        "rhip",      # 12
        "lknee",     # 13
        "rknee",     # 14
        "lankle",    # 15
        "rankle",    # 16
        "neck",      # 17  "neck",      # 12
        "headtop",   # 18  "headtop",   # 13
    ]

def get_common_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 2 ],
            [ 8, 9 ],
            [ 9, 3 ],
            [ 2, 3 ],
            [ 8, 12],
            [ 9, 10],
            [12, 9 ],
            [10, 11],
            [12, 13],
        ]
    )

def get_coco_joint_names():
    return [
        "nose",      # 0
        "leye",      # 1
        "reye",      # 2
        "lear",      # 3
        "rear",      # 4
        "lshoulder", # 5
        "rshoulder", # 6
        "lelbow",    # 7
        "relbow",    # 8
        "lwrist",    # 9
        "rwrist",    # 10
        "lhip",      # 11
        "rhip",      # 12
        "lknee",     # 13
        "rknee",     # 14
        "lankle",    # 15
        "rankle",    # 16
    ]

def get_coco_skeleton():
    # 0  - nose,
    # 1  - leye,
    # 2  - reye,
    # 3  - lear,
    # 4  - rear,
    # 5  - lshoulder,
    # 6  - rshoulder,
    # 7  - lelbow,
    # 8  - relbow,
    # 9  - lwrist,
    # 10 - rwrist,
    # 11 - lhip,
    # 12 - rhip,
    # 13 - lknee,
    # 14 - rknee,
    # 15 - lankle,
    # 16 - rankle,
    return np.array(
        [
            [15, 13], 
            [13, 11],
            [16, 14], 
            [14, 12], 
            [11, 12], 
            [ 5, 11],
            [ 6, 12],
            [ 5, 6 ],
            [ 5, 7 ],
            [ 6, 8 ],
            [ 7, 9 ],
            [ 8, 10],
            [ 1, 2 ],
            [ 0, 1 ],
            [ 0, 2 ],
            [ 1, 3 ],
            [ 2, 4 ],
            [ 3, 5 ],
            [ 4, 6 ]
        ]
    )

def get_coco_body_skeleton():
    # 0  - nose,
    # 1  - leye,
    # 2  - reye,
    # 3  - lear,
    # 4  - rear,
    # 5  - lshoulder,
    # 6  - rshoulder,
    # 7  - lelbow,
    # 8  - relbow,
    # 9  - lwrist,
    # 10 - rwrist,
    # 11 - lhip,
    # 12 - rhip,
    # 13 - lknee,
    # 14 - rknee,
    # 15 - lankle,
    # 16 - rankle,
    return np.array(
        [
            [15, 13], 
            [13, 11],
            [16, 14], 
            [14, 12], 
            [11, 12], 
            [ 5, 11],
            [ 6, 12],
            [ 5, 6 ],
            [ 5, 7 ],
            [ 6, 8 ],
            [ 7, 9 ],
            [ 8, 10],
        ]
    )


def get_coco_bone_skeleton():
    # 0  - nose,
    # 1  - leye,
    # 2  - reye,
    # 3  - lear,
    # 4  - rear,
    # 5  - lshoulder,
    # 6  - rshoulder,
    # 7  - lelbow,
    # 8  - relbow,
    # 9  - lwrist,
    # 10 - rwrist,
    # 11 - lhip,
    # 12 - rhip,
    # 13 - lknee,
    # 14 - rknee,
    # 15 - lankle,
    # 16 - rankle,
    return np.array(
        [
            [15, 13], 
            [13, 11],
            [16, 14], 
            [14, 12], 
            [11, 12], 
            [ 5, 6 ],
            [ 5, 7 ],
            [ 6, 8 ],
            [ 7, 9 ],
            [ 8, 10],
            [ 1, 2 ],
            [ 0, 1 ],
            [ 0, 2 ],
            [ 1, 3 ],
            [ 2, 4 ],
        ]
    )

def get_mpii_joint_names():
    return [
        "rankle",    # 0
        "rknee",     # 1
        "rhip",      # 2
        "lhip",      # 3
        "lknee",     # 4
        "lankle",    # 5
        "hip",       # 6
        "thorax",    # 7
        "neck",      # 8
        "headtop",   # 9
        "rwrist",    # 10
        "relbow",    # 11
        "rshoulder", # 12
        "lshoulder", # 13
        "lelbow",    # 14
        "lwrist",    # 15
    ]

def get_mpii_skeleton():
    # 0  - rankle,
    # 1  - rknee,
    # 2  - rhip,
    # 3  - lhip,
    # 4  - lknee,
    # 5  - lankle,
    # 6  - hip,
    # 7  - thorax,
    # 8  - neck,
    # 9  - headtop,
    # 10 - rwrist,
    # 11 - relbow,
    # 12 - rshoulder,
    # 13 - lshoulder,
    # 14 - lelbow,
    # 15 - lwrist,
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 2, 6 ],
            [ 6, 3 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 9 ],
            [ 7, 12],
            [12, 11],
            [11, 10],
            [ 7, 13],
            [13, 14],
            [14, 15]
        ]
    )

def get_aich_joint_names():
    return [
        "rshoulder", # 0
        "relbow",    # 1
        "rwrist",    # 2
        "lshoulder", # 3
        "lelbow",    # 4
        "lwrist",    # 5
        "rhip",      # 6
        "rknee",     # 7
        "rankle",    # 8
        "lhip",      # 9
        "lknee",     # 10
        "lankle",    # 11
        "headtop",   # 12
        "neck",      # 13
    ]

def get_aich_skeleton():
    # 0  - rshoulder,
    # 1  - relbow,
    # 2  - rwrist,
    # 3  - lshoulder,
    # 4  - lelbow,
    # 5  - lwrist,
    # 6  - rhip,
    # 7  - rknee,
    # 8  - rankle,
    # 9  - lhip,
    # 10 - lknee,
    # 11 - lankle,
    # 12 - headtop,
    # 13 - neck,
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 9, 10],
            [10, 11],
            [12, 13],
            [13, 0 ],
            [13, 3 ],
            [ 0, 6 ],
            [ 3, 9 ]
        ]
    )

def get_3dpw_joint_names():
    return [
        "nose",      # 0
        "thorax",    # 1
        "rshoulder", # 2
        "relbow",    # 3
        "rwrist",    # 4
        "lshoulder", # 5
        "lelbow",    # 6
        "lwrist",    # 7
        "rhip",      # 8
        "rknee",     # 9
        "rankle",    # 10
        "lhip",      # 11
        "lknee",     # 12
        "lankle",    # 13
    ]

def get_3dpw_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 2, 3 ],
            [ 3, 4 ],
            [ 1, 5 ],
            [ 5, 6 ],
            [ 6, 7 ],
            [ 2, 8 ],
            [ 5, 11],
            [ 8, 11],
            [ 8, 9 ],
            [ 9, 10],
            [11, 12],
            [12, 13]
        ]
    )

def get_smplcoco_joint_names():
    return [
        "rankle",    # 0
        "rknee",     # 1
        "rhip",      # 2
        "lhip",      # 3
        "lknee",     # 4
        "lankle",    # 5
        "rwrist",    # 6
        "relbow",    # 7
        "rshoulder", # 8
        "lshoulder", # 9
        "lelbow",    # 10
        "lwrist",    # 11
        "neck",      # 12
        "headtop",   # 13
        "nose",      # 14
        "leye",      # 15
        "reye",      # 16
        "lear",      # 17
        "rear",      # 18
    ]

def get_smplcoco_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 12],
            [12, 9 ],
            [ 9, 10],
            [10, 11],
            [12, 13],
            [14, 15],
            [15, 17],
            [16, 18],
            [14, 16],
            [ 8, 2 ],
            [ 9, 3 ],
            [ 2, 3 ],
        ]
    )

def get_smpl_joint_names():
    return [
        'hips',            # 0
        'leftUpLeg',       # 1
        'rightUpLeg',      # 2
        'spine',           # 3
        'leftLeg',         # 4
        'rightLeg',        # 5
        'spine1',          # 6
        'leftFoot',        # 7
        'rightFoot',       # 8
        'spine2',          # 9
        'leftToeBase',     # 10
        'rightToeBase',    # 11
        'neck',            # 12
        'leftShoulder',    # 13
        'rightShoulder',   # 14
        'head',            # 15
        'leftArm',         # 16
        'rightArm',        # 17
        'leftForeArm',     # 18
        'rightForeArm',    # 19
        'leftHand',        # 20
        'rightHand',       # 21
        'leftHandIndex1',  # 22
        'rightHandIndex1', # 23
    ]

def get_smpl_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 0, 2 ],
            [ 0, 3 ],
            [ 1, 4 ],
            [ 2, 5 ],
            [ 3, 6 ],
            [ 4, 7 ],
            [ 5, 8 ],
            [ 6, 9 ],
            [ 7, 10],
            [ 8, 11],
            [ 9, 12],
            [ 9, 13],
            [ 9, 14],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 22],
            [21, 23],
        ]
    )


def draw_kps(image, keypoints, keypoint_names, skeleton, with_text=False, point_color=(0, 255, 0), line_color=(255, 0, 0), point_radius=5, line_thickness=2):
    keypoints = keypoints.copy()[:, :2].astype(np.int32)
    img = image.copy()
    for connection in skeleton:
        start_idx, end_idx = connection
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_point = tuple(keypoints[start_idx]) 
            end_point = tuple(keypoints[end_idx])      
            if np.all(start_point) and np.all(end_point):  
                cv2.line(img, start_point, end_point, line_color, line_thickness)

    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            cv2.circle(img, (x, y), point_radius, point_color, -1) 
            if with_text:
                cv2.putText(img, keypoint_names[i], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color, 1, cv2.LINE_AA) 

    return img


def smooth_and_interpolate(kp2d, confidence, confidence_threshold=0.5, sigma=2):
    """
    Smooth temporal keypoints and interpolate low-confidence values.
    
    Args:
        kp2d (np.ndarray): temporal keypoint data with shape (N, J, 2), where N is timesteps and J is joint count.
        confidence (np.ndarray): confidence scores with shape (N, J).
        confidence_threshold (float): confidence threshold; points below this value are interpolated.
        sigma (float): Gaussian smoothing sigma.
    
    Returns:
        smoothed_kp2d (np.ndarray): smoothed and interpolated keypoints with shape (N, J, 2).
    """
    N, J, _ = kp2d.shape
    smoothed_kp2d = np.copy(kp2d)

    for j in range(J):
        for d in range(2):
            keypoint_series = kp2d[:, j, d]
            confidence_series = confidence[:, j]

            high_conf_mask = confidence_series >= confidence_threshold

            if np.sum(high_conf_mask) < 2:
                continue

            valid_indices = np.where(high_conf_mask)[0]
            valid_values = keypoint_series[high_conf_mask]

            interp_func = interp1d(valid_indices, valid_values, kind='linear', bounds_error=False, fill_value="extrapolate")

            interpolated_series = interp_func(np.arange(N))

            keypoint_series[~high_conf_mask] = interpolated_series[~high_conf_mask]
            
            smoothed_kp2d[:, j, d] = keypoint_series

        smoothed_kp2d[:, j, 0] = gaussian_filter1d(smoothed_kp2d[:, j, 0], sigma=sigma)
        smoothed_kp2d[:, j, 1] = gaussian_filter1d(smoothed_kp2d[:, j, 1], sigma=sigma)

    return smoothed_kp2d


def visualize_kp3d_to_video(kp3d, skeletons, video_path="output.mp4",
                            image_size=(480, 480), fps=10, title="3D Keypoints Animation"):
    """
    Visualize a 3D keypoint sequence as a Matplotlib animation and save as mp4.

    Args:
        kp3d: 3D keypoint data with shape (N, J, 3), where N is frame count and J is joint count.
        skeletons: skeleton edges, e.g. [(0, 1), (1, 2), ...].
        video_path: output video path, default "output.mp4".
        image_size: image size in pixels, default (480, 480).
        fps: video FPS, default 10.
        title: video title, default "3D Keypoints Animation".

    Notes:
        Camera view is fixed across frames and does not rotate with time.
    """
    N, J, _ = kp3d.shape

    dpi = 100
    fig_width = image_size[0] / dpi
    fig_height = image_size[1] / dpi

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    all_points = kp3d.reshape(-1, 3)
    x_min, y_min, z_min = np.min(all_points, axis=0)
    x_max, y_max, z_max = np.max(all_points, axis=0)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    margin = 0.1 * max_range
    valid_points = set()
    for bone in skeletons:
        valid_points.add(bone[0])
        valid_points.add(bone[1])
    valid_points = list(valid_points)
    def update(frame):
        """
        Per-frame drawing update function
        """
        ax.cla()

        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{title} - Frame {frame + 1}/{N}")

        keypoints = kp3d[frame]  # (J, 3)

        ax.scatter(keypoints[valid_points, 0], keypoints[valid_points, 1], keypoints[valid_points, 2],
                   color='r', s=20, depthshade=True)

        for (i, j) in skeletons:
            if i >= J or j >= J:
                continue
            pt_i = keypoints[i]
            pt_j = keypoints[j]
            xs = [pt_i[0], pt_j[0]]
            ys = [pt_i[1], pt_j[1]]
            zs = [pt_i[2], pt_j[2]]
            ax.plot(xs, ys, zs, color='b', linewidth=2)

        ax.view_init(elev=20, azim=30)

    ani = animation.FuncAnimation(fig, update, frames=N, interval=1000/fps, repeat=False)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Matplotlib'), bitrate=1800)
    ani.save(video_path, writer=writer)

    plt.close(fig)
    print(f"Video wrote to {video_path}")


def triangulate_point(P1, P2, x1, x2):
    """
    Solve a single 3D point with linear triangulation.

    Args:
        P1: camera 1 projection matrix, shape (3, 4)
        P2: camera 2 projection matrix, shape (3, 4)
        x1: 2D keypoint in camera 1, shape (2,)
        x2: 2D keypoint in camera 2, shape (2,)

    Returns:
        X: triangulated 3D point, shape (3,)
    """
    A = np.zeros((4, 4))
    A[0] = x1[0] * P1[2, :] - P1[0, :]
    A[1] = x1[1] * P1[2, :] - P1[1, :]
    A[2] = x2[0] * P2[2, :] - P2[0, :]
    A[3] = x2[1] * P2[2, :] - P2[1, :]
    
    _, _, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[-1]
    X_homogeneous = X_homogeneous / X_homogeneous[3]
    return X_homogeneous[:3]


def triangulate_sequence(K1, K2, R1_seq, T1_seq, R2_seq, T2_seq, kp2d_1_seq, kp2d_2_seq):
    """
    Triangulate human 2D keypoints from stereo views into 3D keypoints.

    Args:
        K1: camera 1 intrinsics, shape (3, 3)
        K2: camera 2 intrinsics, shape (3, 3)
        R1_seq: camera 1 per-frame rotation matrices, shape (N, 3, 3)
        T1_seq: camera 1 per-frame translations, shape (N, 3)
        R2_seq: camera 2 per-frame rotation matrices, shape (N, 3, 3)
        T2_seq: camera 2 per-frame translations, shape (N, 3)
        kp2d_1_seq: camera 1 human 2D keypoints per frame, shape (N, J, 2)
        kp2d_2_seq: camera 2 human 2D keypoints per frame, shape (N, J, 2)

    Returns:
        kp3d: triangulated 3D keypoints, shape (N, J, 3)
    """
    N = kp2d_1_seq.shape[0]
    J = kp2d_1_seq.shape[1]

    kp3d = np.zeros((N, J, 3), dtype=np.float32)

    for n in range(N):
        P1 = K1 @ np.hstack((R1_seq[n], T1_seq[n].reshape(3, 1)))
        P2 = K2 @ np.hstack((R2_seq[n], T2_seq[n].reshape(3, 1)))

        for j in range(J):
            p1 = kp2d_1_seq[n, j]  # (2,)
            p2 = kp2d_2_seq[n, j]  # (2,)

            X = triangulate_point(P1, P2, p1, p2)
            kp3d[n, j] = X
    return kp3d
