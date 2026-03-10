from __future__ import annotations

import os.path as osp
from collections import defaultdict

import numpy as np

import scipy.signal as signal
from progress.bar import Bar

from ultralytics import YOLO
from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    get_track_id
)

VIS_THRESH = 0.3
BBOX_CONF = 0.5
TRACKING_THR = 0.1
MINIMUM_FRMAES = 30
MINIMUM_JOINTS = 6

class DetectionModel(object):
    def __init__(self, bbox_model_ckpt, pose_model_ckpt, vit_cfg, device):
        
        # ViTPose
        pose_model_cfg = osp.join(vit_cfg)
        self.pose_model = init_pose_model(pose_model_cfg, pose_model_ckpt, device=device.lower())
        
        # YOLO
        self.bbox_model = YOLO(bbox_model_ckpt)
        
        self.device = device
        self.initialize_tracking()
        
    def initialize_tracking(self, ):
        self.next_id = 0
        self.frame_id = 0
        self.pose_results_last = []
        self.tracking_results = {
            'id': [],
            'frame_id': [],
            'bbox': [],
            'keypoints': []
        }
        
    def xyxy_to_cxcys(self, bbox, s_factor=1.05):
        cx, cy = bbox[[0, 2]].mean(), bbox[[1, 3]].mean()
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200 * s_factor
        return np.array([[cx, cy, scale]])
        
    def compute_bboxes_from_keypoints(self, s_factor=1.2):
        X = self.tracking_results['keypoints'].copy()
        mask = X[..., -1] > VIS_THRESH
        bbox = np.zeros((len(X), 3))
        for i, (kp, m) in enumerate(zip(X, mask)):
            bb = [kp[m, 0].min(), kp[m, 1].min(),
                  kp[m, 0].max(), kp[m, 1].max()]
            cx, cy = [(bb[2]+bb[0])/2, (bb[3]+bb[1])/2]
            bb_w = bb[2] - bb[0]
            bb_h = bb[3] - bb[1]
            s = np.stack((bb_w, bb_h)).max()
            bb = np.array((cx, cy, s))
            bbox[i] = bb
        
        bbox[:, 2] = bbox[:, 2] * s_factor / 200.0
        self.tracking_results['bbox'] = bbox

    def detect(self, img, fps):
        # bbox detection
        bboxes = self.bbox_model.predict(
            img, device=self.device, classes=0, conf=BBOX_CONF, save=False, verbose=False
        )[0].boxes.xyxy.detach().cpu().numpy()
        if len(bboxes > 1):
            width = bboxes[:, 2] - bboxes[:, 0]  # x2 - x1
            height = bboxes[:, 3] - bboxes[:, 1]  # y2 - y1
            area = width * height
            max_area_idx = np.argmax(area)
            bboxes = bboxes[max_area_idx:max_area_idx+1]
        bboxes = [{'bbox': bbox} for bbox in bboxes]
        
        # keypoints detection
        pose_results, _ = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_results=bboxes,
            format='xyxy',
            return_heatmap=False,
            outputs=None)
        
        for pose_result in pose_results:
            n_valid = (pose_result['keypoints'][:, -1] > VIS_THRESH).sum()
            if n_valid < MINIMUM_JOINTS: continue
            
            _id = 0
            xyxy = pose_result['bbox']
            bbox = self.xyxy_to_cxcys(xyxy)
            
            self.tracking_results['id'].append(_id)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['bbox'].append(bbox)
            self.tracking_results['keypoints'].append(pose_result['keypoints'])
        
        self.frame_id += 1
        self.pose_results_last = pose_results
    
    
    def track(self, img, fps):
        
        # bbox detection
        bboxes = self.bbox_model.predict(
            img, device=self.device, classes=0, conf=BBOX_CONF, save=False, verbose=False
        )[0].boxes.xyxy.detach().cpu().numpy()
        bboxes = [{'bbox': bbox} for bbox in bboxes]
        
        # keypoints detection
        pose_results, _ = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_results=bboxes,
            format='xyxy',
            return_heatmap=False,
            outputs=None)
        
        # person identification
        pose_results, self.next_id = get_track_id(
            pose_results,
            self.pose_results_last,
            self.next_id,
            use_oks=False,
            tracking_thr=TRACKING_THR,
            use_one_euro=True,
            fps=fps)
        
        for pose_result in pose_results:
            n_valid = (pose_result['keypoints'][:, -1] > VIS_THRESH).sum()
            if n_valid < MINIMUM_JOINTS: continue
            
            _id = pose_result['track_id']
            xyxy = pose_result['bbox']
            bbox = self.xyxy_to_cxcys(xyxy)
            
            self.tracking_results['id'].append(_id)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['bbox'].append(bbox)
            self.tracking_results['keypoints'].append(pose_result['keypoints'])
        
        self.frame_id += 1
        self.pose_results_last = pose_results
    
    def process(self, fps):
        for key in ['id', 'frame_id', 'keypoints']:
            self.tracking_results[key] = np.array(self.tracking_results[key])
        self.compute_bboxes_from_keypoints()
            
        output = defaultdict(lambda: defaultdict(list))
        ids = np.unique(self.tracking_results['id'])
        for _id in ids:
            idxs = np.where(self.tracking_results['id'] == _id)[0]
            for key, val in self.tracking_results.items():
                if key == 'id': continue
                output[_id][key] = val[idxs]
        
        # Smooth bounding box detection
        ids = list(output.keys())
        for _id in ids:
            if len(output[_id]['bbox']) < MINIMUM_FRMAES:
                del output[_id]
                continue
            
            kernel = int(int(fps/2) / 2) * 2 + 1
            smoothed_bbox = np.array([signal.medfilt(param, kernel) for param in output[_id]['bbox'].T]).T
            output[_id]['bbox'] = smoothed_bbox
        
        return output