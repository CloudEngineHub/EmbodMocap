from __future__ import annotations

from collections import defaultdict

import torch
import math
import numpy as np
from progress.bar import Bar

from .backbone.hmr2 import hmr2
from .backbone.utils import process_image
from human.utils.imutils import flip_kp, flip_bbox
from PIL import Image


class PoseEstimator(object):
    def __init__(self, device, hmr2_ckpt,  flip_eval=False):
        
        self.device = device
        self.flip_eval = flip_eval

        self.model = hmr2(hmr2_ckpt).to(device).eval()

    def run(self, imgs, tracking_results, patch_h=256, patch_w=256):
        length = len(imgs)
        height, width = imgs[0].shape[:2]
        bar = Bar('HMR2.0 extraction ...', fill='#', max=length)
        with torch.no_grad():
            for frame_id, img in enumerate(imgs):
                
                for _id, val in tracking_results.items():
                    if not frame_id in val['frame_id']: 
                        continue
                    tracking_results[_id].setdefault('global_orient', [])
                    tracking_results[_id].setdefault('body_pose', [])
                    tracking_results[_id].setdefault('betas', [])
                    tracking_results[_id].setdefault('pred_cam', [])

                    frame_id2 = np.where(val['frame_id'] == frame_id)[0][0]
                    bbox = val['bbox'][frame_id2]
                    cx, cy, scale = bbox
                    
                    norm_img, crop_img = process_image(img[..., ::-1], [cx, cy], scale, patch_h, patch_w)
                    norm_img = torch.from_numpy(norm_img).unsqueeze(0).to(self.device)

                    global_orient, body_pose, betas, pred_cam = self.model(norm_img, encode=False)
                    tracking_results[_id]['global_orient'].append(global_orient)
                    tracking_results[_id]['body_pose'].append(body_pose)
                    tracking_results[_id]['betas'].append(betas)
                    tracking_results[_id]['pred_cam'].append(pred_cam)

                    if self.flip_eval:
                        flipped_bbox = flip_bbox(bbox, width, height)
                        tracking_results[_id]['flipped_bbox'].append(flipped_bbox)
                        
                        keypoints = val['keypoints'][frame_id2]
                        flipped_keypoints = flip_kp(keypoints, width)
                        tracking_results[_id]['flipped_keypoints'].append(flipped_keypoints)
                        
                        flipped_features = self.model(torch.flip(norm_img, (3, )), encode=True)
                        tracking_results[_id]['flipped_features'].append(flipped_features.cpu())
                        
                        if frame_id2 == 0:
                            tracking_results = self.predict_init(torch.flip(norm_img, (3, )), tracking_results, _id, flip_eval=True)
                        torch.cuda.empty_cache() 
                bar.next()
        return self.process(tracking_results)
    
    def process(self, tracking_results):
        output = defaultdict(dict)
        for _id, results in tracking_results.items():
            for key, val in results.items():
                if isinstance(val, list):
                    if isinstance(val[0], torch.Tensor):
                        val = torch.cat(val)
                    elif isinstance(val[0], np.ndarray):
                        val = np.array(val)
                output[_id][key] = val
        
        return output